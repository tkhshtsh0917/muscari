"""A self-contained Top-K SAE training loop.

"Shizuku" refers to mahiyama's Japanese SAE-SPLADE models. Their training code is not public.
This is an independent implementation of the recipe that the SAE model card describes: a
Top-K SAE with AuxK, Adam, 5% linear warmup followed by cosine decay, and batches counted in
tokens. The defaults are the values reported on that card. The SAE mechanics (tied
initialisation, unit-norm decoder, AuxK, dead-latent bookkeeping) follow Gao et al. (2024).
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..devices import resolve_device
from ..sae import TopKSAE

CHECKPOINT_NAME = "checkpoint.pt"
CHECKPOINT_VERSION = 1

# A resumed run must agree with its checkpoint on these; the rest may change (for example
# ``total_tokens`` to train for longer).
_RESUME_KEYS = ("num_latents", "k", "batch_tokens")


@dataclass
class ShizukuTrainerConfig:
    """Hyperparameters of ``ShizukuTrainer``.

    Attributes:
        num_latents: dictionary size ``M``.
        k: active latents per token.
        aux_k: dead latents used by the auxiliary reconstruction; 0 disables AuxK.
        aux_coef: weight of the auxiliary loss.
        dead_after_tokens: a latent is dead once it has not fired for this many tokens.
        lr: peak learning rate of Adam.
        warmup_frac: share of the steps spent on linear warmup.
        adam_betas: Adam ``(beta1, beta2)``.
        adam_eps: Adam epsilon.
        total_tokens: training length in tokens; the number of steps is this over
            ``batch_tokens``.
        batch_tokens: activations per optimisation step.
        amp: run the SAE forward pass under bfloat16 autocast. Only used on CUDA.
        log_every: steps between log records.
        checkpoint_every: steps between checkpoints, with one more at the end; 0 writes none.
        seed: seed for the SAE initialisation.
        device: ``"auto"`` or a device string.
    """

    num_latents: int = 65_536
    k: int = 16
    aux_k: int = 256
    aux_coef: float = 1 / 32
    dead_after_tokens: int = 10_000_000
    lr: float = 5e-4
    warmup_frac: float = 0.05
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    total_tokens: int = 1_000_000_000
    batch_tokens: int = 8192
    amp: bool = False
    log_every: int = 100
    checkpoint_every: int = 0
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.num_latents < 1 or self.k < 1 or self.k > self.num_latents:
            raise ValueError("Need 1 <= k <= num_latents.")
        if self.aux_k < 0 or self.aux_coef < 0:
            raise ValueError("aux_k and aux_coef must not be negative.")
        if self.batch_tokens < 1 or self.total_tokens < self.batch_tokens:
            raise ValueError("Need batch_tokens >= 1 and total_tokens >= batch_tokens.")
        if self.lr <= 0 or not 0 <= self.warmup_frac < 1:
            raise ValueError("Need lr > 0 and 0 <= warmup_frac < 1.")
        if self.dead_after_tokens < 1 or self.log_every < 1 or self.checkpoint_every < 0:
            raise ValueError("Need dead_after_tokens >= 1, log_every >= 1 and checkpoint_every >= 0.")

    @property
    def total_steps(self) -> int:
        return self.total_tokens // self.batch_tokens

    @property
    def warmup_steps(self) -> int:
        return int(self.warmup_frac * self.total_steps)


def warmup_cosine(step: int, total_steps: int, warmup_steps: int) -> float:
    """Learning-rate multiplier for the 0-based ``step``: linear warmup, then cosine decay to zero."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class ShizukuTrainer:
    """Trains a ``TopKSAE`` on batches of activations.

    Args:
        cfg: hyperparameters.
        d_in: width of the activations.

    Attributes:
        sae: the SAE being trained, on ``device``.
        step: optimisation steps taken so far.
        tokens_seen: activations trained on so far.
        tokens_since_fired: per-latent count of tokens since the latent last fired.
    """

    def __init__(self, cfg: ShizukuTrainerConfig, d_in: int) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)

        torch.manual_seed(cfg.seed)
        self.sae = TopKSAE(d_in, cfg.num_latents, cfg.k).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.sae.parameters(),
            lr=cfg.lr,
            betas=cfg.adam_betas,
            eps=cfg.adam_eps,
        )

        self.tokens_since_fired = torch.zeros(cfg.num_latents, dtype=torch.long, device=self.device)
        self.step = 0
        self.tokens_seen = 0

    def dead_mask(self) -> torch.Tensor:
        """Boolean ``(num_latents,)`` mask of the latents currently considered dead."""
        return self.tokens_since_fired > self.cfg.dead_after_tokens

    def train_step(self, x: torch.Tensor) -> dict[str, float]:
        """Take one optimisation step on ``x`` ``(batch, d_in)`` and return its metrics.

        The first step also initialises ``b_dec`` from ``x``. ``dead_frac`` describes the
        dictionary before this step; ``lr`` is the rate this step used.
        """
        cfg = self.cfg
        x = x.to(self.device)
        if self.step == 0:
            self.sae.init_b_dec(x)

        learning_rate = cfg.lr * warmup_cosine(self.step, cfg.total_steps, cfg.warmup_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

        dead = self.dead_mask()
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if cfg.amp and self.device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            out = self.sae.loss(x, dead_mask=dead, aux_k=cfg.aux_k)
        loss = out["fvu"] + cfg.aux_coef * out["aux"]

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.sae.remove_parallel_decoder_grad()
        self.optimizer.step()
        self.sae.normalize_decoder()

        # A latent fires when it is selected with a positive activation.
        fired = out["indices"][out["values"] > 0]
        self.tokens_since_fired += x.shape[0]
        self.tokens_since_fired[fired] = 0

        self.step += 1
        self.tokens_seen += x.shape[0]
        return {
            "step": self.step,
            "tokens": self.tokens_seen,
            "fvu": out["fvu"].item(),
            "aux": out["aux"].item(),
            "dead_frac": dead.float().mean().item(),
            "lr": learning_rate,
        }

    def fit(
        self,
        batches: Iterable[torch.Tensor],
        *,
        out_dir: str | Path | None = None,
        log_fn: Callable[[dict[str, float]], None] | None = None,
        data_state: Callable[[], Mapping[str, int]] | None = None,
    ) -> TopKSAE:
        """Train until ``total_tokens`` have been seen or ``batches`` runs out.

        Args:
            batches: activation batches ``(batch_tokens, d_in)``.
            out_dir: if given, ``trainer_config.json`` and ``train_log.jsonl`` are written
                there, plus ``checkpoint.pt`` every ``checkpoint_every`` steps and at the end.
            log_fn: called with each log record.
            data_state: called at checkpoint time; its result is stored in the checkpoint so
                the caller can reposition its data stream when resuming.

        Returns:
            The trained SAE (the same object as ``self.sae``).
        """
        cfg = self.cfg
        if self.step >= cfg.total_steps:
            return self.sae  # a resumed run that is already complete must not pull a batch

        out_path = Path(out_dir) if out_dir is not None else None
        log_file = None
        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)
            (out_path / "trainer_config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
            log_file = (out_path / "train_log.jsonl").open("a")

        started = time.monotonic()
        tokens_at_start = self.tokens_seen
        try:
            for x in batches:
                if self.step >= cfg.total_steps:
                    break
                metrics = self.train_step(x)

                if self.step % cfg.log_every == 0 or self.step == cfg.total_steps:
                    elapsed = max(1e-9, time.monotonic() - started)
                    metrics["tokens_per_sec"] = (self.tokens_seen - tokens_at_start) / elapsed
                    if log_file is not None:
                        log_file.write(json.dumps(metrics) + "\n")
                        log_file.flush()
                    if log_fn is not None:
                        log_fn(metrics)

                if out_path is not None and cfg.checkpoint_every and self.step % cfg.checkpoint_every == 0:
                    self.save_checkpoint(
                        out_path / CHECKPOINT_NAME,
                        data=data_state() if data_state is not None else None,
                    )

            # A final checkpoint lets a later run extend the training length from here.
            if out_path is not None and cfg.checkpoint_every and self.step % cfg.checkpoint_every != 0:
                self.save_checkpoint(
                    out_path / CHECKPOINT_NAME,
                    data=data_state() if data_state is not None else None,
                )
        finally:
            if log_file is not None:
                log_file.close()
        return self.sae

    def state_dict(self) -> dict[str, Any]:
        """Everything needed to continue training, as tensors and plain Python values."""
        return {
            "version": CHECKPOINT_VERSION,
            "config": asdict(self.cfg),
            "d_in": self.sae.d_in,
            "sae": self.sae.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "tokens_since_fired": self.tokens_since_fired,
            "step": self.step,
            "tokens_seen": self.tokens_seen,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a state produced by ``state_dict``.

        Raises:
            ValueError: if the state comes from an unknown checkpoint version or from an SAE of
                a different shape or batch size.
        """
        if state.get("version") != CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported checkpoint version: {state.get('version')!r}.")
        if state["d_in"] != self.sae.d_in:
            raise ValueError(f"Checkpoint d_in {state['d_in']} does not match {self.sae.d_in}.")

        current = asdict(self.cfg)
        for key in _RESUME_KEYS:
            if state["config"][key] != current[key]:
                raise ValueError(f"Checkpoint {key}={state['config'][key]} does not match {key}={current[key]}.")

        self.sae.load_state_dict(state["sae"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.tokens_since_fired = state["tokens_since_fired"].to(self.device)
        self.step = int(state["step"])
        self.tokens_seen = int(state["tokens_seen"])

    def save_checkpoint(self, path: str | Path, *, data: Mapping[str, int] | None = None) -> None:
        """Write a checkpoint atomically: a crash mid-write leaves the previous file intact."""
        target = Path(path)
        temporary = target.with_name(target.name + ".tmp")

        state = self.state_dict()
        state["data"] = dict(data) if data is not None else {}
        torch.save(state, temporary)
        os.replace(temporary, target)

    def load_checkpoint(self, path: str | Path) -> dict[str, int]:
        """Restore from ``save_checkpoint`` and return the stored data-stream state."""
        state = torch.load(Path(path), map_location=self.device, weights_only=True)
        self.load_state_dict(state)
        return dict(state.get("data", {}))
