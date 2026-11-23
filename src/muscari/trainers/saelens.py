"""Train with SAELens' ``SAETrainer`` on the same encoder activations.

SAELens' stock runner loads decoder LLMs through TransformerLens. ``SAETrainer`` itself only
needs an iterator of ``(batch, d_in)`` activation tensors, which is what
``muscari.activations`` produces, so the encoder never has to be wrapped as a hooked model.

Differences from ``TopKTrainer`` that are fixed inside SAELens (checked against 6.56):

* the auxiliary loss uses ``d_in // 2`` dead latents instead of a configurable ``aux_k`` and
  is not variance-normalised, so ``aux_loss_coefficient`` lives on a different scale;
* dead latents are counted in training steps (``dead_feature_window``), not tokens;
* ``batchtopk`` keeps the top ``k * batch`` activations across the whole batch and is exported
  as a JumpReLU SAE with one threshold per latent, so the number of active latents varies from
  token to token.

One thing is changed on purpose: the decoder bias starts at the geometric median of the first
batch instead of zero. Encoder hidden states can have a large common component; with a zero
bias the top-k selection then locks onto the few latents aligned with it. Pass
``init_b_dec=False`` for SAELens' own initialisation.

Requires the ``saelens`` extra: ``pip install "muscari[saelens]"``.
"""

from __future__ import annotations

import itertools
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..devices import resolve_device
from ..export import save_saelens_sae
from ..sae import geometric_median

ARCHITECTURES = ("topk", "batchtopk")


@dataclass
class SAELensTrainerConfig:
    """Hyperparameters passed on to SAELens.

    Attributes:
        architecture: ``"topk"`` or ``"batchtopk"``.
        num_latents: dictionary size (``d_sae`` in SAELens).
        k: active latents per token (on average for ``batchtopk``).
        aux_loss_coefficient: SAELens' weight of its auxiliary loss. Not comparable to
            ``TopKTrainerConfig.aux_coef``.
        dead_after_tokens: converted to SAELens' ``dead_feature_window`` in steps.
        lr: peak learning rate.
        lr_end: learning rate at the end of the cosine schedule.
        warmup_frac: share of the steps spent on linear warmup.
        adam_betas: Adam ``(beta1, beta2)``.
        total_tokens: training length in tokens.
        batch_tokens: activations per optimisation step.
        amp: let SAELens run under autocast.
        init_b_dec: start the decoder bias at the geometric median of the first batch, as the
            ``topk`` trainer does. SAELens itself starts it at zero.
        seed: seed for the SAE initialisation.
        log_to_wandb: send SAELens' metrics to Weights & Biases.
        wandb_project: project name used when logging is on.
        device: ``"auto"`` or a device string.
    """

    architecture: str = "topk"
    num_latents: int = 65_536
    k: int = 16
    aux_loss_coefficient: float = 1.0
    dead_after_tokens: int = 10_000_000
    lr: float = 5e-4
    lr_end: float = 5e-6
    warmup_frac: float = 0.05
    adam_betas: tuple[float, float] = (0.9, 0.999)
    total_tokens: int = 1_000_000_000
    batch_tokens: int = 8192
    amp: bool = False
    init_b_dec: bool = True
    seed: int = 42
    log_to_wandb: bool = False
    wandb_project: str = "muscari"
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}, got {self.architecture!r}.")
        if self.num_latents < 1 or self.k < 1 or self.k > self.num_latents:
            raise ValueError("Need 1 <= k <= num_latents.")
        if self.batch_tokens < 1 or self.total_tokens < self.batch_tokens:
            raise ValueError("Need batch_tokens >= 1 and total_tokens >= batch_tokens.")
        if self.lr <= 0 or not 0 <= self.warmup_frac < 1:
            raise ValueError("Need lr > 0 and 0 <= warmup_frac < 1.")

    @property
    def total_steps(self) -> int:
        return self.total_tokens // self.batch_tokens


def _on_device(batches: Iterable[torch.Tensor], device: torch.device) -> Iterator[torch.Tensor]:
    for batch in batches:
        yield batch.to(device)


def train_with_saelens(
    cfg: SAELensTrainerConfig,
    d_in: int,
    batches: Iterable[torch.Tensor],
    out_dir: str | Path,
    export_config: dict[str, Any] | None = None,
) -> Path:
    """Run ``SAETrainer.fit()`` and write the result in muscari's common format.

    Args:
        cfg: hyperparameters.
        d_in: width of the activations.
        batches: activation batches ``(batch_tokens, d_in)``; must provide at least
            ``cfg.total_steps`` of them.
        out_dir: where ``sae.safetensors`` and ``sae_config.json`` are written.
        export_config: extra provenance stored in ``sae_config.json``.

    Returns:
        The output directory.

    Raises:
        ImportError: if SAELens is not installed.
    """
    try:
        from sae_lens import BatchTopKTrainingSAEConfig, TopKTrainingSAEConfig
        from sae_lens.config import LoggingConfig, SAETrainerConfig
        from sae_lens.saes.sae import TrainingSAE
        from sae_lens.training.sae_trainer import SAETrainer
    except ImportError as error:
        raise ImportError('The SAELens backend needs the extra: pip install "muscari[saelens]"') from error

    device = resolve_device(cfg.device)
    config_classes = {"topk": TopKTrainingSAEConfig, "batchtopk": BatchTopKTrainingSAEConfig}
    sae_config = config_classes[cfg.architecture](
        d_in=d_in,
        d_sae=cfg.num_latents,
        k=cfg.k,
        aux_loss_coefficient=cfg.aux_loss_coefficient,
        normalize_activations="none",
        device=str(device),
    )
    torch.manual_seed(cfg.seed)
    sae = TrainingSAE.from_dict(sae_config.to_dict())

    stream = iter(batches)
    if cfg.init_b_dec:
        first = next(stream)
        with torch.no_grad():
            sae.b_dec.copy_(geometric_median(first[:32_768]).to(sae.b_dec))
        stream = itertools.chain([first], stream)

    trainer_config = SAETrainerConfig(
        total_training_samples=cfg.total_steps * cfg.batch_tokens,
        train_batch_size_samples=cfg.batch_tokens,
        device=str(device),
        lr=cfg.lr,
        lr_end=cfg.lr_end,
        lr_scheduler_name="cosineannealing",
        lr_warm_up_steps=int(cfg.warmup_frac * cfg.total_steps),
        adam_beta1=cfg.adam_betas[0],
        adam_beta2=cfg.adam_betas[1],
        autocast=cfg.amp,
        dead_feature_window=max(1, cfg.dead_after_tokens // cfg.batch_tokens),
        feature_sampling_window=max(1, min(2000, cfg.total_steps)),
        n_checkpoints=0,
        checkpoint_path=None,
        save_final_checkpoint=False,
        logger=LoggingConfig(
            log_to_wandb=cfg.log_to_wandb,
            wandb_project=cfg.wandb_project,
        ),
    )
    trainer = SAETrainer(
        cfg=trainer_config,
        sae=sae,
        data_provider=_on_device(stream, device),
    )
    trained = trainer.fit()

    with tempfile.TemporaryDirectory() as temporary:
        trained.save_inference_model(temporary)
        return save_saelens_sae(
            out_dir,
            temporary,
            config={
                **(export_config or {}),
                "trainer": f"saelens-{cfg.architecture}",
                "b_dec_init": "geometric_median" if cfg.init_b_dec else "zero",
            },
        )
