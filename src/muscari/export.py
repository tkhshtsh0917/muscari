"""One on-disk format for SAEs from either trainer, plus loading and evaluation.

``sae.safetensors`` holds

    encoder.weight  (num_latents, d_in)
    encoder.bias    (num_latents,)   with the b_dec subtraction folded in: b_enc - W_enc @ b_dec
    decoder.weight  (d_in, num_latents)
    decoder.bias    (d_in,)          = b_dec
    threshold       (num_latents,)   only for JumpReLU-style SAEs (SAELens BatchTopK)

so encoding needs no decoder tensor: ``pre = x @ encoder.weight.T + encoder.bias``.
``sae_config.json`` records the sizes, the activation rule and how the SAE was made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .devices import resolve_device
from .sae import TopKSAE

WEIGHTS_NAME = "sae.safetensors"
CONFIG_NAME = "sae_config.json"
FORMAT_VERSION = 1

# A latent that fires on more than this share of tokens is reported as "dense".
DENSE_FREQUENCY = 0.1

# A latent that fires less often than this fraction of the average latent is reported as
# "rarely used". The average firing frequency of a latent is ``l0 / num_latents``.
RARE_RELATIVE_FREQUENCY = 0.1


def save_sae(
    out_dir: str | Path,
    *,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
    decoder_bias: torch.Tensor,
    config: dict[str, Any],
    threshold: torch.Tensor | None = None,
) -> Path:
    """Write the tensors and config in the common format and return the directory.

    ``config`` must contain ``k`` unless ``threshold`` is given. ``d_in``, ``num_latents``,
    ``activation`` and ``format_version`` are derived here and override any such keys.

    Raises:
        ValueError: if the tensor shapes disagree or a Top-K SAE has no ``k``.
    """
    num_latents, d_in = encoder_weight.shape
    if decoder_weight.shape != (d_in, num_latents):
        raise ValueError(f"decoder.weight must be {(d_in, num_latents)}, got {tuple(decoder_weight.shape)}.")
    if encoder_bias.shape != (num_latents,) or decoder_bias.shape != (d_in,):
        raise ValueError("encoder.bias must be (num_latents,) and decoder.bias must be (d_in,).")
    if threshold is None and "k" not in config:
        raise ValueError("A Top-K SAE needs 'k' in its config.")

    tensors = {
        "encoder.weight": encoder_weight,
        "encoder.bias": encoder_bias,
        "decoder.weight": decoder_weight,
        "decoder.bias": decoder_bias,
    }
    if threshold is not None:
        tensors["threshold"] = threshold

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.detach().float().cpu().contiguous() for name, tensor in tensors.items()},
        out / WEIGHTS_NAME,
    )

    full_config = {
        **config,
        "format_version": FORMAT_VERSION,
        "d_in": d_in,
        "num_latents": num_latents,
        "activation": "jumprelu" if threshold is not None else "topk",
    }
    (out / CONFIG_NAME).write_text(
        json.dumps(full_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def save_topk_sae(out_dir: str | Path, sae: TopKSAE, config: dict[str, Any]) -> Path:
    """Export a ``muscari.sae.TopKSAE``, folding ``b_dec`` into the encoder bias."""
    weight = sae.encoder.weight.detach()
    folded_bias = sae.encoder.bias.detach() - weight @ sae.b_dec.detach()

    return save_sae(
        out_dir,
        encoder_weight=weight,
        encoder_bias=folded_bias,
        decoder_weight=sae.decoder.detach().T,
        decoder_bias=sae.b_dec.detach(),
        config={**config, "k": sae.k},
    )


def save_saelens_sae(
    out_dir: str | Path,
    saelens_inference_dir: str | Path,
    config: dict[str, Any],
) -> Path:
    """Convert a directory written by SAELens ``save_inference_model``.

    SAELens stores ``W_enc`` as ``(d_in, d_sae)`` and ``W_dec`` as ``(d_sae, d_in)``; its
    inference export has already folded the decoder norms into the weights. A BatchTopK SAE
    arrives as a JumpReLU SAE with a per-latent ``threshold``.

    Raises:
        ValueError: if the SAE was trained with activation normalisation, which cannot be
            folded into this format.
    """
    source = Path(saelens_inference_dir)
    saelens_config = json.loads((source / "cfg.json").read_text(encoding="utf-8"))
    if saelens_config.get("normalize_activations", "none") != "none":
        raise ValueError("normalize_activations must be 'none'; a scaler cannot be folded here.")

    state = load_file(source / "sae_weights.safetensors")
    w_enc, w_dec, b_enc, b_dec = state["W_enc"], state["W_dec"], state["b_enc"], state["b_dec"]
    apply_b_dec = saelens_config.get("apply_b_dec_to_input", False)
    folded_bias = b_enc - b_dec @ w_enc if apply_b_dec else b_enc

    extra = {"k": saelens_config["k"]} if "k" in saelens_config else {}
    return save_sae(
        out_dir,
        encoder_weight=w_enc.T,
        encoder_bias=folded_bias,
        decoder_weight=w_dec.T,
        decoder_bias=b_dec,
        threshold=state.get("threshold"),
        config={
            **config,
            **extra,
            "saelens_architecture": saelens_config.get("architecture"),
        },
    )


class ExportedSAE(nn.Module):
    """Inference-only view of a saved SAE; the same code serves both trainers' outputs.

    Attributes:
        config: the parsed ``sae_config.json``.
        encoder_weight: ``(num_latents, d_in)``.
        encoder_bias: ``(num_latents,)`` with ``b_dec`` folded in.
        decoder_weight: ``(d_in, num_latents)``.
        decoder_bias: ``(d_in,)``.
        threshold: ``(num_latents,)`` for JumpReLU SAEs, otherwise ``None``.
    """

    encoder_weight: torch.Tensor
    encoder_bias: torch.Tensor
    decoder_weight: torch.Tensor
    decoder_bias: torch.Tensor
    threshold: torch.Tensor | None

    def __init__(self, tensors: dict[str, torch.Tensor], config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("encoder_weight", tensors["encoder.weight"])
        self.register_buffer("encoder_bias", tensors["encoder.bias"])
        self.register_buffer("decoder_weight", tensors["decoder.weight"])
        self.register_buffer("decoder_bias", tensors["decoder.bias"])
        self.register_buffer("threshold", tensors.get("threshold"))

    @property
    def num_latents(self) -> int:
        return int(self.encoder_weight.shape[0])

    @property
    def d_in(self) -> int:
        return int(self.encoder_weight.shape[1])

    def encode(self, x: torch.Tensor, k: int | None = None) -> torch.Tensor:
        """Dense, non-negative feature activations ``(..., num_latents)`` for ``x`` ``(..., d_in)``.

        A Top-K SAE keeps the ``k`` largest pre-activations per token after ReLU; ``k``
        defaults to the training value. A JumpReLU SAE keeps every pre-activation above its
        latent's threshold and ignores ``k``.
        """
        pre = x @ self.encoder_weight.T + self.encoder_bias
        if self.threshold is not None:
            return pre * (pre > self.threshold)

        top_k = int(self.config["k"]) if k is None else k
        values, indices = pre.topk(top_k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, indices, values.relu())

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Reconstruct activations ``(..., d_in)`` from features ``(..., num_latents)``."""
        return features @ self.decoder_weight.T + self.decoder_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def load_sae(sae_dir: str | Path, device: str | torch.device = "cpu") -> ExportedSAE:
    """Load a directory written by ``save_sae``.

    Raises:
        FileNotFoundError: if the weights or the config are missing.
        ValueError: if the files were written by a newer format version.
    """
    source = Path(sae_dir)
    config = json.loads((source / CONFIG_NAME).read_text(encoding="utf-8"))
    if int(config.get("format_version", FORMAT_VERSION)) > FORMAT_VERSION:
        raise ValueError(f"{source} uses format version {config['format_version']}; upgrade muscari to read it.")

    tensors = load_file(source / WEIGHTS_NAME)
    return ExportedSAE(tensors, config).to(resolve_device(device))


@torch.no_grad()
def evaluate_sae(
    sae: ExportedSAE,
    activations: torch.Tensor,
    batch_size: int = 1024,
) -> dict[str, float]:
    """Measure reconstruction and how evenly the dictionary is used on ``activations`` ``(n, d_in)``.

    Returns:
        ``fvu``: squared reconstruction error over the variance around the mean activation.
        ``l0``: mean number of active latents per token.
        ``never_fired_frac``: share of latents that never fire on these tokens.
        ``rare_frac``: share of latents that fire less than a tenth as often as the average
        latent (the average frequency is ``l0 / num_latents``); never-fired latents included.
        ``dense_frac``: share of latents that fire on more than 10% of the tokens.
        ``median_frequency``: the firing frequency of the median latent.
        ``max_frequency``: the largest share of tokens any single latent fires on.
        ``eval_tokens``: number of activations evaluated.

    The frequency figures depend on ``eval_tokens``: with few tokens many rare latents have had
    no chance to fire, so compare ``never_fired_frac`` only between equally sized samples.

    Raises:
        ValueError: if ``activations`` is empty or has the wrong width.
    """
    if activations.ndim != 2 or activations.shape[0] == 0 or activations.shape[1] != sae.d_in:
        raise ValueError(f"activations must be a non-empty (n, {sae.d_in}) tensor.")

    device = sae.encoder_weight.device
    n_tokens = activations.shape[0]
    mean = activations.mean(dim=0, keepdim=True).to(device)

    residual = 0.0
    total = 0.0
    fire_counts = torch.zeros(sae.num_latents, dtype=torch.long, device=device)

    for start in range(0, n_tokens, batch_size):
        x = activations[start : start + batch_size].to(device)
        features = sae.encode(x)

        residual += (x - sae.decode(features)).pow(2).sum().item()
        total += (x - mean).pow(2).sum().item()
        fire_counts += (features > 0).sum(dim=0)

    frequencies = fire_counts.float() / n_tokens
    l0 = fire_counts.sum().item() / n_tokens
    rare_below = RARE_RELATIVE_FREQUENCY * l0 / sae.num_latents
    return {
        "fvu": residual / max(total, 1e-12),
        "l0": l0,
        "never_fired_frac": (fire_counts == 0).float().mean().item(),
        "rare_frac": (frequencies < rare_below).float().mean().item(),
        "dense_frac": (frequencies > DENSE_FREQUENCY).float().mean().item(),
        "median_frequency": frequencies.median().item(),
        "max_frequency": frequencies.max().item(),
        "eval_tokens": n_tokens,
    }
