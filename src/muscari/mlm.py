"""Masked-language-model loss recovered: how much of an encoder's own task survives the SAE.

For an encoder that still has its language-model head, the final hidden states can be replaced
before the head reads them. Replacing them with the SAE's reconstruction costs some loss;
replacing them with one constant vector costs nearly all the head can lose. Where the SAE falls
between the untouched encoder and that constant is the loss recovered.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from .activations import EncoderActivationSource
from .export import ExportedSAE

Replacement = Callable[[torch.Tensor], torch.Tensor]


def load_masked_lm(name_or_path: str, *, revision: str | None = None) -> Any:
    """Load an encoder together with its language-model head.

    Raises:
        ValueError: if the checkpoint holds no trained head, as with encoders saved after
            fine-tuning as embedding models. A head initialised at random would yield a loss
            that means nothing.
    """
    from transformers import AutoModelForMaskedLM

    model, info = AutoModelForMaskedLM.from_pretrained(name_or_path, revision=revision, output_loading_info=True)
    # The output projection is usually tied to the input embeddings and then absent from the file.
    missing = sorted(key for key in info["missing_keys"] if not key.endswith(("decoder.weight", "decoder.bias")))
    if missing:
        raise ValueError(f"{name_or_path} has no trained language-model head (missing weights: {missing[:3]}).")
    return model


@torch.no_grad()
def mlm_loss_recovered(
    sae: ExportedSAE,
    model: Any,
    source: EncoderActivationSource,
    texts: Sequence[tuple[str, str]],
    *,
    mask_probability: float = 0.15,
    batch_size: int = 4,
    seed: int = 42,
) -> dict[str, float]:
    """Compare the masked-language-model loss with and without the SAE in the path.

    Each content token is replaced by the mask token with probability ``mask_probability``.
    The loss is the mean cross-entropy over the masked positions, measured three times on the
    same masked inputs: with the encoder untouched (``loss_clean``), with the final hidden
    state of every content position replaced by the SAE's reconstruction (``loss_spliced``),
    and with it replaced by the mean of those states over all evaluated texts
    (``loss_mean``). Special and prefix positions are never replaced.

    Args:
        sae: an exported SAE trained on ``last_hidden_state`` of this encoder.
        model: the encoder with its head, from ``load_masked_lm``. It is moved to the device of
            ``source``.
        source: supplies the tokenizer, the prefixes, the truncation length and the content
            mask. Its own model is not used.
        texts: ``(kind, text)`` pairs, as for the other methods of ``source``.
        mask_probability: share of content tokens masked.
        batch_size: texts per forward pass. The head produces one logit per vocabulary entry
            and position, so keep this small.
        seed: seed of the mask.

    Returns:
        ``loss_clean``, ``loss_spliced``, ``loss_mean``, ``recovered``
        ``= (loss_mean - loss_spliced) / (loss_mean - loss_clean)``, ``masked_tokens`` and
        ``texts``.

    Raises:
        ValueError: if the tokenizer has no mask token, ``texts`` is empty or nothing was masked.
    """
    tokenizer = source.tokenizer
    if getattr(tokenizer, "mask_token_id", None) is None:
        raise ValueError("The tokenizer has no mask token.")
    if not texts:
        raise ValueError("texts is empty.")
    if not 0 < mask_probability < 1:
        raise ValueError("mask_probability must be between 0 and 1.")

    device = source.device
    model = model.to(device).eval()
    generator = torch.Generator().manual_seed(seed)
    ordered = sorted(texts, key=lambda item: len(item[1]))

    batches = []
    for start in range(0, len(ordered), batch_size):
        chunk = ordered[start : start + batch_size]
        encoding, content = source.tokenize([kind for kind, _ in chunk], [text for _, text in chunk])
        masked = content & (torch.rand(content.shape, generator=generator) < mask_probability)
        if not masked.any():
            continue
        input_ids = encoding["input_ids"].clone()
        labels = input_ids[masked]
        input_ids[masked] = tokenizer.mask_token_id
        batches.append((input_ids, encoding["attention_mask"], content, masked, labels))
    if not batches:
        raise ValueError("No token was masked; give more or longer texts.")

    def loss(replace: Replacement | None, observe: Callable[[torch.Tensor], None] | None = None) -> float:
        """Sum of the cross-entropy over all masked positions, with the content states replaced."""
        total = 0.0
        for input_ids, attention_mask, content, masked, labels in batches:
            content_on_device = content.to(device)

            def hook(_module: Any, _inputs: Any, output: Any, where: torch.Tensor = content_on_device) -> Any:
                hidden = output.last_hidden_state
                if observe is not None:
                    observe(hidden[where].float())
                if replace is not None:
                    hidden = hidden.clone()
                    hidden[where] = replace(hidden[where].float()).to(hidden.dtype)
                    output.last_hidden_state = hidden
                return output

            handle = model.base_model.register_forward_hook(hook)
            try:
                logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits
            finally:
                handle.remove()
            total += F.cross_entropy(logits[masked.to(device)].float(), labels.to(device), reduction="sum").item()
        return total

    state_sum = torch.zeros(sae.d_in, device=device, dtype=torch.float64)
    state_count = 0

    def accumulate(states: torch.Tensor) -> None:
        nonlocal state_count
        state_sum.add_(states.double().sum(dim=0))
        state_count += states.shape[0]

    sae = sae.to(device)
    masked_tokens = sum(int(masked.sum()) for *_, masked, _ in batches)
    clean = loss(None, accumulate) / masked_tokens
    spliced = loss(sae.forward) / masked_tokens
    mean_state = (state_sum / state_count).float()
    ablated = loss(lambda states: mean_state.expand_as(states)) / masked_tokens

    return {
        "loss_clean": clean,
        "loss_spliced": spliced,
        "loss_mean": ablated,
        "recovered": (ablated - spliced) / (ablated - clean),
        "masked_tokens": masked_tokens,
        "texts": len(ordered),
    }
