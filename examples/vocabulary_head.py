"""A vocabulary head that pools the same tokens as the SAE head, for a controlled comparison.

SPLADE turns the logits of the masked-language-model head into a sparse vector:
``w_j = max_i log(1 + ReLU(L_ij))`` over the tokens ``i`` of a text, for every vocabulary entry
``j``. Two changes from SPARSEUP (Formal, Erbacher and Rulli, arXiv 2611.09680) make the start
of training sparser: a logit shift, ``ReLU(L_ij - theta)`` with ``theta`` near the 99.9th
percentile of the logits, and a per-position fan-out that keeps only the ``f`` largest shifted
logits of each token. SPARSEUP reports ``theta = 15`` for ModernBERT and ``f = 12``.

This module applies both to the logits of the content tokens only, chosen exactly as
``muscari.retrieval.SAEPooling`` chooses them (no framing tokens, no prompt tokens), so that a
vocabulary head and an SAE head differ in their dictionary and in nothing else. With
``shift=0``, no fan-out, ``special_token_ids=[]`` and ``exclude_prompt=False`` it equals
Sentence Transformers' ``SpladePooling("max")``.

What this project chose where the paper summary is silent: the gradient of the fan-out flows
only through the kept logits (the selection itself is not differentiated), and ``theta`` is
estimated from a random sample of the content-token logits of a few hundred texts.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from sentence_transformers.sparse_encoder import SparseEncoder
from sentence_transformers.sparse_encoder.modules import Module, Transformer

from muscari.retrieval import SAEPooling


class VocabularyPooling(Module):
    """Shifted, fanned-out SPLADE pooling of masked-language-model logits over content tokens.

    Args:
        vocabulary_size: width of the logits, the dimension of the output.
        shift: ``theta``, subtracted from every logit before ReLU.
        fan_out: logits kept per token after the shift; ``None`` keeps all of them.
        exclude_prompt: leave the prompt's tokens out when the model passes a prompt length.
        special_token_ids: ids that are never pooled; filled in from the tokenizer's framing
            tokens when the module becomes part of a model. ``[]`` pools every attended token.

    The input ``token_embeddings`` are the logits ``(batch, length, vocabulary_size)`` of a
    ``Transformer`` with ``transformer_task="fill-mask"``. The pooling runs in float32.
    """

    config_keys: list[str] = ["vocabulary_size", "shift", "fan_out", "exclude_prompt", "special_token_ids"]

    def __init__(
        self,
        vocabulary_size: int,
        shift: float = 0.0,
        fan_out: int | None = None,
        exclude_prompt: bool = True,
        special_token_ids: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if fan_out is not None and not 0 < fan_out <= vocabulary_size:
            raise ValueError("fan_out must be between 1 and vocabulary_size, or None.")
        self.vocabulary_size = vocabulary_size
        self.shift = float(shift)
        self.fan_out = fan_out
        self.exclude_prompt = exclude_prompt
        self.special_token_ids = None if special_token_ids is None else sorted(int(i) for i in special_token_ids)

    # The SAE head's rule for which tokens are content; it reads only the two attributes above.
    _content_tokens = SAEPooling._content_tokens
    on_model_ready = SAEPooling.on_model_ready

    def forward(self, features: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Add ``sentence_embedding`` ``(batch, vocabulary_size)`` to ``features``."""
        logits, rows, batch_size = self._content_tokens(features)
        device = logits.device
        # Apple MPS does not cover this scatter reduction in every supported PyTorch release.
        pool_device = torch.device("cpu") if device.type == "mps" else device
        pooled = torch.zeros(batch_size, self.vocabulary_size, device=pool_device)

        with torch.autocast(device_type=device.type, enabled=False):
            shifted = logits.float() - self.shift
            if self.fan_out is None:
                weights = torch.log1p(F.relu(shifted))
                keys = None
            else:
                kept, keys = shifted.topk(self.fan_out, dim=1)
                weights = torch.log1p(F.relu(kept))
            weights, rows = weights.to(pool_device), rows.to(pool_device)
            outputs = []
            for row in range(batch_size):
                mine = rows == row
                if keys is None:
                    outputs.append(weights[mine].amax(dim=0) if mine.any() else pooled[row])
                else:
                    outputs.append(
                        pooled[row].scatter_reduce(
                            0, keys[mine.to(keys.device)].reshape(-1).to(pool_device), weights[mine].reshape(-1), "amax"
                        )
                    )
            pooled = torch.stack(outputs).to(device) if outputs else pooled.to(device)

        features["sentence_embedding"] = pooled
        return features

    def get_embedding_dimension(self) -> int:
        return self.vocabulary_size

    def save(self, output_path: str, *args: Any, **kwargs: Any) -> None:
        self.save_config(output_path)

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> VocabularyPooling:
        config = cls.load_config(
            model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        return cls(**config)


def output_size(transformer: Transformer) -> int:
    """Width of the logits of a fill-mask ``Transformer``; the vocabulary may be padded."""
    head = transformer.auto_model.get_output_embeddings()
    return int(head.out_features if hasattr(head, "out_features") else head.weight.shape[0])


def build_vocabulary_encoder(
    model_name_or_path: str,
    *,
    shift: float | str = 0.0,
    fan_out: int | None = None,
    exclude_prompt: bool = True,
    keep_special_tokens: bool = False,
    query_prompt: str | None = None,
    document_prompt: str | None = None,
    max_seq_length: int | None = None,
    shift_texts: Sequence[str] | None = None,
    shift_quantile: float = 0.999,
    seed: int = 42,
    device: str | None = None,
) -> SparseEncoder:
    """A ``SparseEncoder`` with ``VocabularyPooling`` on the masked-language-model head.

    ``shift="auto"`` estimates ``theta`` as the ``shift_quantile`` of the content-token logits of
    ``shift_texts`` (documents, with the document prompt) before the head is attached.
    """
    transformer = Transformer(model_name_or_path, transformer_task="fill-mask", max_seq_length=max_seq_length)
    head = VocabularyPooling(
        output_size(transformer),
        fan_out=fan_out,
        exclude_prompt=exclude_prompt,
        special_token_ids=[] if keep_special_tokens else None,
    )
    prompts = {name: prompt for name, prompt in (("query", query_prompt), ("document", document_prompt)) if prompt}
    model = SparseEncoder(modules=[transformer, head], prompts=prompts or None, similarity_fn_name="dot", device=device)
    if shift == "auto":
        if not shift_texts:
            raise ValueError('shift="auto" needs shift_texts.')
        head.shift = estimate_shift(model, shift_texts, quantile=shift_quantile, seed=seed)
    else:
        head.shift = float(shift)
    return model


@torch.no_grad()
def estimate_shift(
    model: SparseEncoder,
    texts: Sequence[str],
    *,
    quantile: float = 0.999,
    samples_per_batch: int = 200_000,
    batch_size: int = 16,
    seed: int = 42,
) -> float:
    """The ``quantile`` of the content-token logits of ``texts`` encoded as documents.

    Every batch contributes a uniform random sample of at most ``samples_per_batch`` of its
    ``(token, vocabulary)`` logits, drawn with ``seed``, so memory does not grow with the
    vocabulary. The model's last module must be a ``VocabularyPooling``.
    """
    head = model[-1]
    if not isinstance(head, VocabularyPooling):
        raise TypeError("The model's last module must be a VocabularyPooling.")
    generator = torch.Generator().manual_seed(seed)
    captured: list[torch.Tensor] = []

    def keep_sample(module: Any, args: Any, output: Any) -> None:
        logits, _, _ = head._content_tokens(args[0])
        flat = logits.float().reshape(-1)
        if flat.numel() > samples_per_batch:
            flat = flat[torch.randint(flat.numel(), (samples_per_batch,), generator=generator).to(flat.device)]
        captured.append(flat.cpu())

    handle = head.register_forward_hook(keep_sample)
    try:
        model.encode_document(list(texts), batch_size=batch_size, convert_to_tensor=True)
    finally:
        handle.remove()
    values = torch.cat(captured)
    if values.numel() > 16_000_000:  # torch.quantile's limit
        values = values[torch.randperm(values.numel(), generator=generator)[:16_000_000]]
    return float(torch.quantile(values, quantile))
