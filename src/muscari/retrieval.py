"""Use an exported SAE as the head of a Sentence Transformers sparse encoder.

A SPLADE model scores every vocabulary entry at every token with its language-model head,
saturates the scores and takes the maximum over the tokens. ``SAEPooling`` does the same with
the SAE's latents in place of the vocabulary: the encoder's final hidden states go through the
SAE's encoder, the active latents are saturated with ``log1p`` and pooled with a maximum over
the content tokens. The output has one dimension per latent and feeds the losses, trainer and
evaluators of ``sentence_transformers.sparse_encoder`` unchanged.

Requires the ``retrieval`` extra: ``pip install "muscari[retrieval]"``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from sentence_transformers.sparse_encoder import SparseEncoder
from sentence_transformers.sparse_encoder.modules import Module, Transformer
from torch import nn

from .export import ExportedSAE, load_sae
from .sae import sparse_topk

ACTIVATIONS = ("topk", "jumprelu")


class SAEPooling(Module):
    """Turn token states into one sparse vector per text with the encoder half of an SAE.

    For every content token ``h``: ``pre = h @ encoder_weight.T + encoder_bias``; a Top-K SAE
    keeps the ``k`` largest entries, a JumpReLU SAE those above their threshold; the kept
    entries pass through ReLU and ``log1p``. A text's vector holds, for each latent, the
    maximum over its content tokens. Padding, special tokens and, by default, the tokens of
    the prompt are not content, which matches what the SAE was trained on.

    Args:
        d_in: width of the token states.
        num_latents: number of latents, the dimension of the output.
        k: latents kept per token; required for ``activation="topk"``.
        activation: ``"topk"`` or ``"jumprelu"``.
        exclude_prompt: leave the prompt's tokens out of the pooling when the model passes a
            prompt length. Sentence Transformers counts the tokens of the prompt on its own;
            where a token straddles the end of the prompt this can differ by one token from
            the character-based rule used when the SAE was trained.
        special_token_ids: ids that are never pooled. Filled in from the tokenizer when the
            module becomes part of a model.
        masked_latents: latents whose output is always zero, for example ones that are active
            in nearly every text and so cannot tell texts apart.
        trainable: train the SAE's encoder together with the transformer. The default keeps it
            frozen, as the models this head is patterned on do.
        token_chunk: content tokens processed at a time; bounds the size of the
            ``(tokens, num_latents)`` product.

    The decoder of the SAE is not needed for retrieval and is not stored.
    """

    config_keys: list[str] = [
        "d_in",
        "num_latents",
        "k",
        "activation",
        "exclude_prompt",
        "special_token_ids",
        "masked_latents",
        "trainable",
        "token_chunk",
    ]

    def __init__(
        self,
        d_in: int,
        num_latents: int,
        k: int | None = None,
        activation: str = "topk",
        exclude_prompt: bool = True,
        special_token_ids: Sequence[int] | None = None,
        masked_latents: Sequence[int] | None = None,
        trainable: bool = False,
        token_chunk: int = 4096,
    ) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {activation!r}.")
        if activation == "topk" and (k is None or not 0 < k <= num_latents):
            raise ValueError("A Top-K head needs k between 1 and num_latents.")
        if token_chunk < 1:
            raise ValueError("token_chunk must be positive.")

        self.d_in = d_in
        self.num_latents = num_latents
        self.k = k
        self.activation = activation
        self.exclude_prompt = exclude_prompt
        self.special_token_ids = None if special_token_ids is None else sorted(int(i) for i in special_token_ids)
        self.masked_latents = sorted(int(i) for i in masked_latents or [])
        self.trainable = trainable
        self.token_chunk = token_chunk

        self.encoder_weight = nn.Parameter(torch.zeros(num_latents, d_in), requires_grad=trainable)
        self.encoder_bias = nn.Parameter(torch.zeros(num_latents), requires_grad=trainable)
        self.threshold: torch.Tensor | None
        self.register_buffer("threshold", torch.zeros(num_latents) if activation == "jumprelu" else None)

    @classmethod
    def from_sae(cls, sae: ExportedSAE | str | Path, **kwargs: Any) -> SAEPooling:
        """Build the head from an exported SAE or the directory that holds one.

        ``kwargs`` are the constructor arguments other than the sizes and the activation rule.
        """
        if not isinstance(sae, ExportedSAE):
            sae = load_sae(sae)

        jumprelu = sae.threshold is not None
        head = cls(
            d_in=sae.d_in,
            num_latents=sae.num_latents,
            k=None if jumprelu else int(sae.config["k"]),
            activation="jumprelu" if jumprelu else "topk",
            **kwargs,
        )
        with torch.no_grad():
            head.encoder_weight.copy_(sae.encoder_weight)
            head.encoder_bias.copy_(sae.encoder_bias)
            if head.threshold is not None and sae.threshold is not None:
                head.threshold.copy_(sae.threshold)
        return head

    def on_model_ready(self, model: Any) -> None:
        """Take the special token ids from the model's tokenizer unless they were given."""
        tokenizer = getattr(model, "tokenizer", None)
        if self.special_token_ids is None and tokenizer is not None:
            self.special_token_ids = sorted(int(i) for i in tokenizer.all_special_ids)

    def _content_tokens(self, features: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return the states of the content tokens, the row of the text each belongs to, and the batch size."""
        if self.special_token_ids is None:
            raise RuntimeError(
                "SAEPooling does not know the special tokens. Build the model with the head among its "
                "modules, call on_model_ready(model), or pass special_token_ids."
            )
        states = features["token_embeddings"]
        device = states.device

        if "cu_seq_lens_q" in features:
            # Sentence Transformers packed the batch into one sequence without padding.
            states = states.reshape(-1, states.shape[-1])
            bounds = features["cu_seq_lens_q"].to(device)
            lengths = bounds[1:] - bounds[:-1]
            batch_size = int(lengths.shape[0])
            rows = torch.repeat_interleave(torch.arange(batch_size, device=device), lengths)
            positions = torch.arange(states.shape[0], device=device) - bounds[:-1][rows]
            keep = torch.ones(states.shape[0], dtype=torch.bool, device=device)
        else:
            batch_size, length, width = states.shape
            attention = features["attention_mask"].to(device).long()
            # Counted from the first attended token, so that left padding is handled as well.
            positions = (attention.cumsum(dim=1) - 1).reshape(-1)
            rows = torch.arange(batch_size, device=device).repeat_interleave(length)
            keep = attention.reshape(-1) > 0
            states = states.reshape(-1, width)

        special = torch.tensor(self.special_token_ids, dtype=torch.long, device=device)
        input_ids = features["input_ids"].to(device).reshape(-1)
        keep = keep & ~(input_ids.unsqueeze(1) == special).any(dim=1)

        prompt_length = features.get("prompt_length") if self.exclude_prompt else None
        if prompt_length is not None:
            # An int at inference; during training a tensor holding the same value for every row.
            if isinstance(prompt_length, torch.Tensor):
                prompt_length = int(prompt_length.reshape(-1)[0].item())
            keep = keep & (positions >= int(prompt_length))

        return states[keep], rows[keep], batch_size

    def forward(self, features: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Add ``sentence_embedding`` ``(batch, num_latents)`` to ``features``."""
        states, rows, batch_size = self._content_tokens(features)
        keys, values = [], []

        # The product over all latents is the expensive part; keep it in float32 and in chunks.
        with torch.autocast(device_type=states.device.type, enabled=False):
            states = states.float()
            for start in range(0, states.shape[0], self.token_chunk):
                chunk = states[start : start + self.token_chunk]
                chunk_rows = rows[start : start + self.token_chunk]
                if self.activation == "topk":
                    assert self.k is not None
                    selected, latents = sparse_topk(chunk, self.encoder_weight, self.encoder_bias, self.k)
                    keys.append((chunk_rows.unsqueeze(1) * self.num_latents + latents).reshape(-1))
                    values.append(selected.reshape(-1))
                else:
                    assert self.threshold is not None
                    pre = F.linear(chunk, self.encoder_weight, self.encoder_bias)
                    tokens, latents = (pre > self.threshold).nonzero(as_tuple=True)
                    keys.append(chunk_rows[tokens] * self.num_latents + latents)
                    values.append(pre[tokens, latents])

            # The maximum per (text, latent) pair is taken on the CPU when the device is Apple MPS,
            # whose scatter does not cover this reduction in every supported PyTorch release.
            pool_device = torch.device("cpu") if states.device.type == "mps" else states.device
            pooled = torch.zeros(batch_size * self.num_latents, device=pool_device)
            if keys:
                activations = torch.log1p(F.relu(torch.cat(values))).to(pool_device)
                pooled = pooled.scatter_reduce(
                    0, torch.cat(keys).to(pool_device), activations, reduce="amax", include_self=True
                )
            pooled = pooled.view(batch_size, self.num_latents).to(states.device)
            if self.masked_latents:
                mask = torch.ones(self.num_latents, dtype=pooled.dtype, device=pooled.device)
                mask[self.masked_latents] = 0
                pooled = pooled * mask

        features["sentence_embedding"] = pooled
        return features

    def get_embedding_dimension(self) -> int:
        return self.num_latents

    def save(self, output_path: str, *args: Any, safe_serialization: bool = True, **kwargs: Any) -> None:
        self.save_config(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        init_defaults: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> SAEPooling:
        hub_kwargs: dict[str, Any] = {
            "subfolder": subfolder,
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        config = cls.load_config(model_name_or_path, **hub_kwargs)
        for key, value in (init_defaults or {}).items():
            config.setdefault(key, value)
        head = cls(**config)
        loaded: SAEPooling = cls.load_torch_weights(model_name_or_path, model=head, **hub_kwargs)
        return loaded


def build_sparse_encoder(
    sae: ExportedSAE | str | Path,
    model_name_or_path: str | None = None,
    *,
    query_prompt: str | None = None,
    document_prompt: str | None = None,
    max_seq_length: int | None = None,
    head_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> SparseEncoder:
    """Put an exported SAE on top of its encoder as a ``SparseEncoder``, ready to encode or to train.

    Args:
        sae: an exported SAE or its directory.
        model_name_or_path: the encoder; default: the base model recorded with the SAE.
        query_prompt, document_prompt: prefixes the encoder expects, registered as the prompts
            ``"query"`` and ``"document"`` so that ``encode_query`` and ``encode_document`` use them.
        max_seq_length: truncation length; default: the length the SAE was trained with.
        head_kwargs: arguments for ``SAEPooling``, for example ``{"masked_latents": [...]}``.
        kwargs: passed on to ``SparseEncoder``.

    Raises:
        ValueError: if no encoder is given and none is recorded, or if the SAE was not trained
            on the final hidden states.
    """
    if not isinstance(sae, ExportedSAE):
        sae = load_sae(sae)
    if sae.config.get("layer", "last_hidden_state") != "last_hidden_state":
        raise ValueError("A retrieval head needs an SAE trained on last_hidden_state.")
    model_name_or_path = model_name_or_path or sae.config.get("base_model")
    if model_name_or_path is None:
        raise ValueError("model_name_or_path is required: the SAE config records no base model.")

    transformer = Transformer(model_name_or_path, max_seq_length=max_seq_length or sae.config.get("max_length"))
    head = SAEPooling.from_sae(sae, **(head_kwargs or {}))
    prompts = {name: prompt for name, prompt in (("query", query_prompt), ("document", document_prompt)) if prompt}
    return SparseEncoder(modules=[transformer, head], prompts=prompts or None, similarity_fn_name="dot", **kwargs)
