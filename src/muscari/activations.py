"""Collect token-level hidden states from a frozen encoder, on the fly.

An encoder is small next to a decoder LLM, so activations are recomputed instead of cached:
one billion tokens of 768-dimensional float32 activations would need about 3 TB of disk.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from typing import Any

import torch

from .devices import resolve_device

# Texts are length-sorted inside pools of this many forward passes to limit padding.
_POOL_BATCHES = 16


class EncoderActivationSource:
    """Runs texts through a frozen encoder and returns the hidden state of each content token.

    Padding, special tokens and (by default) the tokens of the instruction prefix are dropped.
    What remains are the positions that carry the text itself, which is what a SPLADE-style
    head pools over.

    Args:
        model: a Hugging Face encoder (``AutoModel``). It is moved to ``device``, put in eval
            mode and frozen.
        tokenizer: its tokenizer. A fast tokenizer gives exact prefix masking through character
            offsets; a slow one falls back to counting the tokens of the prefix alone.
        max_length: truncation length in tokens, special tokens included.
        text_batch_size: number of texts per forward pass.
        layer: ``None`` uses ``last_hidden_state``, the tensor Sentence Transformers exposes as
            ``token_embeddings``. An integer indexes ``hidden_states`` instead (0 is the
            embedding output).
        exclude_prefix: drop the tokens that lie entirely inside the prefix.
        device: ``"auto"``, a device string or a ``torch.device``.
        amp_dtype: run the encoder under autocast with this dtype, for example
            ``torch.bfloat16`` on CUDA. Activations are always returned as float32.

    Attributes:
        d_in: hidden size of the encoder, the input width of the SAE.
        texts_seen: number of texts encoded so far.
        tokens_seen: number of content-token activations returned so far.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_length: int = 1024,
        text_batch_size: int = 64,
        layer: int | None = None,
        exclude_prefix: bool = True,
        device: str | torch.device = "auto",
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        if max_length < 1 or text_batch_size < 1:
            raise ValueError("max_length and text_batch_size must be positive.")

        self.device = resolve_device(device)
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_batch_size = text_batch_size
        self.layer = layer
        self.exclude_prefix = exclude_prefix
        self.amp_dtype = amp_dtype
        self.d_in = int(model.config.hidden_size)

        self.texts_seen = 0
        self.tokens_seen = 0
        self._prefix_token_counts: dict[str, int] = {}

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        revision: str | None = None,
        **kwargs: Any,
    ) -> EncoderActivationSource:
        """Load the encoder and tokenizer with ``AutoModel`` and ``AutoTokenizer``.

        ``kwargs`` are the constructor arguments other than ``model`` and ``tokenizer``.
        """
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name_or_path, revision=revision)
        model = AutoModel.from_pretrained(name_or_path, revision=revision)
        return cls(model, tokenizer, **kwargs)

    def _content_mask(self, encoding: Any, prefixes: list[str]) -> torch.Tensor:
        """Boolean ``(batch, seq)`` mask of the positions whose activations are kept."""
        mask = encoding["attention_mask"].bool() & ~encoding["special_tokens_mask"].bool()
        if not self.exclude_prefix or not any(prefixes):
            return mask

        if "offset_mapping" in encoding:
            # A token belongs to the prefix when its character span ends inside the prefix. A
            # token that straddles the boundary carries text, so it is kept.
            prefix_chars = torch.tensor([len(prefix) for prefix in prefixes]).unsqueeze(1)
            return mask & (encoding["offset_mapping"][..., 1] > prefix_chars)

        # Slow tokenizers report no offsets: drop the first n content tokens of each row, where
        # n is the number of tokens the prefix yields when tokenized on its own.
        for row, prefix in enumerate(prefixes):
            if prefix not in self._prefix_token_counts:
                ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
                self._prefix_token_counts[prefix] = len(ids)

            content_positions = mask[row].nonzero().squeeze(-1)
            mask[row, content_positions[: self._prefix_token_counts[prefix]]] = False
        return mask

    @torch.no_grad()
    def encode_batch(self, prefixes: list[str], texts: list[str]) -> torch.Tensor:
        """Encode one batch and return float32 activations ``(n_content_tokens, d_in)``.

        The rows follow the batch order, then the token order inside each text. The result
        lives on ``self.device`` and may have zero rows when every text is empty.

        Raises:
            ValueError: if ``prefixes`` and ``texts`` differ in length.
        """
        encoding = self.tokenizer(
            [prefix + text for prefix, text in zip(prefixes, texts, strict=True)],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
            return_offsets_mapping=bool(getattr(self.tokenizer, "is_fast", False)),
            return_tensors="pt",
        )
        mask = self._content_mask(encoding, prefixes).to(self.device)

        autocast = (
            torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)
            if self.amp_dtype is not None
            else nullcontext()
        )
        with autocast:
            output = self.model(
                input_ids=encoding["input_ids"].to(self.device),
                attention_mask=encoding["attention_mask"].to(self.device),
                output_hidden_states=self.layer is not None,
            )
        hidden = output.last_hidden_state if self.layer is None else output.hidden_states[self.layer]
        activations = hidden[mask].float()

        self.texts_seen += len(texts)
        self.tokens_seen += activations.shape[0]
        return activations

    def stream(self, texts: Iterable[tuple[str, str]]) -> Iterator[torch.Tensor]:
        """Yield one activation tensor per forward pass.

        Texts are gathered into pools and sorted by length inside each pool so that a forward
        pass pads as little as possible. The output therefore follows the document order only
        loosely and consecutive rows are strongly correlated; pass it through
        ``shuffled_batches`` before training.
        """
        pool: list[tuple[str, str]] = []
        pool_size = self.text_batch_size * _POOL_BATCHES

        for item in texts:
            pool.append(item)
            if len(pool) >= pool_size:
                yield from self._encode_pool(pool)
                pool = []

        if pool:
            yield from self._encode_pool(pool)

    def _encode_pool(self, pool: list[tuple[str, str]]) -> Iterator[torch.Tensor]:
        pool.sort(key=lambda item: len(item[0]) + len(item[1]))

        for start in range(0, len(pool), self.text_batch_size):
            chunk = pool[start : start + self.text_batch_size]
            activations = self.encode_batch(
                [prefix for prefix, _ in chunk],
                [text for _, text in chunk],
            )
            if activations.shape[0]:
                yield activations

    def collect(self, texts: Iterable[tuple[str, str]], n_tokens: int) -> torch.Tensor:
        """Collect up to ``n_tokens`` activations into one CPU tensor, for evaluation.

        Raises:
            ValueError: if the text source produces no activations at all.
        """
        chunks: list[torch.Tensor] = []
        total = 0

        for activations in self.stream(texts):
            chunks.append(activations.cpu())
            total += activations.shape[0]
            if total >= n_tokens:
                break

        if not chunks:
            raise ValueError("No activations collected; the text source is empty.")
        return torch.cat(chunks)[:n_tokens]


def shuffled_batches(
    stream: Iterable[torch.Tensor],
    *,
    batch_tokens: int,
    buffer_tokens: int,
    seed: int = 0,
) -> Iterator[torch.Tensor]:
    """Mix activations from many documents and emit batches of exactly ``batch_tokens`` rows.

    Tokens of one document are highly correlated. They are accumulated in a buffer of
    ``buffer_tokens`` rows; whenever it fills up, a random half is emitted and the other half
    stays to be mixed with the next refill. When the stream ends the rest is emitted and the
    final partial batch is dropped. Every activation is emitted at most once.

    The buffer is allocated once on the device of the first tensor. Peak memory is about 1.5
    times ``buffer_tokens * d * 4`` bytes.

    Raises:
        ValueError: if ``buffer_tokens`` is smaller than two batches.
    """
    if batch_tokens < 1:
        raise ValueError("batch_tokens must be positive.")
    if buffer_tokens < 2 * batch_tokens:
        raise ValueError("buffer_tokens must be at least 2 * batch_tokens.")

    generator = torch.Generator().manual_seed(seed)
    buffer: torch.Tensor | None = None
    filled = 0

    def emit(storage: torch.Tensor, n_rows: int, n_emit: int) -> Iterator[torch.Tensor]:
        """Emit ``n_emit`` random rows of ``storage[:n_rows]`` and compact the rest to the front."""
        nonlocal filled
        permutation = torch.randperm(n_rows, generator=generator).to(storage.device)

        for start in range(0, n_emit, batch_tokens):
            yield storage[permutation[start : start + batch_tokens]]

        kept = storage[permutation[n_emit:]]
        filled = kept.shape[0]
        storage[:filled] = kept

    for activations in stream:
        if buffer is None:
            buffer = activations.new_empty((buffer_tokens, activations.shape[1]))

        offset = 0
        while offset < activations.shape[0]:
            take = min(buffer_tokens - filled, activations.shape[0] - offset)
            buffer[filled : filled + take] = activations[offset : offset + take]
            filled += take
            offset += take

            if filled == buffer_tokens:
                half = (buffer_tokens // 2 // batch_tokens) * batch_tokens
                yield from emit(buffer, filled, half)

    if buffer is not None:
        yield from emit(buffer, filled, (filled // batch_tokens) * batch_tokens)
