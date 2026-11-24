"""Collect token-level hidden states from a frozen encoder, on the fly.

An encoder is small next to a decoder LLM, so activations are recomputed instead of cached:
one billion tokens of 768-dimensional float32 activations would need about 3 TB of disk.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import nullcontext
from typing import Any, NamedTuple

import torch

from .devices import resolve_device, synchronize

# Texts are length-sorted inside pools of this many forward passes to limit padding.
_POOL_BATCHES = 16


class EncodedBatch(NamedTuple):
    """Content-token activations of one forward pass.

    Attributes:
        activations: float32 ``(n_content_tokens, d_in)`` on the encoder's device; rows follow
            the batch order, then the token order inside each text.
        tokens_per_text: CPU tensor ``(batch,)`` with the number of rows of each text.
        token_ids: CPU tensor ``(n_content_tokens,)`` with the vocabulary id behind each row.
    """

    activations: torch.Tensor
    tokens_per_text: torch.Tensor
    token_ids: torch.Tensor


class EncodedText(NamedTuple):
    """Content-token activations of one text, on the CPU.

    ``prefix`` is the kind of the text: its prefix, or its label where labels are used.
    """

    prefix: str
    activations: torch.Tensor
    token_ids: torch.Tensor


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
        prefixes: the prefix to prepend for each label, for texts that arrive as
            ``(label, text)``. A kind that is not listed is itself the prefix. With
            ``{"query": "", "document": ""}`` an encoder without prefixes still gets its
            statistics split into queries and documents.
        device: ``"auto"``, a device string or a ``torch.device``.
        amp_dtype: run the encoder under autocast with this dtype, for example
            ``torch.bfloat16`` on CUDA. Activations are always returned as float32.

    Attributes:
        d_in: hidden size of the encoder, the input width of the SAE.
        texts_seen: number of texts encoded so far.
        tokens_seen: number of content-token activations returned so far.
        tokens_by_prefix: the same count split by the kind of the text (its prefix or label).
        encode_seconds: wall-clock time spent tokenizing and running the encoder.
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
        prefixes: Mapping[str, str] | None = None,
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
        self.prefixes = dict(prefixes or {})
        self.amp_dtype = amp_dtype
        self.d_in = int(model.config.hidden_size)

        self.reset_counters()
        self._prefix_token_counts: dict[str, int] = {}

    def reset_counters(self) -> None:
        """Start the usage counters again from zero."""
        self.texts_seen = 0
        self.tokens_seen = 0
        self.tokens_by_prefix: dict[str, int] = defaultdict(int)
        self.encode_seconds = 0.0

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
    def encode_texts(self, kinds: list[str], texts: list[str]) -> EncodedBatch:
        """Encode one batch and return its content-token activations.

        ``kinds`` holds, for each text, the prefix to prepend, or a label listed in
        ``prefixes``. A text may contribute zero rows, for example when it is empty.

        Raises:
            ValueError: if ``kinds`` and ``texts`` differ in length.
        """
        started = time.perf_counter()
        prefixes = [self.prefixes.get(kind, kind) for kind in kinds]
        encoding = self.tokenizer(
            [prefix + text for prefix, text in zip(prefixes, texts, strict=True)],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
            return_offsets_mapping=bool(getattr(self.tokenizer, "is_fast", False)),
            return_tensors="pt",
        )
        mask = self._content_mask(encoding, prefixes)
        tokens_per_text = mask.sum(dim=1)

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
        activations = hidden[mask.to(self.device)].float()
        synchronize(self.device)

        self.texts_seen += len(texts)
        self.tokens_seen += activations.shape[0]
        for kind, count in zip(kinds, tokens_per_text.tolist(), strict=True):
            self.tokens_by_prefix[kind] += count
        self.encode_seconds += time.perf_counter() - started
        return EncodedBatch(activations, tokens_per_text, encoding["input_ids"][mask])

    def encode_batch(self, kinds: list[str], texts: list[str]) -> torch.Tensor:
        """Encode one batch and return only the activations of ``encode_texts``."""
        return self.encode_texts(kinds, texts).activations

    def stream(self, texts: Iterable[tuple[str, str]]) -> Iterator[torch.Tensor]:
        """Yield one activation tensor per forward pass.

        Texts are gathered into pools and sorted by length inside each pool so that a forward
        pass pads as little as possible. The output therefore follows the document order only
        loosely and consecutive rows are strongly correlated; pass it through
        ``shuffled_batches`` before training.
        """
        for pool in self._pools(texts):
            for batch, _ in self._encode_pool(pool):
                yield batch.activations

    def _encode_pool(
        self,
        pool: list[tuple[str, str]],
    ) -> Iterator[tuple[EncodedBatch, list[str]]]:
        """Yield ``(batch, kinds)`` for each forward pass over ``pool`` that produced activations."""
        pool.sort(key=lambda item: len(self.prefixes.get(item[0], item[0])) + len(item[1]))

        for start in range(0, len(pool), self.text_batch_size):
            chunk = pool[start : start + self.text_batch_size]
            kinds = [kind for kind, _ in chunk]
            batch = self.encode_texts(kinds, [text for _, text in chunk])
            if batch.activations.shape[0]:
                yield batch, kinds

    def _pools(self, texts: Iterable[tuple[str, str]]) -> Iterator[list[tuple[str, str]]]:
        pool: list[tuple[str, str]] = []
        pool_size = self.text_batch_size * _POOL_BATCHES

        for item in texts:
            pool.append(item)
            if len(pool) >= pool_size:
                yield pool
                pool = []

        if pool:
            yield pool

    def collect(self, texts: Iterable[tuple[str, str]], n_tokens: int) -> torch.Tensor:
        """Collect up to ``n_tokens`` activations into one CPU tensor, for evaluation.

        Raises:
            ValueError: if the text source produces no activations at all.
        """
        return torch.cat([text.activations for text in self.collect_texts(texts, n_tokens)])

    def collect_by_prefix(self, texts: Iterable[tuple[str, str]], n_tokens: int) -> dict[str, torch.Tensor]:
        """Collect up to ``n_tokens`` activations in total, grouped by the prefix of their text.

        The groups keep the proportions in which the prefixes occur in ``texts``, so that an
        evaluation can be reported per side (queries, documents) as well as overall. Each value
        is a CPU tensor ``(n, d_in)``.

        Raises:
            ValueError: if the text source produces no activations at all.
        """
        return group_by_prefix(self.collect_texts(texts, n_tokens))

    def collect_texts(self, texts: Iterable[tuple[str, str]], n_tokens: int) -> list[EncodedText]:
        """Collect texts one by one until ``n_tokens`` activations have been gathered.

        Text boundaries are kept, which is what an evaluation of pooled, per-text vectors
        needs. Texts without content tokens are left out, and the last text is cut where the
        token budget ends.

        Raises:
            ValueError: if the text source produces no activations at all.
        """
        collected: list[EncodedText] = []
        remaining = n_tokens

        for pool in self._pools(texts):
            for batch, prefixes in self._encode_pool(pool):
                lengths = batch.tokens_per_text.tolist()
                rows = batch.activations.cpu().split(lengths)
                ids = batch.token_ids.split(lengths)
                for prefix, text_rows, text_ids in zip(prefixes, rows, ids, strict=True):
                    if remaining <= 0:
                        return collected
                    if text_rows.shape[0]:
                        collected.append(EncodedText(prefix, text_rows[:remaining], text_ids[:remaining]))
                        remaining -= min(remaining, text_rows.shape[0])

        if not collected:
            raise ValueError("No activations collected; the text source is empty.")
        return collected


def group_by_prefix(texts: Iterable[EncodedText]) -> dict[str, torch.Tensor]:
    """Concatenate the activations of ``texts`` per prefix, in order of first appearance."""
    groups: dict[str, list[torch.Tensor]] = defaultdict(list)
    for text in texts:
        groups[text.prefix].append(text.activations)
    return {prefix: torch.cat(parts) for prefix, parts in groups.items()}


def shuffled_batches(
    stream: Iterable[torch.Tensor],
    *,
    batch_tokens: int,
    buffer_tokens: int,
    seed: int = 42,
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
