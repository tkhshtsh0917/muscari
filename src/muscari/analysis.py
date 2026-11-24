"""Look at an SAE the way a retrieval index would: per text, and per latent.

A sparse retriever does not store token activations. It pools them into one vector per query
or document, usually with a maximum over tokens, and an inverted index then pays for every
non-zero dimension. The functions here measure the dictionary after that pooling, and list what
each latent responds to.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any

import torch

from .activations import EncodedText
from .export import DENSE_FREQUENCY, ExportedSAE


def _token_batches(texts: Sequence[EncodedText], batch_tokens: int) -> Iterator[list[EncodedText]]:
    """Group whole texts into batches of roughly ``batch_tokens`` activations."""
    batch: list[EncodedText] = []
    size = 0

    for text in texts:
        batch.append(text)
        size += text.activations.shape[0]
        if size >= batch_tokens:
            yield batch
            batch, size = [], 0

    if batch:
        yield batch


def _firings(sae: ExportedSAE, batch: Sequence[EncodedText]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode the tokens of ``batch`` and return the row, latent and value of every firing, on the CPU.

    Only the encoding runs on the device of the SAE; the bookkeeping that follows is small.
    """
    features = sae.encode(torch.cat([text.activations for text in batch]).to(sae.encoder_weight.device))
    rows, latents = (features > 0).nonzero(as_tuple=True)
    return rows.cpu(), latents.cpu(), features[rows, latents].float().cpu()


def _token_layout(batch: Sequence[EncodedText]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the number of tokens of each text in ``batch`` and, for every token, the row of its text."""
    lengths = torch.tensor([text.activations.shape[0] for text in batch])
    return lengths, torch.repeat_interleave(torch.arange(len(batch)), lengths)


@torch.no_grad()
def evaluate_texts(
    sae: ExportedSAE,
    texts: Sequence[EncodedText],
    batch_tokens: int = 1024,
) -> dict[str, Any]:
    """Measure the pooled, per-text vectors that a sparse index would store.

    A latent is active in a text when it is active on at least one of its tokens, which is the
    sparsity pattern of max pooling. Saturating the values, for example with ``log1p``, does
    not change it.

    Returns:
        ``by_prefix``: for each prefix, the number of texts, the mean tokens per text, the mean,
        median and 95th percentile of active latents per text (``l0``), the share of latents
        never active (``never_fired_frac``), the share active in more than 10% of the texts
        (``dense_frac``) and the largest share of texts one latent is active in
        (``max_document_frequency``).

        ``flops``: for each pair of prefixes, the sum over latents of the product of their
        activation probabilities. This estimates the number of multiplications when a text of
        one kind is scored against a text of the other through an inverted index. It is the
        FLOPS figure as naver/splade computes it from a query index and a document index.

    Raises:
        ValueError: if ``texts`` is empty.
    """
    if not texts:
        raise ValueError("texts is empty.")

    num_latents = sae.num_latents
    prefixes = list(dict.fromkeys(text.prefix for text in texts))
    prefix_index = {prefix: index for index, prefix in enumerate(prefixes)}
    text_counts = torch.zeros(len(prefixes), dtype=torch.long)
    token_counts = torch.zeros(len(prefixes), dtype=torch.long)
    document_counts = torch.zeros((len(prefixes), num_latents), dtype=torch.long)
    active_per_text: list[torch.Tensor] = []
    text_prefixes: list[torch.Tensor] = []

    for batch in _token_batches(texts, batch_tokens):
        lengths, text_index = _token_layout(batch)
        batch_prefix = torch.tensor([prefix_index[text.prefix] for text in batch])
        rows, latents, _ = _firings(sae, batch)

        # One entry per (text, latent) pair, however many tokens of the text fire the latent.
        pairs = torch.unique(text_index[rows] * num_latents + latents)
        pair_text, pair_latent = pairs // num_latents, pairs % num_latents

        document_counts.view(-1).index_add_(
            0, batch_prefix[pair_text] * num_latents + pair_latent, torch.ones_like(pairs)
        )
        active_per_text.append(torch.bincount(pair_text, minlength=len(batch)))
        text_prefixes.append(batch_prefix)
        text_counts += torch.bincount(batch_prefix, minlength=len(prefixes))
        token_counts.index_add_(0, batch_prefix, lengths)

    active = torch.cat(active_per_text).float()
    text_prefix = torch.cat(text_prefixes)
    probabilities = {
        prefix: document_counts[index].float() / text_counts[index] for prefix, index in prefix_index.items()
    }
    by_prefix = {}
    for prefix, index in prefix_index.items():
        l0 = active[text_prefix == index]
        probability = probabilities[prefix]
        by_prefix[prefix] = {
            "texts": int(text_counts[index]),
            "tokens_per_text": int(token_counts[index]) / int(text_counts[index]),
            "l0": l0.mean().item(),
            "l0_median": l0.median().item(),
            "l0_p95": l0.quantile(0.95).item(),
            "never_fired_frac": (probability == 0).float().mean().item(),
            "dense_frac": (probability > DENSE_FREQUENCY).float().mean().item(),
            "max_document_frequency": probability.max().item(),
        }

    flops = [
        {"prefixes": [first, second], "flops": (probabilities[first] * probabilities[second]).sum().item()}
        for position, first in enumerate(prefixes)
        for second in prefixes[position:]
    ]
    return {"by_prefix": by_prefix, "flops": flops}


@torch.no_grad()
def latent_statistics(
    sae: ExportedSAE,
    texts: Sequence[EncodedText],
    top_n: int = 10,
    batch_tokens: int = 1024,
) -> list[dict[str, Any]]:
    """For every latent, how often it fires, where in a text, and on which vocabulary tokens.

    A latent can follow a token, a kind of text or a position rather than a meaning. The record
    of a latent holds what is needed to tell these apart:

    ``latent``: its index.
    ``frequency``: the share of tokens it fires on.
    ``frequency_by_prefix``: the same share among the tokens of each prefix.
    ``text_frequency``: the share of texts in which it fires on at least one token.
    ``mean_activation``: its mean value where it fires (0 if it never does).
    ``first_token_share``, ``last_token_share``: the part of its firings that fall on the first
    and on the last content token of a text. Compare with ``1 / tokens per text``.
    ``mean_position``: the mean relative position of its firings, from 0 for the first token
    of a text to 1 for the last (0.5 if it never fires).
    ``top_tokens``: up to ``top_n`` entries ``{"token_id", "share", "token_rate"}``, where
    ``share`` is the part of the latent's firings that fall on this token and ``token_rate`` is
    the part of this token's occurrences on which the latent fires. A latent with one token at
    a ``share`` and ``token_rate`` near 1 behaves like a vocabulary entry for that token.

    Records are sorted by ``frequency``, highest first. A text cut by ``max_length`` or by the
    token budget of the collection ends early, so its last token is not the end of the original.

    The number of distinct ``(latent, token)`` pairs is kept in memory; a few million evaluated
    tokens are fine.

    Raises:
        ValueError: if ``texts`` is empty or ``top_n`` is not positive.
    """
    if not texts:
        raise ValueError("texts is empty.")
    if top_n < 1:
        raise ValueError("top_n must be positive.")

    num_latents = sae.num_latents
    prefixes = list(dict.fromkeys(text.prefix for text in texts))
    prefix_index = {prefix: index for index, prefix in enumerate(prefixes)}
    vocabulary = int(max(int(text.token_ids.max()) for text in texts)) + 1

    fire_counts = torch.zeros((len(prefixes), num_latents), dtype=torch.long)
    prefix_tokens = torch.zeros(len(prefixes), dtype=torch.long)
    text_counts = torch.zeros(num_latents, dtype=torch.long)
    activation_sums = torch.zeros(num_latents, dtype=torch.float64)
    position_sums = torch.zeros(num_latents, dtype=torch.float64)
    first_counts = torch.zeros(num_latents, dtype=torch.long)
    last_counts = torch.zeros(num_latents, dtype=torch.long)
    token_counts = torch.zeros(vocabulary, dtype=torch.long)
    pair_keys = []

    for batch in _token_batches(texts, batch_tokens):
        lengths, text_index = _token_layout(batch)
        starts = lengths.cumsum(0) - lengths
        position = torch.arange(int(lengths.sum())) - starts[text_index]
        last = lengths[text_index] - 1
        relative = position.double() / last.clamp_min(1)
        token_prefix = torch.tensor([prefix_index[text.prefix] for text in batch])[text_index]
        token_ids = torch.cat([text.token_ids for text in batch]).cpu()
        rows, latents, values = _firings(sae, batch)

        fire_counts.view(-1).index_add_(0, token_prefix[rows] * num_latents + latents, torch.ones_like(latents))
        prefix_tokens += torch.bincount(token_prefix, minlength=len(prefixes))
        text_pairs = torch.unique(text_index[rows] * num_latents + latents)
        text_counts += torch.bincount(text_pairs % num_latents, minlength=num_latents)
        activation_sums.index_add_(0, latents, values.double())
        position_sums.index_add_(0, latents, relative[rows])
        first_counts += torch.bincount(latents[position[rows] == 0], minlength=num_latents)
        last_counts += torch.bincount(latents[position[rows] == last[rows]], minlength=num_latents)
        token_counts += torch.bincount(token_ids, minlength=vocabulary)
        pair_keys.append(latents * vocabulary + token_ids[rows])

    keys, pair_counts = torch.cat(pair_keys).unique(return_counts=True)
    pair_latents = keys // vocabulary
    pair_tokens = keys % vocabulary

    # Order the pairs by latent, and by count inside each latent, then keep the first top_n.
    by_count = pair_counts.argsort(descending=True, stable=True)
    order = by_count[pair_latents[by_count].argsort(stable=True)]
    sorted_latents = pair_latents[order]
    group_start = torch.searchsorted(sorted_latents, torch.arange(num_latents))
    rank = torch.arange(order.shape[0]) - group_start[sorted_latents]
    kept = order[rank < top_n]

    fired = fire_counts.sum(0)
    top_tokens: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for latent, token, count in zip(
        pair_latents[kept].tolist(),
        pair_tokens[kept].tolist(),
        pair_counts[kept].tolist(),
        strict=True,
    ):
        top_tokens[latent].append(
            {
                "token_id": token,
                "share": count / int(fired[latent]),
                "token_rate": count / int(token_counts[token]),
            }
        )

    n_tokens = int(prefix_tokens.sum())
    denominator = fired.clamp_min(1)
    by_prefix = (fire_counts.double() / prefix_tokens.unsqueeze(1)).tolist()
    mean_activations = (activation_sums / denominator).tolist()
    mean_positions = torch.where(
        fired > 0, position_sums / denominator, torch.tensor(0.5, dtype=torch.float64)
    ).tolist()
    first_shares = (first_counts.double() / denominator).tolist()
    last_shares = (last_counts.double() / denominator).tolist()
    text_frequencies = (text_counts.double() / len(texts)).tolist()
    fired_list = fired.tolist()

    return [
        {
            "latent": latent,
            "frequency": fired_list[latent] / n_tokens,
            "frequency_by_prefix": {prefix: by_prefix[index][latent] for index, prefix in enumerate(prefixes)},
            "text_frequency": text_frequencies[latent],
            "mean_activation": mean_activations[latent],
            "first_token_share": first_shares[latent],
            "last_token_share": last_shares[latent],
            "mean_position": mean_positions[latent],
            "top_tokens": top_tokens.get(latent, []),
        }
        for latent in fired.argsort(descending=True, stable=True).tolist()
    ]
