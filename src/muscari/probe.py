"""Rerank candidates with a frozen encoder and an SAE, without any retrieval training.

Reconstruction error says how close the SAE's output is to the encoder's. It does not say
whether what is lost matters. This module asks two questions that do bear on retrieval, on a
reranking dataset where every query comes with its own candidate passages:

* Does the encoder still rank well when its token states are replaced by the SAE's
  reconstruction? This is the counterpart, for an embedding model, of the language-model loss
  recovered that is reported for SAEs on language models.
* How well do the SAE's features rank on their own, pooled the way an SAE-based sparse
  retriever pools them, before that retriever is trained?
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from .activations import EncoderActivationSource
from .export import ExportedSAE

VARIANTS = ("original", "spliced", "empty")
# Weights of the dense score in the weighted fusion of the reranking probe.
HYBRID_DENSE_WEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass(frozen=True)
class RerankingExample:
    """One query with its candidate passages; ``labels`` are 1 for relevant and 0 otherwise.

    ``query_id`` identifies the query across runs, so per-query results can be paired.
    """

    query: str
    passages: tuple[str, ...]
    labels: tuple[int, ...]
    query_id: str | None = None


class PooledText(NamedTuple):
    """What a retriever would keep of one text.

    Attributes:
        indices: ``(nnz,)`` latents active on at least one content token, ascending.
        values: ``(nnz,)`` the maximum over the content tokens of ``log1p(activation)``.
        embeddings: ``(3, d_in)`` mean-pooled sentence embeddings in the order of ``VARIANTS``:
            from the encoder's own token states, with the content tokens replaced by the SAE's
            reconstruction, and with them replaced by the SAE's decoder bias, which is what the
            SAE returns for an empty code.
    """

    indices: torch.Tensor
    values: torch.Tensor
    embeddings: torch.Tensor


def group_reranking_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    query_column: str,
    passage_column: str,
    label_column: str,
    group_column: str | None = None,
    title_column: str | None = None,
    max_queries: int | None = None,
) -> list[RerankingExample]:
    """Group long-format rows, one candidate passage per row, into one example per query.

    Rows belong together when they share the value of ``group_column``, or the query text if
    no such column is given; that value becomes the ``query_id``. Queries are taken in order
    of first appearance; with ``max_queries`` only the first ones are kept, so shuffle the rows
    beforehand to draw a sample. Queries without a relevant candidate cannot be scored and are
    left out.

    With ``title_column`` a passage is ``"{title} {text}"``, the form the JQaRA evaluator uses.

    Raises:
        ValueError: if a column is missing or no query with a relevant candidate is found.
    """
    key_column = group_column or query_column
    needed = {key_column, query_column, passage_column, label_column} | ({title_column} if title_column else set())
    grouped: dict[Any, tuple[str, list[str], list[int]]] = {}

    for row in rows:
        if missing := sorted(needed - set(row)):
            raise ValueError(f"The data has no column {missing}; it has {sorted(row)}.")
        key = row[key_column]
        if key not in grouped:
            if max_queries is not None and len(grouped) >= max_queries:
                continue
            grouped[key] = (str(row[query_column]), [], [])
        passage = str(row[passage_column])
        if title_column:
            passage = f"{row[title_column]} {passage}"
        grouped[key][1].append(passage)
        grouped[key][2].append(int(row[label_column] > 0))

    examples = [
        RerankingExample(query, tuple(passages), tuple(labels), str(key))
        for key, (query, passages, labels) in grouped.items()
        if any(labels)
    ]
    if not examples:
        raise ValueError("No query with a relevant candidate was found; check the column names.")
    return examples


@torch.no_grad()
def pool_texts(
    source: EncoderActivationSource,
    sae: ExportedSAE,
    kind: str,
    texts: Sequence[str],
    token_chunk: int = 2048,
) -> list[PooledText]:
    """Encode ``texts`` and reduce each one to its sparse vector and its sentence embeddings.

    The sparse vector is the one an SAE-based sparse retriever starts from: ``log1p`` of the
    SAE's activations, then the maximum over the content tokens. The sentence embeddings are
    means over every attended position, which is the pooling of the ruri family and of most
    encoders trained as embedding models; they mean little for an encoder that was not.

    The SAE must have been trained on ``last_hidden_state`` for the embeddings to be valid.
    """
    device = sae.encoder_weight.device
    num_latents = sae.num_latents
    pooled: list[PooledText | None] = [None] * len(texts)
    order = sorted(range(len(texts)), key=lambda position: len(texts[position]))

    for start in range(0, len(order), source.text_batch_size):
        positions = order[start : start + source.text_batch_size]
        batch = source.encode_texts([kind] * len(positions), [texts[position] for position in positions])
        activations = batch.activations.to(device)
        text_index = torch.repeat_interleave(torch.arange(len(positions)), batch.tokens_per_text)

        # Per text: the reconstructed and the original content states summed, and every firing.
        reconstructed = torch.zeros((len(positions), sae.d_in), device=device)
        original = torch.zeros_like(reconstructed)
        keys, values = [], []
        for begin in range(0, activations.shape[0], token_chunk):
            chunk = activations[begin : begin + token_chunk]
            rows_of_chunk = text_index[begin : begin + token_chunk]
            features = sae.encode(chunk)
            reconstructed.index_add_(0, rows_of_chunk.to(device), sae.decode(features))
            original.index_add_(0, rows_of_chunk.to(device), chunk)

            rows, latents = (features > 0).nonzero(as_tuple=True)
            keys.append(rows_of_chunk[rows.cpu()] * num_latents + latents.cpu())
            values.append(torch.log1p(features[rows, latents]).float().cpu())

        if keys:
            unique_keys, inverse = torch.cat(keys).unique(return_inverse=True)
            maxima = torch.zeros(unique_keys.shape[0]).scatter_reduce_(0, inverse, torch.cat(values), "amax")
        else:
            unique_keys, maxima = torch.zeros(0, dtype=torch.long), torch.zeros(0)
        boundaries = torch.searchsorted(unique_keys, torch.arange(len(positions) + 1) * num_latents)

        attended = batch.attended.to(device).unsqueeze(1)
        content = batch.tokens_per_text.to(device).unsqueeze(1)
        sums = batch.hidden_sums.to(device)
        embeddings = torch.stack(
            [
                sums / attended,
                (sums - original + reconstructed) / attended,
                (sums - original + content * sae.decoder_bias) / attended,
            ],
            dim=1,
        ).cpu()

        for row, position in enumerate(positions):
            span = slice(int(boundaries[row]), int(boundaries[row + 1]))
            pooled[position] = PooledText(unique_keys[span] % num_latents, maxima[span], embeddings[row])

    return [text for text in pooled if text is not None]


def ranking_metrics(scores: torch.Tensor, labels: torch.Tensor, k: int = 10) -> dict[str, float]:
    """nDCG@k and MRR@k of one query with binary ``labels``, ranking by descending ``scores``.

    Candidates with equal scores keep their input order, so shuffle them first if that order
    could carry information. The ideal ranking for nDCG puts every relevant candidate first.

    Raises:
        ValueError: if no candidate is relevant.
    """
    relevant = int(labels.sum())
    if relevant == 0:
        raise ValueError("The query has no relevant candidate.")

    top = labels[scores.argsort(descending=True, stable=True)[:k]].double()
    discounts = 1.0 / torch.log2(torch.arange(2, top.shape[0] + 2).double())
    ideal = discounts[: min(relevant, k)].sum()
    hits = top.nonzero()
    return {
        f"ndcg@{k}": float((top * discounts).sum() / ideal),
        f"mrr@{k}": 1.0 / (int(hits[0]) + 1) if hits.numel() else 0.0,
    }


def sparse_scores(query: PooledText, passages: Sequence[PooledText], weights: torch.Tensor) -> torch.Tensor:
    """Dot products of the query's sparse vector with each passage's, with one weight per latent."""
    dense_query = torch.zeros(weights.shape[0], dtype=torch.float64)
    dense_query[query.indices] = query.values.double()
    dense_query *= weights

    lengths = torch.tensor([passage.indices.shape[0] for passage in passages])
    owner = torch.repeat_interleave(torch.arange(len(passages)), lengths)
    products = dense_query[torch.cat([passage.indices for passage in passages])] * torch.cat(
        [passage.values for passage in passages]
    )
    return torch.zeros(len(passages), dtype=torch.float64).index_add_(0, owner, products)


def fuse_scores(dense: torch.Tensor, sparse: torch.Tensor, dense_weight: float) -> torch.Tensor:
    """``dense_weight * z(dense) + (1 - dense_weight) * z(sparse)`` over one query's candidates.

    ``z`` standardises a score list to zero mean and unit variance among the candidates, so the
    two scales become comparable; a constant list contributes nothing. The standardisation
    needs all candidates of the query, as in reranking.
    """

    def standardise(scores: torch.Tensor) -> torch.Tensor:
        scores = scores.double()
        spread = scores.std() if scores.numel() > 1 else torch.tensor(0.0)
        return (scores - scores.mean()) / spread if spread > 0 else torch.zeros_like(scores)

    return dense_weight * standardise(dense) + (1.0 - dense_weight) * standardise(sparse)


def reciprocal_rank_fusion(score_lists: Sequence[torch.Tensor], constant: float = 60.0) -> torch.Tensor:
    """Sum over the lists of ``1 / (constant + rank)``, with rank 1 for the highest score.

    Reciprocal rank fusion (Cormack et al., 2009) uses only the ranks, so it has nothing to
    tune; 60 is the constant of that paper. Equal scores keep their input order.
    """
    fused = torch.zeros(score_lists[0].shape[0], dtype=torch.float64)
    for scores in score_lists:
        ranks = torch.empty_like(fused)
        ranks[scores.argsort(descending=True, stable=True)] = torch.arange(1, fused.shape[0] + 1).double()
        fused += 1.0 / (constant + ranks)
    return fused


def _summary(values: Sequence[float]) -> dict[str, float]:
    ordered = torch.tensor(sorted(values), dtype=torch.float64)
    return {"mean": float(ordered.mean()), "p05": float(ordered.quantile(0.05)), "min": float(ordered[0])}


def probe_reranking(
    source: EncoderActivationSource,
    sae: ExportedSAE,
    examples: Sequence[RerankingExample],
    *,
    query_kind: str = "",
    passage_kind: str = "",
    k: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Rerank the candidates of every query densely and sparsely, with nothing trained.

    Args:
        source: the encoder the SAE was trained on.
        sae: the exported SAE.
        examples: queries with candidates and binary labels.
        query_kind, passage_kind: the prefix to prepend to queries and passages, or a label
            that ``source`` maps to a prefix.
        k: cut-off of nDCG and MRR.
        seed: candidates are shuffled with this seed before ranking, so that ties are not
            broken by the order of the dataset.

    Returns:
        Every nDCG@k and MRR@k below is a mean over the queries and comes with its standard
        error under the key ``<metric>_stderr``.

        ``dense``: for each of ``original``, ``spliced`` and ``empty`` (see ``PooledText``),
        nDCG@k and MRR@k of ranking by cosine similarity of the sentence embeddings, and
        ``recovered``: ``(spliced - empty) / (original - empty)`` for each metric, the share
        of the encoder's ranking quality that survives the SAE, with 1 for all of it.
        ``cosine`` holds the similarity between the original and the spliced embedding of the
        same text.

        ``hybrid``: nDCG@k and MRR@k of rankings that fuse ``dense.original`` with
        ``sparse.plain``, both of which come from one pass through the encoder. ``rrf`` is
        reciprocal rank fusion, which has nothing to tune. ``weighted`` maps a weight of the
        dense score, from 0.1 to 0.9, to the ranking by ``fuse_scores``; choose the weight on
        one split and report it on another. Every entry also holds ``over_dense`` and
        ``over_sparse``: the mean per-query gain over each fused ranking, with the standard
        error of that paired difference.

        ``sparse``: nDCG@k and MRR@k of ranking by the dot product of the pooled sparse
        vectors: ``plain`` as they are, and ``idf`` with every latent weighted by
        ``ln(1 + (N - df + 0.5) / (df + 0.5))``, where ``df`` is the number of the ``N``
        distinct candidate passages it is active in. ``l0`` is the mean number of active
        latents of a query and of a passage.

    Raises:
        ValueError: if ``examples`` is empty.
    """
    if not examples:
        raise ValueError("examples is empty.")

    queries = list(dict.fromkeys(example.query for example in examples))
    passages = list(dict.fromkeys(passage for example in examples for passage in example.passages))
    query_vectors = dict(zip(queries, pool_texts(source, sae, query_kind, queries), strict=True))
    passage_vectors = dict(zip(passages, pool_texts(source, sae, passage_kind, passages), strict=True))

    document_frequency = torch.zeros(sae.num_latents, dtype=torch.float64)
    for vector in passage_vectors.values():
        document_frequency[vector.indices] += 1
    n_passages = len(passages)
    weights = {
        "plain": torch.ones(sae.num_latents, dtype=torch.float64),
        "idf": torch.log1p((n_passages - document_frequency + 0.5) / (document_frequency + 0.5)),
    }

    generator = torch.Generator().manual_seed(seed)
    per_query: dict[str, dict[str, list[float]]] = {}

    def add(name: str, scores: torch.Tensor, labels: torch.Tensor) -> None:
        for metric, value in ranking_metrics(scores, labels, k).items():
            per_query.setdefault(name, {}).setdefault(metric, []).append(value)

    for example in examples:
        shuffle = torch.randperm(len(example.passages), generator=generator)
        candidates = [passage_vectors[example.passages[int(position)]] for position in shuffle]
        labels = torch.tensor(example.labels)[shuffle]
        query = query_vectors[example.query]

        passage_embeddings = torch.stack([candidate.embeddings for candidate in candidates]).double()
        for index, variant in enumerate(VARIANTS):
            similarity = torch.cosine_similarity(
                passage_embeddings[:, index], query.embeddings[index].double().unsqueeze(0), dim=1
            )
            add(variant, similarity, labels)
            if variant == "original":
                dense_scores = similarity
        for name, weight in weights.items():
            scores = sparse_scores(query, candidates, weight)
            add(name, scores, labels)
            if name == "plain":
                plain_scores = scores
        add("rrf", reciprocal_rank_fusion([dense_scores, plain_scores]), labels)
        for dense_weight in HYBRID_DENSE_WEIGHTS:
            add(f"weighted-{dense_weight}", fuse_scores(dense_scores, plain_scores, dense_weight), labels)

    def cosine(vectors: Iterable[PooledText]) -> dict[str, float]:
        return _summary([float(torch.cosine_similarity(v.embeddings[0], v.embeddings[1], dim=0)) for v in vectors])

    totals: dict[str, dict[str, float]] = {}
    recovered = {}
    for name, metrics in per_query.items():
        totals[name] = {}
        for metric, values in metrics.items():
            scores = torch.tensor(values, dtype=torch.float64)
            totals[name][metric] = float(scores.mean())
            totals[name][f"{metric}_stderr"] = float(scores.std() / math.sqrt(len(values))) if len(values) > 1 else 0.0
    for metric in per_query["original"]:
        gap = totals["original"][metric] - totals["empty"][metric]
        recovered[metric] = (totals["spliced"][metric] - totals["empty"][metric]) / gap if gap else math.nan

    def hybrid(name: str) -> dict[str, Any]:
        """The fused ranking and its paired gain over each of the two rankings it fuses."""
        gains: dict[str, dict[str, float]] = {}
        for baseline, key in (("original", "over_dense"), ("plain", "over_sparse")):
            gains[key] = {}
            for metric, values in per_query[name].items():
                gain = torch.tensor(values, dtype=torch.float64) - torch.tensor(per_query[baseline][metric]).double()
                gains[key][metric] = float(gain.mean())
                gains[key][f"{metric}_stderr"] = float(gain.std() / math.sqrt(len(values))) if len(values) > 1 else 0.0
        return {**totals[name], **gains}

    return {
        "queries": len(examples),
        "passages": n_passages,
        "candidates_per_query": sum(len(example.passages) for example in examples) / len(examples),
        "relevant_per_query": sum(sum(example.labels) for example in examples) / len(examples),
        "dense": {
            **{variant: totals[variant] for variant in VARIANTS},
            "recovered": recovered,
            "cosine": {"query": cosine(query_vectors.values()), "passage": cosine(passage_vectors.values())},
        },
        "hybrid": {
            "rrf": hybrid("rrf"),
            "weighted": {str(weight): hybrid(f"weighted-{weight}") for weight in HYBRID_DENSE_WEIGHTS},
        },
        "sparse": {
            "plain": totals["plain"],
            "idf": totals["idf"],
            "l0": {
                "query": sum(v.indices.shape[0] for v in query_vectors.values()) / len(queries),
                "passage": sum(v.indices.shape[0] for v in passage_vectors.values()) / n_passages,
            },
        },
    }
