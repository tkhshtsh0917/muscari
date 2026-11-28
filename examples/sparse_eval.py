#!/usr/bin/env python3
"""Evaluate a sparse encoder: rerank candidates, retrieve from a corpus, and cap the vectors.

Any Sentence Transformers ``SparseEncoder`` can be evaluated: a model trained with
``train_retriever.py``, an untrained SAE head (``--sae``), or a published sparse retriever.

Reranking ranks each query's own candidates (JQaRA: 100 per query) by the dot product of the
sparse vectors. Retrieval ranks a whole corpus in the MTEB layout (configurations ``corpus``,
``queries`` and ``qrels``) and keeps the best ``--top-k`` documents per query while the corpus
is streamed through the encoder in chunks.

Every metric is reported for each pair of inference-time caps in ``--caps``: ``full`` keeps the
vectors as they are, ``64/400`` keeps the 64 largest weights of a query and the 400 largest of a
document. Each result also carries the per-query values, keyed by query id, so that two runs can
be compared on the same queries (``compare_runs.py``).

    python examples/sparse_eval.py --model runs/retriever/model \\
        --benchmark jqara --split test --retrieval mteb/MIRACLJaRetrievalLite \\
        --caps full 64/256 64/128 64/64 --out runs/retriever/eval.json

Published models whose modules are not part of Sentence Transformers need
``--trust-remote-code``, which runs code from the model repository; read that code first.
A local directory written by this project loads without it. Needs the "retrieval" extra.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from sentence_transformers.sparse_encoder import SparseEncoder

from muscari.cli import BENCHMARKS, load_rows
from muscari.probe import RerankingExample, group_reranking_rows, ranking_metrics
from muscari.retrieval import build_sparse_encoder

Cap = tuple[int | None, int | None]
FULL: Cap = (None, None)
# Modules that Sentence Transformers, this project or its examples provide; anything else is remote code.
LOCAL_MODULE_PREFIXES = ("sentence_transformers.", "muscari.", "vocabulary_head.")


# --------------------------------------------------------------------------------------- caps


def parse_cap(value: str) -> Cap:
    """``full`` or ``QUERY/DOCUMENT``, where either side may be ``full``: ``64/400``, ``full/256``."""
    if value == "full":
        return FULL
    parts = value.split("/")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"A cap is 'full' or 'QUERY/DOCUMENT', not {value!r}.")
    sides: list[int | None] = []
    for part in parts:
        if part == "full":
            sides.append(None)
        elif part.isdigit() and int(part) > 0:
            sides.append(int(part))
        else:
            raise argparse.ArgumentTypeError(f"A cap side is a positive integer or 'full', not {part!r}.")
    return sides[0], sides[1]


def cap_label(cap: Cap) -> str:
    if cap == FULL:
        return "full"
    return "/".join("full" if side is None else str(side) for side in cap)


def keep_largest(vectors: torch.Tensor, count: int | None) -> torch.Tensor:
    """Each row with all but its ``count`` largest weights set to zero; ``None`` keeps everything."""
    if count is None or count >= vectors.shape[1]:
        return vectors
    values, indices = vectors.topk(count, dim=1)
    return torch.zeros_like(vectors).scatter_(1, indices, values)


class _Sparsity:
    """Running number of active dimensions per text and share of texts per dimension."""

    def __init__(self, dimension: int) -> None:
        self.texts = 0
        self.active = 0.0
        self.presence = torch.zeros(dimension, dtype=torch.float64)

    def add(self, vectors: torch.Tensor) -> None:
        active = (vectors > 0).float()
        self.texts += int(vectors.shape[0])
        self.active += float(active.sum().item())
        self.presence += active.sum(dim=0).cpu().double()

    @property
    def l0(self) -> float:
        return self.active / max(self.texts, 1)

    @property
    def share(self) -> torch.Tensor:
        return self.presence / max(self.texts, 1)


def _sparsity_summary(queries: _Sparsity, documents: _Sparsity) -> dict[str, float]:
    return {
        "l0_query": queries.l0,
        "l0_document": documents.l0,
        "flops": float((queries.share * documents.share).sum()),
    }


# ------------------------------------------------------------------------------------ encoding


def encode(encode_fn: Callable[..., Any], texts: Sequence[str], batch_size: int) -> torch.Tensor:
    """Dense vectors of ``texts`` on the model's device, in float32."""
    vectors: torch.Tensor = encode_fn(
        list(texts), batch_size=batch_size, convert_to_tensor=True, convert_to_sparse_tensor=False
    )
    return vectors.float()


def _chunks(count: int, size: int, what: str = "") -> Iterable[tuple[int, int]]:
    """``(start, end)`` of consecutive chunks; with ``what`` set, progress goes to stderr."""
    started = time.perf_counter()
    for index, start in enumerate(range(0, count, size)):
        yield start, min(start + size, count)
        if what and (index % 10 == 9 or start + size >= count):
            done = min(start + size, count)
            print(
                f"encoded {done}/{count} {what} in {time.perf_counter() - started:.0f} s", file=sys.stderr, flush=True
            )


# ----------------------------------------------------------------------------------- reranking


@torch.no_grad()
def rerank(
    model: SparseEncoder,
    examples: Sequence[RerankingExample],
    *,
    caps: Sequence[Cap] = (FULL,),
    batch_size: int = 32,
    chunk: int = 256,
    seed: int = 42,
    k: int = 10,
) -> dict[str, Any]:
    """nDCG@k and MRR@k of ranking each query's candidates, for each pair of caps.

    Each distinct passage is encoded once. The query vectors stay on the model's device and the
    passages are encoded ``chunk`` at a time; only the (query, candidate) pairs that are ranked
    are scored, so a vocabulary-sized output, which can be nearly dense before training, is
    never held for the whole collection and no sparse tensor is built on the accelerator, whose
    sparse support varies (Apple MPS).

    Candidates are shuffled with ``seed`` before ranking, as ``muscari probe-sae`` does, so an
    untrained SAE head gives that command's ``sparse.plain`` on the same queries, up to float32
    rounding of the scores.
    """
    queries = list(dict.fromkeys(example.query for example in examples))
    passages = list(dict.fromkeys(passage for example in examples for passage in example.passages))
    query_row = {query: row for row, query in enumerate(queries)}
    passage_row = {passage: row for row, passage in enumerate(passages)}

    # Every ranked pair, sorted by passage so that each chunk owns a contiguous slice.
    pair_query = torch.tensor([query_row[example.query] for example in examples for _ in example.passages])
    pair_passage = torch.tensor([passage_row[passage] for example in examples for passage in example.passages])
    order = pair_passage.argsort(stable=True)
    sorted_passage = pair_passage[order]

    query_vectors = encode(model.encode_query, queries, batch_size)
    dimension = int(query_vectors.shape[1])
    capped_queries = [keep_largest(query_vectors, cap[0]) for cap in caps]
    query_sparsity = [_Sparsity(dimension) for _ in caps]
    for sparsity, vectors in zip(query_sparsity, capped_queries, strict=True):
        sparsity.add(vectors)
    passage_sparsity = [_Sparsity(dimension) for _ in caps]
    scores = torch.empty(len(caps), pair_query.shape[0], dtype=torch.float64)
    pairs_per_step = max(1, 2**24 // dimension)

    for start, end in _chunks(len(passages), chunk, "passages"):
        vectors = encode(model.encode_document, passages[start:end], batch_size)
        low, high = (int(i) for i in torch.searchsorted(sorted_passage, torch.tensor([start, end])))
        slots = order[low:high]
        local = (pair_passage[slots] - start).to(vectors.device)
        owners = pair_query[slots].to(vectors.device)
        for index, cap in enumerate(caps):
            capped = keep_largest(vectors, cap[1])
            passage_sparsity[index].add(capped)
            for first in range(0, slots.shape[0], pairs_per_step):
                part = slice(first, first + pairs_per_step)
                products = (capped[local[part]] * capped_queries[index][owners[part]]).sum(dim=1)
                scores[index, slots[part]] = products.cpu().double()

    generator = torch.Generator().manual_seed(seed)
    shuffles = [torch.randperm(len(example.passages), generator=generator) for example in examples]
    result: dict[str, Any] = {"queries": len(examples), "passages": len(passages), "dimensions": dimension}
    result["caps"], result["per_query"] = {}, {}
    for index, cap in enumerate(caps):
        per_query: dict[str, dict[str, float]] = {}
        offset = 0
        for position, (example, shuffle) in enumerate(zip(examples, shuffles, strict=True)):
            candidate_scores = scores[index, offset : offset + len(example.passages)]
            offset += len(example.passages)
            labels = torch.tensor(example.labels)
            query_id = example.query_id if example.query_id is not None else str(position)
            per_query[query_id] = ranking_metrics(candidate_scores[shuffle], labels[shuffle], k)
        label = cap_label(cap)
        result["caps"][label] = _mean(per_query) | _sparsity_summary(query_sparsity[index], passage_sparsity[index])
        result["per_query"][label] = per_query
    return result


def _mean(per_query: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    metrics = next(iter(per_query.values())).keys()
    return {metric: sum(values[metric] for values in per_query.values()) / len(per_query) for metric in metrics}


def reranking_examples(
    *,
    benchmark: str | None = None,
    data_files: Sequence[str] | None = None,
    split: str | None = None,
    query_column: str = "query",
    passage_column: str = "text",
    label_column: str = "label",
    group_column: str | None = None,
    title_column: str | None = None,
    without_title: bool = False,
    max_queries: int | None = None,
    seed: int = 42,
) -> list[RerankingExample]:
    """Examples of a known benchmark (its preset wins over the column arguments) or of local files.

    With ``max_queries`` a sample of queries is drawn after shuffling the rows with ``seed``; the
    whole split is used otherwise.
    """
    preset = BENCHMARKS.get(benchmark or "", {})
    columns = {
        "query_column": preset.get("query_column", query_column),
        "passage_column": preset.get("passage_column", passage_column),
        "label_column": preset.get("label_column", label_column),
    }
    group = preset.get("group_column", group_column)
    title = None if without_title else preset.get("title_column", title_column)
    rows = load_rows(
        dataset=preset.get("dataset"),
        subsets=None,
        data_files=None if preset else data_files,
        split=split or preset.get("split", "test"),
        column_names=[*columns.values(), *(name for name in (group, title) if name)],
        seed=seed,
    )
    return group_reranking_rows(rows, **columns, group_column=group, title_column=title, max_queries=max_queries)


# ----------------------------------------------------------------------------------- retrieval


def retrieval_metrics(
    ranked: Sequence[str], relevance: Mapping[str, int], *, k: int = 10, depth: int = 100
) -> dict[str, float]:
    """nDCG@k with graded gains, MRR@k and recall@depth of one ranked list of document ids.

    Gains are the qrels scores, as in ``pytrec_eval``; documents with a score of zero or less
    are not relevant.

    Raises:
        ValueError: if no document is relevant.
    """
    gains = sorted((score for score in relevance.values() if score > 0), reverse=True)
    if not gains:
        raise ValueError("The query has no relevant document.")
    dcg = sum(max(relevance.get(doc, 0), 0) / math.log2(rank + 2) for rank, doc in enumerate(ranked[:k]))
    ideal = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains[:k]))
    first = next((rank for rank, doc in enumerate(ranked[:k]) if relevance.get(doc, 0) > 0), None)
    found = sum(1 for doc in ranked[:depth] if relevance.get(doc, 0) > 0)
    return {
        f"ndcg@{k}": dcg / ideal,
        f"mrr@{k}": 0.0 if first is None else 1.0 / (first + 1),
        f"recall@{depth}": found / len(gains),
    }


@torch.no_grad()
def retrieve(
    model: SparseEncoder,
    queries: Mapping[str, str],
    corpus: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    caps: Sequence[Cap] = (FULL,),
    top_k: int = 100,
    batch_size: int = 32,
    chunk: int = 1024,
    k: int = 10,
) -> dict[str, Any]:
    """Rank the whole ``corpus`` for every query that has a relevant document in ``qrels``.

    The corpus is encoded ``chunk`` documents at a time and merged into a running top-``top_k``
    per query on the model's device, so neither the corpus vectors nor the full score matrix
    are kept.
    """
    query_ids = [query_id for query_id in queries if any(score > 0 for score in qrels.get(query_id, {}).values())]
    if not query_ids:
        raise ValueError("No query has a relevant document in the qrels.")
    document_ids = list(corpus)
    query_vectors = encode(model.encode_query, [queries[query_id] for query_id in query_ids], batch_size)
    dimension = int(query_vectors.shape[1])
    device = query_vectors.device
    depth = min(top_k, len(document_ids))

    capped_queries = [keep_largest(query_vectors, cap[0]) for cap in caps]
    query_sparsity = [_Sparsity(dimension) for _ in caps]
    for sparsity, vectors in zip(query_sparsity, capped_queries, strict=True):
        sparsity.add(vectors)
    document_sparsity = [_Sparsity(dimension) for _ in caps]
    best_scores = [torch.full((len(query_ids), 0), -math.inf, device=device) for _ in caps]
    best_rows = [torch.empty((len(query_ids), 0), dtype=torch.long, device=device) for _ in caps]

    for start, end in _chunks(len(document_ids), chunk, "documents"):
        vectors = encode(model.encode_document, [corpus[doc] for doc in document_ids[start:end]], batch_size)
        rows = torch.arange(start, end, device=device).expand(len(query_ids), -1)
        for index, cap in enumerate(caps):
            capped = keep_largest(vectors, cap[1])
            document_sparsity[index].add(capped)
            merged_scores = torch.cat([best_scores[index], capped_queries[index] @ capped.T], dim=1)
            merged_rows = torch.cat([best_rows[index], rows], dim=1)
            kept = merged_scores.topk(min(depth, merged_scores.shape[1]), dim=1)
            best_scores[index] = kept.values
            best_rows[index] = merged_rows.gather(1, kept.indices)

    result: dict[str, Any] = {"queries": len(query_ids), "documents": len(document_ids), "dimensions": dimension}
    result["caps"], result["per_query"] = {}, {}
    for index, cap in enumerate(caps):
        ranked_rows = best_rows[index].cpu().tolist()
        per_query = {
            query_id: retrieval_metrics([document_ids[row] for row in ranked], qrels[query_id], k=k, depth=top_k)
            for query_id, ranked in zip(query_ids, ranked_rows, strict=True)
        }
        label = cap_label(cap)
        result["caps"][label] = _mean(per_query) | _sparsity_summary(query_sparsity[index], document_sparsity[index])
        result["per_query"][label] = per_query
    return result


def retrieval_data(
    source: str, *, split: str = "test", without_title: bool = False, max_queries: int | None = None, seed: int = 42
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    """Queries, corpus and qrels of a dataset in the MTEB layout, from the Hub or a local directory.

    A Hub dataset has the configurations ``corpus`` (``id``, ``text``, optional ``title``),
    ``queries`` (``id``, ``text``) and ``qrels`` (``query-id``, ``corpus-id``, ``score``); a local
    directory has ``corpus.jsonl``, ``queries.jsonl`` and ``qrels.jsonl`` with those columns.
    A document with a title is ``"{title} {text}"``. With ``max_queries`` a sample of the queries
    that have a relevant document is drawn with ``seed``; the corpus is always used whole.
    """
    from datasets import load_dataset

    directory = Path(source)

    def table(name: str) -> Any:
        if directory.is_dir():
            return load_dataset("json", data_files=str(directory / f"{name}.jsonl"), split="train")
        return load_dataset(source, name, split=split)

    corpus: dict[str, str] = {}
    for row in table("corpus"):
        title = "" if without_title else str(row.get("title") or "").strip()
        corpus[str(row["id"])] = f"{title} {row['text']}" if title else str(row["text"])
    qrels: dict[str, dict[str, int]] = {}
    for row in table("qrels"):
        qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    queries = {str(row["id"]): str(row["text"]) for row in table("queries")}

    if max_queries:
        answerable = [query_id for query_id in queries if any(score > 0 for score in qrels.get(query_id, {}).values())]
        generator = torch.Generator().manual_seed(seed)
        chosen = {answerable[int(i)] for i in torch.randperm(len(answerable), generator=generator)[:max_queries]}
        queries = {query_id: text for query_id, text in queries.items() if query_id in chosen}
    return queries, corpus, qrels


# ---------------------------------------------------------------------------------------- model


def needs_remote_code(path: str) -> bool:
    """Whether a saved model names modules or architectures that this environment does not provide.

    A local directory whose ``modules.json`` refers only to Sentence Transformers, this project or
    its examples, that holds no Python file and whose ``config.json`` has no ``auto_map`` loads
    without running code from the model.
    """
    directory = Path(path)
    if not directory.is_dir() or any(directory.rglob("*.py")):
        return True
    modules_file = directory / "modules.json"
    modules = json.loads(modules_file.read_text(encoding="utf-8")) if modules_file.exists() else []
    if any(not str(module.get("type", "")).startswith(LOCAL_MODULE_PREFIXES) for module in modules):
        return True
    for config_file in directory.rglob("config.json"):
        if "auto_map" in json.loads(config_file.read_text(encoding="utf-8")):
            return True
    return False


def load_model(args: argparse.Namespace) -> SparseEncoder:
    """The encoder to evaluate, with ``--query-prompt`` / ``--document-prompt`` overriding its own."""
    if args.sae:
        model = build_sparse_encoder(
            args.sae,
            args.model,
            query_prompt=args.query_prompt or None,
            document_prompt=args.document_prompt or None,
            max_seq_length=args.max_seq_length,
            device=args.device,
        )
    else:
        if not args.model:
            raise SystemExit("Give --model, or --sae for an untrained SAE head.")
        trusted = args.trust_remote_code or not needs_remote_code(args.model)
        model = SparseEncoder(args.model, device=args.device, trust_remote_code=trusted)
        if args.max_seq_length:
            model.max_seq_length = args.max_seq_length
        for name, prompt in (("query", args.query_prompt), ("document", args.document_prompt)):
            if prompt is not None:
                model.prompts[name] = prompt
    model.eval()
    return model


# ------------------------------------------------------------------------------------------ cli


def add_evaluation_arguments(parser: argparse.ArgumentParser, *, prefix: str = "") -> None:
    """Options shared with ``train_retriever.py``; ``prefix`` keeps them apart from training options."""
    group = parser.add_argument_group("reranking")
    group.add_argument(f"--{prefix}benchmark", choices=sorted(BENCHMARKS), help="a known reranking dataset")
    group.add_argument(f"--{prefix}files", nargs="+", help="local reranking data, one candidate per row")
    group.add_argument(f"--{prefix}split", help="split of the benchmark; default: its preset (JQaRA: test)")
    group.add_argument(f"--{prefix}query-column", default="query")
    group.add_argument(f"--{prefix}passage-column", default="text")
    group.add_argument(f"--{prefix}label-column", default="label")
    group.add_argument(f"--{prefix}group-column")
    group.add_argument(f"--{prefix}title-column")
    group.add_argument(f"--{prefix}without-title", action="store_true", help="ignore the benchmark's titles")
    group.add_argument(f"--{prefix}queries", type=int, default=0, help="queries to sample; 0 uses all of them")

    group = parser.add_argument_group("retrieval")
    group.add_argument(f"--{prefix}retrieval", nargs="+", default=[], help="MTEB-layout datasets or directories")
    group.add_argument(f"--{prefix}retrieval-split", default="test")
    group.add_argument(f"--{prefix}retrieval-queries", type=int, default=0, help="queries to sample; 0 uses all")
    group.add_argument(f"--{prefix}top-k", type=int, default=100, help="documents kept per query")

    group = parser.add_argument_group("evaluation")
    group.add_argument(
        f"--{prefix}caps",
        nargs="+",
        type=parse_cap,
        default=[FULL],
        help="inference-time caps QUERY/DOCUMENT on the active dimensions, or full",
    )
    group.add_argument(f"--{prefix}batch-size", type=int, default=32)
    group.add_argument(f"--{prefix}chunk", type=int, default=1024, help="texts encoded between two scorings")


def evaluate(model: SparseEncoder, args: argparse.Namespace, *, prefix: str = "", seed: int = 42) -> dict[str, Any]:
    """Run every evaluation the options ask for; empty when they ask for none."""

    def option(name: str) -> Any:
        return getattr(args, prefix.replace("-", "_") + name)

    result: dict[str, Any] = {}
    caps = list(dict.fromkeys(option("caps")))
    if option("benchmark") or option("files"):
        examples = reranking_examples(
            benchmark=option("benchmark"),
            data_files=option("files"),
            split=option("split"),
            query_column=option("query_column"),
            passage_column=option("passage_column"),
            label_column=option("label_column"),
            group_column=option("group_column"),
            title_column=option("title_column"),
            without_title=option("without_title"),
            max_queries=option("queries") or None,
            seed=seed,
        )
        started = time.perf_counter()
        reranking = rerank(
            model, examples, caps=caps, batch_size=option("batch_size"), chunk=option("chunk"), seed=seed
        )
        reranking["seconds"] = time.perf_counter() - started
        reranking["dataset"] = option("benchmark") or "files"
        reranking["split"] = option("split") or BENCHMARKS.get(option("benchmark") or "", {}).get("split", "test")
        reranking["titles"] = not option("without_title") and bool(
            BENCHMARKS.get(option("benchmark") or "", {}).get("title_column") or option("title_column")
        )
        result["reranking"] = reranking
    for source in option("retrieval"):
        queries, corpus, qrels = retrieval_data(
            source,
            split=option("retrieval_split"),
            without_title=option("without_title"),
            max_queries=option("retrieval_queries") or None,
            seed=seed,
        )
        started = time.perf_counter()
        retrieval = retrieve(
            model,
            queries,
            corpus,
            qrels,
            caps=caps,
            top_k=option("top_k"),
            batch_size=option("batch_size"),
            chunk=option("chunk"),
        )
        retrieval["seconds"] = time.perf_counter() - started
        result.setdefault("retrieval", {})[_public_name(source)] = retrieval
    return result


def _public_name(name: str) -> str:
    """Record a local path by its last component so that outputs carry no directory layout."""
    return Path(name).name if Path(name).exists() else name


def summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """The averaged metrics of an evaluation, without the per-query values."""
    short: dict[str, Any] = {}
    if "reranking" in result:
        short["reranking"] = result["reranking"]["caps"]
    for name, retrieval in result.get("retrieval", {}).items():
        short.setdefault("retrieval", {})[name] = retrieval["caps"]
    return short


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    group = parser.add_argument_group("encoder")
    group.add_argument("--model", help="SparseEncoder directory or Hub id; with --sae: the base encoder")
    group.add_argument("--sae", help="directory of an exported SAE, evaluated as an untrained SAE head")
    group.add_argument("--trust-remote-code", action="store_true", help="run code from the model repository")
    group.add_argument("--query-prompt", help="prefix for queries; default: the model's own")
    group.add_argument("--document-prompt", help="prefix for documents; default: the model's own")
    group.add_argument("--max-seq-length", type=int, help="default: the model's own")
    group.add_argument("--device", help="default: chosen by Sentence Transformers")
    group.add_argument("--seed", type=int, default=42, help="seed for query samples and tie-breaking")
    group.add_argument("--out", required=True, help="JSON file for the results")
    add_evaluation_arguments(parser)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    model = load_model(args)
    result: dict[str, Any] = {
        "model": _public_name(args.model) if args.model else None,
        "sae": _public_name(args.sae) if args.sae else None,
        "max_seq_length": model.max_seq_length,
        "seed": args.seed,
    }
    result |= evaluate(model, args, seed=args.seed)
    if "reranking" not in result and "retrieval" not in result:
        raise SystemExit("Nothing to evaluate: give --benchmark, --files or --retrieval.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary(result), ensure_ascii=False), flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
