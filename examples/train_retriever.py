#!/usr/bin/env python3
"""Train a sparse retriever with Sentence Transformers and rerank a benchmark before and after.

Two heads can be trained with the same data, loss and schedule, so that they can be compared:

    --head sae   the latents of an exported SAE as the vocabulary (``muscari.retrieval``)
    --head mlm   the tokenizer's vocabulary through the language-model head, as in SPLADE

The loss is a contrastive loss with in-batch negatives plus a FLOPS regulariser on queries and
documents whose weight rises quadratically from zero. Before and after training, the candidates
of a reranking dataset are ranked by the dot product of the sparse vectors.

    python examples/train_retriever.py --head sae --sae runs/sae \\
        --query-prompt "検索クエリ: " --document-prompt "検索文書: " \\
        --dataset cl-nagoya/ruri-v3-dataset-ft --subsets jsquad \\
        --query-column anc --positive-column pos --negative-column neg \\
        --benchmark jqara --out runs/retriever

Needs the "retrieval" extra. The defaults are a starting point, not tuned values.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, SparseEncoderTrainingArguments
from sentence_transformers.sparse_encoder.callbacks import SpladeRegularizerWeightSchedulerCallback
from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss
from sentence_transformers.sparse_encoder.modules import SpladePooling, Transformer

from muscari.cli import BENCHMARKS, load_rows
from muscari.probe import PooledText, RerankingExample, group_reranking_rows, ranking_metrics, sparse_scores
from muscari.retrieval import build_sparse_encoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--head", choices=["sae", "mlm"], required=True)
    parser.add_argument("--sae", help="directory of an exported SAE; required for --head sae")
    parser.add_argument("--model", help="encoder; default for --head sae: the base model recorded with the SAE")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--query-prompt", default="", help="prefix the encoder expects for queries")
    parser.add_argument("--document-prompt", default="", help="prefix the encoder expects for documents")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--device", help="default: chosen by Sentence Transformers")

    group = parser.add_argument_group("training data")
    source = group.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="Hugging Face dataset id")
    source.add_argument("--data-files", nargs="+", help="local .jsonl, .parquet or .csv files")
    group.add_argument("--subsets", nargs="*", help="dataset configurations to concatenate")
    group.add_argument("--split", default="train")
    group.add_argument("--query-column", default="query")
    group.add_argument("--positive-column", default="positive")
    group.add_argument("--negative-column", help="column with one hard negative or a list of them")
    group.add_argument("--negatives", type=int, default=1, help="hard negatives used per query")

    group = parser.add_argument_group("training")
    group.add_argument("--max-steps", type=int, default=1000)
    group.add_argument("--batch-size", type=int, default=32)
    group.add_argument("--learning-rate", type=float, default=2e-5)
    group.add_argument("--warmup-ratio", type=float, default=0.05)
    group.add_argument("--weight-decay", type=float, default=0.01)
    group.add_argument("--regularizer-weight", type=float, default=3.33e-5, help="FLOPS weight for both sides")
    group.add_argument("--regularizer-warmup", type=float, default=0.3, help="share of steps over which it rises")
    group.add_argument("--bf16", action="store_true", help="bfloat16 autocast (CUDA)")
    group.add_argument("--seed", type=int, default=42)

    group = parser.add_argument_group("evaluation")
    group.add_argument("--benchmark", choices=sorted(BENCHMARKS), help="a known reranking dataset")
    group.add_argument("--eval-files", nargs="+", help="local reranking data, one candidate per row")
    group.add_argument("--eval-query-column", default="query")
    group.add_argument("--eval-passage-column", default="text")
    group.add_argument("--eval-label-column", default="label")
    group.add_argument("--eval-group-column")
    group.add_argument("--eval-queries", type=int, default=200, help="queries to sample; 0 uses all of them")
    group.add_argument("--eval-batch-size", type=int, default=32)
    return parser


def build_model(args: argparse.Namespace) -> SparseEncoder:
    """The encoder with the requested head; prompts are registered as "query" and "document"."""
    if args.head == "sae":
        if not args.sae:
            raise SystemExit("--head sae needs --sae.")
        return build_sparse_encoder(
            args.sae,
            args.model,
            query_prompt=args.query_prompt or None,
            document_prompt=args.document_prompt or None,
            max_seq_length=args.max_seq_length,
            device=args.device,
        )
    if not args.model:
        raise SystemExit("--head mlm needs --model.")

    prompts = {
        name: prompt for name, prompt in (("query", args.query_prompt), ("document", args.document_prompt)) if prompt
    }
    return SparseEncoder(
        modules=[
            Transformer(args.model, transformer_task="fill-mask", max_seq_length=args.max_seq_length),
            SpladePooling(pooling_strategy="max"),
        ],
        prompts=prompts or None,
        similarity_fn_name="dot",
        device=args.device,
    )


def training_rows(rows: Iterable[Mapping[str, Any]], args: argparse.Namespace) -> Dataset:
    """Rows of ``anchor``, ``positive`` and ``negative_1..n``; rows that lack a field are skipped."""
    wanted = args.max_steps * args.batch_size
    columns: dict[str, list[str]] = {"anchor": [], "positive": []}
    columns.update({f"negative_{index + 1}": [] for index in range(args.negatives if args.negative_column else 0)})

    def first_texts(cell: Any, count: int) -> list[str]:
        texts = [cell] if isinstance(cell, str) else list(cell or [])
        return [text for text in texts if isinstance(text, str) and text][:count]

    for row in rows:
        query = first_texts(row.get(args.query_column), 1)
        positive = first_texts(row.get(args.positive_column), 1)
        negatives = first_texts(row.get(args.negative_column), args.negatives) if args.negative_column else []
        if not query or not positive or len(negatives) < len(columns) - 2:
            continue
        columns["anchor"].append(query[0])
        columns["positive"].append(positive[0])
        for index, negative in enumerate(negatives):
            columns[f"negative_{index + 1}"].append(negative)
        if len(columns["anchor"]) >= wanted:
            break

    if not columns["anchor"]:
        raise SystemExit("No training row has the requested columns.")
    return Dataset.from_dict(columns)


def _rows_of(embeddings: torch.Tensor) -> list[PooledText]:
    """Split a sparse ``(texts, dimensions)`` tensor into one vector per text."""
    embeddings = embeddings.cpu().coalesce()
    rows, dimensions = embeddings.indices()
    values = embeddings.values()
    bounds = torch.searchsorted(rows.contiguous(), torch.arange(embeddings.shape[0] + 1))
    return [
        PooledText(dimensions[start:end], values[start:end].float(), torch.empty(0))
        for start, end in zip(bounds[:-1].tolist(), bounds[1:].tolist(), strict=True)
    ]


def rerank(
    model: SparseEncoder,
    examples: Sequence[RerankingExample],
    *,
    batch_size: int = 32,
    seed: int = 42,
) -> dict[str, float]:
    """nDCG@10 and MRR@10 of ranking each query's candidates, with the sparsity of the vectors.

    Candidates are shuffled with ``seed`` before ranking, as ``muscari probe-sae`` does, so an
    untrained SAE head gives that command's ``sparse.plain`` on the same queries.
    """
    queries = list(dict.fromkeys(example.query for example in examples))
    passages = list(dict.fromkeys(passage for example in examples for passage in example.passages))
    query_embeddings = model.encode_query(queries, batch_size=batch_size)
    query_vectors = dict(zip(queries, _rows_of(query_embeddings), strict=True))
    passage_vectors = dict(zip(passages, _rows_of(model.encode_document(passages, batch_size=batch_size)), strict=True))

    dimension = int(query_embeddings.shape[1])
    weights = torch.ones(dimension, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    per_query: dict[str, list[float]] = {}
    for example in examples:
        shuffle = torch.randperm(len(example.passages), generator=generator)
        candidates = [passage_vectors[example.passages[int(position)]] for position in shuffle]
        scores = sparse_scores(query_vectors[example.query], candidates, weights)
        for metric, value in ranking_metrics(scores, torch.tensor(example.labels)[shuffle]).items():
            per_query.setdefault(metric, []).append(value)

    def presence(vectors: Iterable[PooledText], count: int) -> torch.Tensor:
        frequency = torch.zeros(dimension, dtype=torch.float64)
        for vector in vectors:
            frequency[vector.indices] += 1
        return frequency / count

    result = {metric: sum(values) / len(values) for metric, values in per_query.items()}
    result["queries"] = len(examples)
    result["dimensions"] = dimension
    result["l0_query"] = sum(vector.indices.numel() for vector in query_vectors.values()) / len(queries)
    result["l0_passage"] = sum(vector.indices.numel() for vector in passage_vectors.values()) / len(passages)
    result["flops"] = float(
        (presence(query_vectors.values(), len(queries)) * presence(passage_vectors.values(), len(passages))).sum()
    )
    return result


def evaluation_examples(args: argparse.Namespace) -> list[RerankingExample]:
    preset = BENCHMARKS.get(args.benchmark, {})
    columns = {
        "query_column": preset.get("query_column", args.eval_query_column),
        "passage_column": preset.get("passage_column", args.eval_passage_column),
        "label_column": preset.get("label_column", args.eval_label_column),
    }
    group_column = preset.get("group_column", args.eval_group_column)
    rows = load_rows(
        dataset=preset.get("dataset"),
        subsets=None,
        data_files=args.eval_files,
        split=preset.get("split", "test"),
        column_names=[*columns.values(), *([group_column] if group_column else [])],
        seed=args.seed,
    )
    return group_reranking_rows(rows, **columns, group_column=group_column, max_queries=args.eval_queries or None)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Train and evaluate; returns what is written to ``results.json``."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = build_model(args)
    examples = evaluation_examples(args) if args.benchmark or args.eval_files else None

    result: dict[str, Any] = {
        "head": args.head,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "regularizer_weight": args.regularizer_weight,
        "seed": args.seed,
    }
    if examples:
        result["before"] = rerank(model, examples, batch_size=args.eval_batch_size, seed=args.seed)
        print(json.dumps({"before": result["before"]}, ensure_ascii=False), flush=True)

    train_rows = load_rows(
        dataset=args.dataset,
        subsets=args.subsets,
        data_files=args.data_files,
        split=args.split,
        column_names=[name for name in (args.query_column, args.positive_column, args.negative_column) if name],
        seed=args.seed,
    )
    train_dataset = training_rows(train_rows, args)
    prompts = {
        column: args.query_prompt if column == "anchor" else args.document_prompt
        for column in train_dataset.column_names
    }

    loss = SpladeLoss(
        model,
        SparseMultipleNegativesRankingLoss(model),
        document_regularizer_weight=args.regularizer_weight,
        query_regularizer_weight=args.regularizer_weight,
    )
    trainer = SparseEncoderTrainer(
        model=model,
        args=SparseEncoderTrainingArguments(
            output_dir=str(out / "checkpoints"),
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=int(args.max_steps * args.warmup_ratio),
            weight_decay=args.weight_decay,
            bf16=args.bf16,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            prompts={column: prompt for column, prompt in prompts.items() if prompt} or None,
            dataloader_drop_last=True,
            logging_steps=max(1, args.max_steps // 20),
            save_strategy="no",
            report_to="none",
            seed=args.seed,
            use_cpu=args.device == "cpu",
        ),
        train_dataset=train_dataset,
        loss=loss,
        callbacks=[SpladeRegularizerWeightSchedulerCallback(loss, "quadratic", warmup_ratio=args.regularizer_warmup)],
    )
    started = time.perf_counter()
    trainer.train()
    result["train_seconds"] = time.perf_counter() - started
    result["train_rows"] = len(train_dataset)
    result["train_log"] = [entry for entry in trainer.state.log_history if "loss" in entry]

    if examples:
        result["after"] = rerank(model, examples, batch_size=args.eval_batch_size, seed=args.seed)
        print(json.dumps({"after": result["after"]}, ensure_ascii=False), flush=True)

    model.save(str(out / "model"))
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
