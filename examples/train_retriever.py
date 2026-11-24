#!/usr/bin/env python3
"""Train a sparse retriever with Sentence Transformers and evaluate it before and after.

Three heads can be trained with the same data, loss and schedule, so that they can be compared:

    --head sae     the latents of an exported SAE as the vocabulary (``muscari.retrieval``),
                   frozen unless ``--trainable-sae``
    --head vocab   the tokenizer's vocabulary through the language-model head, pooled over the
                   same content tokens as the SAE head, with SPARSEUP's logit shift
                   (``--shift``, ``auto`` estimates it) and per-token fan-out (``--fan-out``);
                   see ``vocabulary_head.py``
    --head mlm     Sentence Transformers' ``SpladePooling`` unchanged: every attended token,
                   prompt and special tokens included

The loss is a contrastive loss with in-batch negatives plus a FLOPS regulariser on queries and
documents whose weight rises quadratically from zero. Before and after training the model is
evaluated with ``sparse_eval.py``, whose options are taken here with an ``--eval-`` prefix:
reranking a benchmark (``--eval-benchmark jqara --eval-split dev``), retrieval from MTEB-layout
corpora (``--eval-retrieval``) and inference-time caps (``--eval-caps``).

Training saves a checkpoint every tenth of the steps. Started again with the same ``--out``, the
script reuses the evaluation before training and continues from the last checkpoint.

    python examples/train_retriever.py --head sae --sae runs/sae \\
        --query-prompt "検索クエリ: " --document-prompt "検索文書: " \\
        --dataset cl-nagoya/ruri-v3-dataset-ft --subsets jsquad \\
        --query-column anc --positive-column pos --negative-column neg \\
        --eval-benchmark jqara --eval-split dev --out runs/retriever

Needs the "retrieval" extra. The defaults are a starting point, not tuned values.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from datasets import Dataset
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, SparseEncoderTrainingArguments
from sentence_transformers.sparse_encoder.callbacks import SpladeRegularizerWeightSchedulerCallback
from sentence_transformers.sparse_encoder.losses import (
    CachedSpladeLoss,
    SparseMultipleNegativesRankingLoss,
    SpladeLoss,
)
from sentence_transformers.sparse_encoder.modules import SpladePooling, Transformer
from sparse_eval import add_evaluation_arguments, evaluate, summary
from vocabulary_head import build_vocabulary_encoder

from muscari.cli import load_rows
from muscari.retrieval import build_sparse_encoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--head", choices=["sae", "vocab", "mlm"], required=True)
    parser.add_argument("--sae", help="directory of an exported SAE; required for --head sae")
    parser.add_argument("--trainable-sae", action="store_true", help="train the SAE encoder with the transformer")
    parser.add_argument("--shift", default="0", help='--head vocab: logit shift theta, or "auto"')
    parser.add_argument("--shift-quantile", type=float, default=0.999, help="quantile that auto takes")
    parser.add_argument("--shift-texts", type=int, default=256, help="training documents auto samples")
    parser.add_argument("--fan-out", type=int, help="--head vocab: logits kept per token; default: all")
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
    group.add_argument(
        "--mini-batch-size",
        type=int,
        help="encode the batch in pieces of this size with GradCache (CachedSpladeLoss); same gradients",
    )
    group.add_argument(
        "--bf16",
        action="store_true",
        help="bfloat16 autocast (CUDA); it lowered the result of the SAE head clearly in one comparison",
    )
    group.add_argument("--seed", type=int, default=42)

    add_evaluation_arguments(parser, prefix="eval-")
    return parser


def build_model(args: argparse.Namespace, documents: Sequence[str] = ()) -> SparseEncoder:
    """The encoder with the requested head; prompts are registered as "query" and "document".

    ``documents`` are the training texts from which ``--shift auto`` is estimated.
    """
    if args.head == "sae":
        if not args.sae:
            raise SystemExit("--head sae needs --sae.")
        return build_sparse_encoder(
            args.sae,
            args.model,
            query_prompt=args.query_prompt or None,
            document_prompt=args.document_prompt or None,
            max_seq_length=args.max_seq_length,
            head_kwargs={"trainable": args.trainable_sae},
            device=args.device,
        )
    if not args.model:
        raise SystemExit(f"--head {args.head} needs --model.")
    if args.head == "vocab":
        return build_vocabulary_encoder(
            args.model,
            shift=args.shift if args.shift == "auto" else float(args.shift),
            fan_out=args.fan_out,
            query_prompt=args.query_prompt or None,
            document_prompt=args.document_prompt or None,
            max_seq_length=args.max_seq_length,
            shift_texts=list(documents)[: args.shift_texts],
            shift_quantile=args.shift_quantile,
            seed=args.seed,
            device=args.device,
        )

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


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Train and evaluate; returns what is written to ``results.json``."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_rows = load_rows(
        dataset=args.dataset,
        subsets=args.subsets,
        data_files=args.data_files,
        split=args.split,
        column_names=[name for name in (args.query_column, args.positive_column, args.negative_column) if name],
        seed=args.seed,
    )
    train_dataset = training_rows(train_rows, args)
    model = build_model(args, documents=train_dataset["positive"])

    result: dict[str, Any] = {
        "head": args.head,
        "head_config": model[-1].get_config_dict(),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "mini_batch_size": args.mini_batch_size,
        "learning_rate": args.learning_rate,
        "regularizer_weight": args.regularizer_weight,
        "bf16": args.bf16,
        "seed": args.seed,
    }
    before_path = out / "before.json"
    if before_path.exists():
        result["before"] = json.loads(before_path.read_text(encoding="utf-8"))
    else:
        result["before"] = evaluate(model, args, prefix="eval-", seed=args.seed)
        before_path.write_text(json.dumps(result["before"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"before": summary(result["before"])}, ensure_ascii=False), flush=True)

    prompts = {
        column: args.query_prompt if column == "anchor" else args.document_prompt
        for column in train_dataset.column_names
    }

    weights = {
        "document_regularizer_weight": args.regularizer_weight,
        "query_regularizer_weight": args.regularizer_weight,
    }
    if args.mini_batch_size:
        loss: SpladeLoss = CachedSpladeLoss(
            model,
            SparseMultipleNegativesRankingLoss(model),
            **weights,
            mini_batch_size=args.mini_batch_size,
        )
    else:
        loss = SpladeLoss(model, SparseMultipleNegativesRankingLoss(model), **weights)
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
            save_strategy="steps",
            save_steps=max(1, args.max_steps // 10),
            save_total_limit=1,
            report_to="none",
            seed=args.seed,
            use_cpu=args.device == "cpu",
        ),
        train_dataset=train_dataset,
        loss=loss,
        callbacks=[SpladeRegularizerWeightSchedulerCallback(loss, "quadratic", warmup_ratio=args.regularizer_warmup)],
    )
    checkpoints = out / "checkpoints"
    resume = checkpoints.is_dir() and any(path.name.startswith("checkpoint-") for path in checkpoints.iterdir())
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=True if resume else None)
    result["resumed"] = resume
    result["train_seconds"] = time.perf_counter() - started
    result["train_rows"] = len(train_dataset)
    result["train_log"] = [entry for entry in trainer.state.log_history if "loss" in entry]

    model.eval()
    result["after"] = evaluate(model, args, prefix="eval-", seed=args.seed)
    print(json.dumps({"after": summary(result["after"])}, ensure_ascii=False), flush=True)

    model.save(str(out / "model"))
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
