#!/usr/bin/env python3
"""Compare two sets of runs on the same queries: paired bootstrap interval and randomization test.

Each run is a JSON file written by ``sparse_eval.py`` or ``train_retriever.py`` (for the latter
``--stage`` picks the evaluation before or after training). Several files per arm, for example
one per training seed, are averaged per query first, so the comparison is between the arms'
expected per-query values and the queries are the unit of resampling.

    python examples/compare_runs.py \\
        --a runs/sae-seed*/results.json --b runs/vocab-seed*/results.json \\
        --task reranking --cap full --metric ndcg@10

Reported: the two means, their difference ``b - a``, a 95% percentile bootstrap interval of the
difference over queries, and the two-sided p-value of a paired sign-flip randomization test.
Only queries present in every run are compared; how many were dropped is reported. The
resampling uses ``--seed`` and is reproducible.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def per_query(result: Mapping[str, Any], *, task: str, cap: str, metric: str, stage: str) -> dict[str, float]:
    """``{query_id: value}`` of one run; ``task`` is ``reranking`` or ``retrieval/<dataset>``.

    Raises:
        KeyError: naming the part of the result that is missing.
    """
    if "after" in result or "before" in result:
        result = result[stage]
    section: Any = result
    for key in task.split("/", 1):
        if key not in section:
            raise KeyError(f"The result has no {task!r}; it has {sorted(section)}.")
        section = section[key]
    if cap not in section["per_query"]:
        raise KeyError(f"No cap {cap!r}; the result has {sorted(section['per_query'])}.")
    return {query_id: float(values[metric]) for query_id, values in section["per_query"][cap].items()}


def average_runs(runs: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """The per-query mean over runs, on the queries every run has."""
    shared = set(runs[0]).intersection(*runs[1:])
    return {query_id: sum(run[query_id] for run in runs) / len(runs) for query_id in sorted(shared)}


def compare(
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Paired comparison of ``b - a`` over the queries both have.

    The interval resamples queries with replacement; the p-value flips the sign of each paired
    difference at random and counts how often the mean is at least as far from zero as the one
    observed (with the observed statistic included, so it is never zero).

    Raises:
        ValueError: if the two share no query.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("The two arms share no query.")
    values_a = torch.tensor([a[query_id] for query_id in shared], dtype=torch.float64)
    values_b = torch.tensor([b[query_id] for query_id in shared], dtype=torch.float64)
    differences = values_b - values_a
    observed = float(differences.mean())
    generator = torch.Generator().manual_seed(seed)

    means = torch.empty(resamples, dtype=torch.float64)
    signs_extreme = 0
    for start in range(0, resamples, 1000):
        count = min(1000, resamples - start)
        picks = torch.randint(len(shared), (count, len(shared)), generator=generator)
        means[start : start + count] = differences[picks].mean(dim=1)
        signs = torch.randint(0, 2, (count, len(shared)), generator=generator, dtype=torch.float64) * 2 - 1
        flipped = (signs * differences).mean(dim=1)
        signs_extreme += int((flipped.abs() >= abs(observed) - 1e-12).sum())

    tail = (1 - confidence) / 2
    low, high = torch.quantile(means, torch.tensor([tail, 1 - tail], dtype=torch.float64)).tolist()
    return {
        "queries": len(shared),
        "mean_a": float(values_a.mean()),
        "mean_b": float(values_b.mean()),
        "difference": observed,
        "ci_low": low,
        "ci_high": high,
        "confidence": confidence,
        "p_value": (signs_extreme + 1) / (resamples + 1),
        "b_better": int((differences > 0).sum()),
        "a_better": int((differences < 0).sum()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--a", nargs="+", required=True, help="result files of the first arm")
    parser.add_argument("--b", nargs="+", required=True, help="result files of the second arm")
    parser.add_argument("--task", default="reranking", help="reranking or retrieval/<dataset>")
    parser.add_argument("--cap", default="full", help="cap label, for example full or 64/400")
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--stage", choices=["before", "after"], default="after", help="for train_retriever results")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", help="also write the comparison to this JSON file")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    arms = {}
    for name, files in (("a", args.a), ("b", args.b)):
        runs = [
            per_query(
                json.loads(Path(file).read_text(encoding="utf-8")),
                task=args.task,
                cap=args.cap,
                metric=args.metric,
                stage=args.stage,
            )
            for file in files
        ]
        arms[name] = (average_runs(runs), max(len(run) for run in runs), len(runs))
    result: dict[str, Any] = {
        "task": args.task,
        "cap": args.cap,
        "metric": args.metric,
        "runs_a": arms["a"][2],
        "runs_b": arms["b"][2],
    }
    result |= compare(arms["a"][0], arms["b"][0], resamples=args.resamples, confidence=args.confidence, seed=args.seed)
    result["queries_dropped"] = max(arms["a"][1], arms["b"][1]) - result["queries"]
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
