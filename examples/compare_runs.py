#!/usr/bin/env python3
"""Compare two sets of runs on the same queries: paired bootstrap interval and randomization test.

Each run is a JSON file written by ``sparse_eval.py`` or ``train_retriever.py`` (for the latter
``--stage`` picks the evaluation before or after training). An arm may have several files, for
example one per training seed.

    python examples/compare_runs.py \\
        --a runs/sae-seed*/results.json --b runs/vocab-seed*/results.json \\
        --task reranking --cap full --metric ndcg@10

Reported: the two means, their difference ``b - a``, a 95% percentile bootstrap interval of the
difference, the two-sided p-value of a paired sign-flip randomization test on the run-averaged
values, and the mean of every run with the spread between the runs of an arm.

The interval resamples the queries and, when an arm has more than one run, also the runs of
each arm, so that it covers the variation between training seeds as well as between queries.
With few runs that part is coarse: three seeds give only ten distinct resamples of an arm.
``--resample queries`` keeps the runs fixed; its interval describes these particular runs.
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


def align_runs(a: Sequence[Mapping[str, float]], b: Sequence[Mapping[str, float]]) -> tuple[torch.Tensor, ...]:
    """The runs of both arms as ``(runs, queries)`` matrices over the queries every run has.

    Raises:
        ValueError: if the runs share no query.
    """
    shared = sorted(set(a[0]).intersection(*a[1:], *b))
    if not shared:
        raise ValueError("The two arms share no query.")
    return tuple(
        torch.tensor([[run[query_id] for query_id in shared] for run in runs], dtype=torch.float64) for runs in (a, b)
    )


def compare(
    a: Sequence[Mapping[str, float]],
    b: Sequence[Mapping[str, float]],
    *,
    resample: str = "runs-and-queries",
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired comparison of ``b - a``; each arm is a list of runs, ``{query_id: value}`` each.

    The interval is a percentile bootstrap. ``"runs-and-queries"`` draws, for every resample,
    the runs of each arm with replacement and the queries with replacement; ``"queries"``
    keeps the runs and draws only queries. The p-value flips the sign of each query's
    difference of run-averaged values at random and counts how often the mean is at least as
    far from zero as the one observed (with the observed statistic included, so it is never
    zero); it treats the runs as given.

    Raises:
        ValueError: if the arms share no query or ``resample`` is unknown.
    """
    if resample not in ("runs-and-queries", "queries"):
        raise ValueError(f"Unknown resample: {resample!r}.")
    values_a, values_b = align_runs(a, b)
    queries = values_a.shape[1]
    differences = values_b.mean(dim=0) - values_a.mean(dim=0)
    observed = float(differences.mean())
    generator = torch.Generator().manual_seed(seed)

    def run_weights(values: torch.Tensor, count: int) -> torch.Tensor:
        """``(count, runs)`` averaging weights: resampled runs, or every run once."""
        runs = values.shape[0]
        if resample == "queries" or runs == 1:
            return torch.full((count, runs), 1.0 / runs, dtype=torch.float64)
        picks = torch.randint(runs, (count, runs), generator=generator)
        return torch.zeros(count, runs, dtype=torch.float64).scatter_add_(
            1, picks, torch.full(picks.shape, 1.0 / runs, dtype=torch.float64)
        )

    means = torch.empty(resamples, dtype=torch.float64)
    signs_extreme = 0
    for start in range(0, resamples, 1000):
        count = min(1000, resamples - start)
        resampled = run_weights(values_b, count) @ values_b - run_weights(values_a, count) @ values_a
        picks = torch.randint(queries, (count, queries), generator=generator)
        means[start : start + count] = resampled.gather(1, picks).mean(dim=1)
        signs = torch.randint(0, 2, (count, queries), generator=generator, dtype=torch.float64) * 2 - 1
        flipped = (signs * differences).mean(dim=1)
        signs_extreme += int((flipped.abs() >= abs(observed) - 1e-12).sum())

    def spread(values: torch.Tensor) -> float:
        return float(values.mean(dim=1).std()) if values.shape[0] > 1 else 0.0

    tail = (1 - confidence) / 2
    low, high = torch.quantile(means, torch.tensor([tail, 1 - tail], dtype=torch.float64)).tolist()
    return {
        "queries": queries,
        "mean_a": float(values_a.mean()),
        "mean_b": float(values_b.mean()),
        "run_means_a": values_a.mean(dim=1).tolist(),
        "run_means_b": values_b.mean(dim=1).tolist(),
        "run_std_a": spread(values_a),
        "run_std_b": spread(values_b),
        "difference": observed,
        "ci_low": low,
        "ci_high": high,
        "confidence": confidence,
        "resample": resample if max(values_a.shape[0], values_b.shape[0]) > 1 else "queries",
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
    parser.add_argument(
        "--resample",
        choices=["runs-and-queries", "queries"],
        default="runs-and-queries",
        help="what the bootstrap draws; runs are drawn only for an arm with more than one",
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", help="also write the comparison to this JSON file")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    arms = {
        name: [
            per_query(
                json.loads(Path(file).read_text(encoding="utf-8")),
                task=args.task,
                cap=args.cap,
                metric=args.metric,
                stage=args.stage,
            )
            for file in files
        ]
        for name, files in (("a", args.a), ("b", args.b))
    }
    result: dict[str, Any] = {
        "task": args.task,
        "cap": args.cap,
        "metric": args.metric,
        "runs_a": len(arms["a"]),
        "runs_b": len(arms["b"]),
    }
    result |= compare(
        arms["a"],
        arms["b"],
        resample=args.resample,
        resamples=args.resamples,
        confidence=args.confidence,
        seed=args.seed,
    )
    result["queries_dropped"] = max(len(run) for runs in arms.values() for run in runs) - result["queries"]
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
