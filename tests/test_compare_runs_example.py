"""The comparison example must pair queries, average seeds and give honest intervals."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
import torch

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def compare_runs():
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    return importlib.import_module("compare_runs")


def _result(values: dict[str, float], *, trained: bool = False) -> dict:
    evaluation = {
        "reranking": {"per_query": {"full": {query: {"ndcg@10": value} for query, value in values.items()}}},
        "retrieval": {"set": {"per_query": {"8/8": {query: {"recall@100": value} for query, value in values.items()}}}},
    }
    return {"before": {}, "after": evaluation} if trained else evaluation


def test_a_constant_shift_is_detected_and_its_interval_is_tight(compare_runs):
    torch.manual_seed(0)
    base = {f"q{i}": float(value) for i, value in enumerate(torch.rand(200))}
    better = {query: value + 0.05 for query, value in base.items()}

    result = compare_runs.compare(base, better, resamples=2000)

    assert result["difference"] == pytest.approx(0.05)
    assert result["ci_low"] == pytest.approx(0.05) and result["ci_high"] == pytest.approx(0.05)
    assert result["p_value"] == pytest.approx(1 / 2001) and result["b_better"] == 200


def test_noise_without_an_effect_is_not_significant(compare_runs):
    generator = torch.Generator().manual_seed(1)
    a = {f"q{i}": float(value) for i, value in enumerate(torch.rand(300, generator=generator))}
    b = {f"q{i}": float(value) for i, value in enumerate(torch.rand(300, generator=generator))}

    result = compare_runs.compare(a, b, resamples=4000, seed=42)

    assert result["ci_low"] < result["difference"] < result["ci_high"]
    assert result["ci_low"] < 0 < result["ci_high"] and result["p_value"] > 0.05
    assert result == compare_runs.compare(a, b, resamples=4000, seed=42)  # reproducible
    with pytest.raises(ValueError, match="no query"):
        compare_runs.compare({"x": 1.0}, {"y": 1.0})


def test_runs_are_read_averaged_over_seeds_and_paired(compare_runs, tmp_path):
    files = {}
    for name, values in {
        "a1": {"q1": 0.2, "q2": 0.4, "q3": 1.0},
        "a2": {"q1": 0.4, "q2": 0.6},
        "b1": {"q1": 0.5, "q2": 0.5, "q3": 0.0},
    }.items():
        files[name] = tmp_path / f"{name}.json"
        files[name].write_text(json.dumps(_result(values, trained=name == "a2")), encoding="utf-8")

    args = compare_runs.build_parser().parse_args(
        [
            *("--a", str(files["a1"]), str(files["a2"])),
            *("--b", str(files["b1"])),
            *("--resamples", "500"),
            *("--out", str(tmp_path / "comparison.json")),
        ]
    )
    result = compare_runs.run(args)

    assert (result["runs_a"], result["runs_b"], result["queries"], result["queries_dropped"]) == (2, 1, 2, 1)
    assert result["mean_a"] == pytest.approx(0.4) and result["mean_b"] == pytest.approx(0.5)
    assert json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))["difference"] == pytest.approx(0.1)

    retrieval = compare_runs.per_query(
        _result({"q1": 0.3}), task="retrieval/set", cap="8/8", metric="recall@100", stage="after"
    )
    assert retrieval == {"q1": 0.3}
    with pytest.raises(KeyError, match="cap"):
        compare_runs.per_query(_result({"q1": 0.3}), task="reranking", cap="4/4", metric="ndcg@10", stage="after")
    with pytest.raises(KeyError, match="retrieval/other"):
        compare_runs.per_query(_result({"q1": 0.3}), task="retrieval/other", cap="8/8", metric="x", stage="after")
