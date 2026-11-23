"""The step benchmark must run, and its two modes must compute the same loss."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

BENCHMARK = Path(__file__).resolve().parents[1] / "benchmarks" / "sae_step.py"
SMALL = ["--device", "cpu", "--d-in", "16", "--num-latents", "128", "--k", "4", "--batch-tokens", "64", "--steps", "3"]


@pytest.fixture(scope="module")
def benchmark():
    spec = importlib.util.spec_from_file_location("muscari_sae_step_benchmark", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("aux_k", ["0", "8"])
def test_sparse_and_dense_modes_follow_the_same_trajectory(benchmark, aux_k, capsys):
    sparse = benchmark.main(["--mode", "sparse", "--aux-k", aux_k, *SMALL])
    dense = benchmark.main(["--mode", "dense", "--aux-k", aux_k, *SMALL])
    capsys.readouterr()

    assert sparse["first_loss"] == pytest.approx(dense["first_loss"], rel=1e-5)
    assert sparse["last_loss"] == pytest.approx(dense["last_loss"], rel=1e-4)
    assert sparse["last_loss"] < sparse["first_loss"]
    assert sparse["median_step_seconds"] > 0 and sparse["peak_memory_gb"] > 0
