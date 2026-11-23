"""The offline example must keep working with both trainers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "train_tiny.py"


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("muscari_train_tiny_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("trainer", ["shizuku", "saelens"])
def test_example_trains_exports_and_evaluates(example, tmp_path, trainer):
    if trainer == "saelens":
        pytest.importorskip("sae_lens")

    metrics = example.run(trainer, tmp_path, "cpu")
    config = json.loads((tmp_path / f"sae-{trainer}" / "sae_config.json").read_text(encoding="utf-8"))

    assert 0 < metrics["fvu"] < 1.5
    assert metrics["part"] == "holdout" and metrics["eval_tokens"] == 1000
    assert config["columns"]["query"] == example.QUERY_PREFIX
    assert config["trainer"].startswith(trainer)
