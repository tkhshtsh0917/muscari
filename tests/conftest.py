"""Shared fixtures: a tiny offline encoder, so no test needs a download."""

from __future__ import annotations

from pathlib import Path

import pytest
from tiny import build_tiny_encoder

from muscari import EncoderActivationSource


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_encoder(tmp_path_factory.mktemp("tiny-encoder"))


@pytest.fixture(scope="session")
def tiny_mlm_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_encoder(tmp_path_factory.mktemp("tiny-mlm"), with_head=True)


@pytest.fixture
def source(tiny_model_dir: Path) -> EncoderActivationSource:
    return EncoderActivationSource.from_pretrained(
        str(tiny_model_dir),
        text_batch_size=8,
        max_length=32,
        device="cpu",
    )
