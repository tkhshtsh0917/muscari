"""Importing an SAE with muscari's tensor layout must reproduce it exactly."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from tiny import HIDDEN_SIZE

from muscari import TopKSAE, load_sae, save_topk_sae

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def importer():
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    return importlib.import_module("import_sae")


@pytest.fixture
def original(tmp_path):
    torch.manual_seed(42)
    sae = TopKSAE(HIDDEN_SIZE, 64, 4)
    with torch.no_grad():
        sae.b_dec.normal_()
    return load_sae(save_topk_sae(tmp_path / "original", sae, {"max_length": 32}))


def test_an_imported_sae_encodes_and_decodes_like_the_original(importer, original, tmp_path):
    weights = tmp_path / "model.safetensors"
    save_file(load_file(str(tmp_path / "original" / "sae.safetensors")), str(weights))

    importer.main(
        [
            *("--weights", str(weights)),
            *("--k", "4"),
            *("--base-model", "some/encoder"),
            *("--max-length", "32"),
            *("--column", "anc:query=検索クエリ: "),
            *("--column", "pos:document=検索文書: "),
            *("--out", str(tmp_path / "imported")),
        ]
    )
    imported = load_sae(tmp_path / "imported")

    states = torch.randn(10, HIDDEN_SIZE)
    assert torch.equal(imported.encode(states), original.encode(states))
    assert torch.equal(imported(states), original(states))
    assert imported.config["columns"] == {"anc": "検索クエリ: ", "pos": "検索文書: "}
    assert imported.config["labels"] == {"anc": "query", "pos": "document"}
    assert imported.config["imported_from"] == "model.safetensors" and imported.config["base_model"] == "some/encoder"


def test_a_raw_encoder_bias_is_folded_on_request(importer, original, tmp_path):
    tensors = load_file(str(tmp_path / "original" / "sae.safetensors"))
    tensors["encoder.bias"] = tensors["encoder.bias"] + tensors["encoder.weight"] @ tensors["decoder.bias"]
    save_file(tensors, str(tmp_path / "raw.safetensors"))

    out = importer.import_sae(
        tmp_path / "raw.safetensors", tmp_path / "folded", k=4, base_model="m", max_length=32, fold_decoder_bias=True
    )

    assert torch.allclose(load_sae(out).encoder_bias, original.encoder_bias, atol=1e-5)


def test_missing_tensors_and_bad_k_are_rejected(importer, original, tmp_path):
    tensors = load_file(str(tmp_path / "original" / "sae.safetensors"))
    save_file({"W_enc": tensors["encoder.weight"]}, str(tmp_path / "other.safetensors"))
    with pytest.raises(ValueError, match="encoder.bias"):
        importer.import_sae(tmp_path / "other.safetensors", tmp_path / "x", k=4, base_model="m", max_length=8)
    with pytest.raises(ValueError, match="k must be"):
        importer.import_sae(
            tmp_path / "original" / "sae.safetensors", tmp_path / "y", k=65, base_model="m", max_length=8
        )
