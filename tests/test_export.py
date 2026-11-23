"""Common on-disk format: round trips, the two activation rules and evaluation metrics."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file

from muscari import TopKSAE, evaluate_sae, load_sae, save_topk_sae
from muscari.export import CONFIG_NAME, WEIGHTS_NAME, save_sae


def signed_identity_sae(tmp_path, d_in: int = 4, **overrides):
    """An SAE that reconstructs exactly: latents i and d+i carry ReLU(x_i) and ReLU(-x_i)."""
    identity = torch.eye(d_in)
    arguments = {
        "encoder_weight": torch.cat([identity, -identity]),
        "encoder_bias": torch.zeros(2 * d_in),
        "decoder_weight": torch.cat([identity, -identity], dim=1),
        "decoder_bias": torch.zeros(d_in),
        "config": {"k": d_in},
    }
    arguments.update(overrides)
    return load_sae(save_sae(tmp_path, **arguments))


def test_exported_topk_sae_reproduces_the_trained_module(tmp_path):
    torch.manual_seed(42)
    sae = TopKSAE(12, 40, 5)
    with torch.no_grad():
        sae.b_dec.normal_()
        sae.encoder.bias.normal_()
    x = torch.randn(33, 12)

    exported = load_sae(save_topk_sae(tmp_path, sae, {"trainer": "test"}))

    with torch.no_grad():
        assert torch.allclose(exported(x), sae(x), atol=1e-5)
    assert exported.config["k"] == 5
    assert exported.config["activation"] == "topk"
    assert (exported.d_in, exported.num_latents) == (12, 40)


def test_file_layout_folds_b_dec_into_the_encoder_bias(tmp_path):
    torch.manual_seed(42)
    sae = TopKSAE(6, 10, 2)
    with torch.no_grad():
        sae.b_dec.normal_()
    save_topk_sae(tmp_path, sae, {})

    tensors = load_file(tmp_path / WEIGHTS_NAME)
    config = json.loads((tmp_path / CONFIG_NAME).read_text(encoding="utf-8"))

    assert {name: tuple(tensor.shape) for name, tensor in tensors.items()} == {
        "encoder.weight": (10, 6),
        "encoder.bias": (10,),
        "decoder.weight": (6, 10),
        "decoder.bias": (6,),
    }
    assert torch.allclose(tensors["encoder.bias"], -(sae.encoder.weight @ sae.b_dec), atol=1e-6)
    assert config["d_in"] == 6 and config["num_latents"] == 10 and config["format_version"] == 1


def test_config_keeps_non_ascii_provenance_readable(tmp_path):
    torch.manual_seed(42)
    save_topk_sae(tmp_path, TopKSAE(4, 8, 2), {"columns": {"anc": "検索クエリ: "}})

    assert "検索クエリ: " in (tmp_path / CONFIG_NAME).read_text(encoding="utf-8")


def test_evaluation_of_an_exact_sae(tmp_path):
    sae = signed_identity_sae(tmp_path)
    torch.manual_seed(42)
    activations = torch.randn(1000, 4)
    activations[:, 0] = activations[:, 0].abs() + 0.1  # dimension 0 is always positive

    metrics = evaluate_sae(sae, activations, batch_size=64)

    assert metrics["fvu"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["l0"] == pytest.approx(4.0)  # exactly one of each latent pair fires
    assert metrics["never_fired_frac"] == pytest.approx(1 / 8)  # the "negative x_0" latent
    assert metrics["max_frequency"] == pytest.approx(1.0)  # the "positive x_0" latent
    assert metrics["dense_frac"] == pytest.approx(7 / 8)
    assert metrics["eval_tokens"] == 1000


def test_fewer_latents_at_inference_reconstruct_worse(tmp_path):
    sae = signed_identity_sae(tmp_path)
    x = torch.randn(50, 4)

    assert torch.allclose(sae.decode(sae.encode(x)), x, atol=1e-6)
    assert (sae.encode(x, k=2) > 0).sum(dim=-1).max() <= 2
    assert not torch.allclose(sae.decode(sae.encode(x, k=2)), x, atol=1e-3)


def test_jumprelu_keeps_preactivations_above_each_threshold(tmp_path):
    sae = signed_identity_sae(tmp_path, threshold=torch.full((8,), 0.5), config={})
    x = torch.tensor([[1.0, 0.4, -0.7, -0.2]])

    features = sae.encode(x)

    assert sae.config["activation"] == "jumprelu"
    assert features.tolist() == [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, pytest.approx(0.7), 0.0]]


def test_inconsistent_tensors_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="decoder.weight"):
        signed_identity_sae(tmp_path, decoder_weight=torch.zeros(8, 4))
    with pytest.raises(ValueError, match="bias"):
        signed_identity_sae(tmp_path, encoder_bias=torch.zeros(3))
    with pytest.raises(ValueError, match="'k'"):
        signed_identity_sae(tmp_path, config={})


def test_newer_format_versions_are_refused(tmp_path):
    signed_identity_sae(tmp_path)
    path = tmp_path / CONFIG_NAME
    config = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**config, "format_version": 99}), encoding="utf-8")

    with pytest.raises(ValueError, match="format version"):
        load_sae(tmp_path)


@pytest.mark.parametrize("shape", [(0, 4), (10, 5), (4,)])
def test_evaluation_rejects_unusable_activations(tmp_path, shape):
    sae = signed_identity_sae(tmp_path)

    with pytest.raises(ValueError):
        evaluate_sae(sae, torch.zeros(shape))
