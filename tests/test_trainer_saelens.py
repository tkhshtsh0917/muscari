"""SAELens backend: the common format must match SAELens' own inference model numerically."""

from __future__ import annotations

import shutil

import pytest
import torch
from tiny import COLUMNS, make_rows

from muscari import SAELensTrainerConfig, evaluate_sae, iter_texts, load_sae, repeat_texts, shuffled_batches

sae_lens = pytest.importorskip("sae_lens")


def activation_batches(source, batch_tokens: int):
    rows = make_rows(400)
    texts = repeat_texts(lambda: iter_texts(rows, COLUMNS, holdout_pct=10, split="train"))
    return shuffled_batches(
        source.stream(texts),
        batch_tokens=batch_tokens,
        buffer_tokens=4 * batch_tokens,
    )


@pytest.mark.parametrize("architecture", ["topk", "batchtopk"])
def test_export_matches_the_native_inference_model(source, tmp_path, monkeypatch, architecture):
    import muscari.trainers.saelens as backend

    cfg = SAELensTrainerConfig(
        architecture=architecture,
        num_latents=256,
        k=8,
        dead_after_tokens=2_000,
        total_tokens=60_000,
        batch_tokens=256,
        device="cpu",
    )
    held_out = source.collect(iter_texts(make_rows(400), COLUMNS, holdout_pct=10, split="holdout"), 2000)

    # Keep SAELens' own inference export so that both forms can be compared.
    native_dir = tmp_path / "native"
    convert = backend.save_saelens_sae

    def keep_native(out_dir, saelens_inference_dir, config):
        shutil.copytree(saelens_inference_dir, native_dir)
        return convert(out_dir, saelens_inference_dir, config)

    monkeypatch.setattr(backend, "save_saelens_sae", keep_native)
    out = backend.train_with_saelens(
        cfg,
        source.d_in,
        activation_batches(source, cfg.batch_tokens),
        tmp_path / "out",
        export_config={"base_model": "tiny"},
    )

    exported = load_sae(out)
    native = sae_lens.SAE.load_from_disk(str(native_dir), device="cpu")
    x = held_out[:128]
    with torch.no_grad():
        assert torch.allclose(exported.encode(x), native.encode(x), atol=1e-4)
        assert torch.allclose(exported(x), native(x), atol=1e-4)

    assert exported.config["activation"] == ("topk" if architecture == "topk" else "jumprelu")
    assert exported.config["trainer"] == f"saelens-{architecture}"
    assert exported.config["base_model"] == "tiny"
    assert evaluate_sae(exported, held_out)["fvu"] < 1.0


@pytest.mark.parametrize("overrides", [{"architecture": "jumprelu"}, {"k": 0}, {"total_tokens": 1}, {"lr": -1.0}])
def test_invalid_configurations_are_rejected(overrides):
    with pytest.raises(ValueError):
        SAELensTrainerConfig(**overrides)
