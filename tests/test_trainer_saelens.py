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


def offset_batches(n_batches: int, *, batch: int = 512, d_in: int = 32, seed: int = 42) -> list[torch.Tensor]:
    """Sparse combinations of fixed directions on top of a common component six times larger."""
    fixed = torch.Generator().manual_seed(1234)
    dictionary = torch.nn.functional.normalize(torch.randn(128, d_in, generator=fixed), dim=-1)
    common = 12.0 * torch.nn.functional.normalize(torch.randn(d_in, generator=fixed), dim=0)

    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        chosen = torch.randint(128, (batch, 4), generator=generator)
        weights = torch.rand(batch, 4, generator=generator) + 0.5
        batches.append((weights.unsqueeze(-1) * dictionary[chosen]).sum(dim=1) + common)
    return batches


def test_decoder_bias_initialisation_prevents_collapse_on_a_common_component(tmp_path):
    import muscari.trainers.saelens as backend

    held_out = torch.cat(offset_batches(8, seed=1))
    metrics = {}
    for init_b_dec in (True, False):
        cfg = SAELensTrainerConfig(
            num_latents=256,
            k=8,
            dead_after_tokens=512 * 50,
            total_tokens=512 * 300,
            batch_tokens=512,
            init_b_dec=init_b_dec,
            device="cpu",
        )
        # Exactly total_steps batches: the batch used for the initialisation must still be trained on.
        out = backend.train_with_saelens(cfg, 32, offset_batches(cfg.total_steps), tmp_path / str(init_b_dec))
        exported = load_sae(out)
        metrics[init_b_dec] = evaluate_sae(exported, held_out)
        assert exported.config["b_dec_init"] == ("geometric_median" if init_b_dec else "zero")

    assert metrics[True]["fvu"] < 0.7
    assert metrics[False]["fvu"] > 1.0  # stock initialisation: worse than predicting the mean
    assert metrics[False]["max_frequency"] == pytest.approx(1.0)
    assert metrics[True]["max_frequency"] < 0.5


def test_the_seed_fixes_the_initialisation(tmp_path):
    import muscari.trainers.saelens as backend

    def train(name: str, seed: int) -> torch.Tensor:
        torch.manual_seed(len(name))  # whatever the global generator holds must not matter
        cfg = SAELensTrainerConfig(
            num_latents=64,
            k=4,
            total_tokens=512 * 5,
            batch_tokens=512,
            seed=seed,
            device="cpu",
        )
        out = backend.train_with_saelens(cfg, 32, offset_batches(cfg.total_steps), tmp_path / name)
        return load_sae(out).encoder_weight

    first = train("a", seed=7)

    assert torch.equal(train("bb", seed=7), first)
    assert not torch.equal(train("ccc", seed=8), first)


@pytest.mark.parametrize("overrides", [{"architecture": "jumprelu"}, {"k": 0}, {"total_tokens": 1}, {"lr": -1.0}])
def test_invalid_configurations_are_rejected(overrides):
    with pytest.raises(ValueError):
        SAELensTrainerConfig(**overrides)
