"""TopKTrainer: schedule, learning, dead-latent bookkeeping and exact resumption."""

from __future__ import annotations

import json
import math

import pytest
import torch
import torch.nn.functional as F

from muscari import TopKTrainer, TopKTrainerConfig
from muscari.trainers import warmup_cosine
from muscari.trainers.topk import CHECKPOINT_NAME


def small_config(**overrides) -> TopKTrainerConfig:
    arguments = {
        "num_latents": 64,
        "k": 2,
        "aux_k": 8,
        "dead_after_tokens": 2_000,
        "total_tokens": 256 * 40,
        "batch_tokens": 256,
        "log_every": 10,
        "device": "cpu",
    }
    arguments.update(overrides)
    return TopKTrainerConfig(**arguments)


def sparse_dictionary_batches(
    n_batches: int, *, batch: int = 256, d_in: int = 16, seed: int = 42
) -> list[torch.Tensor]:
    """Activations that are sums of two out of 32 fixed directions with positive weights."""
    dictionary = F.normalize(torch.randn(32, d_in, generator=torch.Generator().manual_seed(1234)), dim=-1)
    generator = torch.Generator().manual_seed(seed)

    batches = []
    for _ in range(n_batches):
        chosen = torch.randint(32, (batch, 2), generator=generator)
        weights = torch.rand(batch, 2, generator=generator) + 0.5
        batches.append((weights.unsqueeze(-1) * dictionary[chosen]).sum(dim=1) + 1.0)
    return batches


def test_schedule_warms_up_linearly_then_decays_to_zero():
    multipliers = [warmup_cosine(step, total_steps=100, warmup_steps=5) for step in range(100)]

    assert multipliers[:5] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert multipliers[5] == pytest.approx(1.0)
    assert all(earlier >= later for earlier, later in zip(multipliers[5:], multipliers[6:], strict=False))
    assert multipliers[52] == pytest.approx(0.5 * (1 + math.cos(math.pi * 47 / 95)))
    assert warmup_cosine(100, total_steps=100, warmup_steps=5) == pytest.approx(0.0)
    assert warmup_cosine(0, total_steps=100, warmup_steps=0) == pytest.approx(1.0)


def test_training_recovers_structure_in_sparse_data():
    cfg = small_config(total_tokens=256 * 1000, lr=3e-3)
    trainer = TopKTrainer(cfg, d_in=16)
    held_out = sparse_dictionary_batches(1, seed=1)[0]

    def fvu() -> float:
        with torch.no_grad():
            residual = (held_out - trainer.sae(held_out)).pow(2).sum()
            return (residual / (held_out - held_out.mean(dim=0)).pow(2).sum()).item()

    trainer.sae.init_b_dec(held_out)
    before = fvu()
    trainer.fit(sparse_dictionary_batches(1000))

    assert trainer.step == cfg.total_steps == 1000
    assert trainer.tokens_seen == 256 * 1000
    assert fvu() < 0.15 < 0.5 * before
    assert torch.allclose(trainer.sae.decoder.norm(dim=-1), torch.ones(64), atol=1e-5)


def test_learning_rate_follows_the_schedule():
    cfg = small_config(lr=1e-3, warmup_frac=0.1)
    trainer = TopKTrainer(cfg, d_in=16)
    rates = [trainer.train_step(batch)["lr"] for batch in sparse_dictionary_batches(cfg.total_steps)]

    assert rates[:4] == pytest.approx([0.25e-3, 0.5e-3, 0.75e-3, 1e-3])
    assert rates[-1] < 1e-5
    assert trainer.optimizer.param_groups[0]["lr"] == rates[-1]


def test_latents_die_after_the_token_window():
    cfg = small_config(num_latents=512, dead_after_tokens=300, aux_k=0)
    trainer = TopKTrainer(cfg, d_in=16)
    batches = sparse_dictionary_batches(3)

    trainer.train_step(batches[0])
    assert set(trainer.tokens_since_fired.tolist()) == {0, 256}
    assert not trainer.dead_mask().any()  # nothing has been silent for more than 300 tokens

    metrics = trainer.train_step(batches[1])
    silent = trainer.tokens_since_fired == 512
    assert silent.any()
    assert torch.equal(trainer.dead_mask(), silent)
    assert metrics["dead_frac"] == 0.0  # reported for the dictionary before the step

    assert trainer.train_step(batches[2])["dead_frac"] == pytest.approx(silent.float().mean().item())


def test_unused_fraction_counts_latents_silent_over_the_window():
    cfg = small_config(num_latents=512, aux_k=0)
    trainer = TopKTrainer(cfg, d_in=16)
    batches = sparse_dictionary_batches(2)

    assert trainer.unused_frac() == 0.0  # nothing has been trained yet

    trainer.train_step(batches[0])
    trainer.train_step(batches[1])
    silent_for_both = (trainer.tokens_since_fired == 512).float().mean().item()
    silent_for_last = (trainer.tokens_since_fired >= 256).float().mean().item()

    assert 0 < silent_for_both < silent_for_last
    assert trainer.unused_frac() == pytest.approx(silent_for_both)  # the window is the whole run so far
    assert trainer.unused_frac(window_tokens=256) == pytest.approx(silent_for_last)
    assert trainer.step_seconds > 0


def test_a_selected_latent_with_zero_activation_does_not_count_as_firing():
    cfg = small_config(num_latents=4, k=4, aux_k=0)
    trainer = TopKTrainer(cfg, d_in=16)
    with torch.no_grad():
        trainer.sae.encoder.bias[0] = -1e6  # always selected because k == num_latents, never positive

    trainer.train_step(sparse_dictionary_batches(1)[0])

    assert trainer.tokens_since_fired[0] == 256
    assert (trainer.tokens_since_fired[1:] == 0).any()


def test_auxiliary_loss_is_active_once_latents_are_dead():
    cfg = small_config(dead_after_tokens=256, total_tokens=256 * 10)
    trainer = TopKTrainer(cfg, d_in=16)
    metrics = [trainer.train_step(batch) for batch in sparse_dictionary_batches(10)]

    assert metrics[0]["aux"] == 0.0
    assert any(record["dead_frac"] > 0 and record["aux"] > 0 for record in metrics)


def test_resuming_from_a_checkpoint_continues_exactly(tmp_path):
    cfg = small_config()
    batches = sparse_dictionary_batches(cfg.total_steps)

    uninterrupted = TopKTrainer(cfg, d_in=16)
    uninterrupted.fit(batches)

    first_half = TopKTrainer(cfg, d_in=16)
    first_half.fit(batches[:20])
    first_half.save_checkpoint(tmp_path / CHECKPOINT_NAME, data={"texts_seen": 123})

    resumed = TopKTrainer(small_config(seed=99), d_in=16)  # a different initialisation
    data_state = resumed.load_checkpoint(tmp_path / CHECKPOINT_NAME)
    resumed.fit(batches[20:])

    assert data_state == {"texts_seen": 123}
    assert (resumed.step, resumed.tokens_seen) == (uninterrupted.step, uninterrupted.tokens_seen)
    assert torch.equal(resumed.tokens_since_fired, uninterrupted.tokens_since_fired)
    for restored, reference in zip(resumed.sae.parameters(), uninterrupted.sae.parameters(), strict=True):
        assert torch.equal(restored, reference)


def test_fit_writes_logs_config_and_periodic_checkpoints(tmp_path):
    cfg = small_config(checkpoint_every=15)
    trainer = TopKTrainer(cfg, d_in=16)
    seen = []

    trainer.fit(
        sparse_dictionary_batches(cfg.total_steps + 5),  # extra batches are left unused
        out_dir=tmp_path,
        log_fn=seen.append,
        data_state=lambda: {"texts_seen": trainer.step * 7},
    )

    log = [json.loads(line) for line in (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert [record["step"] for record in log] == [10, 20, 30, 40]
    assert log == seen
    assert log[-1]["tokens"] == cfg.total_tokens and log[-1]["tokens_per_sec"] > 0
    assert json.loads((tmp_path / "trainer_config.json").read_text())["num_latents"] == 64

    checkpoint = torch.load(tmp_path / CHECKPOINT_NAME, weights_only=True)
    assert checkpoint["step"] == 40 and checkpoint["data"] == {"texts_seen": 280}  # 15, 30 and the end
    assert not (tmp_path / (CHECKPOINT_NAME + ".tmp")).exists()


def test_a_completed_run_does_not_consume_batches():
    trainer = TopKTrainer(small_config(), d_in=16)
    trainer.fit(sparse_dictionary_batches(40))

    def fail():
        raise AssertionError("no batch should be requested")
        yield

    trainer.fit(fail())

    assert trainer.step == 40


def test_fit_stops_early_when_the_batches_run_out():
    trainer = TopKTrainer(small_config(), d_in=16)
    trainer.fit(sparse_dictionary_batches(7))

    assert trainer.step == 7


@pytest.mark.parametrize(
    ("overrides", "d_in"),
    [({"num_latents": 128}, 16), ({"k": 3}, 16), ({"batch_tokens": 128}, 16), ({}, 8)],
)
def test_a_checkpoint_from_a_different_shape_is_refused(tmp_path, overrides, d_in):
    TopKTrainer(small_config(), d_in=16).save_checkpoint(tmp_path / CHECKPOINT_NAME)

    with pytest.raises(ValueError, match="does not match"):
        TopKTrainer(small_config(**overrides), d_in=d_in).load_checkpoint(tmp_path / CHECKPOINT_NAME)


def test_training_length_may_be_extended_on_resume(tmp_path):
    TopKTrainer(small_config(), d_in=16).save_checkpoint(tmp_path / CHECKPOINT_NAME)

    TopKTrainer(small_config(total_tokens=256 * 80), d_in=16).load_checkpoint(tmp_path / CHECKPOINT_NAME)


@pytest.mark.parametrize(
    "overrides",
    [
        {"k": 0},
        {"k": 65},
        {"aux_k": -1},
        {"batch_tokens": 0},
        {"total_tokens": 100},
        {"lr": 0.0},
        {"warmup_frac": 1.0},
        {"dead_after_tokens": 0},
        {"log_every": 0},
        {"checkpoint_every": -1},
    ],
)
def test_invalid_configurations_are_rejected(overrides):
    with pytest.raises(ValueError):
        small_config(**overrides)
