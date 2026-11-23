"""Hardware checks for Apple MPS; skipped wherever PyTorch cannot use it."""

from __future__ import annotations

import pytest
import torch
from tiny import DOC_PREFIX, QUERY_PREFIX

from muscari import EncoderActivationSource, ShizukuTrainer, ShizukuTrainerConfig, TopKSAE, resolve_device

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available"),
]


def test_auto_device_prefers_mps_over_cpu():
    assert resolve_device("auto").type in ("cuda", "mps")
    assert resolve_device("mps").type == "mps"


def test_loss_and_gradients_match_the_cpu():
    x = torch.randn(256, 24)
    dead = torch.zeros(512, dtype=torch.bool)
    dead[400:] = True
    results = {}

    for device in ("cpu", "mps"):
        torch.manual_seed(0)
        sae = TopKSAE(24, 512, 8).to(device)
        out = sae.loss(x.to(device), dead_mask=dead.to(device), aux_k=32)
        (out["fvu"] + out["aux"]).backward()
        results[device] = (
            out["fvu"].item(),
            out["aux"].item(),
            [parameter.grad.cpu() for parameter in sae.parameters()],
        )

    assert results["mps"][0] == pytest.approx(results["cpu"][0], rel=1e-4)
    assert results["mps"][1] == pytest.approx(results["cpu"][1], rel=1e-4)
    for on_mps, on_cpu in zip(results["mps"][2], results["cpu"][2], strict=True):
        assert torch.allclose(on_mps, on_cpu, atol=1e-5)


def test_trainer_learns_on_mps():
    cfg = ShizukuTrainerConfig(
        num_latents=256,
        k=4,
        aux_k=16,
        dead_after_tokens=4_000,
        total_tokens=512 * 200,
        batch_tokens=512,
        lr=3e-3,
        device="mps",
    )
    trainer = ShizukuTrainer(cfg, d_in=24)
    generator = torch.Generator().manual_seed(0)
    directions = torch.nn.functional.normalize(torch.randn(64, 24, generator=generator), dim=-1)

    def batch() -> torch.Tensor:
        chosen = torch.randint(64, (512, 2), generator=generator)
        return directions[chosen].sum(dim=1)

    first = trainer.train_step(batch())
    for _ in range(199):
        last = trainer.train_step(batch())

    assert trainer.sae.decoder.device.type == "mps"
    assert last["fvu"] < 0.5 * first["fvu"]


def test_activations_match_the_cpu(tiny_model_dir):
    batch = ([DOC_PREFIX, QUERY_PREFIX], ["東京 求人 検索", "名古屋 営業"])
    on_cpu = EncoderActivationSource.from_pretrained(str(tiny_model_dir), device="cpu").encode_batch(*batch)
    on_mps = EncoderActivationSource.from_pretrained(str(tiny_model_dir), device="mps").encode_batch(*batch)

    assert on_mps.device.type == "mps"
    assert torch.allclose(on_mps.cpu(), on_cpu, atol=1e-4)
