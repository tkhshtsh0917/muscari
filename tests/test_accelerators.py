"""Hardware checks for CUDA and Apple MPS; skipped where PyTorch can use neither."""

from __future__ import annotations

import pytest
import torch
from tiny import DOC_PREFIX, QUERY_PREFIX

from muscari import (
    EncodedText,
    EncoderActivationSource,
    TopKSAE,
    TopKTrainer,
    TopKTrainerConfig,
    evaluate_texts,
    latent_statistics,
    load_sae,
    resolve_device,
    save_topk_sae,
)

ACCELERATORS = [
    name
    for name, available in (("cuda", torch.cuda.is_available()), ("mps", torch.backends.mps.is_available()))
    if available
]

pytestmark = [
    pytest.mark.accelerator,
    pytest.mark.skipif(not ACCELERATORS, reason="neither CUDA nor MPS is available"),
]


@pytest.fixture(params=ACCELERATORS or ["unavailable"])
def device(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def sparse_batch(generator: torch.Generator, directions: torch.Tensor) -> torch.Tensor:
    chosen = torch.randint(directions.shape[0], (512, 2), generator=generator)
    return directions[chosen].sum(dim=1)


def test_auto_device_prefers_an_accelerator():
    assert resolve_device("auto").type == ("cuda" if "cuda" in ACCELERATORS else "mps")


def test_explicit_device_is_accepted(device):
    assert resolve_device(device).type == device


@pytest.mark.parametrize("n_dead", [112, 500], ids=["few-dead", "most-dead"])
def test_loss_and_gradients_match_the_cpu(device, n_dead):
    x = torch.randn(256, 24)
    dead = torch.zeros(512, dtype=torch.bool)
    dead[512 - n_dead :] = True
    results = {}

    for name in ("cpu", device):
        torch.manual_seed(42)
        sae = TopKSAE(24, 512, 8).to(name)
        out = sae.loss(x.to(name), dead_mask=dead.to(name), aux_k=4)  # 500 dead > 64 * 4: the gathered path off CUDA
        (out["fvu"] + out["aux"]).backward()
        results[name] = (
            out["fvu"].item(),
            out["aux"].item(),
            [parameter.grad.cpu() for parameter in sae.parameters()],
        )

    assert results[device][0] == pytest.approx(results["cpu"][0], rel=1e-4)
    assert results[device][1] == pytest.approx(results["cpu"][1], rel=1e-4)
    for accelerated, reference in zip(results[device][2], results["cpu"][2], strict=True):
        assert torch.allclose(accelerated, reference, atol=1e-5)


@pytest.mark.parametrize("amp", [False, True], ids=["fp32", "bf16"])
def test_trainer_learns(device, amp):
    if amp and device != "cuda":
        pytest.skip("bfloat16 autocast is only used on CUDA")

    cfg = TopKTrainerConfig(
        num_latents=256,
        k=4,
        aux_k=16,
        dead_after_tokens=4_000,
        total_tokens=512 * 200,
        batch_tokens=512,
        lr=3e-3,
        amp=amp,
        device=device,
    )
    trainer = TopKTrainer(cfg, d_in=24)
    generator = torch.Generator().manual_seed(42)
    directions = torch.nn.functional.normalize(torch.randn(64, 24, generator=generator), dim=-1)

    first = trainer.train_step(sparse_batch(generator, directions))
    for _ in range(199):
        last = trainer.train_step(sparse_batch(generator, directions))

    assert trainer.sae.decoder.device.type == device
    assert trainer.sae.decoder.dtype == torch.float32
    assert last["fvu"] < 0.5 * first["fvu"]


def test_bfloat16_loss_stays_close_to_float32():
    if "cuda" not in ACCELERATORS:
        pytest.skip("bfloat16 autocast is only used on CUDA")

    torch.manual_seed(42)
    sae = TopKSAE(24, 512, 8).cuda()
    x = torch.randn(256, 24, device="cuda")
    dead = torch.zeros(512, dtype=torch.bool, device="cuda")
    dead[400:] = True

    exact = sae.loss(x, dead_mask=dead, aux_k=32)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mixed = sae.loss(x, dead_mask=dead, aux_k=32)
    (mixed["fvu"] + mixed["aux"]).backward()

    assert mixed["fvu"].item() == pytest.approx(exact["fvu"].item(), rel=0.02)
    assert mixed["aux"].item() == pytest.approx(exact["aux"].item(), rel=0.02)
    assert all(parameter.grad.dtype == torch.float32 for parameter in sae.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in sae.parameters())


def test_activations_match_the_cpu(tiny_model_dir, device):
    batch = ([DOC_PREFIX, QUERY_PREFIX], ["東京 求人 検索", "名古屋 営業"])
    on_cpu = EncoderActivationSource.from_pretrained(str(tiny_model_dir), device="cpu").encode_batch(*batch)
    accelerated = EncoderActivationSource.from_pretrained(str(tiny_model_dir), device=device).encode_batch(*batch)

    assert accelerated.device.type == device
    assert torch.allclose(accelerated.cpu(), on_cpu, atol=1e-4)


def test_text_and_latent_statistics_match_the_cpu(tmp_path, device):
    torch.manual_seed(42)
    out = save_topk_sae(tmp_path, TopKSAE(16, 64, 4), {})
    texts = [
        EncodedText(prefix, torch.randn(length, 16), torch.randint(50, (length,)))
        for prefix, length in ((QUERY_PREFIX, 3), (DOC_PREFIX, 40), (QUERY_PREFIX, 1), (DOC_PREFIX, 25))
    ]
    on_cpu, accelerated = load_sae(out, "cpu"), load_sae(out, device)

    assert evaluate_texts(accelerated, texts, batch_tokens=32) == evaluate_texts(on_cpu, texts, batch_tokens=32)
    for fast, exact in zip(latent_statistics(accelerated, texts), latent_statistics(on_cpu, texts), strict=True):
        assert fast.pop("mean_activation") == pytest.approx(exact.pop("mean_activation"), rel=1e-4)
        assert fast == exact
