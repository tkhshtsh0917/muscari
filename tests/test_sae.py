"""Top-K SAE: sparse autograd against dense oracles, initialisation, AuxK, decoder norms."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from muscari import TopKSAE, geometric_median, normalized_mse
from muscari import sae as sae_module
from muscari.sae import _SparseDecode, _SparseEncode


def make_sae(*, d_in: int = 12, num_latents: int = 40, k: int = 5) -> TopKSAE:
    torch.manual_seed(0)
    sae = TopKSAE(d_in, num_latents, k)
    with torch.no_grad():
        sae.b_dec.normal_()
        sae.encoder.bias.normal_(std=0.1)
    return sae


def dense_loss(sae: TopKSAE, x: torch.Tensor, dead: torch.Tensor | None, aux_k: int) -> dict[str, torch.Tensor]:
    """The textbook computation with dense codes and plain autograd, as an independent oracle."""
    pre = (x - sae.b_dec) @ sae.encoder.weight.T + sae.encoder.bias
    values, indices = pre.topk(sae.k, dim=-1)
    code = torch.zeros_like(pre).scatter(-1, indices, values.relu())
    recon = code @ sae.decoder + sae.b_dec
    fvu = normalized_mse(recon, x)

    aux = recon.new_zeros(())
    if dead is not None and aux_k > 0 and dead.any():
        aux_values, aux_indices = pre.masked_fill(~dead, float("-inf")).topk(min(aux_k, int(dead.sum())), dim=-1)
        aux_code = torch.zeros_like(pre).scatter(-1, aux_indices, aux_values.relu())
        aux = normalized_mse(aux_code @ sae.decoder, (x - recon).detach())
    return {"fvu": fvu, "aux": aux, "recon": recon}


def gradients(sae: TopKSAE, loss: torch.Tensor) -> list[torch.Tensor]:
    return list(torch.autograd.grad(loss, list(sae.parameters())))


def test_forward_matches_the_dense_formula():
    sae = make_sae()
    x = torch.randn(17, sae.d_in)

    assert torch.allclose(sae(x), dense_loss(sae, x, None, 0)["recon"], atol=1e-5)


def test_encode_returns_the_k_largest_preactivations_after_relu():
    sae = make_sae()
    x = torch.randn(9, sae.d_in)
    values, indices = sae.encode(x)

    pre = (x - sae.b_dec) @ sae.encoder.weight.T + sae.encoder.bias
    expected = pre.topk(sae.k, dim=-1).values.relu()

    assert values.shape == indices.shape == (9, sae.k)
    assert torch.allclose(values.sort(dim=-1, descending=True).values, expected, atol=1e-6)
    assert torch.allclose(pre.gather(-1, indices).relu(), values, atol=1e-6)


@pytest.mark.parametrize("dense_aux_ratio", [0, 10**6], ids=["gathered-aux", "dense-aux"])
@pytest.mark.parametrize("aux_k", [0, 8, 100])
def test_loss_and_gradients_match_dense_autograd(monkeypatch, aux_k, dense_aux_ratio):
    monkeypatch.setattr(sae_module, "_DENSE_AUX_RATIO", dense_aux_ratio)
    sae = make_sae()
    x = torch.randn(64, sae.d_in)
    dead = torch.zeros(sae.num_latents, dtype=torch.bool)
    dead[20:] = True

    sparse = sae.loss(x, dead_mask=dead, aux_k=aux_k)
    dense = dense_loss(sae, x, dead, aux_k)

    assert sparse["fvu"].item() == pytest.approx(dense["fvu"].item(), rel=1e-5)
    assert sparse["aux"].item() == pytest.approx(dense["aux"].item(), rel=1e-5)
    for ours, reference in zip(
        gradients(sae, sparse["fvu"] + 0.5 * sparse["aux"]),
        gradients(sae, dense["fvu"] + 0.5 * dense["aux"]),
        strict=True,
    ):
        assert torch.allclose(ours, reference, atol=1e-6)


def test_gradients_are_independent_of_the_chunk_size(monkeypatch):
    monkeypatch.setattr(sae_module, "_DENSE_AUX_RATIO", 0)  # route the auxiliary path through gathers
    sae = make_sae()
    x = torch.randn(64, sae.d_in)
    dead = torch.zeros(sae.num_latents, dtype=torch.bool)
    dead[20:] = True

    def run() -> list[torch.Tensor]:
        out = sae.loss(x, dead_mask=dead, aux_k=8)
        return gradients(sae, out["fvu"] + out["aux"])

    whole = run()
    monkeypatch.setattr(sae_module, "_CHUNK_ELEMENTS", 1)  # one row per chunk
    row_by_row = run()

    for chunked, reference in zip(row_by_row, whole, strict=True):
        assert torch.allclose(chunked, reference, atol=1e-6)


def test_sparse_encode_passes_a_numerical_gradient_check(monkeypatch):
    torch.manual_seed(0)
    x = torch.randn(6, 5, dtype=torch.double, requires_grad=True)
    weight = torch.randn(11, 5, dtype=torch.double, requires_grad=True)
    bias = torch.randn(11, dtype=torch.double, requires_grad=True)
    b_dec = torch.randn(5, dtype=torch.double, requires_grad=True)
    dead = torch.zeros(11, dtype=torch.bool)
    dead[6:] = True

    def encode(x_, weight_, bias_, b_dec_):
        values, _, aux_values, _ = _SparseEncode.apply(x_, weight_, bias_, b_dec_, dead.nonzero().squeeze(1), 3, 2)
        return values, aux_values

    for dense_aux_ratio in (0, 10**6):
        monkeypatch.setattr(sae_module, "_DENSE_AUX_RATIO", dense_aux_ratio)
        assert torch.autograd.gradcheck(encode, (x, weight, bias, b_dec))


def test_sparse_decode_passes_a_numerical_gradient_check():
    torch.manual_seed(0)
    values = torch.rand(7, 4, dtype=torch.double, requires_grad=True)
    indices = torch.randint(10, (7, 4))
    indices[0] = 3  # a row that selects the same latent several times
    decoder = torch.randn(10, 5, dtype=torch.double, requires_grad=True)

    def decode(values_, decoder_):
        return _SparseDecode.apply(values_, indices, decoder_)

    assert torch.allclose(decode(values, decoder), torch.einsum("bn,bnd->bd", values, decoder[indices]))
    assert torch.autograd.gradcheck(decode, (values, decoder))


@pytest.mark.parametrize("dense_aux_ratio", [0, 10**6], ids=["gathered-aux", "dense-aux"])
def test_a_latent_in_both_selections_accumulates_both_gradients(monkeypatch, dense_aux_ratio):
    monkeypatch.setattr(sae_module, "_DENSE_AUX_RATIO", dense_aux_ratio)
    sae = make_sae(num_latents=6, k=6)  # every latent is selected, so dead ones are in both
    x = torch.randn(32, sae.d_in)
    dead = torch.tensor([True, True, False, False, False, False])

    sparse = sae.loss(x, dead_mask=dead, aux_k=2)
    dense = dense_loss(sae, x, dead, 2)

    for ours, reference in zip(
        gradients(sae, sparse["fvu"] + sparse["aux"]),
        gradients(sae, dense["fvu"] + dense["aux"]),
        strict=True,
    ):
        assert torch.allclose(ours, reference, atol=1e-6)


def test_loss_runs_under_bfloat16_autocast():
    sae = make_sae()
    x = torch.randn(64, sae.d_in)
    dead = torch.zeros(sae.num_latents, dtype=torch.bool)
    dead[20:] = True

    exact = sae.loss(x, dead_mask=dead, aux_k=8)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        mixed = sae.loss(x, dead_mask=dead, aux_k=8)
    (mixed["fvu"] + mixed["aux"]).backward()

    assert mixed["fvu"].item() == pytest.approx(exact["fvu"].item(), rel=0.05)
    assert all(parameter.grad.dtype == torch.float32 for parameter in sae.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in sae.parameters())


def test_initialisation_is_tied_unit_norm_and_scale_calibrated():
    torch.manual_seed(0)
    sae = TopKSAE(64, 512, 8)

    assert torch.allclose(sae.decoder.norm(dim=-1), torch.ones(512), atol=1e-6)
    cosine = F.cosine_similarity(sae.encoder.weight, sae.decoder, dim=-1)
    assert torch.allclose(cosine, torch.ones(512), atol=1e-5)
    assert torch.count_nonzero(sae.encoder.bias) == 0 and torch.count_nonzero(sae.b_dec) == 0

    probes = F.normalize(torch.randn(2048, 64), dim=-1)
    with torch.no_grad():
        mean_norm = sae(probes).norm(dim=-1).mean().item()
    assert 0.9 < mean_norm < 1.1


def test_b_dec_starts_at_the_geometric_median():
    sae = make_sae()
    sample = torch.randn(500, sae.d_in) + 3.0
    sae.init_b_dec(sample)

    assert torch.allclose(sae.b_dec, geometric_median(sample), atol=1e-6)


def test_geometric_median_is_the_center_of_symmetric_points():
    center = torch.tensor([1.0, -2.0, 0.5])
    offsets = torch.randn(50, 3)
    points = torch.cat([center + offsets, center - offsets])

    assert torch.allclose(geometric_median(points), center, atol=1e-4)


def test_geometric_median_resists_an_outlier_that_moves_the_mean():
    torch.manual_seed(0)
    points = torch.randn(200, 4)
    points[0] = 1e4

    assert geometric_median(points).norm() < 0.5
    assert points.mean(dim=0).norm() > 50


def test_geometric_median_of_collinear_points_is_their_median():
    points = torch.tensor([[0.0], [1.0], [2.0], [10.0], [100.0]])

    assert torch.allclose(geometric_median(points, max_iter=500, tol=1e-9), torch.tensor([2.0]), atol=1e-3)


def test_geometric_median_rejects_empty_input():
    with pytest.raises(ValueError):
        geometric_median(torch.empty(0, 3))


def test_normalized_mse_is_zero_when_exact_and_one_for_the_batch_mean():
    target = torch.randn(100, 6) + 2.0

    assert normalized_mse(target, target).item() == 0.0
    assert normalized_mse(target.mean(dim=0, keepdim=True).expand_as(target), target).item() == pytest.approx(1.0)


def test_auxk_trains_only_dead_latents():
    sae = make_sae()
    dead = torch.zeros(sae.num_latents, dtype=torch.bool)
    dead[30:] = True

    out = sae.loss(torch.randn(32, sae.d_in), dead_mask=dead, aux_k=4)
    out["aux"].backward()

    assert out["aux"].item() > 0
    for gradient in (sae.encoder.weight.grad, sae.encoder.bias.grad, sae.decoder.grad):
        assert torch.count_nonzero(gradient[~dead]) == 0
        assert torch.count_nonzero(gradient[dead]) > 0


def test_auxk_is_zero_without_dead_latents_or_when_disabled():
    sae = make_sae()
    x = torch.randn(16, sae.d_in)
    none_dead = torch.zeros(sae.num_latents, dtype=torch.bool)
    some_dead = none_dead.clone()
    some_dead[:5] = True

    assert sae.loss(x, dead_mask=none_dead, aux_k=4)["aux"].item() == 0.0
    assert sae.loss(x, dead_mask=some_dead, aux_k=0)["aux"].item() == 0.0
    assert sae.loss(x)["aux"].item() == 0.0


def test_auxk_uses_at_most_the_available_dead_latents():
    sae = make_sae()
    dead = torch.zeros(sae.num_latents, dtype=torch.bool)
    dead[:3] = True

    out = sae.loss(torch.randn(16, sae.d_in), dead_mask=dead, aux_k=10)

    assert torch.isfinite(out["aux"])


def test_decoder_gradient_projection_and_renormalisation_keep_unit_rows():
    sae = make_sae()
    sae.loss(torch.randn(32, sae.d_in))["fvu"].backward()

    sae.remove_parallel_decoder_grad()
    parallel = (sae.decoder.grad * sae.decoder).sum(dim=-1)
    assert torch.allclose(parallel, torch.zeros_like(parallel), atol=1e-6)

    with torch.no_grad():
        sae.decoder.mul_(3.0)
    sae.normalize_decoder()
    assert torch.allclose(sae.decoder.norm(dim=-1), torch.ones(sae.num_latents), atol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_in": 4, "num_latents": 8, "k": 9},
        {"d_in": 4, "num_latents": 8, "k": 0},
        {"d_in": 0, "num_latents": 8, "k": 1},
        {"d_in": 4, "num_latents": 0, "k": 1},
    ],
)
def test_invalid_sizes_are_rejected(kwargs):
    with pytest.raises(ValueError):
        TopKSAE(**kwargs)
