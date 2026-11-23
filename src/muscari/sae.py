"""Top-K sparse autoencoder (Gao et al., 2024, "Scaling and evaluating sparse autoencoders").

    z     = TopK_k(ReLU(W_enc (x - b_dec) + b_enc))
    x_hat = W_dec z + b_dec

The decoder is stored as ``(num_latents, d_in)`` so that each row is one feature direction;
rows are kept at unit norm by the trainer.

Only ``k`` of the ``num_latents`` latents are active per token, and the gradient of the loss is
zero for every other latent. Plain autograd does not use this: it materialises a dense
``(batch, num_latents)`` gradient and multiplies it with the weights twice, which costs twice
the forward pass and gigabytes of memory. The autograd functions below propagate through the
selected entries only, so a training step is dominated by the single dense product
``W_enc (x - b_dec)`` of the forward pass. They use only gather, matmul, einsum, scatter and
``index_add_`` and therefore run on CPU, CUDA and MPS alike.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

# Upper bound on the elements of one gathered ``(rows, n, d_in)`` block.
_CHUNK_ELEMENTS = 1 << 25

# The auxiliary reconstruction touches ``aux_k`` latents per token, far more than ``k``. A
# gather moves data much more slowly than a matrix product multiplies it, so while the dead
# latents number at most this multiple of ``aux_k`` the auxiliary path uses dense products
# restricted to the dead columns. Beyond that it falls back to gathers, whose cost does not
# grow with the number of dead latents. Both give the same result.
_DENSE_AUX_RATIO = 64


def _use_dense_aux(n_dead: int, aux_k: int) -> bool:
    return n_dead <= _DENSE_AUX_RATIO * aux_k


def normalized_mse(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared error divided by the squared error of predicting the batch mean of ``target``.

    For a reconstruction this is the fraction of variance unexplained (FVU) of the batch.
    """
    residual = (target - recon).pow(2).sum()
    total = (target - target.mean(dim=0, keepdim=True)).pow(2).sum()
    return residual / total.clamp_min(1e-12)


@torch.no_grad()
def geometric_median(points: torch.Tensor, *, max_iter: int = 100, tol: float = 1e-5) -> torch.Tensor:
    """Geometric median of the rows of ``points`` ``(n, d)`` by Weiszfeld's algorithm.

    The geometric median minimises the sum of Euclidean distances to the rows, which makes it
    less sensitive to a few extreme activations than the mean. Iteration stops when the update
    is smaller than ``tol`` relative to the spread of the data, or after ``max_iter`` rounds.

    Raises:
        ValueError: if ``points`` is not a non-empty matrix.
    """
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("points must be a non-empty (n, d) tensor.")

    points = points.float()
    median = points.mean(dim=0)
    scale = (points - median).norm(dim=1).mean().clamp_min(1e-12)

    for _ in range(max_iter):
        distances = (points - median).norm(dim=1).clamp_min(1e-8 * scale)
        weights = 1.0 / distances
        updated = (weights.unsqueeze(1) * points).sum(dim=0) / weights.sum()

        converged = (updated - median).norm() <= tol * scale
        median = updated
        if converged:
            break
    return median


def _rows_per_chunk(indices: torch.Tensor, width: int) -> int:
    return max(1, _CHUNK_ELEMENTS // max(1, indices.shape[1] * width))


def _sum_selected_rows(weights: torch.Tensor, indices: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """``out[b] = sum_j weights[b, j] * matrix[indices[b, j]]``, computed in row chunks.

    ``weights`` and ``indices`` are ``(batch, n)``, ``matrix`` is ``(rows, width)`` and the
    result is ``(batch, width)``.
    """
    out = matrix.new_empty((weights.shape[0], matrix.shape[1]))
    step = _rows_per_chunk(indices, matrix.shape[1])

    for start in range(0, weights.shape[0], step):
        rows = slice(start, start + step)
        out[rows] = torch.einsum("bn,bnd->bd", weights[rows], matrix[indices[rows]])
    return out


def _add_outer_to_rows(
    target: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    vectors: torch.Tensor,
) -> None:
    """``target[indices[b, j]] += weights[b, j] * vectors[b]`` for all ``b, j``, in row chunks.

    ``target`` is ``(rows, width)``, ``indices`` and ``weights`` are ``(batch, n)`` and
    ``vectors`` is ``(batch, width)``. Repeated indices accumulate.
    """
    step = _rows_per_chunk(indices, target.shape[1])

    for start in range(0, weights.shape[0], step):
        rows = slice(start, start + step)
        contributions = weights[rows].unsqueeze(-1) * vectors[rows].unsqueeze(1)
        target.index_add_(0, indices[rows].reshape(-1), contributions.reshape(-1, target.shape[1]))


class _SparseEncode(torch.autograd.Function):
    """Top-k pre-activations of ``W (x - b_dec) + b`` with a backward pass over selected entries.

    Returns ``(values, indices, aux_values, aux_indices)``. ``values`` holds the ``k`` largest
    pre-activations per token *before* ReLU. The auxiliary pair holds the ``aux_k`` largest
    pre-activations among the latents listed in ``dead_index`` and has zero columns when
    ``aux_k`` is 0. All indices are latent ids. The dense ``(batch, num_latents)``
    pre-activations exist only inside ``forward``.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        b_dec: torch.Tensor,
        dead_index: torch.Tensor,
        k: int,
        aux_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pre = F.linear(x - b_dec, weight, bias)
        values, indices = pre.topk(k, dim=-1, sorted=False)

        if aux_k > 0:
            aux_values, aux_local = pre.index_select(1, dead_index).topk(aux_k, dim=-1, sorted=False)
            aux_indices = dead_index[aux_local]
        else:
            aux_values = pre.new_empty((x.shape[0], 0))
            aux_local = indices.new_empty((x.shape[0], 0))
            aux_indices = aux_local

        ctx.save_for_backward(x, weight, b_dec, dead_index, indices, aux_indices)
        ctx.aux_local = aux_local
        ctx.mark_non_differentiable(indices, aux_indices)
        return values, indices, aux_values, aux_indices

    @staticmethod
    def backward(
        ctx: Any,
        grad_values: torch.Tensor | None,
        _grad_indices: None,
        grad_aux_values: torch.Tensor | None,
        _grad_aux_indices: None,
    ) -> tuple[torch.Tensor | None, ...]:
        x, weight, b_dec, dead_index, indices, aux_indices = ctx.saved_tensors
        need_x, need_weight, need_bias, need_b_dec = ctx.needs_input_grad[:4]
        centered = x - b_dec

        # With g = d loss / d pre restricted to the selected entries:
        #   d pre[b, m] / d bias[m]   = 1            ->  grad_bias[m]   = sum_b g[b, m]
        #   d pre[b, m] / d weight[m] = x[b] - b_dec ->  grad_weight[m] = sum_b g[b, m] (x[b] - b_dec)
        #   d pre[b, m] / d x[b]      = weight[m]    ->  grad_x[b]      = sum_m g[b, m] weight[m]
        grad_bias = torch.zeros_like(weight[:, 0])
        grad_weight = torch.zeros_like(weight) if need_weight else None
        grad_x = torch.zeros_like(x) if need_x else None

        if grad_values is not None:
            grads = grad_values.to(weight.dtype)
            grad_bias.index_add_(0, indices.reshape(-1), grads.reshape(-1))
            if grad_weight is not None:
                _add_outer_to_rows(grad_weight, indices, grads, centered)
            if grad_x is not None:
                grad_x += _sum_selected_rows(grads, indices, weight)

        if grad_aux_values is not None and aux_indices.shape[1] > 0:
            grads = grad_aux_values.to(weight.dtype)
            if _use_dense_aux(dead_index.numel(), aux_indices.shape[1]):
                dense = grads.new_zeros((x.shape[0], dead_index.numel())).scatter_(1, ctx.aux_local, grads)
                grad_bias.index_add_(0, dead_index, dense.sum(dim=0))
                if grad_weight is not None:
                    grad_weight.index_add_(0, dead_index, dense.T @ centered)
            else:
                grad_bias.index_add_(0, aux_indices.reshape(-1), grads.reshape(-1))
                if grad_weight is not None:
                    _add_outer_to_rows(grad_weight, aux_indices, grads, centered)
            if grad_x is not None:
                grad_x += _sum_selected_rows(grads, aux_indices, weight)

        # b_dec enters every token's pre-activation with the opposite sign of x.
        grad_b_dec = -(grad_bias @ weight) if need_b_dec else None
        return grad_x, grad_weight, grad_bias if need_bias else None, grad_b_dec, None, None, None


class _SparseDecode(torch.autograd.Function):
    """``out[b] = sum_j values[b, j] * decoder[indices[b, j]]`` without a saved ``(batch, n, d)`` block.

    Advanced indexing would keep the gathered block alive for the backward pass: 6 GB for 8192
    tokens, 256 auxiliary latents and 768 dimensions. Here only ``values``, ``indices`` and
    ``decoder`` are saved and both passes walk over row chunks.
    """

    @staticmethod
    def forward(
        ctx: Any,
        values: torch.Tensor,
        indices: torch.Tensor,
        decoder: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(values, indices, decoder)
        return _sum_selected_rows(values, indices, decoder)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, None, torch.Tensor | None]:
        values, indices, decoder = ctx.saved_tensors
        need_values, _, need_decoder = ctx.needs_input_grad

        grad_values = None
        if need_values:
            grad_values = torch.empty_like(values)
            step = _rows_per_chunk(indices, decoder.shape[1])
            for start in range(0, values.shape[0], step):
                rows = slice(start, start + step)
                grad_values[rows] = torch.einsum("bd,bnd->bn", grad_out[rows], decoder[indices[rows]])

        grad_decoder = None
        if need_decoder:
            grad_decoder = torch.zeros_like(decoder)
            _add_outer_to_rows(grad_decoder, indices, values, grad_out)

        return grad_values, None, grad_decoder


class TopKSAE(nn.Module):
    """Trainable Top-K SAE.

    Args:
        d_in: width of the activations being reconstructed.
        num_latents: dictionary size ``M``.
        k: number of latents kept per token.

    Shapes:
        ``encoder.weight`` is ``(num_latents, d_in)``, ``encoder.bias`` is ``(num_latents,)``,
        ``decoder`` is ``(num_latents, d_in)`` and ``b_dec`` is ``(d_in,)``.

    Raises:
        ValueError: if a size is not positive or ``k`` exceeds ``num_latents``.
    """

    def __init__(self, d_in: int, num_latents: int, k: int) -> None:
        super().__init__()
        if d_in < 1 or num_latents < 1 or k < 1:
            raise ValueError("d_in, num_latents and k must be positive.")
        if k > num_latents:
            raise ValueError(f"k ({k}) cannot exceed num_latents ({num_latents}).")

        self.d_in = d_in
        self.num_latents = num_latents
        self.k = k

        self.encoder = nn.Linear(d_in, num_latents)
        self.decoder = nn.Parameter(torch.empty(num_latents, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialise with unit-norm random directions and a tied, rescaled encoder.

        The encoder starts as the transpose of the decoder. It is then scaled by one factor so
        that random unit-norm inputs are reconstructed with unit norm on average. This follows
        the public reference implementation of Gao et al., whose authors note that they found
        no clear evidence that the rescaling helps.
        """
        directions = F.normalize(torch.randn(self.num_latents, self.d_in), dim=-1)
        self.decoder.copy_(directions)
        self.encoder.weight.copy_(directions)
        self.encoder.bias.zero_()
        self.b_dec.zero_()

        probes = F.normalize(torch.randn(256, self.d_in), dim=-1).to(self.decoder)
        recon_norm = self(probes).norm(dim=-1).mean()
        self.encoder.weight.div_(recon_norm.clamp_min(1e-8))

    @torch.no_grad()
    def init_b_dec(self, sample: torch.Tensor, *, max_rows: int = 32_768) -> None:
        """Start ``b_dec`` at the geometric median of ``sample`` ``(n, d_in)``.

        The SAE then models deviations from a typical activation. At most ``max_rows`` rows
        are used.
        """
        median = geometric_median(sample[:max_rows])
        self.b_dec.copy_(median.to(self.b_dec))

    def _select(
        self,
        x: torch.Tensor,
        dead_index: torch.Tensor | None = None,
        aux_k: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Activations after ReLU and their latents, for the main and the auxiliary selection."""
        if dead_index is None:
            dead_index = torch.empty(0, dtype=torch.long, device=x.device)

        values, indices, aux_values, aux_indices = _SparseEncode.apply(
            x,
            self.encoder.weight,
            self.encoder.bias,
            self.b_dec,
            dead_index,
            self.k,
            min(aux_k, dead_index.numel()),
        )
        return F.relu(values), indices, F.relu(aux_values), aux_indices

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(values, indices)``, both ``(batch, k)``, for ``x`` of shape ``(batch, d_in)``.

        These are the ``k`` largest pre-activations per token after ReLU and the latents they
        belong to, in no particular order. A value of zero means the latent was selected but
        did not fire.
        """
        values, indices, _, _ = self._select(x)
        return values, indices

    def decode_sparse(
        self,
        values: torch.Tensor,
        indices: torch.Tensor,
        *,
        add_bias: bool = True,
    ) -> torch.Tensor:
        """Sum ``values[b, j] * decoder[indices[b, j]]`` over ``j``; both inputs are ``(batch, n)``."""
        recon = _SparseDecode.apply(values.to(self.decoder.dtype), indices, self.decoder)
        return recon + self.b_dec if add_bias else recon

    def _decode_auxiliary(
        self,
        aux_values: torch.Tensor,
        aux_indices: torch.Tensor,
        dead_index: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruction from the selected dead latents, without the decoder bias."""
        if not _use_dense_aux(dead_index.numel(), aux_indices.shape[1]):
            return self.decode_sparse(aux_values, aux_indices, add_bias=False)

        # Position of every dead latent inside ``dead_index``; other entries are never read.
        position = torch.zeros(self.num_latents, dtype=torch.long, device=dead_index.device)
        position[dead_index] = torch.arange(dead_index.numel(), device=dead_index.device)

        code = aux_values.new_zeros((aux_values.shape[0], dead_index.numel()), dtype=self.decoder.dtype)
        code = code.scatter(1, position[aux_indices], aux_values.to(self.decoder.dtype))
        return code @ self.decoder.index_select(0, dead_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``x`` ``(batch, d_in)``."""
        values, indices = self.encode(x)
        return self.decode_sparse(values, indices)

    def loss(
        self,
        x: torch.Tensor,
        *,
        dead_mask: torch.Tensor | None = None,
        aux_k: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Reconstruction loss plus the AuxK term that revives dead latents.

        AuxK: the ``aux_k`` strongest *dead* latents try to reconstruct what the live latents
        missed. The residual is detached, so this term trains only the dead latents and pulls
        them back towards directions that are still unexplained.

        Args:
            x: activations ``(batch, d_in)``.
            dead_mask: boolean ``(num_latents,)``, true for latents considered dead.
            aux_k: number of dead latents used by the auxiliary reconstruction; 0 disables it.
                At most the number of dead latents is used.

        Returns:
            ``fvu``: batch-normalised reconstruction error (scalar).
            ``aux``: auxiliary error, normalised by the variance of the residual (scalar, zero
            when there are no dead latents or the residual is degenerate).
            ``values`` and ``indices``: the selected activations, ``(batch, k)``.
        """
        dead_index = None
        if dead_mask is not None and aux_k > 0:
            dead_index = dead_mask.nonzero().squeeze(1)

        values, indices, aux_values, aux_indices = self._select(x, dead_index, aux_k)
        recon = self.decode_sparse(values, indices)
        fvu = normalized_mse(recon, x)

        aux = recon.new_zeros(())
        if dead_index is not None and aux_indices.shape[1] > 0:
            residual = (x - recon).detach()
            aux_recon = self._decode_auxiliary(aux_values, aux_indices, dead_index)
            aux = torch.nan_to_num(normalized_mse(aux_recon, residual), nan=0.0, posinf=0.0)

        return {"fvu": fvu, "aux": aux, "values": values, "indices": indices}

    @torch.no_grad()
    def remove_parallel_decoder_grad(self) -> None:
        """Project out the gradient component that would only change a decoder row's norm."""
        grad = self.decoder.grad
        if grad is not None:
            grad.sub_((grad * self.decoder).sum(dim=-1, keepdim=True) * self.decoder)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Rescale every decoder row to unit norm."""
        self.decoder.div_(self.decoder.norm(dim=-1, keepdim=True).clamp_min(1e-8))
