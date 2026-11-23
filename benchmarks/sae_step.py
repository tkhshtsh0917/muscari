#!/usr/bin/env python3
"""Time one optimisation step of the Top-K SAE: sparse backward versus dense autograd.

Both modes compute the same loss on the same random activations and take the same Adam step.
``sparse`` uses ``muscari.TopKSAE``; ``dense`` builds the ``(batch, num_latents)`` code and
lets autograd differentiate through it. A fixed share of the latents is flagged dead so that
the auxiliary loss is exercised from the first step.

    python benchmarks/sae_step.py --mode sparse
    python benchmarks/sae_step.py --mode dense --batch-tokens 2048

Random activations say nothing about SAE quality; only time and memory are meaningful here.
Report the device, thread count and library versions next to any number you quote.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time

import torch

from muscari import TopKSAE, normalized_mse, resolve_device


def dense_loss(sae: TopKSAE, x: torch.Tensor, dead_mask: torch.Tensor, aux_k: int) -> torch.Tensor:
    """The loss of ``TopKSAE.loss`` written with dense codes and plain autograd."""
    pre = (x - sae.b_dec) @ sae.encoder.weight.T + sae.encoder.bias
    values, indices = pre.topk(sae.k, dim=-1)
    code = torch.zeros_like(pre).scatter(-1, indices, values.relu())
    recon = code @ sae.decoder + sae.b_dec
    loss = normalized_mse(recon, x)

    n_aux = min(aux_k, int(dead_mask.sum()))
    if n_aux > 0:
        aux_values, aux_indices = pre.masked_fill(~dead_mask, float("-inf")).topk(n_aux, dim=-1)
        aux_code = torch.zeros_like(pre).scatter(-1, aux_indices, aux_values.relu())
        loss = loss + normalized_mse(aux_code @ sae.decoder, (x - recon).detach())
    return loss


def sparse_loss(sae: TopKSAE, x: torch.Tensor, dead_mask: torch.Tensor, aux_k: int) -> torch.Tensor:
    out = sae.loss(x, dead_mask=dead_mask, aux_k=aux_k)
    return out["fvu"] + out["aux"]


def peak_memory_gb(device: torch.device) -> float:
    """Peak memory in GB: allocator peak on CUDA, driver allocation on MPS, process RSS on CPU."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e9
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


def run(args: argparse.Namespace) -> dict[str, object]:
    device = resolve_device(args.device)
    torch.manual_seed(0)

    sae = TopKSAE(args.d_in, args.num_latents, args.k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-4)
    x = torch.randn(args.batch_tokens, args.d_in, device=device)
    dead_mask = torch.zeros(args.num_latents, dtype=torch.bool, device=device)
    dead_mask[: int(args.dead_frac * args.num_latents)] = True
    loss_fn = sparse_loss if args.mode == "sparse" else dense_loss

    seconds = []
    losses = []
    for _ in range(args.steps):
        started = time.perf_counter()
        loss = loss_fn(sae, x, dead_mask, args.aux_k)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())  # also waits for the device to finish
        seconds.append(time.perf_counter() - started)

    return {
        "mode": args.mode,
        "device": str(device),
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "d_in": args.d_in,
        "num_latents": args.num_latents,
        "k": args.k,
        "aux_k": args.aux_k,
        "dead_frac": args.dead_frac,
        "batch_tokens": args.batch_tokens,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "median_step_seconds": statistics.median(seconds[1:] or seconds),
        "peak_memory_gb": peak_memory_gb(device),
    }


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["sparse", "dense"], default="sparse")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--d-in", type=int, default=768)
    parser.add_argument("--num-latents", type=int, default=65_536)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--aux-k", type=int, default=256, help="0 disables the auxiliary loss")
    parser.add_argument("--dead-frac", type=float, default=0.1, help="share of latents flagged dead")
    parser.add_argument("--batch-tokens", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=5, help="the first step is excluded from the median")
    result = run(parser.parse_args(argv))

    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
