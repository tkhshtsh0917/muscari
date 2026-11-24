"""Pick the device that training and activation extraction run on."""

from __future__ import annotations

import torch


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Turn ``"auto"`` into the best available device; pass anything else through.

    ``"auto"`` prefers CUDA, then Apple MPS, then the CPU. An explicit request is checked so
    that a missing accelerator fails here with a clear message instead of deep inside PyTorch.

    Raises:
        ValueError: if CUDA or MPS is requested but PyTorch cannot use it on this machine.
    """
    if isinstance(device, torch.device):
        return device

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is False.")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but torch.backends.mps.is_available() is False.")
    return resolved


def synchronize(device: torch.device) -> None:
    """Wait for queued work on ``device`` to finish, so that wall-clock timings mean something."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
