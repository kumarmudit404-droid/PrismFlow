"""CPU/GPU auto-detection."""

from __future__ import annotations


def get_device(prefer: str = "auto"):
    """Return a torch.device, auto-detecting CUDA availability.

    prefer: "auto" picks cuda if available else cpu; "cpu" or "cuda" force
    that device (raises if "cuda" is requested but unavailable).
    """
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if prefer != "auto":
        raise ValueError(f"Unknown device preference: {prefer!r}")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
