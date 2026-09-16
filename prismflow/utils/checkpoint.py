"""Checkpoint save/load helpers.

No paths are hard-coded: callers pass in a directory (typically sourced from
a Config object) rather than this module assuming a location.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_checkpoint(state: dict[str, Any], checkpoint_dir: str | Path, name: str) -> Path:
    """Save a checkpoint dict to <checkpoint_dir>/<name>.pt and return the path."""
    import torch

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{name}.pt"
    torch.save(state, path)
    return path


def load_checkpoint(checkpoint_dir: str | Path, name: str, map_location: str | None = None) -> dict[str, Any]:
    """Load a checkpoint dict from <checkpoint_dir>/<name>.pt."""
    import torch

    path = Path(checkpoint_dir) / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}")
    return torch.load(path, map_location=map_location)
