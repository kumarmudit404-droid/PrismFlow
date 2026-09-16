"""Deterministic seeding across python / numpy / torch (incl. CUDA)."""

from __future__ import annotations

import os
import random


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy, and torch (CPU + all CUDA devices).

    With deterministic=True, also configures torch/cuDNN for bit-reproducible
    results at the cost of some performance.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    except ImportError:
        pass
