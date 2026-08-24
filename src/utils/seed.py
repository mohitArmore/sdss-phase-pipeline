"""Global seed control.

Reproducibility contract: every experiment sets the seed once at entry, and
every stochastic component (numpy, torch, python random, dataloader workers)
reads from this single source of truth.
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Set seeds for python, numpy, and (if available) torch.

    Parameters
    ----------
    seed : int
        The seed value. 42 by default, override per-experiment via config.
    deterministic_torch : bool
        If True, sets torch to fully deterministic mode. This is slower but
        essential for the ablation harness in Phase 3. Turn OFF only if you
        need max training throughput and don't care about bit-exact repro.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Newer PyTorch prefers this:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except AttributeError:
                pass
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:
    """Passed to torch DataLoader(worker_init_fn=seed_worker) for deterministic loading."""
    import numpy as _np
    import random as _random
    worker_seed = (int.from_bytes(os.urandom(4), "little")) % (2**32)
    _np.random.seed(worker_seed)
    _random.seed(worker_seed)
