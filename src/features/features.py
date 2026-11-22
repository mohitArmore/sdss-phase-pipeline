"""Microstructure features from class masks (0 austenite, 1 ferrite, 2 sigma, 255 ignore)."""
from __future__ import annotations

import numpy as np

from src.data.annotations import IGNORE

PHASES = {"austenite": 0, "ferrite": 1, "sigma": 2}


def phase_fractions(mask: np.ndarray) -> dict[str, float]:
    """Area fractions over pixels != 255, plus ignored_fraction over all pixels."""
    valid = mask != IGNORE
    n = int(valid.sum())
    out = {name: (float((mask == c).sum()) / n if n else float("nan")) for name, c in PHASES.items()}
    out["ignored_fraction"] = 1.0 - n / mask.size if mask.size else float("nan")
    return out
