"""Scale-bar calibration, measured on ORIGINAL figure pixels.

measure_bar_px: inside a tight bbox around the bar, binarise (Otsu), take the
bar polarity, and return the longest horizontal run of bar pixels. Text
("200 µm") inside the bbox is fine as long as the bar is the longest run.
Uncertainty: +-1 px at each end -> rel ~ 2 / bar_px.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from skimage.filters import threshold_otsu


def _longest_run(row: np.ndarray) -> int:
    if not row.any():
        return 0
    padded = np.concatenate([[0], row.astype(np.int8), [0]])
    d = np.diff(padded)
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


def measure_bar_px(img: np.ndarray, bbox: list[int], polarity: str = "auto") -> int:
    """Length in px of the scale bar inside bbox [x0,y0,x1,y1] (original coords).
    polarity: 'dark' | 'light' | 'auto' (auto = minority class, i.e. bar+text
    on background)."""
    x0, y0, x1, y1 = (int(v) for v in bbox)
    region = img[y0:y1, x0:x1]
    if region.ndim == 3:
        region = region[..., :3].mean(axis=2)
    if region.size == 0 or region.max() == region.min():
        raise ValueError(f"scale-bar bbox {bbox} is empty or uniform")
    light = region > threshold_otsu(region)
    if polarity == "auto":
        polarity = "light" if light.mean() < 0.5 else "dark"
    bar = light if polarity == "light" else ~light
    return max(_longest_run(r) for r in bar)


def calibrate(scale_bar: dict[str, Any] | None, img: np.ndarray | None,
              polarity: str = "auto") -> dict[str, Any] | None:
    """scale_bar from meta.yaml: {um, px} (manual) or {um, bbox} (measured).
    Returns {um, bar_px, um_per_px, rel_uncertainty, method} or None if incomplete."""
    sb = scale_bar or {}
    um = sb.get("um")
    if not um:
        return None
    if sb.get("px"):
        bar_px, method = float(sb["px"]), "manual"
    elif sb.get("bbox") and img is not None:
        bar_px, method = float(measure_bar_px(img, sb["bbox"], polarity)), "measured"
    else:
        return None
    if bar_px <= 0:
        return None
    return {"um": float(um), "bar_px": bar_px, "um_per_px": float(um) / bar_px,
            "rel_uncertainty": 2.0 / bar_px, "method": method}
