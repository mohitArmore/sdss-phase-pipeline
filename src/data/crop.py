"""Crop a panel out of an original figure.

Boxes are [x0, y0, x1, y1] in original-figure pixels, x1/y1 exclusive.
The crop is shrunk by an inner margin (config crop.margin_px) to drop panel
frame lines / anti-aliased edges. Crops are never resampled.
"""
from __future__ import annotations

import numpy as np


def effective_box(crop_box: list[int], margin_px: int, shape: tuple[int, ...]) -> list[int]:
    h, w = shape[:2]
    x0, y0, x1, y1 = (int(v) for v in crop_box)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(f"crop_box {crop_box} outside image of size {w}x{h}")
    m = int(margin_px)
    box = [x0 + m, y0 + m, x1 - m, y1 - m]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"crop_box {crop_box} too small for margin {m}px")
    return box


def crop_panel(img: np.ndarray, crop_box: list[int], margin_px: int = 0,
               min_side_px: int = 0) -> tuple[np.ndarray, list[int], list[str]]:
    """Return (crop, effective_box, qc_flags). Flags 'small_crop' if either side < min_side_px."""
    box = effective_box(crop_box, margin_px, img.shape)
    x0, y0, x1, y1 = box
    crop = np.ascontiguousarray(img[y0:y1, x0:x1])
    flags = ["small_crop"] if min(crop.shape[:2]) < int(min_side_px) else []
    return crop, box, flags
