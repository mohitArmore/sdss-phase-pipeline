"""Ignore regions: map ignore_boxes from original-figure coordinates into the
crop frame and rasterise them as a 255 mask."""
from __future__ import annotations

import numpy as np

from src.legacy.ang_to_mask import IGNORE_INDEX  # one definition of 255 across the repo


def to_crop_frame(boxes: list[list[int]] | None, eff_box: list[int]) -> list[list[int]]:
    """Translate boxes by the crop origin and clip to the crop; drop boxes fully outside."""
    cx0, cy0, cx1, cy1 = eff_box
    w, h = cx1 - cx0, cy1 - cy0
    out = []
    for b in boxes or []:
        x0, y0, x1, y1 = (int(b[0]) - cx0, int(b[1]) - cy0, int(b[2]) - cx0, int(b[3]) - cy0)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 > x0 and y1 > y0:
            out.append([x0, y0, x1, y1])
    return out


def ignore_mask(shape: tuple[int, ...], crop_boxes: list[list[int]]) -> np.ndarray:
    """uint8 (H, W): 255 inside any box, 0 elsewhere."""
    m = np.zeros(shape[:2], dtype=np.uint8)
    for x0, y0, x1, y1 in crop_boxes:
        m[y0:y1, x0:x1] = IGNORE_INDEX
    return m
