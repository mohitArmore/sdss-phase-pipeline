"""EBSD phase-map figure (RGB) -> class mask, by nearest palette colour in CIELAB.

Steps:
  1. RGB -> CIELAB. For `ebsd_phase_iq` (phase colours modulated by an image-
     quality grey overlay) classify on a*/b* only, since IQ changes L*.
  2. For every pixel, delta-E to each palette colour (phase colours + 'ignore'
     colours such as black boundaries / white non-indexed). Nearest class wins.
  3. 255 if min delta-E > max_delta_e, or nearest/second-nearest (different
     class) > ambiguity_ratio (anti-aliased pixels between two phases).
  4. Mode filter over phase pixels, remove islands < min_island_px (-> 255).
  5. boundary_policy: 'ignore' keeps unclassified pixels at 255; 'reassign'
     grows neighbouring phases into them (skimage expand_labels), except inside
     the human ignore boxes, which always stay 255.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.color import rgb2lab
from skimage.morphology import remove_small_objects
from skimage.segmentation import expand_labels

from src.data.annotations import IGNORE

CLASS_IDS = {"austenite": 0, "ferrite": 1, "sigma": 2}


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = str(h).lstrip("#")
    if len(h) != 6:
        raise ValueError(f"palette colour must be #rrggbb, got {h!r}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def palette_lab(palette: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """-> (lab colours (K,3), class id per colour (K,), 255 for ignore colours)."""
    cols, ids = [], []
    for name, value in palette.items():
        cid = IGNORE if name == "ignore" else CLASS_IDS.get(name)
        if cid is None:
            raise ValueError(f"unknown palette key {name!r} (use ferrite/austenite/sigma/ignore)")
        for h in ([value] if isinstance(value, str) else list(value or [])):
            cols.append(_hex_to_rgb(h))
            ids.append(cid)
    rgb = np.array(cols, dtype=np.uint8).reshape(1, -1, 3)
    return rgb2lab(rgb)[0].astype(np.float32), np.array(ids, dtype=np.int32)


def classify_colours(img: np.ndarray, palette: dict[str, Any], iq_mode: bool = False,
                     max_delta_e: float = 20.0, ambiguity_ratio: float = 0.8) -> np.ndarray:
    lab = rgb2lab(img[..., :3]).astype(np.float32)
    pal, ids = palette_lab(palette)
    sl = slice(1, 3) if iq_mode else slice(0, 3)
    # ponytail: in iq_mode grey 'ignore' colours (black/white) all sit at a*=b*=0, so dark
    # low-chroma pixels become ambiguous/ignored rather than misclassified — conservative.
    classes = np.unique(ids)
    # per-class min distance (H, W, C); memory ~ H*W*C*4 bytes
    dist = np.stack([
        np.min(np.stack([np.linalg.norm(lab[..., sl] - pal[k, sl], axis=-1)
                         for k in np.flatnonzero(ids == c)]), axis=0)
        for c in classes], axis=-1)
    order = np.argsort(dist, axis=-1)
    d1 = np.take_along_axis(dist, order[..., :1], -1)[..., 0]
    label = classes[order[..., 0]].astype(np.uint8)
    if len(classes) > 1:
        d2 = np.take_along_axis(dist, order[..., 1:2], -1)[..., 0]
        ambiguous = d1 > ambiguity_ratio * np.maximum(d2, 1e-6)
    else:
        ambiguous = np.zeros(label.shape, bool)
    label[(d1 > max_delta_e) | ambiguous] = IGNORE
    return label


def mode_filter(mask: np.ndarray, size: int) -> np.ndarray:
    """Majority vote among phase classes in a size x size window; 255 stays 255."""
    if size is None or size < 2:
        return mask
    phases = [c for c in np.unique(mask) if c != IGNORE]
    if len(phases) < 2:
        return mask
    counts = np.stack([uniform_filter((mask == c).astype(np.float32), size) for c in phases], -1)
    out = np.array(phases, dtype=np.uint8)[np.argmax(counts, axis=-1)]
    return np.where(mask == IGNORE, IGNORE, out).astype(np.uint8)


def remove_islands(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Connected phase regions smaller than min_px -> 255."""
    if not min_px or min_px < 2:
        return mask
    out = mask.copy()
    for c in np.unique(mask):
        if c == IGNORE:
            continue
        sel = mask == c
        out[sel & ~remove_small_objects(sel, min_size=int(min_px))] = IGNORE
    return out


def apply_boundary_policy(mask: np.ndarray, policy: str, hard_ignore: np.ndarray | None = None,
                          max_px: int = 3) -> np.ndarray:
    if policy == "ignore":
        out = mask.copy()
    elif policy == "reassign":
        lab = np.where(mask == IGNORE, 0, mask.astype(np.int32) + 1)
        grown = expand_labels(lab, distance=int(max_px))
        out = np.where(grown == 0, IGNORE, grown - 1).astype(np.uint8)
    else:
        raise ValueError(f"boundary_policy must be 'ignore' or 'reassign', got {policy!r}")
    if hard_ignore is not None:
        out[hard_ignore > 0] = IGNORE
    return out


def ebsd_to_mask(img: np.ndarray, palette: dict[str, Any], image_type: str = "ebsd_phase",
                 ignore: np.ndarray | None = None, max_delta_e: float = 20.0,
                 ambiguity_ratio: float = 0.8, mode_filter_size: int = 3, min_island_px: int = 4,
                 boundary_policy: str = "ignore", reassign_max_px: int = 3) -> np.ndarray:
    mask = classify_colours(img, palette, image_type == "ebsd_phase_iq", max_delta_e, ambiguity_ratio)
    if ignore is not None:
        mask[ignore > 0] = IGNORE
    mask = mode_filter(mask, mode_filter_size)
    mask = remove_islands(mask, min_island_px)
    return apply_boundary_policy(mask, boundary_policy, ignore, reassign_max_px)
