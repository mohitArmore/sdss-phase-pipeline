"""Etched optical / SEM micrograph -> class mask by classical thresholding.

Pipeline: best_channel (most bimodal of R,G,B,L*,a*,b*) -> flat-field
(subtract large Gaussian blur) -> NL-means denoise -> threshold (otsu |
multiotsu | local | kmeans) -> which side is ferrite from `dark_phase`
(decided on L*, so it also works when a*/b* is the best channel) ->
morphological open/close -> small-island removal -> human ignore boxes = 255.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2lab
from skimage.filters import threshold_local, threshold_multiotsu, threshold_otsu
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_holes, remove_small_objects

from src.data.annotations import IGNORE
from src.segment.ebsd_colour_to_mask import remove_islands

METHODS = ("otsu", "multiotsu", "local", "kmeans")


def channels(img: np.ndarray) -> dict[str, np.ndarray]:
    rgb = img[..., :3]
    lab = rgb2lab(rgb)
    return {"R": rgb[..., 0].astype(float), "G": rgb[..., 1].astype(float),
            "B": rgb[..., 2].astype(float), "L": lab[..., 0], "a": lab[..., 1], "b": lab[..., 2]}


def separability(x: np.ndarray) -> float:
    """Otsu's eta: between-class variance / total variance (0..1, 1 = perfectly bimodal)."""
    x = np.asarray(x, float).ravel()
    if x.size < 2 or x.var() == 0:
        return 0.0
    t = threshold_otsu(x)
    lo, hi = x[x <= t], x[x > t]
    if lo.size == 0 or hi.size == 0:
        return 0.0
    w0, w1 = lo.size / x.size, hi.size / x.size
    return float(w0 * w1 * (lo.mean() - hi.mean()) ** 2 / x.var())


def best_channel(img: np.ndarray, valid: np.ndarray | None = None) -> tuple[str, np.ndarray, dict[str, float]]:
    chs = channels(img)
    v = np.ones(img.shape[:2], bool) if valid is None else valid
    scores = {k: separability(c[v]) for k, c in chs.items()}
    name = max(scores, key=scores.get)
    return name, chs[name], scores


def to_uint8(ch: np.ndarray, valid: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(ch[valid], [0.5, 99.5]) if valid.any() else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def preprocess(ch: np.ndarray, valid: np.ndarray, flatfield_sigma_px: float = 50,
               denoise_h: float = 7) -> np.ndarray:
    f = ch.astype(float).copy()
    if valid.any():
        f[~valid] = np.median(f[valid])   # legend/label boxes must not bias the background
    if flatfield_sigma_px and flatfield_sigma_px > 0:
        bg = gaussian_filter(f, flatfield_sigma_px)
        f = f - bg + bg[valid].mean() if valid.any() else f - bg
    u8 = to_uint8(f, valid)
    if denoise_h and denoise_h > 0:
        u8 = cv2.fastNlMeansDenoising(u8, None, h=float(denoise_h), templateWindowSize=7,
                                      searchWindowSize=21)
    return u8


def _kmeans_threshold(x: np.ndarray, attempts: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    if x.size > 200_000:                       # ponytail: subsample; 200k px is plenty for 2 centres
        x = rng.choice(x, 200_000, replace=False)
    cv2.setRNGSeed(int(seed))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    _, _, centres = cv2.kmeans(x.reshape(-1, 1).astype(np.float32), 2, None, crit,
                               int(attempts), cv2.KMEANS_PP_CENTERS)
    return float(centres.mean())


def segment_micrograph(img: np.ndarray, dark_phase: str, method: str = "otsu",
                       ignore: np.ndarray | None = None, flatfield_sigma_px: float = 50,
                       denoise_h: float = 7, morph_radius_px: int = 1, min_island_px: int = 256,
                       local_block_px: int = 51, local_offset: float = 0.0,
                       threshold_shift: float = 0.0, sigma_tone: str | None = None,
                       kmeans_attempts: int = 5, seed: int = 42) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (mask, info). threshold_shift scales the threshold by (1 + shift),
    used for the +-10 % sensitivity runs."""
    if dark_phase not in ("ferrite", "austenite"):
        raise ValueError("dark_phase must be 'ferrite' or 'austenite' (from meta.yaml)")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    valid = np.ones(img.shape[:2], bool) if ignore is None else ignore == 0
    name, ch, scores = best_channel(img, valid)
    x = preprocess(ch, valid, flatfield_sigma_px, denoise_h).astype(float)
    L = channels(img)["L"]
    k = 1.0 + float(threshold_shift)
    info: dict[str, Any] = {"channel": name, "channel_scores": scores, "method": method}

    if method == "multiotsu" and sigma_tone in ("dark", "bright"):
        t = threshold_multiotsu(x[valid], classes=3) * k
        cls = np.digitize(x, t)                                     # 0,1,2 by channel value
        order = np.argsort([L[valid & (cls == c)].mean() if (valid & (cls == c)).any() else np.inf
                            for c in range(3)])                     # darkest -> brightest on L*
        darkest, middle, brightest = order
        sigma_c = darkest if sigma_tone == "dark" else brightest
        rest = [c for c in (darkest, middle, brightest) if c != sigma_c]   # dark first
        lut = np.zeros(3, np.uint8)
        lut[sigma_c] = 2
        lut[rest[0]] = 1 if dark_phase == "ferrite" else 0
        lut[rest[1]] = 0 if dark_phase == "ferrite" else 1
        mask = lut[cls]
        mask[~valid] = IGNORE
        mask = remove_islands(mask, min_island_px)
        info["thresholds"] = [float(v) for v in t]
        return mask, info

    if method in ("otsu", "multiotsu"):            # multiotsu without sigma_tone = 2-class Otsu
        t = threshold_otsu(x[valid]) * k
        high = x > t
    elif method == "local":
        block = int(local_block_px) | 1
        t = threshold_local(x, block_size=block, offset=float(local_offset)) * k
        high = x > t
    else:
        t = _kmeans_threshold(x[valid], kmeans_attempts, seed) * k
        high = x > t
    info["threshold"] = float(np.mean(t))

    hv, lv = valid & high, valid & ~high
    high_is_dark = hv.any() and lv.any() and L[hv].mean() < L[lv].mean()
    dark = high if high_is_dark else ~high
    fer = dark if dark_phase == "ferrite" else ~dark

    if morph_radius_px and morph_radius_px > 0:
        se = disk(int(morph_radius_px))
        fer = binary_closing(binary_opening(fer, se), se)
    if min_island_px and min_island_px > 1:
        fer = remove_small_objects(fer, min_size=int(min_island_px))
        fer = remove_small_holes(fer, area_threshold=int(min_island_px))
    mask = fer.astype(np.uint8)
    mask[~valid] = IGNORE
    return mask, info


def load_ilastik_probabilities(h5_path: str | Path, class_order: tuple[str, ...] = ("austenite", "ferrite"),
                               dataset: str = "exported_data") -> np.ndarray | None:
    """Argmax mask from an ilastik probability export (H, W, C). Returns None
    (with no error) if h5py is not installed — it is an optional dependency."""
    try:
        import h5py
    except ImportError:
        return None
    ids = {"austenite": 0, "ferrite": 1, "sigma": 2}
    with h5py.File(h5_path, "r") as f:
        p = np.asarray(f[dataset])
    p = p.reshape(p.shape[-3], p.shape[-2], p.shape[-1])
    lut = np.array([ids[c] for c in class_order], np.uint8)
    return lut[np.argmax(p, axis=-1)]
