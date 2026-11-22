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


def grain_stats(mask: np.ndarray, cls: int, um_per_px: float | None = None,
                min_region_px: int = 20, exclude_border: bool = True) -> "pd.DataFrame":
    """Per-region stats for phase `cls` (4-connected regions).

    ponytail: regions are connected phase areas, not crystallographic grains —
    a colour/etch map can't separate touching same-phase grains. Use EBSD grain
    maps if true grain size is needed.

    Excludes regions touching the image border or an ignore (255) area (truncated)
    and regions < min_region_px. Lengths in um when um_per_px is given, else NaN.
    """
    import pandas as pd
    from scipy.ndimage import binary_dilation
    from skimage.measure import label, regionprops_table
    from skimage.segmentation import clear_border

    lab = label(mask == cls, connectivity=1)
    if exclude_border:
        lab = clear_border(lab)
        touching = np.unique(lab[binary_dilation(mask == IGNORE)])
        lab[np.isin(lab, touching[touching > 0])] = 0
    cols = ["label", "area", "equivalent_diameter_area", "axis_major_length",
            "axis_minor_length", "orientation"]
    t = pd.DataFrame(regionprops_table(lab, properties=cols))
    t = t[t["area"] >= min_region_px].reset_index(drop=True)
    t["aspect_ratio"] = t["axis_major_length"] / t["axis_minor_length"].replace(0, np.nan)
    s = float(um_per_px) if um_per_px else np.nan
    t["ecd_um"] = t["equivalent_diameter_area"] * s
    t["major_um"] = t["axis_major_length"] * s
    t["minor_um"] = t["axis_minor_length"] * s
    t["area_um2"] = t["area"] * s * s
    t["orientation_deg"] = np.degrees(t["orientation"])
    return t


def summarize_grains(t: "pd.DataFrame", prefix: str) -> dict[str, float]:
    def stat(col, fn):
        return float(fn(t[col])) if len(t) and t[col].notna().any() else float("nan")
    return {f"{prefix}_n_regions": int(len(t)),
            f"{prefix}_ecd_um_mean": stat("ecd_um", np.nanmean),
            f"{prefix}_ecd_um_median": stat("ecd_um", np.nanmedian),
            f"{prefix}_ecd_px_mean": stat("equivalent_diameter_area", np.nanmean),
            f"{prefix}_aspect_mean": stat("aspect_ratio", np.nanmean),
            f"{prefix}_orientation_deg_median": stat("orientation_deg", np.nanmedian)}


def boundary_density(mask: np.ndarray, um_per_px: float | None = None) -> dict[str, float]:
    """Phase-boundary length per unit valid area (Sv-like, 2D L_A).

    Counts 4-neighbour pixel pairs with different phase (both valid), times pi/4
    to correct pixel-edge counting for isotropic boundaries. Returns per px and
    per um (if scale known)."""
    v = mask != IGNORE
    h = (mask[:, 1:] != mask[:, :-1]) & v[:, 1:] & v[:, :-1]
    w = (mask[1:, :] != mask[:-1, :]) & v[1:, :] & v[:-1, :]
    n = int(v.sum())
    la_px = (np.pi / 4) * (int(h.sum()) + int(w.sum())) / n if n else float("nan")
    return {"boundary_density_per_px": la_px,
            "boundary_density_per_um": la_px / um_per_px if um_per_px else float("nan")}


def mask_features(mask: np.ndarray, um_per_px: float | None = None, min_region_px: int = 20,
                  exclude_border: bool = True) -> dict[str, float]:
    """One flat row: fractions, boundary density, grain stats per present phase."""
    row: dict[str, float] = dict(phase_fractions(mask))
    row.update(boundary_density(mask, um_per_px))
    for name, c in PHASES.items():
        if name == "sigma" and not (mask == c).any():
            continue
        row.update(summarize_grains(grain_stats(mask, c, um_per_px, min_region_px, exclude_border), name))
    return row
