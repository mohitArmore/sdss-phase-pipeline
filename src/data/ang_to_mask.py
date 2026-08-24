"""Convert a raw .ang EBSD file to a binary phase mask.

Design notes (small-data-first):
  - We use `orix` for parsing (pure Python, no MATLAB in the critical path).
  - We threshold on Confidence Index (CI) to drop unreliable pixels — this is
    standard EBSD hygiene (Wright et al., Microscopy & Microanalysis 2011).
  - We do LIGHT morphological cleanup only: dilation of "unindexed" so bad
    pixels don't survive as fake tiny grains, and median filtering on the
    phase label. Heavy morphology is *rejected* because it destroys the
    minority-phase (austenite) islands we most care about.
  - We emit a sidecar JSON with EVERY parameter used, so viva/paper claims
    about phase fractions are auditable.

What we DO NOT do here:
  - Grain reconstruction (that's a Phase 4 feature-extraction concern; the
    segmentation model doesn't need grain IDs, just per-pixel phase).
  - Orientation analysis (Phase 4).
  - Registration to SEM (that's the next module: sem_ebsd_registration.py).

Phase mapping convention:
  Class 0 = background / austenite (γ, FCC, m-3m)
  Class 1 = ferrite (δ or α, BCC, m-3m)
  Class 255 = unindexed / ignore (this is the standard PyTorch "ignore_index"
              value; the loss will skip these pixels)

We fix ferrite=1 because in a duplex steel, austenite is typically the
minority phase in as-built LPBF, and it's convention in the metallurgy DL
literature to put the minority/foreground class at label 1. But the class
IDs are configurable — see configs/data/default.yaml.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, median_filter

# orix is a hard dependency for this module; import lazily so unit tests
# that don't need it can still import the module structure.
try:
    from orix.io import load as orix_load
    _ORIX_AVAILABLE = True
except ImportError:
    _ORIX_AVAILABLE = False


# Standard EBSD "ignore" value used by PyTorch loss functions.
IGNORE_INDEX = 255


@dataclass
class MaskingConfig:
    """All knobs for .ang → mask conversion.

    Every value here is loaded from configs/data/*.yaml — nothing hardcoded.
    """
    ci_threshold: float = 0.1
    """Minimum Confidence Index. Pixels below this are marked unindexed.
    0.1 is the community default from Wright et al. (2011); tune per-dataset."""

    iq_threshold: float | None = None
    """Optional Image Quality threshold. None = don't filter on IQ."""

    apply_median_filter: bool = True
    """Whether to run a small median filter to clean single-pixel noise."""

    median_filter_size: int = 3
    """Median filter kernel size (odd). 3 = light cleanup, doesn't dissolve grains."""

    dilate_unindexed: bool = True
    """Whether to slightly grow the 'unindexed' region so we don't train on
    the ambiguous borders of low-confidence areas."""

    unindexed_dilation_pixels: int = 1
    """How much to grow. 1 is conservative and generally safe."""

    # ---- Phase ID mapping ----
    # Different .ang files number phases differently depending on the OIM
    # software version and the acquisition template. We resolve names, not IDs.
    ferrite_names: tuple[str, ...] = (
        "Ferrite", "Iron bcc", "Iron (Alpha)", "Iron-BCC (Old)",
        "Iron (BCC)", "Alpha-Iron", "δ-Ferrite", "delta-Ferrite",
    )
    austenite_names: tuple[str, ...] = (
        "Austenite", "Iron fcc", "Iron (Gamma)", "Iron-FCC",
        "Iron (FCC)", "Gamma-Iron", "γ-Austenite",
    )
    sigma_names: tuple[str, ...] = ("Sigma", "σ", "Sigma phase")

    ferrite_class_id: int = 1
    austenite_class_id: int = 0
    sigma_class_id: int = 2   # only used if 3-class mode is on

    three_class_mode: bool = False
    """If True, produce a 3-class mask (ferrite/austenite/sigma). Default is
    binary ferrite-vs-austenite as per Phase 0 recommendation."""


@dataclass
class MaskResult:
    """Output of .ang → mask, ready to be saved and manifest-recorded."""
    mask: np.ndarray                     # uint8, shape (H, W), values in {0,1,[2,]255}
    phase_fractions: dict[str, float]    # {"ferrite": 0.94, "austenite": 0.05, "unindexed": 0.01, ...}
    ci_stats: dict[str, float]           # {"mean": .., "median": .., "pct_above_thresh": ..}
    step_size_um: float                  # from the .ang header
    shape: tuple[int, int]               # (H, W)
    phases_found: list[str]              # names from the .ang, for logging
    config: MaskingConfig                # exact settings used


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match_phase_name(candidate: str, aliases: tuple[str, ...]) -> bool:
    """Case- and whitespace-insensitive match, tolerant of unicode oddities."""
    c = candidate.strip().lower().replace(" ", "")
    for a in aliases:
        if a.strip().lower().replace(" ", "") == c:
            return True
    return False


def _resolve_phase_map(phases_from_ang, cfg: MaskingConfig) -> dict[int, str]:
    """Return {phase_id_in_ang: canonical_name} where canonical_name is one of
    'ferrite', 'austenite', 'sigma', or 'other'."""
    resolved: dict[int, str] = {}
    for phase_id, phase in phases_from_ang:
        name = str(phase.name)
        if _match_phase_name(name, cfg.ferrite_names):
            resolved[phase_id] = "ferrite"
        elif _match_phase_name(name, cfg.austenite_names):
            resolved[phase_id] = "austenite"
        elif _match_phase_name(name, cfg.sigma_names):
            resolved[phase_id] = "sigma"
        else:
            resolved[phase_id] = "other"
    return resolved


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def ang_to_mask(ang_path: str | Path, cfg: MaskingConfig | None = None) -> MaskResult:
    """Convert one .ang file to a phase mask.

    Parameters
    ----------
    ang_path : str | Path
        Path to the .ang file.
    cfg : MaskingConfig
        Preprocessing knobs. If None, uses defaults.

    Returns
    -------
    MaskResult
    """
    if not _ORIX_AVAILABLE:
        raise ImportError(
            "orix is required for .ang parsing. `pip install orix==0.13.3`. "
            "If you're on Colab and orix install failed, see docs/troubleshooting.md."
        )

    cfg = cfg or MaskingConfig()
    ang_path = Path(ang_path)
    if not ang_path.exists():
        raise FileNotFoundError(ang_path)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    # orix returns a CrystalMap with .phase_id, .prop['ci'], .prop['iq'], etc.
    # NOTE: field names in prop dict can vary — TSL .ang typically uses 'ci' and 'iq'.
    xmap = orix_load(str(ang_path))

    # The step size lives in xmap.dx (x-step) and xmap.dy (y-step) in orix.
    # Convention: units are microns for .ang. If they aren't, the .ang header is nonstandard.
    try:
        step_um = float(xmap.dx)
    except Exception:
        step_um = float("nan")

    # Shape of the map on disk (rows, cols).
    shape = xmap.shape  # tuple

    # ------------------------------------------------------------------
    # 2. Confidence-index filtering
    # ------------------------------------------------------------------
    # Grab CI. orix stores per-pixel properties under xmap.prop.
    prop = getattr(xmap, "prop", {}) or {}
    ci = None
    for key in ("ci", "CI", "confidence_index", "Confidence Index"):
        if key in prop:
            ci = np.asarray(prop[key]).reshape(shape)
            break
    if ci is None:
        # Some .ang variants don't include CI. Treat everything as passing.
        ci = np.ones(shape, dtype=np.float32)

    ci_stats = {
        "mean": float(np.nanmean(ci)),
        "median": float(np.nanmedian(ci)),
        "min": float(np.nanmin(ci)),
        "max": float(np.nanmax(ci)),
        "pct_above_thresh": float(100.0 * np.mean(ci >= cfg.ci_threshold)),
    }

    low_ci = ci < cfg.ci_threshold

    # Optional IQ filter (fires only if configured)
    if cfg.iq_threshold is not None:
        iq = None
        for key in ("iq", "IQ", "image_quality"):
            if key in prop:
                iq = np.asarray(prop[key]).reshape(shape)
                break
        if iq is not None:
            low_iq = iq < cfg.iq_threshold
            low_ci = low_ci | low_iq

    # ------------------------------------------------------------------
    # 3. Phase ID → semantic class
    # ------------------------------------------------------------------
    phases_iter = list(xmap.phases)  # list of (id, Phase)
    phase_map = _resolve_phase_map(phases_iter, cfg)

    phase_id_grid = np.asarray(xmap.phase_id).reshape(shape)

    # Start with everything as "unindexed". Fill in the recognized phases.
    mask = np.full(shape, IGNORE_INDEX, dtype=np.uint8)
    for pid, canonical in phase_map.items():
        sel = phase_id_grid == pid
        if canonical == "ferrite":
            mask[sel] = cfg.ferrite_class_id
        elif canonical == "austenite":
            mask[sel] = cfg.austenite_class_id
        elif canonical == "sigma" and cfg.three_class_mode:
            mask[sel] = cfg.sigma_class_id
        elif canonical == "sigma" and not cfg.three_class_mode:
            # In binary mode we choose to lump sigma with ferrite by default,
            # because sigma nucleates FROM ferrite in duplex steels. This is a
            # documented modeling choice — override in config if you want it
            # marked as ignore instead.
            mask[sel] = cfg.ferrite_class_id
        else:
            # Unknown phase → treat as unindexed (safer than guessing).
            mask[sel] = IGNORE_INDEX

    # Apply CI/IQ mask AFTER phase assignment so low-confidence pixels win.
    mask[low_ci] = IGNORE_INDEX

    # ------------------------------------------------------------------
    # 4. Light morphological cleanup
    # ------------------------------------------------------------------
    if cfg.dilate_unindexed and cfg.unindexed_dilation_pixels > 0:
        unindexed = mask == IGNORE_INDEX
        unindexed = binary_dilation(unindexed, iterations=cfg.unindexed_dilation_pixels)
        mask[unindexed] = IGNORE_INDEX

    if cfg.apply_median_filter:
        # Median filter only over VALID (non-ignore) pixels. We do this by
        # separating: filter the phase labels, then restore ignore mask on top.
        valid = mask != IGNORE_INDEX
        # Temporarily set ignore pixels to the mode of their neighborhood so
        # median doesn't drag them in. Easiest: filter mask directly, then
        # re-apply ignore mask.
        filtered = median_filter(mask, size=cfg.median_filter_size, mode="nearest")
        mask = np.where(valid, filtered, IGNORE_INDEX).astype(np.uint8)

    # ------------------------------------------------------------------
    # 5. Compute phase fractions
    # ------------------------------------------------------------------
    total = mask.size
    fractions: dict[str, float] = {}
    counts = {
        "ferrite":   int(np.sum(mask == cfg.ferrite_class_id)),
        "austenite": int(np.sum(mask == cfg.austenite_class_id)),
        "unindexed": int(np.sum(mask == IGNORE_INDEX)),
    }
    if cfg.three_class_mode:
        counts["sigma"] = int(np.sum(mask == cfg.sigma_class_id))
    for k, v in counts.items():
        fractions[k] = round(v / total, 6)

    phases_found = [str(p.name) for _, p in phases_iter]

    return MaskResult(
        mask=mask,
        phase_fractions=fractions,
        ci_stats=ci_stats,
        step_size_um=step_um,
        shape=tuple(int(s) for s in shape),
        phases_found=phases_found,
        config=cfg,
    )


def save_mask(result: MaskResult, mask_path: str | Path, meta_path: str | Path | None = None) -> None:
    """Save the mask as PNG and its metadata as JSON sidecar."""
    mask_path = Path(mask_path)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    # PIL preserves uint8 losslessly as a single-channel PNG — perfect for masks.
    from PIL import Image
    Image.fromarray(result.mask, mode="L").save(mask_path)

    if meta_path is None:
        meta_path = mask_path.with_suffix(".json")
    meta_path = Path(meta_path)

    payload: dict[str, Any] = {
        "shape": result.shape,
        "step_size_um": result.step_size_um,
        "phase_fractions": result.phase_fractions,
        "ci_stats": result.ci_stats,
        "phases_found_in_ang": result.phases_found,
        "config": {
            "ci_threshold": result.config.ci_threshold,
            "iq_threshold": result.config.iq_threshold,
            "apply_median_filter": result.config.apply_median_filter,
            "median_filter_size": result.config.median_filter_size,
            "dilate_unindexed": result.config.dilate_unindexed,
            "unindexed_dilation_pixels": result.config.unindexed_dilation_pixels,
            "three_class_mode": result.config.three_class_mode,
            "ferrite_class_id": result.config.ferrite_class_id,
            "austenite_class_id": result.config.austenite_class_id,
            "sigma_class_id": result.config.sigma_class_id,
        },
        "class_id_convention": {
            "0": "austenite (γ, FCC)",
            "1": "ferrite (δ/α, BCC)",
            "2": "sigma (σ)  [only in 3-class mode]",
            str(IGNORE_INDEX): "unindexed / ignore",
        },
    }
    with meta_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
