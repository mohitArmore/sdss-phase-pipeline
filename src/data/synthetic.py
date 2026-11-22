"""Synthetic two-phase images with KNOWN phase fractions (tests + fixture paper).

Not real data. Used to check that the extraction pipeline recovers a known
fraction from (a) an EBSD-style colour map with black boundaries and (b) a
grey etched-style micrograph with noise and uneven illumination.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from skimage.segmentation import find_boundaries

FERRITE_RGB = (255, 0, 0)
AUSTENITE_RGB = (0, 0, 255)


def phase_field(shape: tuple[int, int], ferrite_frac: float, rng: np.random.Generator,
                blob_sigma: float = 10.0, valid: np.ndarray | None = None) -> np.ndarray:
    """Random blobs; threshold chosen so the ferrite fraction over `valid` pixels is exact."""
    field = gaussian_filter(rng.standard_normal(shape), blob_sigma)
    vals = field[valid] if valid is not None else field.ravel()
    return field > np.quantile(vals, 1.0 - ferrite_frac)


def render_ebsd(ferrite: np.ndarray, boundary: bool = True) -> np.ndarray:
    img = np.where(ferrite[..., None], FERRITE_RGB, AUSTENITE_RGB).astype(np.uint8)
    if boundary:
        img[find_boundaries(ferrite, mode="thick")] = 0   # black, ~1 px each side
    return img


def render_micrograph(ferrite: np.ndarray, rng: np.random.Generator, dark: float = 70,
                      bright: float = 185, noise_sd: float = 12, illum: float = 0.35,
                      dark_is_ferrite: bool = True) -> np.ndarray:
    h, w = ferrite.shape
    base = np.where(ferrite == dark_is_ferrite, dark, bright).astype(float)
    yy, xx = np.mgrid[0:h, 0:w]
    gain = 1.0 + illum * ((xx / w) - 0.5) + 0.5 * illum * ((yy / h) - 0.5)   # uneven illumination
    img = gaussian_filter(base, 1.0) * gain + rng.normal(0, noise_sd, (h, w))
    return np.clip(img, 0, 255).astype(np.uint8)


def _true_fraction(rep: dict) -> float | None:
    unit_pct = rep.get("unit", "percent") == "percent"
    if rep.get("ferrite") is not None:
        f = float(rep["ferrite"])
    elif rep.get("austenite") is not None:
        f = (100.0 if unit_pct else 1.0) - float(rep["austenite"])
    else:
        return None
    return f / 100.0 if unit_pct else f


def make_fixture_paper(meta_path: str | Path, out_dir: str | Path, seed: int = 0) -> Path:
    """Render the figures described by a fixture meta.yaml into out_dir.

    Truth comes from meta.yaml itself: each panel's reported ferrite (or 1 -
    austenite) is the exact ferrite fraction drawn over the crop minus its
    ignore boxes. Scale bars are drawn as a black bar inside a white box at
    scale_bar.bbox, 10 px shorter than the bbox at each end (bar_px = width - 20).
    """
    meta_path, out_dir = Path(meta_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(meta_path, out_dir / "meta.yaml")
    meta = yaml.safe_load(meta_path.read_text())
    rng = np.random.default_rng(seed)
    for fig in meta["figures"]:
        panels = fig["panels"]
        W = max(p["crop_box"][2] for p in panels) + 20
        H = max(p["crop_box"][3] for p in panels) + 20
        canvas = np.full((H, W, 3), 255, np.uint8)
        for p in panels:
            x0, y0, x1, y1 = p["crop_box"]
            valid = np.ones((y1 - y0, x1 - x0), bool)
            for bx0, by0, bx1, by1 in p.get("ignore_boxes") or []:
                valid[max(by0 - y0, 0):by1 - y0, max(bx0 - x0, 0):bx1 - x0] = False
            fer = phase_field(valid.shape, _true_fraction(p["reported"]), rng, valid=valid)
            if p["image_type"].startswith("ebsd"):
                tile = render_ebsd(fer)
            else:
                g = render_micrograph(fer, rng, dark_is_ferrite=p["dark_phase"] == "ferrite")
                tile = np.repeat(g[..., None], 3, axis=2)
            canvas[y0:y1, x0:x1] = tile
            for bx0, by0, bx1, by1 in p.get("ignore_boxes") or []:
                canvas[by0:by1, bx0:bx1] = 255          # legend/label area
            sb = p.get("scale_bar") or {}
            if sb.get("bbox"):
                bx0, by0, bx1, by1 = sb["bbox"]
                cy = (by0 + by1) // 2
                canvas[cy - 2:cy + 3, bx0 + 10:bx1 - 10] = 0
        im = Image.fromarray(canvas)
        ImageDraw.Draw(im).text((4, 4), f"Fig. {fig['figure_id']} (synthetic)", fill=(0, 0, 0))
        dst = out_dir / fig["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.suffix.lower() in (".jpg", ".jpeg"):
            im.save(dst, format="JPEG", quality=75)   # creating a test JPEG, not re-encoding one
        else:
            im.save(dst, format="PNG")
    return out_dir
