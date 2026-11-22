"""Image IO with provenance: never re-encode JPEGs, derived images are PNG.

- load_image(path) -> (uint8 RGB array, info dict incl. format / is_jpeg /
  estimated JPEG quality / chroma subsampling).
- save_png(path, arr): the ONLY writer for derived images and masks.
- sha256_file(path): hash of the original bytes (for provenance and the
  "original never rewritten" check).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# IJG standard luminance quantisation table (quality 50), natural (row-major) order.
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
], dtype=float)
# Pillow (>=8.3) returns quantization tables in natural order, same as above.
_SUBSAMPLING = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def estimate_jpeg_quality(img: Image.Image) -> int | None:
    """Estimate IJG quality (1-100) from the luminance quantisation table.
    ponytail: assumes IJG-style scaled tables; non-IJG encoders give a rough number."""
    tables = getattr(img, "quantization", None)
    if not tables or 0 not in tables or len(tables[0]) != 64:
        return None
    q = np.asarray(tables[0], dtype=float)
    scale = float(np.mean(q / _STD_LUMA * 100.0))
    quality = (200.0 - scale) / 2.0 if scale <= 100 else 5000.0 / scale
    return int(round(min(max(quality, 1), 100)))


def load_image(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load any PNG/JPEG/TIFF as uint8 RGB (H, W, 3). The file is only read."""
    path = Path(path)
    with Image.open(path) as img:
        fmt = img.format
        info: dict[str, Any] = {
            "path": str(path), "format": fmt, "is_jpeg": fmt in ("JPEG", "MPO"),
            "mode": img.mode, "width": img.width, "height": img.height,
            "jpeg_quality_est": None, "chroma_subsampling": None,
        }
        if info["is_jpeg"]:
            info["jpeg_quality_est"] = estimate_jpeg_quality(img)
            try:
                from PIL import JpegImagePlugin
                info["chroma_subsampling"] = _SUBSAMPLING.get(JpegImagePlugin.get_sampling(img))
            except Exception:  # noqa: BLE001 — metadata is best-effort
                pass
        if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            a = np.asarray(img, dtype=np.float64)
            lo, hi = float(a.min()), float(a.max())
            a = np.zeros_like(a) if hi == lo else (a - lo) / (hi - lo) * 255.0
            arr = np.repeat(a.astype(np.uint8)[..., None], 3, axis=2)
        elif img.mode in ("RGBA", "LA", "P", "PA"):
            # Composite transparency on white (figures are printed on white).
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            arr = np.asarray(Image.alpha_composite(bg, rgba).convert("RGB"))
        else:
            arr = np.asarray(img.convert("RGB"))
    return np.ascontiguousarray(arr, dtype=np.uint8), info


def save_png(path: str | Path, arr: np.ndarray) -> Path:
    """Write a derived image/mask as lossless PNG. Refuses any other extension."""
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError(f"Derived images must be PNG, got {path.name}")
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        raise ValueError(f"save_png expects uint8, got {arr.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="PNG")
    return path


def load_mask(path: str | Path) -> np.ndarray:
    """Read a single-channel PNG (mask / ignore map) exactly as stored."""
    with Image.open(path) as img:
        return np.asarray(img).copy()
