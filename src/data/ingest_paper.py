"""Paper registry: read/validate meta.yaml, hash originals, extract PDF images,
draw coordinate-grid previews for manual cropping.

All paper facts come from the human-filled meta.yaml. Missing facts produce
warnings (never invented values); structural problems produce errors.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.data.manifest import ACQUISITION_ROUTES, IMAGE_TYPES, PF_METHODS, PF_SCOPES
from src.io.image_io import load_image, save_png, sha256_file

ROUTE_CODES = {"publisher_figure": "pub", "pdf_embedded": "pdfemb", "pdf_render": "pdfrend",
               "supplementary": "supp", "screenshot": "scr", "author_supplied": "auth"}
PAPER_FACTS = ("doi", "citation", "licence")
# Keys a figure may set as defaults for its panels.
_INHERITED = ("image_type", "condition", "build_plane", "etchant", "dark_phase", "sigma_tone",
              "ebsd_palette", "ebsd_step_um", "magnification", "scale_bar", "reported",
              "ignore_boxes", "properties")


def load_meta(paper_dir: str | Path) -> dict[str, Any]:
    p = Path(paper_dir) / "meta.yaml"
    with p.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
    meta.setdefault("paper_id", Path(paper_dir).name)
    return meta


def _token(s: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(s)).strip("-") or "na"


def make_sample_id(paper_id: str, figure_id: Any, panel: Any, route: str | None,
                   condition: Any, image_type: str | None) -> str:
    return "_".join([
        _token(paper_id), f"f{_token(figure_id)}{_token(panel)}",
        ROUTE_CODES.get(route or "", "na"), _token(condition or "na"), _token(image_type or "na"),
    ])


def panel_contexts(paper_dir: str | Path, meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten meta.yaml into one dict per panel (figure defaults merged in)."""
    paper_dir = Path(paper_dir)
    out = []
    for fig in meta.get("figures") or []:
        for pan in fig.get("panels") or []:
            ctx = {k: fig[k] for k in _INHERITED if k in fig}
            ctx.update(pan)
            ctx.update({
                "paper_id": meta["paper_id"], "paper_dir": str(paper_dir),
                "doi": meta.get("doi"), "citation": meta.get("citation"),
                "licence": meta.get("licence"), "material": str(meta.get("material") or "2507"),
                "process": meta.get("process"), "paper_notes": meta.get("notes"),
                "figure_id": str(fig.get("figure_id")), "file": fig.get("file"),
                "figure_path": str(paper_dir / fig.get("file", "")),
                "acquisition_route": fig.get("acquisition_route"),
                "source_url": fig.get("source_url"),
                "retrieved_on": None if fig.get("retrieved_on") is None else str(fig.get("retrieved_on")),
            })
            ctx["sample_id"] = make_sample_id(meta["paper_id"], ctx["figure_id"], ctx.get("panel"),
                                              ctx["acquisition_route"], ctx.get("condition"),
                                              ctx.get("image_type"))
            out.append(ctx)
    return out


def _box_ok(b: Any) -> bool:
    return (isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, int) for v in b)
            and b[2] > b[0] and b[3] > b[1] and min(b) >= 0)


def validate_meta(paper_dir: str | Path, meta: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors = cannot process; warnings = missing facts."""
    errors: list[str] = []
    warnings: list[str] = []
    pid = meta.get("paper_id")
    for k in PAPER_FACTS:
        if not meta.get(k):
            warnings.append(f"{pid}: '{k}' is null — fill it from the paper")
    if not meta.get("figures"):
        errors.append(f"{pid}: no figures listed")
    for fig in meta.get("figures") or []:
        fid = fig.get("figure_id")
        if fid is None:
            errors.append(f"{pid}: figure without figure_id")
        f = Path(paper_dir) / str(fig.get("file", ""))
        if not fig.get("file") or not f.is_file():
            errors.append(f"{pid} fig {fid}: file not found: {f}")
        if fig.get("acquisition_route") not in ACQUISITION_ROUTES:
            errors.append(f"{pid} fig {fid}: acquisition_route must be one of {ACQUISITION_ROUTES}")
        for k in ("source_url", "retrieved_on"):
            if not fig.get(k):
                warnings.append(f"{pid} fig {fid}: '{k}' is null")
    seen: set[str] = set()
    for c in panel_contexts(paper_dir, meta):
        tag = c["sample_id"]
        if tag in seen:
            errors.append(f"{tag}: duplicate sample_id (panel letters must be unique per figure)")
        seen.add(tag)
        if not _box_ok(c.get("crop_box")):
            errors.append(f"{tag}: crop_box must be [x0,y0,x1,y1] ints with x1>x0, y1>y0")
        for b in c.get("ignore_boxes") or []:
            if not _box_ok(b):
                errors.append(f"{tag}: bad ignore box {b}")
        it = c.get("image_type")
        if it not in IMAGE_TYPES:
            errors.append(f"{tag}: image_type must be one of {IMAGE_TYPES}")
        elif it.startswith("ebsd"):
            pal = c.get("ebsd_palette") or {}
            if not (pal.get("ferrite") and pal.get("austenite")):
                errors.append(f"{tag}: ebsd_palette needs ferrite and austenite colours")
        elif c.get("dark_phase") not in ("ferrite", "austenite"):
            warnings.append(f"{tag}: dark_phase is null — micrograph cannot be segmented")
        if not c.get("condition"):
            warnings.append(f"{tag}: condition is null")
        sb = c.get("scale_bar") or {}
        if not (sb.get("um") and (sb.get("px") or _box_ok(sb.get("bbox")))):
            warnings.append(f"{tag}: scale_bar missing/incomplete — no pixel size, grain sizes in px only")
        rep = c.get("reported") or {}
        if all(rep.get(k) is None for k in ("ferrite", "austenite", "sigma")):
            warnings.append(f"{tag}: no reported phase fraction — cannot validate")
        else:
            if rep.get("method") not in PF_METHODS:
                warnings.append(f"{tag}: reported.method should be one of {PF_METHODS}")
            if rep.get("scope") not in PF_SCOPES:
                warnings.append(f"{tag}: reported.scope should be one of {PF_SCOPES}")
            if not rep.get("location_in_paper"):
                warnings.append(f"{tag}: reported.location_in_paper is null")
    return errors, warnings


def ingest_paper_dir(paper_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Validate meta.yaml, hash every figure file, record format / is_jpeg.
    Writes <out_dir>/<paper_id>.json. If a previous ingest exists and a hash
    changed, that is reported as an error (originals must never change)."""
    paper_dir = Path(paper_dir)
    meta = load_meta(paper_dir)
    errors, warnings = validate_meta(paper_dir, meta)
    out_path = Path(out_dir) / f"{meta['paper_id']}.json"
    previous = json.loads(out_path.read_text()) if out_path.exists() else {}
    prev_hash = {f["file"]: f["sha256"] for f in previous.get("files", [])}
    files = []
    for fig in meta.get("figures") or []:
        f = paper_dir / str(fig.get("file", ""))
        if not f.is_file():
            continue
        _, info = load_image(f)
        rec = {"figure_id": str(fig.get("figure_id")), "file": fig["file"],
               "sha256": sha256_file(f), "format": info["format"], "is_jpeg": info["is_jpeg"],
               "jpeg_quality_est": info["jpeg_quality_est"],
               "chroma_subsampling": info["chroma_subsampling"],
               "width": info["width"], "height": info["height"]}
        old = prev_hash.get(fig["file"])
        if old and old != rec["sha256"]:
            errors.append(f"{meta['paper_id']}: original {fig['file']} CHANGED since last ingest "
                          f"({old[:12]} -> {rec['sha256'][:12]}); originals must be immutable")
        files.append(rec)
    result = {"paper_id": meta["paper_id"], "files": files, "errors": errors, "warnings": warnings}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# PDF image extraction (lossless)
# ---------------------------------------------------------------------------

def extract_pdf_images(pdf_path: str | Path, out_dir: str | Path,
                       use_pdfimages: bool = True) -> Path:
    """Extract every embedded image with its original bytes. Uses
    `pdfimages -all` when installed, else PyMuPDF doc.extract_image(xref)
    (soft masks are written as separate files, never composited). Returns the
    listing CSV path (page, xref, smask_xref, width, height, format, file, sha256)."""
    pdf_path, out_dir = Path(pdf_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if use_pdfimages and shutil.which("pdfimages"):
        subprocess.run(["pdfimages", "-all", "-p", str(pdf_path), str(out_dir / "img")], check=True)
        from PIL import Image
        for f in sorted(out_dir.glob("img-*")):
            m = re.match(r"img-(\d+)-(\d+)\.(\w+)$", f.name)
            if not m:
                continue
            try:
                with Image.open(f) as im:
                    w, h = im.size
            except Exception:  # noqa: BLE001 — e.g. JBIG2/CCITT streams PIL can't open
                w = h = None
            rows.append({"page": int(m.group(1)), "xref": None, "smask_xref": None, "width": w,
                         "height": h, "format": m.group(3), "file": f.name,
                         "sha256": sha256_file(f)})
    else:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        seen: set[int] = set()
        for pno in range(doc.page_count):
            for img in doc.get_page_images(pno, full=True):
                xref, smask = img[0], img[1]
                if xref in seen:
                    continue
                seen.add(xref)
                d = doc.extract_image(xref)
                if not d or not d.get("image"):
                    continue
                name = f"p{pno + 1:03d}_x{xref}.{d['ext']}"
                (out_dir / name).write_bytes(d["image"])
                rows.append({"page": pno + 1, "xref": xref, "smask_xref": smask or None,
                             "width": d.get("width"), "height": d.get("height"),
                             "format": d["ext"], "file": name,
                             "sha256": sha256_file(out_dir / name)})
                if smask:
                    sm = doc.extract_image(smask)
                    if sm and sm.get("image"):
                        sname = f"p{pno + 1:03d}_x{xref}_smask{smask}.{sm['ext']}"
                        (out_dir / sname).write_bytes(sm["image"])
        doc.close()
    listing = out_dir / "listing.csv"
    with listing.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["page", "xref", "smask_xref", "width", "height",
                                          "format", "file", "sha256"])
        w.writeheader()
        w.writerows(rows)
    return listing


# ---------------------------------------------------------------------------
# Grid preview for manual cropping
# ---------------------------------------------------------------------------

def grid_preview(img: np.ndarray, grid_px: int = 100,
                 boxes: list[tuple[list[int], tuple[int, int, int]]] | None = None) -> np.ndarray:
    """Copy of img with a labelled pixel grid every grid_px (original pixel
    coordinates, so values can be typed straight into meta.yaml) and optional
    coloured boxes for already-entered crop/ignore/scale-bar boxes."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img).convert("RGB")
    h, w = img.shape[:2]
    pad = 40  # margin for axis labels, so labels never hide image pixels
    canvas = Image.new("RGB", (w + pad, h + pad), (255, 255, 255))
    canvas.paste(im, (pad, pad))
    d = ImageDraw.Draw(canvas)
    for x in range(0, w, grid_px):
        d.line([(x + pad, pad), (x + pad, h + pad)], fill=(0, 200, 255), width=1)
        d.text((x + pad + 2, 2), str(x), fill=(0, 0, 0))
    for y in range(0, h, grid_px):
        d.line([(pad, y + pad), (w + pad, y + pad)], fill=(0, 200, 255), width=1)
        d.text((2, y + pad + 2), str(y), fill=(0, 0, 0))
    for box, colour in boxes or []:
        if _box_ok(box):
            x0, y0, x1, y1 = box
            d.rectangle([x0 + pad, y0 + pad, x1 - 1 + pad, y1 - 1 + pad], outline=colour, width=2)
    d.text((2, h + pad - 12), f"grid {grid_px}px, origin top-left, offset {pad}px", fill=(255, 0, 0))
    return np.asarray(canvas)


def write_previews(paper_dir: str | Path, out_dir: str | Path, grid_px: int = 100) -> list[Path]:
    paper_dir = Path(paper_dir)
    meta = load_meta(paper_dir)
    out: list[Path] = []
    for fig in meta.get("figures") or []:
        f = paper_dir / str(fig.get("file", ""))
        if not f.is_file():
            continue
        img, _ = load_image(f)
        boxes: list[tuple[list[int], tuple[int, int, int]]] = []
        for pan in fig.get("panels") or []:
            boxes.append((pan.get("crop_box"), (0, 200, 0)))
            boxes += [(b, (255, 0, 0)) for b in pan.get("ignore_boxes") or []]
            sb = pan.get("scale_bar") or {}
            if sb.get("bbox"):
                boxes.append((sb["bbox"], (255, 0, 255)))
        p = Path(out_dir) / meta["paper_id"] / f"fig{_token(fig.get('figure_id'))}_grid.png"
        save_png(p, grid_preview(img, grid_px, boxes))
        out.append(p)
    return out
