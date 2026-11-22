"""Paper figure -> crop -> class mask -> features -> validation.

One function per stage; every stage reads/updates the manifest record for a
sample (upsert by sample_id), so scripts can be run one stage at a time or all
together via scripts/run_paper_pipeline.py.

Layout (relative to data_root):
  raw/papers/<paper_id>/meta.yaml + figures/   originals, never written
  interim/ingest/<paper_id>.json               hashes + meta validation result
  interim/crops/<sample_id>.png, _ignore.png   crop + 255 ignore map
  interim/masks/<sample_id>.png                class mask (0 aus, 1 fer, 2 sigma, 255 ignore)
  interim/overlays/<sample_id>.png             QC overlay
  processed/features/features.csv
  processed/manifests/manifest.json            schema v2
"""
from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.data.annotations import IGNORE, ignore_mask, to_crop_frame
from src.data.crop import crop_panel
from src.data.ingest_paper import ingest_paper_dir, load_meta, panel_contexts
from src.data.manifest import (
    Extraction, ImageInfo, PhaseFractions, Properties, Record, Source,
    add_or_update_sample, get_record, load_manifest,
)
from src.data.scalebar import calibrate
from src.eval.validate import boundary_band_exclude, evaluate, normalise_reported, write_report
from src.features.features import mask_features, phase_fractions
from src.io.image_io import load_image, load_mask, save_png, sha256_file
from src.segment.ebsd_colour_to_mask import ebsd_to_mask
from src.segment.micrograph_segment import load_ilastik_probabilities, segment_micrograph


class SkipPanel(Exception):
    """Panel cannot be processed for a data reason (logged, flagged, run continues)."""


def rel(p: str | Path, root: Path) -> str:
    return Path(os.path.relpath(Path(p), root)).as_posix()


def paper_dirs(paths: dict[str, Path], paper: str | None = None) -> list[Path]:
    if paper:
        d = paths["papers"] / paper
        if not (d / "meta.yaml").exists():
            raise FileNotFoundError(f"{d / 'meta.yaml'} not found")
        return [d]
    if not paths["papers"].exists():
        return []
    return sorted(d for d in paths["papers"].iterdir() if (d / "meta.yaml").exists())


def select_contexts(paths: dict[str, Path], paper: str | None = None,
                    sample: str | None = None) -> list[dict[str, Any]]:
    ctxs = [c for d in paper_dirs(paths, paper) for c in panel_contexts(d, load_meta(d))]
    if sample:
        ctxs = [c for c in ctxs if c["sample_id"] == sample]
        if not ctxs:
            raise KeyError(f"sample_id {sample} not found in any meta.yaml")
    return ctxs


def _out(paths: dict[str, Path], kind: str, sid: str, suffix: str = "") -> Path:
    return paths["interim"] / kind / f"{sid}{suffix}.png"


def _record(paths: dict[str, Path], sid: str) -> dict[str, Any]:
    rec = get_record(paths["manifest"], sid)
    if rec is None:
        raise SkipPanel(f"{sid}: no manifest record — run crop_panels.py first")
    return rec


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_ingest(paper_dir: Path, cfg: dict, paths: dict[str, Path], log) -> bool:
    res = ingest_paper_dir(paper_dir, paths["interim"] / "ingest")
    if not cfg.get("io", {}).get("allow_jpeg", True):
        res["errors"] += [f"{f['file']}: JPEG not allowed (io.allow_jpeg=false)"
                          for f in res["files"] if f["is_jpeg"]]
    if load_meta(paper_dir).get("paper_id") != paper_dir.name:
        res["warnings"].append(f"{paper_dir.name}: folder name differs from meta paper_id")
    for w in res["warnings"]:
        log.warning(w)
    for e in res["errors"]:
        log.error(e)
    return not res["errors"]


def stage_crop(ctx: dict, cfg: dict, paths: dict[str, Path], log) -> dict[str, Any]:
    sid, root = ctx["sample_id"], paths["data_root"]
    ccfg = cfg.get("crop", {})
    fig = Path(ctx["figure_path"])
    img, info = load_image(fig)
    crop, box, flags = crop_panel(img, ctx["crop_box"], ccfg.get("margin_px", 0), ccfg.get("min_side_px", 0))
    ign = ignore_mask(crop.shape, to_crop_frame(ctx.get("ignore_boxes"), box))
    crop_p, ign_p = save_png(_out(paths, "crops", sid), crop), save_png(_out(paths, "crops", sid, "_ignore"), ign)
    reported, rflags = normalise_reported(ctx.get("reported"),
                                          cfg.get("validation", {}).get("sum_tolerance_pp", 2.0))
    old = get_record(paths["manifest"], sid) or {}
    props = {k: v for k, v in (ctx.get("properties") or {}).items() if k in asdict(Properties())}
    rec = Record(
        sample_id=sid,
        paths={"original": rel(fig, root), "crop": rel(crop_p, root), "ignore": rel(ign_p, root),
               "mask": None, "overlay": None},
        material=ctx["material"], process=ctx.get("process"), condition=ctx.get("condition"),
        build_plane=ctx.get("build_plane"),
        image=ImageInfo(image_type=ctx.get("image_type"), original_format=info["format"],
                        is_jpeg=info["is_jpeg"], crop_box=list(ctx["crop_box"]),
                        width_px=int(crop.shape[1]), height_px=int(crop.shape[0]),
                        etchant=ctx.get("etchant"), dark_phase=ctx.get("dark_phase"),
                        ebsd_palette=ctx.get("ebsd_palette"), ebsd_step_um=ctx.get("ebsd_step_um"),
                        magnification=ctx.get("magnification")),
        source=Source(provenance="paper", paper_doi=ctx.get("doi"), citation=ctx.get("citation"),
                      licence=ctx.get("licence"), figure_id=ctx["figure_id"], panel=str(ctx.get("panel")),
                      acquisition_route=ctx.get("acquisition_route"), source_url=ctx.get("source_url"),
                      retrieved_on=ctx.get("retrieved_on"), sha256_original=sha256_file(fig),
                      notes=ctx.get("notes")),
        reported=PhaseFractions(**reported),
        qc_flags=sorted(set(flags + rflags)),
        properties=Properties(**props),
        split=old.get("split"), cv_fold=old.get("cv_fold"),
    )
    add_or_update_sample(paths["manifest"], rec)
    log.info("[crop] %s %dx%d box=%s flags=%s", sid, crop.shape[1], crop.shape[0], box, rec.qc_flags)
    return asdict(rec)


def stage_scale(ctx: dict, cfg: dict, paths: dict[str, Path], log) -> dict[str, Any]:
    sid = ctx["sample_id"]
    rec = _record(paths, sid)
    img, _ = load_image(ctx["figure_path"])          # original pixels, not the crop
    cal = calibrate(ctx.get("scale_bar"), img, cfg.get("scalebar", {}).get("polarity", "auto"))
    im = rec["image"]
    if cal is None:
        log.warning("[scale] %s: no usable scale_bar in meta.yaml — pixel size unknown", sid)
        im.update(scale_bar_um=None, scale_bar_px=None, pixel_size_um=None, scale_uncertainty_rel=None)
    else:
        im.update(scale_bar_um=cal["um"], scale_bar_px=cal["bar_px"], pixel_size_um=cal["um_per_px"],
                  scale_uncertainty_rel=cal["rel_uncertainty"])
        log.info("[scale] %s %.4f um/px (bar %.0f px, %s, ±%.1f%%)", sid, cal["um_per_px"],
                 cal["bar_px"], cal["method"], 100 * cal["rel_uncertainty"])
    add_or_update_sample(paths["manifest"], rec)
    return rec


def segment_panel(crop: np.ndarray, ign: np.ndarray, ctx: dict, cfg: dict,
                  boundary_policy: str | None = None, threshold_shift: float = 0.0,
                  method: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Dispatch on image_type. Returns (mask, {method, params, boundary_policy})."""
    it = ctx.get("image_type")
    if it in ("ebsd_phase", "ebsd_phase_iq"):
        e = cfg.get("ebsd", {})
        pol = boundary_policy or e.get("boundary_policy", "ignore")
        params = {"max_delta_e": float(e.get("max_delta_e", 20)) * (1 + threshold_shift),
                  "ambiguity_ratio": e.get("ambiguity_ratio", 0.8),
                  "mode_filter_size": e.get("mode_filter", 3),
                  "min_island_px": e.get("min_island_px", 4),
                  "reassign_max_px": e.get("reassign_max_px", 3)}
        mask = ebsd_to_mask(crop, ctx["ebsd_palette"], it, ign, boundary_policy=pol, **params)
        return mask, {"method": "ebsd_palette_deltaE", "params": params, "boundary_policy": pol}
    if it in ("om", "sem"):
        if ctx.get("ilastik_probabilities") and method is None and threshold_shift == 0:
            m = load_ilastik_probabilities(Path(ctx["paper_dir"]) / ctx["ilastik_probabilities"])
            if m is not None:
                m = m.copy()
                m[ign > 0] = IGNORE
                return m, {"method": "ilastik", "params": {}, "boundary_policy": None}
        if ctx.get("dark_phase") not in ("ferrite", "austenite"):
            raise SkipPanel(f"{ctx['sample_id']}: dark_phase not set in meta.yaml — cannot segment")
        m = cfg.get("micrograph", {})
        meth = method or (m.get("methods") or ["otsu"])[0]
        params = {k: m[k] for k in ("flatfield_sigma_px", "denoise_h", "morph_radius_px", "min_island_px",
                                    "local_block_px", "local_offset", "kmeans_attempts") if k in m}
        mask, info = segment_micrograph(crop, ctx["dark_phase"], meth, ign, threshold_shift=threshold_shift,
                                        sigma_tone=ctx.get("sigma_tone"), seed=int(cfg.get("seed", 42)),
                                        **params)
        params.update({k: v for k, v in info.items() if k != "method"})
        return mask, {"method": meth, "params": params, "boundary_policy": None}
    raise SkipPanel(f"{ctx['sample_id']}: unsupported image_type {it!r}")


def _hex(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)


def overlay(crop: np.ndarray, mask: np.ndarray, colours: dict[str, str], alpha: float = 0.45) -> np.ndarray:
    out = crop[..., :3].astype(float).copy()
    for name, cid in (("austenite", 0), ("ferrite", 1), ("sigma", 2), ("ignore", IGNORE)):
        sel = mask == cid
        if sel.any() and colours.get(name):
            out[sel] = (1 - alpha) * out[sel] + alpha * _hex(colours[name])
    return np.clip(out, 0, 255).astype(np.uint8)


def _load_crop(rec: dict, paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    root = paths["data_root"]
    crop, _ = load_image(root / rec["paths"]["crop"])
    return crop, load_mask(root / rec["paths"]["ignore"])


def stage_mask(ctx: dict, cfg: dict, paths: dict[str, Path], h: str, log) -> dict[str, Any]:
    sid, root = ctx["sample_id"], paths["data_root"]
    rec = _record(paths, sid)
    crop, ign = _load_crop(rec, paths)
    try:
        mask, info = segment_panel(crop, ign, ctx, cfg)
    except SkipPanel:
        rec["qc_flags"] = sorted(set(rec["qc_flags"]) | {"not_segmented"})
        add_or_update_sample(paths["manifest"], rec)
        raise
    mp = save_png(_out(paths, "masks", sid), mask)
    ov = cfg.get("overlay", {})
    op = save_png(_out(paths, "overlays", sid), overlay(crop, mask, ov.get("colours", {}), ov.get("alpha", 0.45)))
    fr = phase_fractions(mask)
    ignored = fr.pop("ignored_fraction")
    rec["paths"].update(mask=rel(mp, root), overlay=rel(op, root))
    rec["qc_flags"] = [f for f in rec["qc_flags"] if f != "not_segmented"]
    rec["extracted"] = asdict(Extraction(method=info["method"], params={**info["params"], "config_hash": h},
                                         boundary_policy=info["boundary_policy"], fractions=fr,
                                         ignored_fraction=ignored))
    add_or_update_sample(paths["manifest"], rec)
    log.info("[mask] %s %s ferrite=%.4f austenite=%.4f ignored=%.3f", sid, info["method"],
             fr["ferrite"], fr["austenite"], ignored)
    return rec


def stage_validate(ctx: dict, cfg: dict, paths: dict[str, Path], log) -> dict[str, Any]:
    sid = ctx["sample_id"]
    rec = _record(paths, sid)
    v = cfg.get("validation", {})
    variants: dict[str, dict[str, Any]] = {}
    if rec.get("extracted") and rec["paths"].get("mask"):
        crop, ign = _load_crop(rec, paths)
        base = load_mask(paths["data_root"] / rec["paths"]["mask"])
        s = float(v.get("threshold_shift_rel", 0.10))
        for sign in (+1, -1):
            m, _ = segment_panel(crop, ign, ctx, cfg, threshold_shift=sign * s)
            variants[f"threshold_{sign * s:+.0%}"] = {"kind": "threshold", "fractions": phase_fractions(m)}
        if ctx.get("image_type", "").startswith("ebsd"):
            other = "reassign" if rec["extracted"]["boundary_policy"] == "ignore" else "ignore"
            m, _ = segment_panel(crop, ign, ctx, cfg, boundary_policy=other)
            variants[f"boundary_policy_{other}"] = {"kind": "boundary", "fractions": phase_fractions(m)}
        elif rec["extracted"]["method"] != "ilastik":
            for meth in (cfg.get("micrograph", {}).get("methods") or [])[1:]:
                m, _ = segment_panel(crop, ign, ctx, cfg, method=meth)
                variants[f"method_{meth}"] = {"kind": "method", "fractions": phase_fractions(m)}
        for px in v.get("boundary_band_px", [1, 2]):
            variants[f"band_exclude_{px}px"] = {"kind": "boundary",
                                                "fractions": phase_fractions(boundary_band_exclude(base, int(px)))}
    evaluate(rec, variants, v)
    add_or_update_sample(paths["manifest"], rec)
    log.info("[validate] %s abs_error_pp=%s flags=%s", sid, rec["abs_error_pp"], rec["qc_flags"])
    return rec


def stage_features(cfg: dict, paths: dict[str, Path], log, sample_ids: list[str]) -> Path:
    import pandas as pd
    fc = cfg.get("features", {})
    ids = set(sample_ids)
    rows = []
    for rec in load_manifest(paths["manifest"]) if paths["manifest"].exists() else []:
        if rec["sample_id"] not in ids or not rec["paths"].get("mask"):
            continue
        mask = load_mask(paths["data_root"] / rec["paths"]["mask"])
        um = rec["image"].get("pixel_size_um")
        row = {"sample_id": rec["sample_id"], "paper_doi": rec["source"].get("paper_doi"),
               "image_type": rec["image"].get("image_type"), "is_jpeg": rec["image"].get("is_jpeg"),
               "condition": rec.get("condition"), "build_plane": rec.get("build_plane"),
               "um_per_px": um, "scale_uncertainty_rel": rec["image"].get("scale_uncertainty_rel")}
        row.update(mask_features(mask, um, fc.get("min_region_px", 20), fc.get("exclude_border", True)))
        rows.append(row)
    out = paths["features"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out.exists():   # keep rows of samples not in this run
        old = pd.read_csv(out)
        df = pd.concat([old[~old["sample_id"].isin(ids)], df], ignore_index=True)
    df.sort_values("sample_id").to_csv(out, index=False)
    log.info("[features] %d rows -> %s", len(rows), out)
    return out


def report(cfg: dict, paths: dict[str, Path], h: str, log, sample_ids: list[str] | None = None) -> dict:
    recs = load_manifest(paths["manifest"]) if paths["manifest"].exists() else []
    if sample_ids is not None:
        ids = set(sample_ids)
        recs = [r for r in recs if r["sample_id"] in ids]
    out = write_report(recs, paths["reports"], cfg.get("validation", {}).get("group_by", "paper_doi"), h)
    log.info("[report] %s", {k: str(v) for k, v in out.items()})
    return out


def run_stage_over(ctxs: list[dict], fn, log, *args) -> list[str]:
    """Apply a per-panel stage; log and collect failures instead of aborting the run."""
    failed = []
    for c in ctxs:
        try:
            fn(c, *args)
        except SkipPanel as e:
            log.warning("[skip] %s", e)
            failed.append(c["sample_id"])
        except Exception as e:  # noqa: BLE001 — one bad panel must not kill a batch run
            log.error("[fail] %s: %s: %s", c["sample_id"], type(e).__name__, e)
            failed.append(c["sample_id"])
    return failed
