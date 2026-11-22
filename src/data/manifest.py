"""Dataset manifest — the single source of truth.

A manifest is a JSON file listing every sample (one image panel) in the dataset.
Every downstream loader reads the manifest; `data/raw/` and `data/interim/` are
never walked directly by training code.

Schema v2 (paper-figure dataset). One record per cropped panel:
{
  "sample_id":   str,   # {paper_id}_f{figure}{panel}_{route}_{condition}_{image_type}
  "paths":       {original, crop, ignore, mask, overlay}   # relative to data root
  "material":    str,   # "2507"
  "process":     str|None,   # "LPBF" | "wrought" | ...
  "condition":   str|None,   # "as_built" | "SA_1100C_1h" | ...
  "build_plane": str|None,   # "XY" | "XZ" | ...
  "image":       ImageInfo,  # type, format, JPEG flag, crop box, scale calibration
  "source":      Source,     # provenance: DOI, citation, licence (verbatim), figure/panel, sha256
  "reported":    PhaseFractions,  # values the PAPER reports (fractions 0..1), from meta.yaml only
  "extracted":   Extraction|None, # what this pipeline measured
  "abs_error_pp": {phase: float}|None,
  "qc_flags":    [str],
  "properties":  Properties,
  "split":       str|None,
  "cv_fold":     int|None,
  "legacy":      dict|None,  # v1-only fields kept losslessly by migrate_v1_to_v2()
}

Schema v1 (.ang era) is still readable: load_manifest() migrates it. The v1
`Sample` dataclass is kept so the legacy .ang script keeps working; saving a
Sample writes it in v2 shape.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Schema version so future changes are handled with migration functions,
# not silent breakage.
MANIFEST_SCHEMA_VERSION = 2

IMAGE_TYPES = ("ebsd_phase", "ebsd_phase_iq", "om", "sem")
ACQUISITION_ROUTES = ("publisher_figure", "pdf_embedded", "pdf_render",
                      "supplementary", "screenshot", "author_supplied")
PF_METHODS = ("EBSD", "XRD", "IA", "Feritscope", "this_work")
PF_SCOPES = ("this_panel", "sample_mean")


# ---------------------------------------------------------------------------
# Shared / v1 dataclasses (v1 kept for the legacy .ang path)
# ---------------------------------------------------------------------------

@dataclass
class Material:
    alloy: str = "SDSS_2507"
    composition: dict[str, float] = field(default_factory=dict)
    condition: str = "unknown"


@dataclass
class Process:
    route: str = "LPBF"
    laser_power_W: float | None = None
    scan_speed_mm_s: float | None = None
    hatch_um: float | None = None
    layer_um: float | None = None
    heat_treatment: str | None = None


@dataclass
class Properties:
    tensile_MPa: float | None = None
    yield_MPa: float | None = None
    elongation_pct: float | None = None
    hardness_HV: float | None = None
    corrosion_uA: float | None = None


@dataclass
class PhaseFractionsV1:
    ferrite: float | None = None
    austenite: float | None = None
    sigma: float | None = None
    unindexed: float | None = None


@dataclass
class Quality:
    ebsd_ci_mean: float | None = None
    ebsd_indexed_pct: float | None = None
    exclude: bool = False
    exclude_reason: str | None = None


@dataclass
class Source:
    """Provenance. v1 used only provenance/citation/notes; the rest is v2."""
    provenance: str = "own"              # "own" | "paper" | "dataset:MetalDAM"
    paper_doi: str | None = None
    citation: str | None = None
    licence: str | None = None           # verbatim from the paper / publisher
    figure_id: str | None = None
    panel: str | None = None
    acquisition_route: str | None = None  # one of ACQUISITION_ROUTES
    source_url: str | None = None
    retrieved_on: str | None = None
    sha256_original: str | None = None
    notes: str | None = None


@dataclass
class Sample:
    """v1 record (.ang era). Saved/loaded in v2 shape via migrate_sample_v1()."""
    sample_id: str
    ang_path: str | None = None
    sem_path: str | None = None
    optical_path: str | None = None
    mask_path: str | None = None
    mask_meta_path: str | None = None
    modality: str = "sem"
    magnification: float = 0.0
    pixel_size_um: float = 0.0
    material: Material = field(default_factory=Material)
    process: Process = field(default_factory=Process)
    properties: Properties = field(default_factory=Properties)
    phase_fractions_ebsd: PhaseFractionsV1 = field(default_factory=PhaseFractionsV1)
    quality: Quality = field(default_factory=Quality)
    split: str | None = None
    cv_fold: int | None = None
    source: Source = field(default_factory=Source)


# ---------------------------------------------------------------------------
# v2 dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ImageInfo:
    image_type: str | None = None        # one of IMAGE_TYPES
    original_format: str | None = None   # "PNG" | "JPEG" | ...
    is_jpeg: bool | None = None
    crop_box: list[int] | None = None    # [x0, y0, x1, y1] in ORIGINAL figure pixels
    width_px: int | None = None          # of the crop
    height_px: int | None = None
    etchant: str | None = None
    dark_phase: str | None = None        # "ferrite" | "austenite" (micrographs)
    ebsd_palette: dict[str, Any] | None = None
    ebsd_step_um: float | None = None
    magnification: float | None = None
    scale_bar_um: float | None = None
    scale_bar_px: float | None = None
    pixel_size_um: float | None = None
    scale_uncertainty_rel: float | None = None


@dataclass
class PhaseFractions:
    """Phase fractions as reported by the paper, stored as fractions 0..1."""
    ferrite: float | None = None
    austenite: float | None = None
    sigma: float | None = None
    method: str | None = None            # one of PF_METHODS
    scope: str | None = None             # one of PF_SCOPES
    sd: float | None = None              # fraction units, like the values
    location_in_paper: str | None = None


@dataclass
class Extraction:
    method: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    boundary_policy: str | None = None
    fractions: dict[str, float] = field(default_factory=dict)
    ignored_fraction: float | None = None
    sensitivity: dict[str, Any] = field(default_factory=dict)


@dataclass
class Record:
    sample_id: str
    paths: dict[str, str | None] = field(default_factory=dict)
    material: str = "2507"
    process: str | None = None
    condition: str | None = None
    build_plane: str | None = None
    image: ImageInfo = field(default_factory=ImageInfo)
    source: Source = field(default_factory=Source)
    reported: PhaseFractions = field(default_factory=PhaseFractions)
    extracted: Extraction | None = None
    abs_error_pp: dict[str, float] | None = None
    qc_flags: list[str] = field(default_factory=list)
    properties: Properties = field(default_factory=Properties)
    split: str | None = None
    cv_fold: int | None = None
    legacy: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate_sample_v1(s: dict[str, Any]) -> dict[str, Any]:
    """Map one v1 sample dict to a v2 record dict. Lossless: anything without a
    v2 home goes into `legacy`."""
    mat = s.get("material") or {}
    proc = s.get("process") or {}
    pf = s.get("phase_fractions_ebsd") or {}
    q = s.get("quality") or {}
    src = s.get("source") or {}
    if s.get("ang_path"):
        image_type = "ebsd_phase"
    elif s.get("modality") == "optical":
        image_type = "om"
    else:
        image_type = "sem"
    has_pf = any(pf.get(k) is not None for k in ("ferrite", "austenite", "sigma"))
    rec = Record(
        sample_id=s["sample_id"],
        paths={"original": s.get("sem_path") or s.get("optical_path"),
               "mask": s.get("mask_path"), "ang": s.get("ang_path")},
        material=mat.get("alloy", "SDSS_2507"),
        process=proc.get("route"),
        condition=mat.get("condition"),
        image=ImageInfo(image_type=image_type,
                        magnification=s.get("magnification") or None,
                        pixel_size_um=s.get("pixel_size_um") or None),
        source=Source(**{k: src.get(k) for k in asdict(Source()) if k in src}),
        # v1 fractions came from our own .ang files -> they are EBSD ground truth.
        reported=PhaseFractions(ferrite=pf.get("ferrite"), austenite=pf.get("austenite"),
                                sigma=pf.get("sigma"),
                                method="EBSD" if has_pf else None,
                                scope="this_panel" if has_pf else None),
        qc_flags=([f"excluded:{q.get('exclude_reason')}"] if q.get("exclude") else []),
        properties=Properties(**(s.get("properties") or {})),
        split=s.get("split"),
        cv_fold=s.get("cv_fold"),
        legacy={"schema_version": 1, "ang_path": s.get("ang_path"),
                "mask_meta_path": s.get("mask_meta_path"), "modality": s.get("modality"),
                "material": mat, "process": proc, "quality": q,
                "unindexed": pf.get("unindexed")},
    )
    return asdict(rec)


def migrate_v1_to_v2(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise ValueError(f"migrate_v1_to_v2 expects schema v1, got {payload.get('schema_version')}")
    return {"schema_version": 2,
            "samples": [migrate_sample_v1(s) for s in payload.get("samples", [])]}


def _as_v2_dict(obj: Sample | Record | dict[str, Any]) -> dict[str, Any]:
    if isinstance(obj, Sample):
        return migrate_sample_v1(asdict(obj))
    if isinstance(obj, Record):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj if "image" in obj else migrate_sample_v1(obj)
    raise TypeError(f"Cannot store {type(obj).__name__} in manifest")


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _write(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump({"schema_version": MANIFEST_SCHEMA_VERSION, "samples": samples},
                  f, indent=2, default=str)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written manifest


def save_manifest(samples: Iterable[Sample | Record | dict[str, Any]], path: str | Path) -> None:
    _write(Path(path), [_as_v2_dict(s) for s in samples])


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Return v2 record dicts. v1 files are migrated in memory (file untouched)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    version = payload.get("schema_version")
    if version == 1:
        payload = migrate_v1_to_v2(payload)
    elif version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Manifest {path} uses schema v{version} but code expects "
            f"v{MANIFEST_SCHEMA_VERSION} (or v1, which is migrated). Add a migration."
        )
    return payload["samples"]


def get_record(manifest_path: str | Path, sample_id: str) -> dict[str, Any] | None:
    p = Path(manifest_path)
    if not p.exists():
        return None
    return next((s for s in load_manifest(p) if s["sample_id"] == sample_id), None)


def add_or_update_sample(manifest_path: str | Path,
                         sample: Sample | Record | dict[str, Any]) -> None:
    """Idempotent upsert by sample_id. Safe to call from a processing loop."""
    manifest_path = Path(manifest_path)
    samples = load_manifest(manifest_path) if manifest_path.exists() else []
    new = _as_v2_dict(sample)
    for i, s in enumerate(samples):
        if s["sample_id"] == new["sample_id"]:
            samples[i] = new
            break
    else:
        samples.append(new)
    _write(manifest_path, samples)


upsert_record = add_or_update_sample
