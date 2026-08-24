"""Dataset manifest — the single source of truth.

A manifest is a JSON file listing every sample in the dataset. It links:
  - the raw .ang file (if present)
  - the raw SEM/optical micrograph
  - the derived phase mask
  - any per-sample metadata (composition, heat treatment, magnification)
  - property labels (tensile strength, hardness, corrosion current, ...)
  - a data-quality flag (e.g., "EBSD indexing quality low, exclude from training")

Every downstream loader reads the manifest. `data/raw/` and `data/interim/`
should never be walked directly by training code.

Schema (per sample):
{
  "sample_id":        str,      # unique, e.g. "asbuilt_LP150_SS600_v001"
  "ang_path":         str|None, # relative to data root, e.g. "raw/ang/foo.ang"
  "sem_path":         str|None,
  "optical_path":     str|None,
  "mask_path":        str|None, # derived by preprocess_ang.py
  "mask_meta_path":   str|None, # sidecar JSON with CI thresholds, class counts

  "modality":         str,      # "sem" | "optical"   -- which image the model sees
  "magnification":    float,    # e.g. 500.0
  "pixel_size_um":    float,    # microns per pixel (from calibration bar OR .ang step)

  "material": {
    "alloy":          str,      # "SDSS_2507"
    "composition":    dict,     # {"Cr": 25.0, "Ni": 7.0, ...}   optional
    "condition":      str,      # "as_built" | "solution_annealed_1100C_1h" | "aged_850C_30min"
  },
  "process": {
    "route":          str,      # "LPBF" | "wrought" | "DED"
    "laser_power_W":  float|None,
    "scan_speed_mm_s":float|None,
    "hatch_um":       float|None,
    "layer_um":       float|None,
    "heat_treatment": str|None, # free text override of material.condition
  },
  "properties": {
    "tensile_MPa":    float|None,
    "yield_MPa":      float|None,
    "elongation_pct": float|None,
    "hardness_HV":    float|None,
    "corrosion_uA":   float|None,
  },
  "phase_fractions_ebsd": {     # ground truth from .ang, populated by preprocess
    "ferrite":        float|None,
    "austenite":      float|None,
    "sigma":          float|None,
    "unindexed":      float|None,
  },
  "quality": {
    "ebsd_ci_mean":       float|None,  # mean confidence index of .ang
    "ebsd_indexed_pct":   float|None,
    "exclude":            bool,
    "exclude_reason":     str|None,
  },
  "split":              str|None,   # "train" | "val" | "test" | null   (set by Phase 2 CV script)
  "cv_fold":            int|None,   # 0..K-1 for k-fold CV
  "source": {
    "provenance":     str,          # "own" | "paper:doi..." | "dataset:MetalDAM"
    "citation":       str|None,
    "notes":          str|None,
  }
}
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


# Schema version so future changes can be handled with migration functions,
# not silent breakage.
MANIFEST_SCHEMA_VERSION = 1


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
class PhaseFractions:
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
    provenance: str = "own"
    citation: str | None = None
    notes: str | None = None


@dataclass
class Sample:
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
    phase_fractions_ebsd: PhaseFractions = field(default_factory=PhaseFractions)
    quality: Quality = field(default_factory=Quality)
    split: str | None = None
    cv_fold: int | None = None
    source: Source = field(default_factory=Source)


def save_manifest(samples: Iterable[Sample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "samples": [asdict(s) for s in samples],
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load a manifest and return the sample dicts. We use dicts (not Sample
    instances) downstream because dataloaders index by column, not by object."""
    path = Path(path)
    with path.open("r") as f:
        payload = json.load(f)
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Manifest {path} uses schema v{payload.get('schema_version')} "
            f"but code expects v{MANIFEST_SCHEMA_VERSION}. Add a migration."
        )
    return payload["samples"]


def add_or_update_sample(manifest_path: str | Path, sample: Sample) -> None:
    """Idempotent upsert by sample_id. Safe to call from a preprocessing loop."""
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        samples = load_manifest(manifest_path)
    else:
        samples = []
    # Replace if sample_id exists, else append.
    for i, s in enumerate(samples):
        if s["sample_id"] == sample.sample_id:
            samples[i] = asdict(sample)
            break
    else:
        samples.append(asdict(sample))
    payload = {"schema_version": MANIFEST_SCHEMA_VERSION, "samples": samples}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
