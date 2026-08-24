"""Preprocess raw .ang files into phase masks and register any paired SEM images.

Usage
-----
    python scripts/preprocess_ang.py --config configs/data/default.yaml

What it does
------------
1. Walks `data/raw/ang/` and finds every .ang file.
2. For each .ang:
   a. Runs the .ang → mask conversion (src.data.ang_to_mask).
   b. Saves the mask as PNG under `data/interim/masks/`, with a JSON sidecar
      containing all preprocessing parameters and computed phase fractions.
   c. If a paired SEM image exists (same stem in data/raw/sem/), registers
      it to the mask and saves the aligned version to data/interim/registered/.
   d. Upserts a row into the manifest at data/processed/manifests/manifest.json.

Reruns are IDEMPOTENT: only files whose .ang mtime is newer than the mask are
re-processed. Use `--force` to reprocess everything.

Design choices explained
------------------------
- We DON'T combine mask generation and registration into one function because
  a lot of the time you have EBSD without a paired SEM (still useful ground
  truth for phase fractions) or SEM without EBSD (needs pseudo-labels).
- We keep raw/, interim/, and processed/ separated for the reproducibility
  contract in README.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the src package is importable when run as a script from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ang_to_mask import MaskingConfig, ang_to_mask, save_mask   # noqa: E402
from src.data.manifest import (                                          # noqa: E402
    Material, Process, Quality, Sample, Source, add_or_update_sample,
    PhaseFractions,
)
from src.data.registration import register                               # noqa: E402
from src.utils.config import config_hash, load_config                    # noqa: E402
from src.utils.logging import get_logger                                 # noqa: E402
from src.utils.seed import set_global_seed                               # noqa: E402


def find_paired_image(ang_path: Path, sem_dir: Path, optical_dir: Path) -> tuple[Path | None, str]:
    """Look for a SEM or optical image with the same stem as the .ang file.
    Returns (path_or_None, modality)."""
    stem = ang_path.stem
    for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"):
        p = sem_dir / f"{stem}{ext}"
        if p.exists():
            return p, "sem"
    for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"):
        p = optical_dir / f"{stem}{ext}"
        if p.exists():
            return p, "optical"
    return None, "sem"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/data/default.yaml"))
    p.add_argument("--data-root", type=Path, default=Path("data"),
                   help="Root of the data tree (contains raw/, interim/, processed/).")
    p.add_argument("--force", action="store_true",
                   help="Reprocess every .ang even if mask exists and is newer.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-registration", action="store_true",
                   help="Only build masks, do not align SEM images.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    log = get_logger("preprocess_ang", run_name="preprocess_ang")

    cfg = load_config(args.config)
    h = config_hash(cfg)
    log.info("Loaded config from %s (hash=%s)", args.config, h)

    # Resolve preprocessing parameters (defaults + overrides from YAML)
    mask_cfg = MaskingConfig(**cfg.get("masking", {}))

    ang_dir = args.data_root / "raw" / "ang"
    sem_dir = args.data_root / "raw" / "sem"
    optical_dir = args.data_root / "raw" / "optical"
    mask_dir = args.data_root / "interim" / "masks"
    reg_dir = args.data_root / "interim" / "registered"
    manifest_path = args.data_root / "processed" / "manifests" / "manifest.json"

    for d in (mask_dir, reg_dir, manifest_path.parent):
        d.mkdir(parents=True, exist_ok=True)

    ang_files = sorted(ang_dir.glob("*.ang"))
    if not ang_files:
        log.warning("No .ang files found in %s — nothing to do.", ang_dir)
        return

    log.info("Found %d .ang files. Registration=%s. Force=%s.",
             len(ang_files), not args.skip_registration, args.force)

    n_processed = 0
    n_skipped = 0
    n_reg_failed = 0

    for ang_path in ang_files:
        stem = ang_path.stem
        mask_path = mask_dir / f"{stem}.png"
        meta_path = mask_dir / f"{stem}.json"

        # Idempotence check
        if not args.force and mask_path.exists() and mask_path.stat().st_mtime > ang_path.stat().st_mtime:
            log.info("[skip] %s (mask up-to-date)", stem)
            n_skipped += 1
            continue

        # --- 1. .ang → mask
        try:
            result = ang_to_mask(ang_path, mask_cfg)
        except Exception as e:
            log.error("[fail] %s: .ang parsing raised %s: %s", stem, type(e).__name__, e)
            continue
        save_mask(result, mask_path, meta_path)
        log.info(
            "[mask] %s  shape=%s  step=%.3fum  fractions=%s  CI_mean=%.3f",
            stem, result.shape, result.step_size_um, result.phase_fractions,
            result.ci_stats["mean"],
        )

        # --- 2. Register paired image (if any)
        paired_path, modality = find_paired_image(ang_path, sem_dir, optical_dir)
        aligned_path: Path | None = None
        if paired_path is not None and not args.skip_registration:
            try:
                reg = register(paired_path, mask_path, method="auto")
                aligned_path = reg_dir / f"{stem}.png"
                import cv2
                cv2.imwrite(str(aligned_path), reg.aligned_image)
                log.info("[reg]  %s  method=%s  ok=%s", stem, reg.method, reg.ok)
                if not reg.ok:
                    n_reg_failed += 1
            except Exception as e:
                log.error("[reg fail] %s: %s", stem, e)
                n_reg_failed += 1

        # --- 3. Manifest upsert
        sample = Sample(
            sample_id=stem,
            ang_path=str(ang_path.relative_to(args.data_root)),
            sem_path=str(paired_path.relative_to(args.data_root)) if (paired_path and modality == "sem") else None,
            optical_path=str(paired_path.relative_to(args.data_root)) if (paired_path and modality == "optical") else None,
            mask_path=str(mask_path.relative_to(args.data_root)),
            mask_meta_path=str(meta_path.relative_to(args.data_root)),
            modality=modality,
            pixel_size_um=result.step_size_um,
            material=Material(alloy="SDSS_2507", condition="unknown"),
            process=Process(route="LPBF"),
            phase_fractions_ebsd=PhaseFractions(
                ferrite=result.phase_fractions.get("ferrite"),
                austenite=result.phase_fractions.get("austenite"),
                sigma=result.phase_fractions.get("sigma"),
                unindexed=result.phase_fractions.get("unindexed"),
            ),
            quality=Quality(
                ebsd_ci_mean=result.ci_stats["mean"],
                ebsd_indexed_pct=result.ci_stats["pct_above_thresh"],
                exclude=False,
            ),
            source=Source(provenance="own"),
        )
        add_or_update_sample(manifest_path, sample)
        n_processed += 1

    log.info("Done. Processed=%d  Skipped=%d  RegistrationFailed=%d",
             n_processed, n_skipped, n_reg_failed)
    log.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
