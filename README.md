# SDSS Phase Pipeline

Automated phase-fraction estimation and property prediction for LPBF-fabricated
2507 super-duplex stainless steel.

**Pipeline:** `paper figure → crop → class mask → features → validation` (→ property prediction → uncertainty, later phases)

The dataset is built from figures in published papers (EBSD phase maps, etched
optical and SEM micrographs; PNG/JPG). Phase fractions (and region size / aspect
ratio) are extracted by image analysis and validated against the fraction each
paper reports. **Step-by-step manual workflow: [docs/PAPER_DATASET_GUIDE.md](docs/PAPER_DATASET_GUIDE.md).**

The earlier `.ang → mask` path is kept, importable, under `src/legacy/` and
`scripts/legacy/preprocess_ang.py`.

## Design principle

Every choice is made for **small-data first, scalable later**. When more labeled
data arrives, no code is rewritten — only config values change (see `configs/`).

## Directory layout

```
sdss-phase-pipeline/
├── configs/data/papers.yaml   # ALL tunables for the paper-figure pipeline
├── templates/                 # paper_meta_template.yaml (human-filled registry entry)
├── data/
│   ├── raw/papers/<paper_id>/ # meta.yaml (committed) + figures/ (never committed, never modified)
│   ├── interim/               # ingest/, previews/, crops/, masks/, overlays/   (PNG only, not committed)
│   ├── processed/
│   │   ├── manifests/         # manifest.json, schema v2 (committed)
│   │   └── features/          # features.csv (committed)
│   └── LICENSES.md            # per-paper licence / redistribution table
├── src/
│   ├── io/image_io.py         # load_image (format, is_jpeg, JPEG quality), save_png, sha256_file
│   ├── data/                  # manifest (v2), ingest_paper, crop, annotations, scalebar, synthetic
│   ├── segment/               # ebsd_colour_to_mask, micrograph_segment
│   ├── features/features.py   # phase fractions, region stats, boundary density
│   ├── eval/validate.py       # errors vs reported, tolerances, sensitivity, QC flags, report
│   ├── pipeline.py            # one function per stage, used by all scripts
│   ├── legacy/                # .ang → mask, SEM↔EBSD registration (old path)
│   └── utils/                 # config hash, logging, seed, CLI setup
├── scripts/                   # ingest_paper, crop_panels, calibrate_scale, make_masks,
│                              # extract_features, validate, run_paper_pipeline, legacy/
├── reports/validation/        # summary.csv, summary.md, bland_altman.png
├── docs/PAPER_DATASET_GUIDE.md
└── tests/                     # synthetic-image tests + fixtures/synthetic_paper/meta.yaml
```

Mask classes: `0 = austenite, 1 = ferrite, 2 = sigma (optional), 255 = ignore`.

## Reproducibility contract

1. `data/raw/` is **never modified** by any processing script; all processing writes to `interim/` or `processed/`. The only writers into `raw/` are acquisition helpers that *add* files: `ingest_paper.py --extract-pdf` (original embedded bytes) and `run_paper_pipeline.py --make-fixture`. Originals are SHA-256-hashed at ingest and re-checked on every ingest.
2. Every experiment is driven by a YAML config in `configs/`. No hardcoded paths or hyperparameters in `src/`.
3. `requirements.txt` gives lower bounds (`>=`). `python -m pip install -r requirements.txt` reproduces the environment on Colab or locally.
4. Random seeds are set in `src/utils/seed.py` and threaded through every dataloader and model.
5. Results are logged to `reports/` with the config hash so any figure can be traced back to the exact code + config that produced it.

## Quick start

```bash
pip install -r requirements.txt
pytest                                   # synthetic-image tests, no real data needed

# Try the whole pipeline on the synthetic fixture paper (scratch data root):
python scripts/run_paper_pipeline.py --all --make-fixture --data-root /tmp/sdss_demo --reports-dir /tmp/sdss_demo/reports

# Real papers (see docs/PAPER_DATASET_GUIDE.md):
python scripts/ingest_paper.py --paper-dir data/raw/papers/<paper_id> --preview
python scripts/run_paper_pipeline.py --all --config configs/data/papers.yaml
```

On Colab: `!python scripts/colab_bootstrap.py --data-root /content/drive/MyDrive/BTP/data`
(installs requirements + poppler-utils, links Drive data). CPU runtime is enough.
