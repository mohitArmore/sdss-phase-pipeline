# SDSS Phase Pipeline

Automated phase-fraction estimation and property prediction for LPBF-fabricated
2507 super-duplex stainless steel.

**Pipeline:** `.ang → phase mask → SEM segmentation → features → property prediction → uncertainty`

## Design principle

Every choice is made for **small-data first, scalable later**. When more labeled
data arrives, no code is rewritten — only config values change (see `configs/`).

## Directory layout

```
sdss_phase_pipeline/
├── data/
│   ├── raw/                   # Immutable inputs, never modified
│   │   ├── ang/               # Raw .ang files from EBSD
│   │   ├── sem/               # Raw SEM micrographs (paired with .ang)
│   │   └── optical/           # Raw optical micrographs
│   ├── interim/               # Intermediate artifacts
│   │   ├── masks/             # Binary phase masks derived from .ang
│   │   └── registered/        # SEM aligned to EBSD frame
│   ├── processed/             # Ready-to-train
│   │   ├── patches/           # Cropped training patches (image + mask)
│   │   └── manifests/         # JSON manifests linking everything
│   └── external/              # Public datasets (MetalDAM, orix-data, etc.)
├── src/
│   ├── data/                  # Loaders, .ang parsing, mask generation, registration
│   ├── models/                # Segmentation architectures (Phase 2)
│   ├── training/              # Train loops, CV, callbacks (Phase 2)
│   ├── evaluation/            # Metrics, visualization (Phase 2)
│   ├── features/              # Feature extraction (Phase 4)
│   ├── uncertainty/           # MC dropout, ensembles, conformal (Phase 3/4)
│   └── utils/                 # Logging, config loading, seed control
├── configs/                   # Hydra-style YAML configs, all knobs live here
├── notebooks/                 # Exploratory work only, never source of truth
├── reports/                   # Figures and tables for the paper
├── scripts/                   # One-shot CLI entry points
└── tests/                     # Unit tests for src/
```

## Reproducibility contract

1. `data/raw/` is **never modified** by any script. All processing writes to `interim/` or `processed/`.
2. Every experiment is driven by a YAML config in `configs/`. No hardcoded paths or hyperparameters in `src/`.
3. `requirements.txt` pins versions. `python -m pip install -r requirements.txt` reproduces the environment on Colab or locally.
4. Random seeds are set in `src/utils/seed.py` and threaded through every dataloader and model.
5. Results are logged to `reports/` with the config hash so any figure can be traced back to the exact code + config that produced it.

## Quick start

```bash
# 1. Clone
git clone <your-repo-url> && cd sdss_phase_pipeline

# 2. Install
pip install -r requirements.txt

# 3. Drop .ang files into data/raw/ang/ and matching SEM images into data/raw/sem/

# 4. Preprocess (generate masks and manifest)
python scripts/preprocess_ang.py --config configs/data/default.yaml

# 5. Later phases:
# python scripts/train.py --config configs/training/phase2_baseline.yaml
```

See `docs/` (to be added in Phase 6) for the full methodology.
