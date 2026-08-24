"""
Colab bootstrap.

Paste this into the first cell of any Colab notebook using this project,
or run: `!python scripts/colab_bootstrap.py`

What it does:
  1. Mounts Google Drive (data lives there — Colab's local disk is ephemeral).
  2. Clones the repo if not present.
  3. Installs pinned requirements.
  4. Verifies GPU availability and reports VRAM.
  5. Sets deterministic seeds.

Usage in Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    !git clone https://github.com/<you>/sdss_phase_pipeline /content/repo
    %cd /content/repo
    !pip install -q -r requirements.txt
    !python scripts/colab_bootstrap.py --data-root /content/drive/MyDrive/BTP/data
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def check_gpu() -> None:
    try:
        import torch
    except ImportError:
        print("[bootstrap] PyTorch not installed yet. Run `pip install -r requirements.txt` first.")
        return
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[bootstrap] GPU: {name}  |  VRAM: {vram_gb:.1f} GB  |  torch {torch.__version__}")
        if vram_gb < 14:
            print("[bootstrap] WARNING: <14 GB VRAM detected. Reduce batch_size in configs/.")
    else:
        print("[bootstrap] No GPU detected. Training will be extremely slow.")
        print("[bootstrap] In Colab: Runtime → Change runtime type → T4 GPU.")


def link_drive_data(data_root: Path, repo_data_dir: Path) -> None:
    """Symlink the Drive data folder into the repo so paths in configs stay portable."""
    if not data_root.exists():
        print(f"[bootstrap] Data root {data_root} does not exist yet. Creating.")
        data_root.mkdir(parents=True, exist_ok=True)
        for sub in ["raw/ang", "raw/sem", "raw/optical",
                    "interim/masks", "interim/registered",
                    "processed/patches", "processed/manifests",
                    "external"]:
            (data_root / sub).mkdir(parents=True, exist_ok=True)
    if repo_data_dir.exists() and not repo_data_dir.is_symlink():
        # Repo shipped with an empty data/ dir; replace it with a symlink to Drive.
        import shutil
        shutil.rmtree(repo_data_dir)
    if not repo_data_dir.exists():
        repo_data_dir.symlink_to(data_root)
        print(f"[bootstrap] Linked {repo_data_dir}  →  {data_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/content/drive/MyDrive/BTP/data"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    print(f"[bootstrap] Python: {sys.version.split()[0]}")
    print(f"[bootstrap] Repo:   {args.repo_root}")

    link_drive_data(args.data_root, args.repo_root / "data")
    check_gpu()

    # Deterministic seeds (matches src/utils/seed.py)
    from src.utils.seed import set_global_seed
    set_global_seed(42)
    print("[bootstrap] Seeds set (42). Ready.")


if __name__ == "__main__":
    main()
