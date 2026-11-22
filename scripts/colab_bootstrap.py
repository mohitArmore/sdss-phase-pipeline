"""
Colab bootstrap.

Paste this into the first cell of any Colab notebook using this project,
or run: `!python scripts/colab_bootstrap.py`

What it does:
  1. Installs requirements.txt (+ poppler-utils for `pdfimages`; non-fatal if apt fails,
     PyMuPDF is the fallback). Skip with --skip-install.
  2. Links the Drive data folder into the repo (Colab's local disk is ephemeral).
  3. Reports GPU (not needed for the paper-figure pipeline — CPU runtime is fine).
  4. Sets deterministic seeds.

Usage in Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    !git clone https://github.com/<you>/sdss_phase_pipeline /content/repo
    %cd /content/repo
    !python scripts/colab_bootstrap.py --data-root /content/drive/MyDrive/BTP/data
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure the src package is importable — same trick used in scripts/preprocess_ang.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def install_deps(repo_root: Path) -> None:
    """pip install requirements; apt-get poppler-utils (pdfimages). Failures are non-fatal."""
    req = repo_root / "requirements.txt"
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])
    print(f"[bootstrap] pip install -r requirements.txt -> exit {r.returncode}")
    if shutil.which("pdfimages"):
        print("[bootstrap] pdfimages already available.")
        return
    try:
        r = subprocess.run("apt-get -qq update && apt-get -qq install -y poppler-utils",
                           shell=True, timeout=600)
        ok = r.returncode == 0 and shutil.which("pdfimages")
    except Exception as e:  # noqa: BLE001 — not on Colab / no root: fall back to PyMuPDF
        print(f"[bootstrap] apt-get failed: {e}")
        ok = False
    print("[bootstrap] pdfimages " + ("installed." if ok else "unavailable — PyMuPDF will be used."))


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
        print("[bootstrap] No GPU detected (fine for the paper-figure pipeline; needed later for training).")


def link_drive_data(data_root: Path, repo_data_dir: Path) -> None:
    """Symlink the Drive data folder into the repo so paths in configs stay portable."""
    if not data_root.exists():
        print(f"[bootstrap] Data root {data_root} does not exist yet. Creating.")
        data_root.mkdir(parents=True, exist_ok=True)
        for sub in ["raw/papers", "interim", "processed/manifests", "processed/features",
                    "processed/patches", "external"]:
            (data_root / sub).mkdir(parents=True, exist_ok=True)
    if repo_data_dir.exists() and not repo_data_dir.is_symlink():
        # Repo's data/ holds committed meta.yaml / manifest / CSVs: copy any file missing
        # on Drive first (never overwrite Drive), then replace data/ with a symlink.
        for f in repo_data_dir.rglob("*"):
            dst = data_root / f.relative_to(repo_data_dir)
            if f.is_file() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
        shutil.rmtree(repo_data_dir)
    if not repo_data_dir.exists():
        repo_data_dir.symlink_to(data_root)
        print(f"[bootstrap] Linked {repo_data_dir}  →  {data_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/content/drive/MyDrive/BTP/data"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()

    print(f"[bootstrap] Python: {sys.version.split()[0]}")
    print(f"[bootstrap] Repo:   {args.repo_root}")

    if not args.skip_install:
        install_deps(args.repo_root)
    link_drive_data(args.data_root, args.repo_root / "data")
    check_gpu()

    # Deterministic seeds (matches src/utils/seed.py)
    from src.utils.seed import set_global_seed
    set_global_seed(42)
    print("[bootstrap] Seeds set (42). Ready.")


if __name__ == "__main__":
    main()
