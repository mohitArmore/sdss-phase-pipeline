"""Shared CLI plumbing for scripts/: config + hash, seed, logger, paths."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils.config import config_hash, load_config
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed


def base_parser(description: str, selection: bool = True, sample: bool = False) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, default=Path("configs/data/papers.yaml"))
    p.add_argument("--data-root", type=Path, default=None,
                   help="Override paths.data_root from the config.")
    p.add_argument("--reports-dir", type=Path, default=None,
                   help="Override paths.reports_dir from the config.")
    if selection:
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--paper", help="paper_id (folder name under raw/papers)")
        g.add_argument("--all", action="store_true", help="every paper with a meta.yaml")
        if sample:
            g.add_argument("--sample", help="a single sample_id")
    return p


def setup(args: argparse.Namespace, name: str) -> tuple[dict[str, Any], str, Any, dict[str, Path]]:
    """Load config, seed, logger; return (cfg, hash, log, paths)."""
    cfg = load_config(args.config)
    h = config_hash(cfg)
    set_global_seed(int(cfg.get("seed", 42)), deterministic_torch=False)
    pc = cfg.get("paths", {})
    root = Path(args.data_root or pc.get("data_root", "data"))
    reports = Path(getattr(args, "reports_dir", None) or pc.get("reports_dir", "reports/validation"))
    paths = {
        "data_root": root,
        "papers": root / pc.get("papers", "raw/papers"),
        "interim": root / pc.get("interim", "interim"),
        "manifest": root / pc.get("manifest", "processed/manifests/manifest.json"),
        "features": root / pc.get("features", "processed/features/features.csv"),
        "reports": reports,
        "logs": Path(pc.get("logs_dir", "reports/logs")),
    }
    log = get_logger(f"sdss.{name}", log_dir=paths["logs"], run_name=f"{name}_{h}")
    log.info("config=%s hash=%s data_root=%s", args.config, h, root)
    return cfg, h, log, paths
