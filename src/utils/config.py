"""Config loading with hash-tracking.

Every artifact produced by the pipeline (mask, model checkpoint, feature CSV)
should be paired with the SHA of the config that produced it. This makes viva
defense simple: "figure X was generated from run Y, config hash Z".
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    path = Path(path)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config file {path} is empty or malformed")
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Deterministic short hash of a config (first 10 chars of SHA-256)."""
    payload = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]


def save_run_config(cfg: dict[str, Any], run_dir: str | Path) -> str:
    """Write the resolved config into the run directory and return its hash."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    h = config_hash(cfg)
    out = run_dir / f"config_{h}.yaml"
    with out.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    return h
