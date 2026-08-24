"""Structured logging.

Everything that writes to stdout also writes to reports/logs/<run_name>.log
so viva questions like "what settings did you use in this experiment" have a
one-file answer.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def get_logger(
    name: str = "sdss",
    log_dir: Path | str = "reports/logs",
    run_name: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a logger that writes to stdout AND to a timestamped file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called twice in same session (e.g. in a notebook).
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_dir / f"{run_name}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("Logger initialized. Log file: %s", log_dir / f"{run_name}.log")
    return logger
