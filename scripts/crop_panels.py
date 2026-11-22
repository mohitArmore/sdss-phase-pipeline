"""Crop every panel listed in meta.yaml (crop_box + ignore_boxes) and upsert manifest records.

    python scripts/crop_panels.py --paper <paper_id> | --all [--config configs/data/papers.yaml]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import pipeline as P  # noqa: E402
from src.utils.cli import base_parser, setup  # noqa: E402


def main() -> int:
    args = base_parser(__doc__.splitlines()[0]).parse_args()
    cfg, h, log, paths = setup(args, "crop_panels")
    ctxs = P.select_contexts(paths, args.paper)
    return 1 if P.run_stage_over(ctxs, P.stage_crop, log, cfg, paths, log) else 0


if __name__ == "__main__":
    sys.exit(main())
