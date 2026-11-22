"""Measure scale bars (on original figure pixels) and store um/px + uncertainty in the manifest.

    python scripts/calibrate_scale.py --paper <paper_id> | --all
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import pipeline as P  # noqa: E402
from src.utils.cli import base_parser, setup  # noqa: E402


def main() -> int:
    args = base_parser(__doc__.splitlines()[0]).parse_args()
    cfg, h, log, paths = setup(args, "calibrate_scale")
    ctxs = P.select_contexts(paths, args.paper)
    return 1 if P.run_stage_over(ctxs, P.stage_scale, log, cfg, paths, log) else 0


if __name__ == "__main__":
    sys.exit(main())
