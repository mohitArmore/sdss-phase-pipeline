"""Segment cropped panels into class masks (0 aus, 1 fer, 2 sigma, 255 ignore) + QC overlays.

    python scripts/make_masks.py --sample <sample_id> | --paper <paper_id> | --all
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import pipeline as P  # noqa: E402
from src.utils.cli import base_parser, setup  # noqa: E402


def main() -> int:
    args = base_parser(__doc__.splitlines()[0], sample=True).parse_args()
    cfg, h, log, paths = setup(args, "make_masks")
    ctxs = P.select_contexts(paths, args.paper, args.sample)
    return 1 if P.run_stage_over(ctxs, P.stage_mask, log, cfg, paths, h, log) else 0


if __name__ == "__main__":
    sys.exit(main())
