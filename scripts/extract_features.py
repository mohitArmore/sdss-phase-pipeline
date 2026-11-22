"""Compute phase fractions, region/grain stats and boundary density -> processed/features/features.csv.

    python scripts/extract_features.py --paper <paper_id> | --all
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import pipeline as P  # noqa: E402
from src.utils.cli import base_parser, setup  # noqa: E402


def main() -> int:
    args = base_parser(__doc__.splitlines()[0]).parse_args()
    cfg, h, log, paths = setup(args, "extract_features")
    ids = [c["sample_id"] for c in P.select_contexts(paths, args.paper)]
    P.stage_features(cfg, paths, log, ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
