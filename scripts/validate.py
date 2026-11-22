"""Compare extracted vs reported phase fractions, run sensitivity checks, write reports/validation/.

    python scripts/validate.py --paper <paper_id> | --all
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import pipeline as P  # noqa: E402
from src.utils.cli import base_parser, setup  # noqa: E402


def main() -> int:
    args = base_parser(__doc__.splitlines()[0]).parse_args()
    cfg, h, log, paths = setup(args, "validate")
    ctxs = P.select_contexts(paths, args.paper)
    failed = P.run_stage_over(ctxs, P.stage_validate, log, cfg, paths, log)
    P.report(cfg, paths, h, log, [c["sample_id"] for c in ctxs])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
