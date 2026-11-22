"""Run ingest -> crop -> scale -> masks -> validation -> features -> report for one paper or all papers.

    python scripts/run_paper_pipeline.py --all --config configs/data/papers.yaml
    python scripts/run_paper_pipeline.py --paper <paper_id>
    python scripts/run_paper_pipeline.py --all --make-fixture   # first renders the synthetic
        fixture paper (tests/fixtures/synthetic_paper) into raw/papers/synthetic_fixture

Papers whose meta.yaml has errors are skipped (logged); a failing panel is logged
and the run continues. Exit code 1 if anything was skipped or failed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import pipeline as P  # noqa: E402
from src.utils.cli import base_parser, setup  # noqa: E402

FIXTURE_META = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic_paper" / "meta.yaml"


def main() -> int:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--make-fixture", action="store_true")
    args = p.parse_args()
    cfg, h, log, paths = setup(args, "run_paper_pipeline")
    if args.make_fixture:
        from src.data.synthetic import make_fixture_paper
        out = make_fixture_paper(FIXTURE_META, paths["papers"] / "synthetic_fixture", int(cfg.get("seed", 42)))
        log.info("Synthetic fixture paper written to %s", out)

    dirs = P.paper_dirs(paths, args.paper)
    if not dirs:
        log.warning("No papers with meta.yaml under %s", paths["papers"])
        return 1
    bad = []
    ids: list[str] = []
    for d in dirs:
        log.info("=== paper %s", d.name)
        if not P.stage_ingest(d, cfg, paths, log):
            log.error("Skipping %s: fix meta.yaml errors above", d.name)
            bad.append(d.name)
            continue
        ctxs = P.select_contexts(paths, d.name)
        ids += [c["sample_id"] for c in ctxs]
        failed = set(P.run_stage_over(ctxs, P.stage_crop, log, cfg, paths, log))
        ok = [c for c in ctxs if c["sample_id"] not in failed]
        failed |= set(P.run_stage_over(ok, P.stage_scale, log, cfg, paths, log))
        failed |= set(P.run_stage_over([c for c in ok if c["sample_id"] not in failed],
                                       P.stage_mask, log, cfg, paths, h, log))
        P.run_stage_over(ok, P.stage_validate, log, cfg, paths, log)   # flags unsegmented panels too
        bad += sorted(failed)
    P.stage_features(cfg, paths, log, ids)
    P.report(cfg, paths, h, log, ids)
    log.info("Done. %d samples, %d skipped/failed: %s", len(ids), len(bad), bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
