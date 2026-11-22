"""Ingest a paper folder: validate meta.yaml, hash originals, extract PDF images, write grid previews.

Usage
-----
    python scripts/ingest_paper.py --paper-dir data/raw/papers/<paper_id>
    python scripts/ingest_paper.py --paper-dir data/raw/papers/<paper_id> --preview
    python scripts/ingest_paper.py --paper-dir data/raw/papers/<paper_id> --extract-pdf paper.pdf

--extract-pdf only needs the paper folder to exist (no meta.yaml yet): images go
to <paper-dir>/figures/extracted/ with listing.csv. Pick the figures you need,
then write meta.yaml (templates/paper_meta_template.yaml).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ingest_paper import extract_pdf_images, ingest_paper_dir, write_previews  # noqa: E402
from src.utils.cli import base_parser, setup  # noqa: E402


def main() -> int:
    p = base_parser(__doc__.splitlines()[0], selection=False)
    p.add_argument("--paper-dir", type=Path, required=True)
    p.add_argument("--extract-pdf", type=Path, default=None)
    p.add_argument("--preview", action="store_true")
    args = p.parse_args()
    cfg, h, log, paths = setup(args, "ingest_paper")

    if args.extract_pdf:
        listing = extract_pdf_images(args.extract_pdf, args.paper_dir / "figures" / "extracted")
        log.info("Extracted PDF images -> %s", listing)
        if not (args.paper_dir / "meta.yaml").exists():
            return 0

    res = ingest_paper_dir(args.paper_dir, paths["interim"] / "ingest")
    if not cfg.get("io", {}).get("allow_jpeg", True):
        res["errors"] += [f"{f['file']}: JPEG not allowed (io.allow_jpeg=false)"
                          for f in res["files"] if f["is_jpeg"]]
    for f in res["files"]:
        log.info("[file] %s %s jpeg=%s q~%s sha256=%s", f["file"], f["format"], f["is_jpeg"],
                 f["jpeg_quality_est"], f["sha256"][:12])
    for w in res["warnings"]:
        log.warning(w)
    for e in res["errors"]:
        log.error(e)

    if args.preview:
        for out in write_previews(args.paper_dir, paths["interim"] / "previews",
                                  int(cfg.get("preview", {}).get("grid_px", 100))):
            log.info("[preview] %s", out)
    return 1 if res["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
