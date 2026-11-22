"""End-to-end: run_paper_pipeline.py on the synthetic fixture paper."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.data.synthetic import make_fixture_paper
from src.io.image_io import sha256_file

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "synthetic_paper" / "meta.yaml"


def test_run_paper_pipeline_end_to_end(tmp_path):
    root, reports = tmp_path / "data", tmp_path / "reports"
    paper = make_fixture_paper(FIXTURE, root / "raw" / "papers" / "synthetic_fixture", seed=42)
    originals = {p: sha256_file(p) for p in (paper / "figures").iterdir()}
    assert any(p.suffix == ".jpg" for p in originals)

    cmd = [sys.executable, "scripts/run_paper_pipeline.py", "--all",
           "--config", "configs/data/papers.yaml", "--data-root", str(root), "--reports-dir", str(reports)]
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]

    # originals (incl. the JPEG) are byte-identical after the run
    assert {p: sha256_file(p) for p in originals} == originals

    manifest = json.loads((root / "processed/manifests/manifest.json").read_text())
    assert manifest["schema_version"] == 2
    recs = {r["sample_id"]: r for r in manifest["samples"]}
    assert len(recs) == 4
    for sid, r in recs.items():
        assert (root / r["paths"]["mask"]).exists() and (root / r["paths"]["overlay"]).exists()
        assert r["paths"]["mask"].endswith(".png") and "interim/overlays/" in r["paths"]["overlay"]
        tol = 4.0 if r["image"]["is_jpeg"] else 2.0
        assert max(r["abs_error_pp"].values()) <= tol, (sid, r["abs_error_pp"])
        assert "out_of_tolerance" not in r["qc_flags"]
        assert r["source"]["sha256_original"] == originals[root / r["paths"]["original"]]
        assert r["extracted"]["params"]["config_hash"]
        assert r["extracted"]["sensitivity"]

    a = recs["synthetic_fixture_f1a_pub_as_built_ebsd_phase"]
    assert abs(a["image"]["pixel_size_um"] - 50 / 110) < 1e-9          # measured bar
    assert recs["synthetic_fixture_f2a_pdfemb_SA_1100C_1h_sem"]["image"]["scale_bar_px"] == 130
    nb = recs["synthetic_fixture_f2b_pdfemb_SA_1100C_1h_ebsd_phase"]
    assert {"missing_scale", "jpeg_source"} <= set(nb["qc_flags"])

    feats = pd.read_csv(root / "processed/features/features.csv")
    assert len(feats) == 4 and {"ferrite", "ferrite_ecd_um_mean", "boundary_density_per_um"} <= set(feats)
    for f in ("summary.csv", "summary.md", "bland_altman.png"):
        assert (reports / f).exists()
    # no derived JPEGs anywhere under interim/
    assert not [p for p in (root / "interim").rglob("*") if p.suffix.lower() in (".jpg", ".jpeg")]

    # idempotent re-run: same records, no duplicates
    res2 = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
    assert res2.returncode == 0
    m2 = json.loads((root / "processed/manifests/manifest.json").read_text())
    assert len(m2["samples"]) == 4
