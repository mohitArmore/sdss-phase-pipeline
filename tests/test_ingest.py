import csv

import numpy as np
import pytest
import yaml
from PIL import Image

from src.data.ingest_paper import (
    extract_pdf_images, grid_preview, ingest_paper_dir, make_sample_id, panel_contexts, load_meta,
)
from src.io.image_io import sha256_file


def _paper(tmp_path, **panel_over):
    d = tmp_path / "p1"
    (d / "figures").mkdir(parents=True)
    Image.fromarray(np.zeros((50, 60, 3), np.uint8)).save(d / "figures" / "f1.jpg", quality=80)
    panel = {"panel": "a", "crop_box": [0, 0, 40, 40], "image_type": "om",
             "condition": "as_built", "dark_phase": "ferrite"}
    panel.update(panel_over)
    meta = {"paper_id": "p1", "doi": None, "citation": None, "licence": None,
            "figures": [{"figure_id": "1", "file": "figures/f1.jpg",
                         "acquisition_route": "publisher_figure", "panels": [panel]}]}
    (d / "meta.yaml").write_text(yaml.safe_dump(meta))
    return d


def test_ingest_hashes_and_warns_on_missing_facts(tmp_path):
    d = _paper(tmp_path)
    h0 = sha256_file(d / "figures" / "f1.jpg")
    res = ingest_paper_dir(d, tmp_path / "ingest")
    assert res["errors"] == []
    assert res["files"][0]["sha256"] == h0 and res["files"][0]["is_jpeg"]
    assert any("'doi' is null" in w for w in res["warnings"])
    assert any("no reported phase fraction" in w for w in res["warnings"])
    # tampering with the original is detected on re-ingest
    (d / "figures" / "f1.jpg").write_bytes((d / "figures" / "f1.jpg").read_bytes() + b"\0")
    assert any("CHANGED" in e for e in ingest_paper_dir(d, tmp_path / "ingest")["errors"])


def test_bad_crop_box_is_error(tmp_path):
    d = _paper(tmp_path, crop_box=[10, 10, 5, 40])
    assert any("crop_box" in e for e in ingest_paper_dir(d, tmp_path / "i")["errors"])


def test_sample_id_and_context():
    assert make_sample_id("gargalis2023met", "16", "a", "publisher_figure", "as_built", "om") \
        == "gargalis2023met_f16a_pub_as_built_om"


def test_figure_defaults_inherited(tmp_path):
    d = _paper(tmp_path)
    meta = load_meta(d)
    meta["figures"][0]["etchant"] = "Beraha"
    c = panel_contexts(d, meta)[0]
    assert c["etchant"] == "Beraha" and c["sample_id"].startswith("p1_f1a_pub")


def test_grid_preview_shape():
    out = grid_preview(np.zeros((120, 230, 3), np.uint8), 100, [([0, 0, 50, 50], (255, 0, 0))])
    assert out.shape == (160, 270, 3)


def test_extract_pdf_lossless(tmp_path):
    fitz = pytest.importorskip("fitz")
    jpg = tmp_path / "e.jpg"
    rng = np.random.default_rng(1)
    Image.fromarray((rng.random((40, 50, 3)) * 255).astype(np.uint8)).save(jpg, quality=70)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(fitz.Rect(0, 0, 200, 160), filename=str(jpg))
    doc.save(tmp_path / "x.pdf")
    listing = extract_pdf_images(tmp_path / "x.pdf", tmp_path / "out", use_pdfimages=False)
    rows = list(csv.DictReader(listing.open()))
    assert len(rows) == 1 and rows[0]["format"] in ("jpeg", "jpg")
    # embedded JPEG stream comes out byte-identical
    assert rows[0]["sha256"] == sha256_file(jpg)
