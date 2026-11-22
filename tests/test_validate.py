import numpy as np
import pytest

from src.eval.validate import (
    boundary_band_exclude, evaluate, normalise_reported, tolerance_pp, write_report,
)

VCFG = {"tol_pp": {"ebsd_same_map": 5, "ia": 8, "xrd": 10, "asbuilt_abs": 2},
        "asbuilt_conditions": ["as_built"], "boundary_sensitive_pp": 2.0}


def test_normalise_units():
    r, f = normalise_reported({"austenite": 0.1, "ferrite": 99.9})
    assert r["ferrite"] == pytest.approx(0.999) and r["austenite"] == pytest.approx(0.001) and f == []
    r, f = normalise_reported({"ferrite": 0.6, "austenite": 0.4})
    assert r["ferrite"] == 0.6 and f == []
    r, f = normalise_reported({"ferrite": 0.5})
    assert r["ferrite"] == pytest.approx(0.005) and f == ["reported_unit_ambiguous"]
    r, f = normalise_reported({"ferrite": 60, "austenite": 30})
    assert "reported_sum_mismatch" in f
    r, f = normalise_reported({"ferrite": 0.5, "unit": "fraction"})
    assert r["ferrite"] == 0.5 and f == []


def test_tolerance_rules():
    assert tolerance_pp({"method": "EBSD", "scope": "this_panel"}, "ebsd_phase", "SA", VCFG)[0] == 5
    assert tolerance_pp({"method": "EBSD", "scope": "sample_mean"}, "om", "SA", VCFG)[0] == 8
    assert tolerance_pp({"method": "XRD"}, "om", "SA", VCFG)[0] == 10
    assert tolerance_pp({"method": "XRD"}, "om", "as_built", VCFG) == (2, "xrd+asbuilt_abs")


def test_band_exclude():
    m = np.zeros((10, 10), np.uint8)
    m[:, 5:] = 1
    assert (boundary_band_exclude(m, 1)[:, 4:6] == 255).all()
    assert (boundary_band_exclude(m, 2)[:, 3:7] == 255).all()
    assert (boundary_band_exclude(m, 2)[:, :3] == 0).all()


def test_evaluate_flags_and_report(tmp_path):
    rec = {"sample_id": "p_f1a_pub_as_built_om", "condition": "as_built",
           "image": {"image_type": "om", "is_jpeg": True, "pixel_size_um": None},
           "source": {"paper_doi": None},
           "reported": {"ferrite": 0.90, "austenite": None, "sigma": None, "method": "IA"},
           "extracted": {"method": "otsu", "fractions": {"ferrite": 0.95, "austenite": 0.05},
                         "params": {}, "ignored_fraction": 0.01},
           "qc_flags": ["small_crop"]}
    variants = {"band_1px": {"kind": "boundary", "fractions": {"ferrite": 0.92}},
                "thr_+10pct": {"kind": "threshold", "fractions": {"ferrite": 0.951}}}
    evaluate(rec, variants, VCFG)
    assert rec["abs_error_pp"]["ferrite"] == pytest.approx(5.0)
    assert set(rec["qc_flags"]) == {"small_crop", "jpeg_source", "missing_scale",
                                    "boundary_sensitive", "out_of_tolerance"}
    out = write_report([rec], tmp_path, "paper_doi", "abc")
    assert out["csv"].exists() and out["bland_altman"].exists()
    assert "MAE_pp" in out["md"].read_text(encoding="utf-8")
