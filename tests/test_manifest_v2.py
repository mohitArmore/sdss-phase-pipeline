"""Manifest schema v2: records, v1 -> v2 migration, upsert."""
import json

import pytest

from src.data.manifest import (
    Extraction, ImageInfo, PhaseFractions, Record, Source,
    add_or_update_sample, get_record, load_manifest, migrate_v1_to_v2, save_manifest,
)

V1_PAYLOAD = {
    "schema_version": 1,
    "samples": [{
        "sample_id": "asbuilt_001",
        "ang_path": "raw/ang/asbuilt_001.ang", "sem_path": "raw/sem/asbuilt_001.png",
        "optical_path": None, "mask_path": "interim/masks/asbuilt_001.png",
        "mask_meta_path": "interim/masks/asbuilt_001.json",
        "modality": "sem", "magnification": 500.0, "pixel_size_um": 0.5,
        "material": {"alloy": "SDSS_2507", "composition": {"Cr": 25.0}, "condition": "as_built"},
        "process": {"route": "LPBF", "laser_power_W": 200.0, "scan_speed_mm_s": None,
                    "hatch_um": None, "layer_um": None, "heat_treatment": None},
        "properties": {"tensile_MPa": 1000.0, "yield_MPa": None, "elongation_pct": None,
                       "hardness_HV": None, "corrosion_uA": None},
        "phase_fractions_ebsd": {"ferrite": 0.94, "austenite": 0.05, "sigma": None, "unindexed": 0.01},
        "quality": {"ebsd_ci_mean": 0.8, "ebsd_indexed_pct": 99.0, "exclude": True,
                    "exclude_reason": "low CI"},
        "split": "train", "cv_fold": 2,
        "source": {"provenance": "own", "citation": None, "notes": "x"},
    }],
}


def test_migrate_v1_to_v2_maps_fields():
    out = migrate_v1_to_v2(V1_PAYLOAD)
    assert out["schema_version"] == 2
    r = out["samples"][0]
    assert r["image"]["image_type"] == "ebsd_phase"
    assert r["image"]["pixel_size_um"] == 0.5
    assert r["condition"] == "as_built" and r["process"] == "LPBF"
    assert r["reported"]["ferrite"] == pytest.approx(0.94)
    assert r["reported"]["method"] == "EBSD"
    assert r["paths"]["mask"] == "interim/masks/asbuilt_001.png"
    assert r["qc_flags"] == ["excluded:low CI"]
    assert r["properties"]["tensile_MPa"] == 1000.0
    assert (r["split"], r["cv_fold"]) == ("train", 2)
    assert r["source"]["notes"] == "x"
    # v1-only data preserved losslessly
    assert r["legacy"]["process"]["laser_power_W"] == 200.0
    assert r["legacy"]["material"]["composition"] == {"Cr": 25.0}
    assert r["legacy"]["unindexed"] == 0.01


def test_load_manifest_accepts_v1_file_without_rewriting(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(V1_PAYLOAD))
    before = p.read_bytes()
    recs = load_manifest(p)
    assert recs[0]["sample_id"] == "asbuilt_001" and "image" in recs[0]
    assert p.read_bytes() == before


def test_upsert_into_v1_file_upgrades_it(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(V1_PAYLOAD))
    add_or_update_sample(p, Record(sample_id="new_one"))
    assert json.loads(p.read_text())["schema_version"] == 2
    assert {r["sample_id"] for r in load_manifest(p)} == {"asbuilt_001", "new_one"}


def test_record_roundtrip_and_upsert(tmp_path):
    p = tmp_path / "m.json"
    r = Record(
        sample_id="p_f1a_pub_as_built_om",
        image=ImageInfo(image_type="om", is_jpeg=True, crop_box=[0, 0, 10, 10]),
        source=Source(provenance="paper", paper_doi=None, figure_id="1", panel="a"),
        reported=PhaseFractions(ferrite=0.5, method="IA", scope="this_panel"),
        extracted=Extraction(method="otsu", fractions={"ferrite": 0.52}),
    )
    save_manifest([r], p)
    r.qc_flags = ["jpeg_source"]
    add_or_update_sample(p, r)
    recs = load_manifest(p)
    assert len(recs) == 1
    got = get_record(p, "p_f1a_pub_as_built_om")
    assert got["qc_flags"] == ["jpeg_source"]
    assert got["extracted"]["fractions"]["ferrite"] == 0.52
    assert got["image"]["crop_box"] == [0, 0, 10, 10]
