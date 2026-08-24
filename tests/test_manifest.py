"""Tests for src.data.manifest — no .ang files required."""
from pathlib import Path

import pytest

from src.data.manifest import (
    Material, Process, Properties, Quality, Sample, Source,
    PhaseFractions, add_or_update_sample, load_manifest, save_manifest,
)


def test_sample_defaults_are_serializable(tmp_path: Path):
    s = Sample(sample_id="test_001")
    save_manifest([s], tmp_path / "m.json")
    loaded = load_manifest(tmp_path / "m.json")
    assert len(loaded) == 1
    assert loaded[0]["sample_id"] == "test_001"
    assert loaded[0]["material"]["alloy"] == "SDSS_2507"
    assert loaded[0]["modality"] == "sem"


def test_upsert_replaces_by_sample_id(tmp_path: Path):
    p = tmp_path / "m.json"
    s1 = Sample(sample_id="A", modality="sem")
    s2 = Sample(sample_id="A", modality="optical")  # same id, different data
    add_or_update_sample(p, s1)
    add_or_update_sample(p, s2)
    loaded = load_manifest(p)
    assert len(loaded) == 1
    assert loaded[0]["modality"] == "optical"


def test_upsert_appends_new(tmp_path: Path):
    p = tmp_path / "m.json"
    add_or_update_sample(p, Sample(sample_id="A"))
    add_or_update_sample(p, Sample(sample_id="B"))
    loaded = load_manifest(p)
    assert {s["sample_id"] for s in loaded} == {"A", "B"}


def test_phase_fractions_round_trip(tmp_path: Path):
    p = tmp_path / "m.json"
    s = Sample(
        sample_id="asbuilt_001",
        phase_fractions_ebsd=PhaseFractions(ferrite=0.94, austenite=0.05, unindexed=0.01),
    )
    add_or_update_sample(p, s)
    loaded = load_manifest(p)
    assert loaded[0]["phase_fractions_ebsd"]["ferrite"] == pytest.approx(0.94)
    assert loaded[0]["phase_fractions_ebsd"]["austenite"] == pytest.approx(0.05)


def test_schema_version_mismatch_raises(tmp_path: Path):
    import json
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"schema_version": 999, "samples": []}))
    with pytest.raises(ValueError, match="schema"):
        load_manifest(p)
