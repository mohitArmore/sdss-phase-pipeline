"""Tests for src.data.ang_to_mask that don't require a real .ang file."""
import numpy as np
import pytest

from src.data.ang_to_mask import IGNORE_INDEX, MaskingConfig, save_mask, MaskResult


def test_masking_config_defaults():
    cfg = MaskingConfig()
    assert cfg.ci_threshold == 0.1
    assert cfg.ferrite_class_id == 1
    assert cfg.austenite_class_id == 0
    assert cfg.three_class_mode is False


def test_save_mask_roundtrip(tmp_path):
    """Building a MaskResult by hand and saving it should produce a valid PNG + JSON."""
    from PIL import Image
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:5, :] = 1
    mask[0, 0] = IGNORE_INDEX

    result = MaskResult(
        mask=mask,
        phase_fractions={"ferrite": 0.5, "austenite": 0.49, "unindexed": 0.01},
        ci_stats={"mean": 0.8, "median": 0.85, "min": 0.1, "max": 1.0, "pct_above_thresh": 99.0},
        step_size_um=0.5,
        shape=(10, 10),
        phases_found=["Austenite", "Ferrite"],
        config=MaskingConfig(),
    )
    save_mask(result, tmp_path / "test_mask.png", tmp_path / "test_mask.json")

    loaded = np.array(Image.open(tmp_path / "test_mask.png"))
    assert loaded.shape == (10, 10)
    assert loaded[0, 0] == IGNORE_INDEX
    assert loaded[6, 0] == 0
    assert loaded[2, 0] == 1

    import json
    meta = json.loads((tmp_path / "test_mask.json").read_text())
    assert meta["step_size_um"] == 0.5
    assert meta["phase_fractions"]["ferrite"] == 0.5
    assert meta["config"]["ci_threshold"] == 0.1


def test_ang_to_mask_missing_file_raises(tmp_path):
    from src.data.ang_to_mask import ang_to_mask, _ORIX_AVAILABLE
    if not _ORIX_AVAILABLE:
        pytest.skip("orix not installed in this environment")
    with pytest.raises(FileNotFoundError):
        ang_to_mask(tmp_path / "does_not_exist.ang")
