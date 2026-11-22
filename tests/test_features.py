import numpy as np
import pytest

from src.features.features import boundary_density, grain_stats, mask_features, phase_fractions


def test_fractions_exclude_ignore():
    m = np.array([[0, 1, 1, 255]], np.uint8)
    f = phase_fractions(m)
    assert f["ferrite"] == pytest.approx(2 / 3) and f["ignored_fraction"] == pytest.approx(0.25)


def test_grain_stats_excludes_border_and_tiny():
    m = np.zeros((100, 100), np.uint8)
    m[20:30, 20:60] = 1          # 10 x 40 interior region, aspect 4
    m[0:10, 0:10] = 1            # touches border -> excluded
    m[80:82, 80:82] = 1          # 4 px -> excluded
    m[50:60, 70:80] = 1
    m[60:62, 70:80] = 255        # touches ignore -> excluded
    t = grain_stats(m, 1, um_per_px=0.5, min_region_px=20)
    assert len(t) == 1
    assert t["aspect_ratio"].iloc[0] == pytest.approx(4.0, rel=0.05)
    assert t["ecd_um"].iloc[0] == pytest.approx(np.sqrt(400 / np.pi) * 2 * 0.5, rel=1e-6)


def test_boundary_density_straight_line():
    m = np.zeros((10, 10), np.uint8)
    m[:, 5:] = 1
    d = boundary_density(m, um_per_px=2.0)
    assert d["boundary_density_per_px"] == pytest.approx(np.pi / 4 * 10 / 100)
    assert d["boundary_density_per_um"] == pytest.approx(d["boundary_density_per_px"] / 2)
    assert "austenite_n_regions" in mask_features(m)
