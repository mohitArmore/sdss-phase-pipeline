import io

import numpy as np
import pytest
from PIL import Image

from src.data.synthetic import phase_field, render_micrograph
from src.features.features import phase_fractions
from src.segment.micrograph_segment import best_channel, segment_micrograph


def _rgb(g):
    return np.repeat(g[..., None], 3, axis=2)


def _jpeg(img, q=75):
    b = io.BytesIO()
    Image.fromarray(img).save(b, format="JPEG", quality=q)
    return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))


@pytest.mark.parametrize("frac", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("method", ["otsu", "kmeans"])
@pytest.mark.parametrize("jpeg,tol_pp", [(False, 2.0), (True, 4.0)])
def test_micrograph_fraction_recovered(frac, method, jpeg, tol_pp):
    rng = np.random.default_rng(int(frac * 10))
    fer = phase_field((512, 512), frac, rng)
    img = _rgb(render_micrograph(fer, rng))
    if jpeg:
        img = _jpeg(img)
    mask, info = segment_micrograph(img, dark_phase="ferrite", method=method)
    assert abs(phase_fractions(mask)["ferrite"] - fer.mean()) * 100 <= tol_pp, info


def test_dark_phase_austenite_and_ignore():
    rng = np.random.default_rng(3)
    fer = phase_field((400, 400), 0.4, rng)
    img = _rgb(render_micrograph(fer, rng, dark_is_ferrite=False))
    ign = np.zeros((400, 400), np.uint8)
    ign[:50, :50] = 255
    mask, _ = segment_micrograph(img, dark_phase="austenite", ignore=ign)
    assert (mask[:50, :50] == 255).all()
    truth = fer[ign == 0].mean()
    assert abs(phase_fractions(mask)["ferrite"] - truth) * 100 <= 2.0


def test_best_channel_prefers_contrast():
    img = np.zeros((50, 50, 3), np.uint8)
    img[:, :25, 1] = 200                 # only green carries the phase contrast
    img[..., 0] = np.random.default_rng(0).integers(0, 30, (50, 50))
    name, _, scores = best_channel(img)
    assert name in ("G", "L", "a") and scores["G"] > scores["R"]


def test_multiotsu_sigma():
    lab = np.zeros((300, 300), np.uint8)
    lab[:, 100:200] = 1
    lab[:, 200:] = 2
    g = np.choose(lab, [160, 60, 240]).astype(float)       # aus bright, fer dark, sigma brightest
    g = np.clip(g + np.random.default_rng(0).normal(0, 6, g.shape), 0, 255).astype(np.uint8)
    mask, _ = segment_micrograph(_rgb(g), "ferrite", method="multiotsu", sigma_tone="bright",
                                 flatfield_sigma_px=0, min_island_px=16)
    f = phase_fractions(mask)
    assert f["sigma"] == pytest.approx(1 / 3, abs=0.02) and f["ferrite"] == pytest.approx(1 / 3, abs=0.02)
