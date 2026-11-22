import io

import numpy as np
import pytest
from PIL import Image

from src.data.synthetic import phase_field, render_ebsd
from src.features.features import phase_fractions
from src.segment.ebsd_colour_to_mask import ebsd_to_mask

PALETTE = {"ferrite": "#ff0000", "austenite": "#0000ff", "ignore": ["#000000", "#ffffff"]}


def _jpeg(img, q=75):
    b = io.BytesIO()
    Image.fromarray(img).save(b, format="JPEG", quality=q)
    return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))


@pytest.mark.parametrize("frac", [0.15, 0.5, 0.9])
@pytest.mark.parametrize("policy", ["ignore", "reassign"])
@pytest.mark.parametrize("jpeg,tol_pp", [(False, 2.0), (True, 4.0)])
def test_ebsd_fraction_recovered(frac, policy, jpeg, tol_pp):
    fer = phase_field((400, 400), frac, np.random.default_rng(int(frac * 100)))
    img = render_ebsd(fer)
    if jpeg:
        img = _jpeg(img)
    mask = ebsd_to_mask(img, PALETTE, boundary_policy=policy)
    got = phase_fractions(mask)["ferrite"]
    assert abs(got - fer.mean()) * 100 <= tol_pp
    if policy == "reassign":
        assert phase_fractions(mask)["ignored_fraction"] < 0.01


def test_ignore_boxes_and_unknown_colour():
    img = np.zeros((20, 20, 3), np.uint8)
    img[:, :10] = (255, 0, 0)
    img[:, 10:] = (0, 255, 0)                       # green: not in palette -> 255
    ign = np.zeros((20, 20), np.uint8)
    ign[:5, :5] = 255
    m = ebsd_to_mask(img, PALETTE, ignore=ign, mode_filter_size=0, min_island_px=0)
    assert (m[:, 10:] == 255).all() and (m[:5, :5] == 255).all() and (m[10:, :10] == 1).all()
    m2 = ebsd_to_mask(img, PALETTE, ignore=ign, boundary_policy="reassign", reassign_max_px=50)
    assert (m2[:5, :5] == 255).all()                # human ignore boxes are never reassigned


def test_iq_mode_uses_chroma_only():
    img = np.zeros((10, 20, 3), np.uint8)
    img[:, :10] = (120, 0, 0)                       # IQ-darkened red
    img[:, 10:] = (0, 0, 140)                       # IQ-darkened blue
    pal = {"ferrite": "#ff0000", "austenite": "#0000ff"}
    m = ebsd_to_mask(img, pal, image_type="ebsd_phase_iq", max_delta_e=60, mode_filter_size=0)
    assert (m[:, :10] == 1).all() and (m[:, 10:] == 0).all()
