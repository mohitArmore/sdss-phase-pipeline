import numpy as np
import pytest

from src.data.scalebar import calibrate, measure_bar_px


@pytest.mark.parametrize("bg,fg", [(255, 0), (0, 255)])
def test_measure_synthetic_bar_with_text(bg, fg):
    img = np.full((60, 300, 3), bg, np.uint8)
    img[40:45, 50:193] = fg            # 143 px bar
    img[10:25, 90:100] = fg            # "text" above the bar
    img[10:25, 110:120] = fg
    assert measure_bar_px(img, [40, 5, 210, 50]) == 143


def test_calibrate_measured_and_manual():
    img = np.full((60, 300), 255, np.uint8)
    img[40:45, 50:250] = 0
    c = calibrate({"um": 100, "bbox": [45, 35, 255, 50]}, img)
    assert c["bar_px"] == 200 and c["um_per_px"] == pytest.approx(0.5)
    assert c["rel_uncertainty"] == pytest.approx(0.01) and c["method"] == "measured"
    m = calibrate({"um": 200, "px": 143}, None)
    assert m["um_per_px"] == pytest.approx(200 / 143) and m["method"] == "manual"
    assert calibrate({"um": None}, img) is None and calibrate(None, img) is None
