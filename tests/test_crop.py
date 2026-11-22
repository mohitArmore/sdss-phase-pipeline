import numpy as np
import pytest

from src.data.annotations import ignore_mask, to_crop_frame
from src.data.crop import crop_panel


def test_crop_and_ignore_box_mapping():
    img = np.zeros((100, 200, 3), np.uint8)
    img[30, 50] = 255                      # marker pixel in original coords
    crop, box, flags = crop_panel(img, [40, 20, 140, 90], margin_px=4, min_side_px=512)
    assert box == [44, 24, 136, 86]
    assert crop.shape == (62, 92, 3)
    assert crop[30 - 24, 50 - 44, 0] == 255   # same pixel, crop frame
    assert flags == ["small_crop"]
    boxes = to_crop_frame([[50, 30, 60, 40], [0, 0, 10, 10], [130, 80, 300, 300]], box)
    assert boxes == [[6, 6, 16, 16], [86, 56, 92, 62]]   # middle box dropped, last clipped
    m = ignore_mask(crop.shape, boxes)
    assert m[6, 6] == 255 and m[16, 16] == 0 and m[61, 91] == 255 and m[5, 5] == 0


def test_crop_box_outside_raises():
    with pytest.raises(ValueError):
        crop_panel(np.zeros((10, 10)), [0, 0, 20, 5])
