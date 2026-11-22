import numpy as np
import pytest
from PIL import Image

from src.io.image_io import load_image, load_mask, save_png, sha256_file


def test_jpeg_info_and_no_rewrite(tmp_path):
    rng = np.random.default_rng(0)
    arr = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    p = tmp_path / "x.jpg"
    Image.fromarray(arr).save(p, quality=75, subsampling=2)
    h0 = sha256_file(p)
    img, info = load_image(p)
    assert img.shape == (64, 64, 3) and img.dtype == np.uint8
    assert info["is_jpeg"] and info["format"] == "JPEG"
    assert abs(info["jpeg_quality_est"] - 75) <= 2
    assert info["chroma_subsampling"] == "4:2:0"
    assert sha256_file(p) == h0


def test_png_gray_and_save_png(tmp_path):
    p = tmp_path / "g.png"
    Image.fromarray(np.full((8, 9), 7, np.uint8)).save(p)
    img, info = load_image(p)
    assert img.shape == (8, 9, 3) and not info["is_jpeg"]
    m = np.array([[0, 1], [255, 2]], np.uint8)
    save_png(tmp_path / "m.png", m)
    assert (load_mask(tmp_path / "m.png") == m).all()
    with pytest.raises(ValueError):
        save_png(tmp_path / "m.jpg", m)
