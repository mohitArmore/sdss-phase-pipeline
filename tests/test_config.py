"""configs/data/papers.yaml carries every tunable named in the spec."""
from src.utils.config import load_config


def test_papers_config_spec_values():
    c = load_config("configs/data/papers.yaml")
    assert c["io"] == {"allow_jpeg": True, "never_reencode_jpeg": True, "derived_format": "png"}
    assert c["crop"] == {"margin_px": 4, "min_side_px": 512}
    e = c["ebsd"]
    assert (e["max_delta_e"], e["ambiguity_ratio"], e["mode_filter"], e["min_island_px"],
            e["boundary_policy"]) == (20, 0.8, 3, 4, "ignore")
    m = c["micrograph"]
    assert m["methods"] == ["otsu", "kmeans"]
    assert (m["flatfield_sigma_px"], m["denoise_h"], m["morph_radius_px"], m["min_island_px"]) == (50, 7, 1, 256)
    v = c["validation"]
    assert v["tol_pp"] == {"ebsd_same_map": 5, "ia": 8, "xrd": 10, "asbuilt_abs": 2}
    assert v["group_by"] == "paper_doi"
    assert c["resample"]["target_um_per_px"] == {"sem": 0.25, "ebsd": 0.5, "om": 0.75}
