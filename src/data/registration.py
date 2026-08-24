"""Register a SEM/optical micrograph to an EBSD phase mask.

Why registration is necessary
-----------------------------
The EBSD scan (.ang) is taken on the same specimen area as the SEM image, but:
  - the coordinate systems can differ by a small rotation (stage tilt),
  - pixel sizes differ (SEM at 1000x might be ~0.1 um/px; .ang step often 0.5 um),
  - the fields of view can differ (SEM often larger than the EBSD scan area).

If we train segmentation on unregistered pairs, the model learns garbage — the
"ground truth" for pixel (i,j) in SEM doesn't correspond to phase (i,j) in the
mask. This module fixes that.

Strategy — three fallbacks in order:
  1. ORB feature matching + RANSAC homography (works when features are dense
     and both images have clear texture).
  2. Enhanced Correlation Coefficient (ECC) alignment (works when features are
     sparse, e.g., low-contrast optical images).
  3. Manual four-point alignment (writes a CSV template you fill in with
     matching (SEM_x, SEM_y, EBSD_x, EBSD_y) coordinates, and we compute the
     homography from that).

Output: the SEM/optical image, warped so pixel (i,j) matches mask pixel (i,j),
and clipped to the mask's field of view.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


@dataclass
class RegistrationResult:
    aligned_image: np.ndarray       # same H,W as mask
    warp_matrix: np.ndarray         # 3x3 homography (or 2x3 affine, promoted)
    method: str                     # "orb" | "ecc" | "manual"
    inlier_count: int | None        # ORB only
    ecc_score: float | None         # ECC only
    ok: bool                        # False = alignment probably failed, review manually


def _promote_to_homography(m: np.ndarray) -> np.ndarray:
    """Turn a 2x3 affine into a 3x3 homography."""
    if m.shape == (3, 3):
        return m
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = m
    return H


def _rescale_mask_to_reference(mask: np.ndarray, sem_shape: tuple[int, int]) -> np.ndarray:
    """The mask (from EBSD) is usually smaller than the SEM image. We do NOT
    upscale the mask (that invents pixels); we scale the SEM DOWN or crop it
    to the mask FOV. Registration works in the mask frame."""
    return mask  # kept for the caller — real logic is in `register()` below


# ---------------------------------------------------------------------------
# ORB + RANSAC
# ---------------------------------------------------------------------------
def _register_orb(
    sem_gray: np.ndarray,
    mask_gray: np.ndarray,
    max_features: int = 5000,
    good_match_ratio: float = 0.15,
) -> tuple[np.ndarray, int]:
    """Feature-based registration. Returns (H, n_inliers)."""
    # Boost contrast in both to help feature detection on low-contrast optical images.
    sem_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(sem_gray)
    mask_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(mask_gray)

    orb = cv2.ORB_create(max_features)
    kp1, desc1 = orb.detectAndCompute(sem_eq, None)
    kp2, desc2 = orb.detectAndCompute(mask_eq, None)
    if desc1 is None or desc2 is None or len(kp1) < 8 or len(kp2) < 8:
        raise RuntimeError("ORB: too few keypoints for reliable homography")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda m: m.distance)
    n_good = max(10, int(len(matches) * good_match_ratio))
    matches = matches[:n_good]
    if len(matches) < 8:
        raise RuntimeError("ORB: fewer than 8 good matches — cannot fit homography")

    pts_sem = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_mask = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(pts_sem, pts_mask, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("ORB: RANSAC failed to find a homography")
    n_inliers = int(inlier_mask.sum())
    return H, n_inliers


# ---------------------------------------------------------------------------
# ECC intensity alignment
# ---------------------------------------------------------------------------
def _register_ecc(
    sem_gray: np.ndarray,
    mask_gray: np.ndarray,
    warp_mode: int = cv2.MOTION_AFFINE,
    n_iter: int = 500,
    eps: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """Intensity-based alignment. Returns (H_3x3, ecc_score)."""
    # ECC works best when both inputs are of similar spatial resolution;
    # resize the SEM to match the mask size first.
    if sem_gray.shape != mask_gray.shape:
        sem_r = cv2.resize(sem_gray, (mask_gray.shape[1], mask_gray.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        sem_r = sem_gray

    warp = np.eye(2, 3, dtype=np.float32) if warp_mode != cv2.MOTION_HOMOGRAPHY else np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, eps)
    score, warp = cv2.findTransformECC(mask_gray, sem_r, warp, warp_mode, criteria, None, 5)
    return _promote_to_homography(warp), float(score)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def register(
    sem_path: str | Path,
    mask_path: str | Path,
    method: Literal["auto", "orb", "ecc", "manual"] = "auto",
    manual_points_csv: str | Path | None = None,
    min_orb_inliers: int = 20,
    min_ecc_score: float = 0.5,
) -> RegistrationResult:
    """Register a SEM (or optical) image to a mask.

    In "auto" mode we try ORB first, fall back to ECC, and fall back again to
    manual points if provided. If nothing works, we return `ok=False` so the
    calling script can flag the sample and move on rather than corrupt training.
    """
    sem = cv2.imread(str(sem_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if sem is None:
        raise FileNotFoundError(sem_path)
    if mask is None:
        raise FileNotFoundError(mask_path)

    # For registration we want a "phase-like" grayscale mask (0 vs 255), not the
    # class-ID mask (0/1/255). Threshold it.
    mask_for_align = np.where(mask == 1, 255, 0).astype(np.uint8)

    last_error: str | None = None

    if method in ("auto", "orb"):
        try:
            H, n_inliers = _register_orb(sem, mask_for_align)
            if n_inliers >= min_orb_inliers:
                aligned = cv2.warpPerspective(sem, H, (mask.shape[1], mask.shape[0]))
                return RegistrationResult(
                    aligned_image=aligned, warp_matrix=H, method="orb",
                    inlier_count=n_inliers, ecc_score=None, ok=True,
                )
            last_error = f"ORB inliers ({n_inliers}) below threshold ({min_orb_inliers})"
        except RuntimeError as e:
            last_error = f"ORB failed: {e}"

    if method in ("auto", "ecc"):
        try:
            H, score = _register_ecc(sem, mask_for_align)
            if score >= min_ecc_score:
                aligned = cv2.warpPerspective(sem, H, (mask.shape[1], mask.shape[0]))
                return RegistrationResult(
                    aligned_image=aligned, warp_matrix=H, method="ecc",
                    inlier_count=None, ecc_score=score, ok=True,
                )
            last_error = f"ECC score ({score:.3f}) below threshold ({min_ecc_score})"
        except cv2.error as e:
            last_error = f"ECC failed: {e}"

    if method in ("auto", "manual") and manual_points_csv is not None:
        H = _homography_from_manual_csv(manual_points_csv)
        aligned = cv2.warpPerspective(sem, H, (mask.shape[1], mask.shape[0]))
        return RegistrationResult(
            aligned_image=aligned, warp_matrix=H, method="manual",
            inlier_count=None, ecc_score=None, ok=True,
        )

    # All methods failed — return the SEM unchanged and flag it.
    return RegistrationResult(
        aligned_image=sem, warp_matrix=np.eye(3), method="failed",
        inlier_count=None, ecc_score=None, ok=False,
    )


def _homography_from_manual_csv(path: str | Path) -> np.ndarray:
    """CSV format:  sem_x,sem_y,mask_x,mask_y  (>=4 rows).

    Generate a template with `write_manual_template()`.
    """
    import csv
    src, dst = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            src.append([float(row["sem_x"]), float(row["sem_y"])])
            dst.append([float(row["mask_x"]), float(row["mask_y"])])
    if len(src) < 4:
        raise ValueError(f"Manual registration needs >=4 point pairs, got {len(src)}")
    src = np.array(src, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.array(dst, dtype=np.float32).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
    if H is None:
        raise RuntimeError("Manual points did not yield a valid homography")
    return H


def write_manual_template(path: str | Path) -> None:
    """Write a CSV template the user fills in with matching corner points."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("sem_x,sem_y,mask_x,mask_y\n")
        f.write("# fill in >=4 rows of matching corner/feature pixel coordinates\n")
