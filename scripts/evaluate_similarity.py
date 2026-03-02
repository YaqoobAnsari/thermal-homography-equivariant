#!/usr/bin/env python
"""
Comprehensive Sim(2) Similarity Estimation Evaluation

Evaluates LogPolarSim2Net against classical baselines on:
1. Per-component accuracy (rotation, scale, translation)
2. Equivariance metrics (flat error curves)
3. Thermal-specific challenges (colormap invariance, low texture)
4. Generalization to unseen transformation ranges

This script generates the paper's main results demonstrating that
equivariant models produce FLAT error curves across all transformation
parameters, while non-equivariant baselines fail outside their training
distribution.

Usage:
    # Test with untrained model (classical FMT baseline)
    python scripts/evaluate_similarity.py --device cuda --n-samples 50

    # Test with trained checkpoint
    python scripts/evaluate_similarity.py --checkpoint checkpoints/best.pt --device cuda

    # Full evaluation for paper
    python scripts/evaluate_similarity.py --checkpoint checkpoints/best.pt \
        --device cuda --n-samples 100 --output-dir outputs/similarity_evaluation

Author: Yaqoob Ansari
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for cluster use
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.log_polar_sim2_net import LogPolarSim2Net, create_log_polar_sim2_net
from src.data.synthetic_generator import (
    SyntheticThermalGenerator,
    generate_asymmetric_pattern,
    generate_arrow_pattern,
    generate_L_shape,
    generate_natural_texture,
    generate_thermal_hotspot,
    generate_multi_hotspot,
)
from src.training.metrics import (
    scale_error,
    sim2_component_errors,
    equivariance_score,
    compute_sim2_metrics,
    corner_error,
)

# Suppress matplotlib font warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# =============================================================================
# Configuration Constants
# =============================================================================

ROTATION_ANGLES = [0, 15, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
SCALE_FACTORS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 2.0]
TRANSLATION_MAGNITUDES = [0, 10, 20, 30, 40, 50]

IMAGE_SIZE = (256, 256)
SEED = 42

# Pattern types for synthetic data generation
EVAL_PATTERNS = ["asymmetric", "arrow", "L_shape", "natural", "multi_hotspot"]

# Colormaps for thermal invariance testing
THERMAL_COLORMAPS = {
    "grayscale": None,
    "hot": cv2.COLORMAP_HOT,
    "jet": cv2.COLORMAP_JET,
    "bone": cv2.COLORMAP_BONE,
}


# =============================================================================
# Utility Functions
# =============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def generate_test_pair(
    pattern_type: str,
    rotation_deg: float,
    scale: float,
    tx_px: float,
    ty_px: float,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    noise_std: float = 5.0,
) -> Dict[str, Any]:
    """
    Generate a single test pair with exact transformation parameters.

    Args:
        pattern_type: Type of pattern to generate
        rotation_deg: Rotation angle in degrees
        scale: Scale factor
        tx_px: Translation x in pixels
        ty_px: Translation y in pixels
        image_size: Image dimensions
        noise_std: Noise standard deviation

    Returns:
        Dictionary with image tensors and ground truth parameters
    """
    H_img, W_img = image_size

    # Generate source image
    generators = {
        "asymmetric": lambda: generate_asymmetric_pattern(size=image_size, noise_std=noise_std),
        "arrow": lambda: generate_arrow_pattern(size=image_size, noise_std=noise_std),
        "L_shape": lambda: generate_L_shape(size=image_size, noise_std=noise_std),
        "natural": lambda: generate_natural_texture(size=image_size, noise_std=noise_std),
        "thermal_hotspot": lambda: generate_thermal_hotspot(size=image_size, noise_std=noise_std),
        "multi_hotspot": lambda: generate_multi_hotspot(size=image_size, noise_std=noise_std),
    }

    gen_fn = generators.get(pattern_type, generators["asymmetric"])
    image_src = gen_fn()

    # Build exact transformation matrix
    angle_rad = rotation_deg * math.pi / 180.0
    cx, cy = W_img / 2.0, H_img / 2.0
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    # H = T(center + t) @ S(s) @ R(theta) @ T(-center)
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    SR = np.array([
        [scale * cos_a, -scale * sin_a, 0],
        [scale * sin_a, scale * cos_a, 0],
        [0, 0, 1],
    ], dtype=np.float64)
    T2 = np.array([[1, 0, cx + tx_px], [0, 1, cy + ty_px], [0, 0, 1]], dtype=np.float64)

    H_gt = T2 @ SR @ T1

    # Warp
    image_tgt = cv2.warpPerspective(
        image_src, H_gt.astype(np.float32), (W_img, H_img),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    # Convert to tensors
    src_tensor = torch.from_numpy(image_src).float().unsqueeze(0) / 255.0
    tgt_tensor = torch.from_numpy(image_tgt).float().unsqueeze(0) / 255.0
    H_tensor = torch.from_numpy(H_gt).float()

    return {
        "image_src": src_tensor,
        "image_tgt": tgt_tensor,
        "homography": H_tensor,
        "rotation": torch.tensor(angle_rad, dtype=torch.float32),
        "scale": torch.tensor(scale, dtype=torch.float32),
        "translation": torch.tensor(
            [tx_px / (W_img / 2), ty_px / (H_img / 2)],
            dtype=torch.float32,
        ),
        "rotation_deg": rotation_deg,
        "scale_factor": scale,
        "tx_px": tx_px,
        "ty_px": ty_px,
    }


def batch_from_samples(samples: List[Dict], device: torch.device) -> Dict[str, Tensor]:
    """Stack a list of sample dicts into a batch dict."""
    batch = {}
    batch["image_src"] = torch.stack([s["image_src"] for s in samples]).to(device)
    batch["image_tgt"] = torch.stack([s["image_tgt"] for s in samples]).to(device)
    batch["homography"] = torch.stack([s["homography"] for s in samples]).to(device)
    batch["rotation"] = torch.stack([s["rotation"] for s in samples]).to(device)
    batch["scale"] = torch.stack([s["scale"] for s in samples]).to(device)
    batch["translation"] = torch.stack([s["translation"] for s in samples]).to(device)
    return batch


# =============================================================================
# Classical Baselines
# =============================================================================

def estimate_sift_ransac(
    src_gray: np.ndarray,
    tgt_gray: np.ndarray,
) -> Dict[str, Any]:
    """
    Estimate similarity transform using SIFT keypoints + RANSAC.

    Args:
        src_gray: Source grayscale image (uint8)
        tgt_gray: Target grayscale image (uint8)

    Returns:
        Dictionary with estimated rotation, scale, translation, and metadata
    """
    result = {
        "rotation_deg": 0.0,
        "scale": 1.0,
        "tx_px": 0.0,
        "ty_px": 0.0,
        "success": False,
        "n_keypoints_src": 0,
        "n_keypoints_tgt": 0,
        "n_matches": 0,
        "n_inliers": 0,
    }

    try:
        sift = cv2.SIFT_create(nfeatures=1000)
        kp1, des1 = sift.detectAndCompute(src_gray, None)
        kp2, des2 = sift.detectAndCompute(tgt_gray, None)

        result["n_keypoints_src"] = len(kp1) if kp1 else 0
        result["n_keypoints_tgt"] = len(kp2) if kp2 else 0

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return result

        # Match descriptors
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)

        # Ratio test
        good_matches = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        result["n_matches"] = len(good_matches)

        if len(good_matches) < 4:
            return result

        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # Estimate homography with RANSAC
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

        if H is None:
            return result

        n_inliers = int(mask.sum()) if mask is not None else 0
        result["n_inliers"] = n_inliers

        # Decompose the affine part of the homography
        # H[:2,:2] = s * R for a similarity transform
        a, b = H[0, 0], H[0, 1]
        c, d = H[1, 0], H[1, 1]

        # Scale = sqrt(det(A)) where A is the 2x2 block
        det = a * d - b * c
        scale = math.sqrt(abs(det))

        # Rotation = atan2(c, a) (from the rotation matrix)
        rotation_rad = math.atan2(c, a)
        rotation_deg = math.degrees(rotation_rad)

        # Translation is complex for centered transforms; extract from H
        cx, cy = src_gray.shape[1] / 2, src_gray.shape[0] / 2
        # The translation in the centered convention:
        # H[0,2] = cx*(1 - s*cos) + cy*s*sin + tx
        # H[1,2] = cy*(1 - s*cos) - cx*s*sin + ty
        cos_r = math.cos(rotation_rad)
        sin_r = math.sin(rotation_rad)
        tx = H[0, 2] - cx * (1 - scale * cos_r) - cy * scale * sin_r
        ty = H[1, 2] - cy * (1 - scale * cos_r) + cx * scale * sin_r

        result["rotation_deg"] = rotation_deg
        result["scale"] = scale
        result["tx_px"] = tx
        result["ty_px"] = ty
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def estimate_orb_ransac(
    src_gray: np.ndarray,
    tgt_gray: np.ndarray,
) -> Dict[str, Any]:
    """
    Estimate similarity transform using ORB keypoints + RANSAC.

    Args:
        src_gray: Source grayscale image (uint8)
        tgt_gray: Target grayscale image (uint8)

    Returns:
        Dictionary with estimated rotation, scale, translation, and metadata
    """
    result = {
        "rotation_deg": 0.0,
        "scale": 1.0,
        "tx_px": 0.0,
        "ty_px": 0.0,
        "success": False,
        "n_keypoints_src": 0,
        "n_keypoints_tgt": 0,
        "n_matches": 0,
        "n_inliers": 0,
    }

    try:
        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(src_gray, None)
        kp2, des2 = orb.detectAndCompute(tgt_gray, None)

        result["n_keypoints_src"] = len(kp1) if kp1 else 0
        result["n_keypoints_tgt"] = len(kp2) if kp2 else 0

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return result

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        result["n_matches"] = len(matches)

        if len(matches) < 4:
            return result

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

        if H is None:
            return result

        n_inliers = int(mask.sum()) if mask is not None else 0
        result["n_inliers"] = n_inliers

        # Decompose
        a, b = H[0, 0], H[0, 1]
        c, d = H[1, 0], H[1, 1]
        det = a * d - b * c
        scale_est = math.sqrt(abs(det))
        rotation_rad = math.atan2(c, a)

        cx, cy = src_gray.shape[1] / 2, src_gray.shape[0] / 2
        cos_r = math.cos(rotation_rad)
        sin_r = math.sin(rotation_rad)
        tx = H[0, 2] - cx * (1 - scale_est * cos_r) - cy * scale_est * sin_r
        ty = H[1, 2] - cy * (1 - scale_est * cos_r) + cx * scale_est * sin_r

        result["rotation_deg"] = math.degrees(rotation_rad)
        result["scale"] = scale_est
        result["tx_px"] = tx
        result["ty_px"] = ty
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def estimate_ecc(
    src_gray: np.ndarray,
    tgt_gray: np.ndarray,
    motion_type: int = cv2.MOTION_EUCLIDEAN,
    n_iterations: int = 200,
    epsilon: float = 1e-6,
) -> Dict[str, Any]:
    """
    Estimate transform using Enhanced Correlation Coefficient (ECC).

    Args:
        src_gray: Source grayscale image (uint8)
        tgt_gray: Target grayscale image (uint8)
        motion_type: cv2 motion model (EUCLIDEAN for rotation+translation)
        n_iterations: Maximum iterations
        epsilon: Convergence threshold

    Returns:
        Dictionary with estimated rotation, scale, translation, and metadata
    """
    result = {
        "rotation_deg": 0.0,
        "scale": 1.0,
        "tx_px": 0.0,
        "ty_px": 0.0,
        "success": False,
        "ecc_value": 0.0,
    }

    try:
        # Initialize warp matrix
        if motion_type == cv2.MOTION_EUCLIDEAN:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        elif motion_type == cv2.MOTION_AFFINE:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(3, 3, dtype=np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            n_iterations,
            epsilon,
        )

        cc, warp_matrix = cv2.findTransformECC(
            src_gray, tgt_gray, warp_matrix, motion_type, criteria,
            inputMask=None, gaussFiltSize=5,
        )

        result["ecc_value"] = float(cc)

        # Extract parameters from warp matrix
        if motion_type in (cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE):
            a, b = warp_matrix[0, 0], warp_matrix[0, 1]
            c, d = warp_matrix[1, 0], warp_matrix[1, 1]

            det = a * d - b * c
            scale_est = math.sqrt(abs(det))
            rotation_rad = math.atan2(c, a)

            result["rotation_deg"] = math.degrees(rotation_rad)
            result["scale"] = scale_est
            result["tx_px"] = float(warp_matrix[0, 2])
            result["ty_px"] = float(warp_matrix[1, 2])
        else:
            # Homography
            a, b = warp_matrix[0, 0], warp_matrix[0, 1]
            c, d = warp_matrix[1, 0], warp_matrix[1, 1]
            det = a * d - b * c
            scale_est = math.sqrt(abs(det))
            rotation_rad = math.atan2(c, a)

            result["rotation_deg"] = math.degrees(rotation_rad)
            result["scale"] = scale_est
            result["tx_px"] = float(warp_matrix[0, 2])
            result["ty_px"] = float(warp_matrix[1, 2])

        result["success"] = True

    except cv2.error:
        # ECC can fail to converge
        pass
    except Exception as e:
        result["error"] = str(e)

    return result


# =============================================================================
# Our Model Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_model_on_samples(
    model: LogPolarSim2Net,
    samples: List[Dict],
    device: torch.device,
    batch_size: int = 16,
) -> List[Dict[str, float]]:
    """
    Run our model on a list of samples and return per-sample results.

    Args:
        model: LogPolarSim2Net model instance
        samples: List of sample dicts from generate_test_pair()
        device: Torch device
        batch_size: Batch size for inference

    Returns:
        List of result dicts with predicted and ground truth values
    """
    model.eval()
    results = []

    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i : i + batch_size]
        batch = batch_from_samples(batch_samples, device)

        output = model(batch["image_src"], batch["image_tgt"])

        B = batch["image_src"].shape[0]
        for j in range(B):
            # Predicted values
            rot_pred_deg = output["rotation_deg"][j].item()
            scale_pred = output["scale"][j].item()

            tx_pred = output["translation_x"][j].item() if "translation_x" in output else output["translation"][j, 0].item()
            ty_pred = output["translation_y"][j].item() if "translation_y" in output else output["translation"][j, 1].item()

            # Ground truth
            sample = batch_samples[j]
            rot_gt_deg = sample["rotation_deg"]
            scale_gt = sample["scale_factor"]
            tx_gt_norm = sample["translation"][0].item()
            ty_gt_norm = sample["translation"][1].item()

            # Compute errors
            rot_err = abs(rot_pred_deg - rot_gt_deg)
            # Handle wraparound
            rot_err = min(rot_err, 360.0 - rot_err)

            scale_err_log = abs(math.log(max(scale_pred, 1e-6)) - math.log(max(scale_gt, 1e-6)))

            H_img, W_img = IMAGE_SIZE
            tx_pred_px = tx_pred * (W_img / 2)
            ty_pred_px = ty_pred * (H_img / 2)
            tx_gt_px = tx_gt_norm * (W_img / 2)
            ty_gt_px = ty_gt_norm * (H_img / 2)
            t_err_px = math.sqrt((tx_pred_px - tx_gt_px) ** 2 + (ty_pred_px - ty_gt_px) ** 2)

            results.append({
                "rotation_pred_deg": rot_pred_deg,
                "rotation_gt_deg": rot_gt_deg,
                "rotation_error_deg": rot_err,
                "scale_pred": scale_pred,
                "scale_gt": scale_gt,
                "scale_error_log": scale_err_log,
                "translation_pred_px": (tx_pred_px, ty_pred_px),
                "translation_gt_px": (tx_gt_px, ty_gt_px),
                "translation_error_px": t_err_px,
                "confidence_sr": output["confidence_sr"][j].item(),
            })

    return results


# =============================================================================
# Sweep Evaluation Functions
# =============================================================================

def sweep_rotation(
    model: Optional[LogPolarSim2Net],
    device: torch.device,
    angles: List[float] = ROTATION_ANGLES,
    n_samples: int = 50,
    fixed_scale: float = 1.0,
    fixed_tx: float = 0.0,
    fixed_ty: float = 0.0,
    pattern: str = "asymmetric",
) -> Dict[str, Any]:
    """
    Sweep rotation angles and measure error for all methods.

    Args:
        model: Our trained model (or None for baselines only)
        device: Torch device
        angles: Rotation angles to test
        n_samples: Samples per angle
        fixed_scale: Fixed scale factor
        fixed_tx: Fixed translation x
        fixed_ty: Fixed translation y
        pattern: Pattern type

    Returns:
        Dictionary with results for each method at each angle
    """
    results = {
        "angles": angles,
        "ours": defaultdict(list),
        "sift_ransac": defaultdict(list),
        "orb_ransac": defaultdict(list),
        "ecc": defaultdict(list),
    }

    for angle in angles:
        print(f"  Rotation sweep: {angle}deg ...", end=" ", flush=True)

        # Generate samples for this angle
        samples = []
        for _ in range(n_samples):
            s = generate_test_pair(
                pattern_type=pattern,
                rotation_deg=angle,
                scale=fixed_scale,
                tx_px=fixed_tx,
                ty_px=fixed_ty,
            )
            samples.append(s)

        # --- Our model ---
        if model is not None:
            model_results = evaluate_model_on_samples(model, samples, device)
            rot_errs = [r["rotation_error_deg"] for r in model_results]
            scale_errs = [r["scale_error_log"] for r in model_results]
            t_errs = [r["translation_error_px"] for r in model_results]
            results["ours"]["rotation_error"].append(float(np.mean(rot_errs)))
            results["ours"]["scale_error"].append(float(np.mean(scale_errs)))
            results["ours"]["translation_error"].append(float(np.mean(t_errs)))

        # --- Classical baselines ---
        for idx in range(n_samples):
            src_np = (samples[idx]["image_src"].squeeze().numpy() * 255).astype(np.uint8)
            tgt_np = (samples[idx]["image_tgt"].squeeze().numpy() * 255).astype(np.uint8)
            gt_rot = samples[idx]["rotation_deg"]
            gt_scale = samples[idx]["scale_factor"]

            # SIFT + RANSAC
            sift_res = estimate_sift_ransac(src_np, tgt_np)
            sift_rot_err = abs(sift_res["rotation_deg"] - gt_rot)
            sift_rot_err = min(sift_rot_err, 360.0 - sift_rot_err)
            sift_scale_err = abs(math.log(max(sift_res["scale"], 1e-6)) - math.log(max(gt_scale, 1e-6)))
            results["sift_ransac"]["rotation_error_all"].append(
                {"angle": angle, "error": sift_rot_err, "success": sift_res["success"]}
            )

            # ORB + RANSAC
            orb_res = estimate_orb_ransac(src_np, tgt_np)
            orb_rot_err = abs(orb_res["rotation_deg"] - gt_rot)
            orb_rot_err = min(orb_rot_err, 360.0 - orb_rot_err)
            results["orb_ransac"]["rotation_error_all"].append(
                {"angle": angle, "error": orb_rot_err, "success": orb_res["success"]}
            )

            # ECC
            ecc_res = estimate_ecc(src_np, tgt_np)
            ecc_rot_err = abs(ecc_res["rotation_deg"] - gt_rot)
            ecc_rot_err = min(ecc_rot_err, 360.0 - ecc_rot_err)
            results["ecc"]["rotation_error_all"].append(
                {"angle": angle, "error": ecc_rot_err, "success": ecc_res["success"]}
            )

        # Aggregate baseline errors per angle
        for method in ["sift_ransac", "orb_ransac", "ecc"]:
            angle_entries = [
                e for e in results[method]["rotation_error_all"]
                if e["angle"] == angle
            ]
            successes = [e for e in angle_entries if e["success"]]
            if successes:
                results[method]["rotation_error"].append(float(np.mean([e["error"] for e in successes])))
            else:
                results[method]["rotation_error"].append(180.0)  # Failure penalty
            results[method]["success_rate"].append(
                len(successes) / max(len(angle_entries), 1)
            )

        print("done")

    # Clean up intermediate data
    for method in ["sift_ransac", "orb_ransac", "ecc"]:
        if "rotation_error_all" in results[method]:
            del results[method]["rotation_error_all"]

    return results


def sweep_scale(
    model: Optional[LogPolarSim2Net],
    device: torch.device,
    scales: List[float] = SCALE_FACTORS,
    n_samples: int = 50,
    fixed_rotation: float = 0.0,
    fixed_tx: float = 0.0,
    fixed_ty: float = 0.0,
    pattern: str = "asymmetric",
) -> Dict[str, Any]:
    """Sweep scale factors and measure error for all methods."""
    results = {
        "scales": scales,
        "ours": defaultdict(list),
        "sift_ransac": defaultdict(list),
        "orb_ransac": defaultdict(list),
        "ecc": defaultdict(list),
    }

    for s in scales:
        print(f"  Scale sweep: {s:.1f}x ...", end=" ", flush=True)

        samples = []
        for _ in range(n_samples):
            sample = generate_test_pair(
                pattern_type=pattern,
                rotation_deg=fixed_rotation,
                scale=s,
                tx_px=fixed_tx,
                ty_px=fixed_ty,
            )
            samples.append(sample)

        # Our model
        if model is not None:
            model_results = evaluate_model_on_samples(model, samples, device)
            scale_errs = [r["scale_error_log"] for r in model_results]
            rot_errs = [r["rotation_error_deg"] for r in model_results]
            results["ours"]["scale_error"].append(float(np.mean(scale_errs)))
            results["ours"]["rotation_error"].append(float(np.mean(rot_errs)))

        # Baselines
        for idx in range(n_samples):
            src_np = (samples[idx]["image_src"].squeeze().numpy() * 255).astype(np.uint8)
            tgt_np = (samples[idx]["image_tgt"].squeeze().numpy() * 255).astype(np.uint8)
            gt_scale = samples[idx]["scale_factor"]

            sift_res = estimate_sift_ransac(src_np, tgt_np)
            sift_scale_err = abs(math.log(max(sift_res["scale"], 1e-6)) - math.log(max(gt_scale, 1e-6)))
            results["sift_ransac"]["scale_error_all"].append(
                {"scale": s, "error": sift_scale_err, "success": sift_res["success"]}
            )

            orb_res = estimate_orb_ransac(src_np, tgt_np)
            orb_scale_err = abs(math.log(max(orb_res["scale"], 1e-6)) - math.log(max(gt_scale, 1e-6)))
            results["orb_ransac"]["scale_error_all"].append(
                {"scale": s, "error": orb_scale_err, "success": orb_res["success"]}
            )

            ecc_res = estimate_ecc(src_np, tgt_np)
            ecc_scale_err = abs(math.log(max(ecc_res["scale"], 1e-6)) - math.log(max(gt_scale, 1e-6)))
            results["ecc"]["scale_error_all"].append(
                {"scale": s, "error": ecc_scale_err, "success": ecc_res["success"]}
            )

        for method in ["sift_ransac", "orb_ransac", "ecc"]:
            entries = [e for e in results[method]["scale_error_all"] if e["scale"] == s]
            successes = [e for e in entries if e["success"]]
            if successes:
                results[method]["scale_error"].append(float(np.mean([e["error"] for e in successes])))
            else:
                results[method]["scale_error"].append(2.0)  # Failure penalty
            results[method]["success_rate"].append(
                len(successes) / max(len(entries), 1)
            )

        print("done")

    for method in ["sift_ransac", "orb_ransac", "ecc"]:
        if "scale_error_all" in results[method]:
            del results[method]["scale_error_all"]

    return results


def sweep_translation(
    model: Optional[LogPolarSim2Net],
    device: torch.device,
    magnitudes: List[float] = TRANSLATION_MAGNITUDES,
    n_samples: int = 50,
    fixed_rotation: float = 0.0,
    fixed_scale: float = 1.0,
    pattern: str = "asymmetric",
) -> Dict[str, Any]:
    """Sweep translation magnitudes and measure error for all methods."""
    results = {
        "magnitudes": magnitudes,
        "ours": defaultdict(list),
        "sift_ransac": defaultdict(list),
        "orb_ransac": defaultdict(list),
        "ecc": defaultdict(list),
    }

    for mag in magnitudes:
        print(f"  Translation sweep: {mag}px ...", end=" ", flush=True)

        samples = []
        for _ in range(n_samples):
            # Random direction for given magnitude
            if mag == 0:
                tx, ty = 0.0, 0.0
            else:
                angle = np.random.uniform(0, 2 * math.pi)
                tx = mag * math.cos(angle)
                ty = mag * math.sin(angle)

            sample = generate_test_pair(
                pattern_type=pattern,
                rotation_deg=fixed_rotation,
                scale=fixed_scale,
                tx_px=tx,
                ty_px=ty,
            )
            samples.append(sample)

        # Our model
        if model is not None:
            model_results = evaluate_model_on_samples(model, samples, device)
            t_errs = [r["translation_error_px"] for r in model_results]
            rot_errs = [r["rotation_error_deg"] for r in model_results]
            results["ours"]["translation_error"].append(float(np.mean(t_errs)))
            results["ours"]["rotation_error"].append(float(np.mean(rot_errs)))

        # Baselines
        for idx in range(n_samples):
            src_np = (samples[idx]["image_src"].squeeze().numpy() * 255).astype(np.uint8)
            tgt_np = (samples[idx]["image_tgt"].squeeze().numpy() * 255).astype(np.uint8)
            gt_tx = samples[idx]["tx_px"]
            gt_ty = samples[idx]["ty_px"]

            sift_res = estimate_sift_ransac(src_np, tgt_np)
            sift_t_err = math.sqrt(
                (sift_res["tx_px"] - gt_tx) ** 2 + (sift_res["ty_px"] - gt_ty) ** 2
            )
            results["sift_ransac"]["translation_error_all"].append(
                {"mag": mag, "error": sift_t_err, "success": sift_res["success"]}
            )

            orb_res = estimate_orb_ransac(src_np, tgt_np)
            orb_t_err = math.sqrt(
                (orb_res["tx_px"] - gt_tx) ** 2 + (orb_res["ty_px"] - gt_ty) ** 2
            )
            results["orb_ransac"]["translation_error_all"].append(
                {"mag": mag, "error": orb_t_err, "success": orb_res["success"]}
            )

            ecc_res = estimate_ecc(src_np, tgt_np)
            ecc_t_err = math.sqrt(
                (ecc_res["tx_px"] - gt_tx) ** 2 + (ecc_res["ty_px"] - gt_ty) ** 2
            )
            results["ecc"]["translation_error_all"].append(
                {"mag": mag, "error": ecc_t_err, "success": ecc_res["success"]}
            )

        for method in ["sift_ransac", "orb_ransac", "ecc"]:
            entries = [e for e in results[method]["translation_error_all"] if e["mag"] == mag]
            successes = [e for e in entries if e["success"]]
            if successes:
                results[method]["translation_error"].append(float(np.mean([e["error"] for e in successes])))
            else:
                results[method]["translation_error"].append(100.0)  # Failure penalty
            results[method]["success_rate"].append(
                len(successes) / max(len(entries), 1)
            )

        print("done")

    for method in ["sift_ransac", "orb_ransac", "ecc"]:
        if "translation_error_all" in results[method]:
            del results[method]["translation_error_all"]

    return results


# =============================================================================
# Thermal-Specific Tests
# =============================================================================

def thermal_colormap_test(
    model: Optional[LogPolarSim2Net],
    device: torch.device,
    n_samples: int = 30,
) -> Dict[str, Any]:
    """
    Test colormap invariance on thermal images.

    Our FFT-magnitude approach should be invariant to colormaps because
    FFT magnitude is intensity-distribution invariant. Keypoint methods
    (SIFT/ORB) should fail on thermal images due to low texture and
    colormap-dependent features.

    Args:
        model: Our model (or None)
        device: Torch device
        n_samples: Samples per colormap

    Returns:
        Results for each method and colormap combination
    """
    print("\n--- Thermal Colormap Invariance Test ---")
    results = {}

    thermal_patterns = ["multi_hotspot", "thermal_hotspot"]

    for cmap_name, cmap_code in THERMAL_COLORMAPS.items():
        print(f"  Colormap: {cmap_name} ...", end=" ", flush=True)

        all_model_rot_errs = []
        all_sift_rot_errs = []
        all_orb_rot_errs = []
        sift_successes = 0
        orb_successes = 0
        sift_kp_counts = []
        orb_kp_counts = []

        for _ in range(n_samples):
            pattern = np.random.choice(thermal_patterns)
            rot = np.random.uniform(-60, 60)
            scale = np.random.uniform(0.9, 1.1)

            sample = generate_test_pair(
                pattern_type=pattern,
                rotation_deg=rot,
                scale=scale,
                tx_px=0.0,
                ty_px=0.0,
            )

            # Apply colormap
            src_gray = (sample["image_src"].squeeze().numpy() * 255).astype(np.uint8)
            tgt_gray = (sample["image_tgt"].squeeze().numpy() * 255).astype(np.uint8)

            if cmap_code is not None:
                src_color = cv2.applyColorMap(src_gray, cmap_code)
                tgt_color = cv2.applyColorMap(tgt_gray, cmap_code)
                # Convert back to grayscale (simulates what happens when
                # keypoint detectors see colormapped thermal)
                src_for_kp = cv2.cvtColor(src_color, cv2.COLOR_BGR2GRAY)
                tgt_for_kp = cv2.cvtColor(tgt_color, cv2.COLOR_BGR2GRAY)
            else:
                src_for_kp = src_gray
                tgt_for_kp = tgt_gray

            # Our model -- always uses raw grayscale (FFT magnitude)
            if model is not None:
                model_res = evaluate_model_on_samples(model, [sample], device)
                all_model_rot_errs.append(model_res[0]["rotation_error_deg"])

            # SIFT on colormapped image
            sift_res = estimate_sift_ransac(src_for_kp, tgt_for_kp)
            sift_rot_err = abs(sift_res["rotation_deg"] - rot)
            sift_rot_err = min(sift_rot_err, 360.0 - sift_rot_err)
            all_sift_rot_errs.append(sift_rot_err)
            sift_successes += int(sift_res["success"])
            sift_kp_counts.append(sift_res["n_keypoints_src"])

            # ORB on colormapped image
            orb_res = estimate_orb_ransac(src_for_kp, tgt_for_kp)
            orb_rot_err = abs(orb_res["rotation_deg"] - rot)
            orb_rot_err = min(orb_rot_err, 360.0 - orb_rot_err)
            all_orb_rot_errs.append(orb_rot_err)
            orb_successes += int(orb_res["success"])
            orb_kp_counts.append(orb_res["n_keypoints_src"])

        results[cmap_name] = {
            "ours_rotation_mae": float(np.mean(all_model_rot_errs)) if all_model_rot_errs else None,
            "sift_rotation_mae": float(np.mean(all_sift_rot_errs)),
            "orb_rotation_mae": float(np.mean(all_orb_rot_errs)),
            "sift_success_rate": sift_successes / n_samples,
            "orb_success_rate": orb_successes / n_samples,
            "sift_mean_keypoints": float(np.mean(sift_kp_counts)),
            "orb_mean_keypoints": float(np.mean(orb_kp_counts)),
        }

        print("done")

    return results


# =============================================================================
# Equivariance Analysis
# =============================================================================

def compute_equivariance_metrics(
    rotation_results: Dict,
    scale_results: Dict,
) -> Dict[str, Any]:
    """
    Compute equivariance scores from sweep results.

    The key claim: equivariant models have FLAT error curves, meaning
    error does not depend on the transformation parameter. This is
    quantified by the coefficient of variation (CV = std/mean).

    Args:
        rotation_results: Output of sweep_rotation()
        scale_results: Output of sweep_scale()

    Returns:
        Equivariance metrics for each method
    """
    print("\n--- Computing Equivariance Scores ---")
    eq_metrics = {}

    # Rotation equivariance
    angles = rotation_results["angles"]
    for method_name in ["ours", "sift_ransac", "orb_ransac", "ecc"]:
        method_data = rotation_results.get(method_name, {})
        rot_errors = method_data.get("rotation_error", [])

        if rot_errors:
            errors_by_angle = dict(zip(angles, rot_errors))
            eq = equivariance_score(errors_by_angle)
            eq_metrics[f"{method_name}_rotation_equivariance"] = eq
            print(
                f"  {method_name:15s} rotation: score={eq['score']:.3f}, "
                f"CV={eq['cv']:.3f}, equivariant={eq['is_equivariant']}"
            )

    # Scale equivariance
    scales = scale_results["scales"]
    for method_name in ["ours", "sift_ransac", "orb_ransac", "ecc"]:
        method_data = scale_results.get(method_name, {})
        scale_errors = method_data.get("scale_error", [])

        if scale_errors:
            errors_by_scale = dict(zip(scales, scale_errors))
            eq = equivariance_score(errors_by_scale)
            eq_metrics[f"{method_name}_scale_equivariance"] = eq
            print(
                f"  {method_name:15s} scale:    score={eq['score']:.3f}, "
                f"CV={eq['cv']:.3f}, equivariant={eq['is_equivariant']}"
            )

    return eq_metrics


# =============================================================================
# Plotting
# =============================================================================

def plot_rotation_sweep(results: Dict, output_dir: str, title_suffix: str = ""):
    """Plot rotation error vs angle for all methods."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    angles = results["angles"]

    # Plot each method
    method_styles = {
        "ours": {"color": "#2196F3", "marker": "o", "linewidth": 2.5, "label": "Ours (LogPolarSim2Net)"},
        "sift_ransac": {"color": "#FF9800", "marker": "s", "linewidth": 1.5, "label": "SIFT + RANSAC"},
        "orb_ransac": {"color": "#4CAF50", "marker": "^", "linewidth": 1.5, "label": "ORB + RANSAC"},
        "ecc": {"color": "#9C27B0", "marker": "D", "linewidth": 1.5, "label": "ECC"},
    }

    for method_name, style in method_styles.items():
        method_data = results.get(method_name, {})
        errors = method_data.get("rotation_error", [])
        if errors and len(errors) == len(angles):
            ax.plot(
                angles, errors,
                color=style["color"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                label=style["label"],
                markersize=6,
            )

    ax.set_xlabel("Ground Truth Rotation Angle (degrees)", fontsize=12)
    ax.set_ylabel("Rotation MAE (degrees)", fontsize=12)
    ax.set_title(f"Rotation Error vs. Rotation Angle{title_suffix}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(angles)
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = os.path.join(output_dir, "rotation_sweep.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    path_png = os.path.join(output_dir, "rotation_sweep.png")
    fig.savefig(path_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_scale_sweep(results: Dict, output_dir: str, title_suffix: str = ""):
    """Plot scale error vs scale factor for all methods."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    scales = results["scales"]

    method_styles = {
        "ours": {"color": "#2196F3", "marker": "o", "linewidth": 2.5, "label": "Ours (LogPolarSim2Net)"},
        "sift_ransac": {"color": "#FF9800", "marker": "s", "linewidth": 1.5, "label": "SIFT + RANSAC"},
        "orb_ransac": {"color": "#4CAF50", "marker": "^", "linewidth": 1.5, "label": "ORB + RANSAC"},
        "ecc": {"color": "#9C27B0", "marker": "D", "linewidth": 1.5, "label": "ECC"},
    }

    for method_name, style in method_styles.items():
        method_data = results.get(method_name, {})
        errors = method_data.get("scale_error", [])
        if errors and len(errors) == len(scales):
            ax.plot(
                scales, errors,
                color=style["color"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                label=style["label"],
                markersize=6,
            )

    ax.set_xlabel("Ground Truth Scale Factor", fontsize=12)
    ax.set_ylabel("Scale Error (log-space)", fontsize=12)
    ax.set_title(f"Scale Error vs. Scale Factor{title_suffix}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "scale_sweep.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    path_png = os.path.join(output_dir, "scale_sweep.png")
    fig.savefig(path_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_translation_sweep(results: Dict, output_dir: str, title_suffix: str = ""):
    """Plot translation error vs translation magnitude for all methods."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    magnitudes = results["magnitudes"]

    method_styles = {
        "ours": {"color": "#2196F3", "marker": "o", "linewidth": 2.5, "label": "Ours (LogPolarSim2Net)"},
        "sift_ransac": {"color": "#FF9800", "marker": "s", "linewidth": 1.5, "label": "SIFT + RANSAC"},
        "orb_ransac": {"color": "#4CAF50", "marker": "^", "linewidth": 1.5, "label": "ORB + RANSAC"},
        "ecc": {"color": "#9C27B0", "marker": "D", "linewidth": 1.5, "label": "ECC"},
    }

    for method_name, style in method_styles.items():
        method_data = results.get(method_name, {})
        errors = method_data.get("translation_error", [])
        if errors and len(errors) == len(magnitudes):
            ax.plot(
                magnitudes, errors,
                color=style["color"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                label=style["label"],
                markersize=6,
            )

    ax.set_xlabel("Ground Truth Translation Magnitude (pixels)", fontsize=12)
    ax.set_ylabel("Translation Error (pixels)", fontsize=12)
    ax.set_title(f"Translation Error vs. Translation Magnitude{title_suffix}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "translation_sweep.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    path_png = os.path.join(output_dir, "translation_sweep.png")
    fig.savefig(path_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_method_comparison_bar(
    all_results: Dict,
    output_dir: str,
):
    """Bar chart comparing overall accuracy of all methods."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    methods = ["Ours", "SIFT+RANSAC", "ORB+RANSAC", "ECC"]
    method_keys = ["ours", "sift_ransac", "orb_ransac", "ecc"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    # Rotation error
    rot_results = all_results.get("rotation_sweep", {})
    rot_errors = []
    for mk in method_keys:
        method_data = rot_results.get(mk, {})
        errs = method_data.get("rotation_error", [])
        rot_errors.append(float(np.mean(errs)) if errs else 0.0)

    axes[0].bar(methods, rot_errors, color=colors)
    axes[0].set_ylabel("Mean Rotation MAE (deg)")
    axes[0].set_title("Rotation Accuracy")
    axes[0].tick_params(axis="x", rotation=30)

    # Scale error
    scale_results = all_results.get("scale_sweep", {})
    scale_errors = []
    for mk in method_keys:
        method_data = scale_results.get(mk, {})
        errs = method_data.get("scale_error", [])
        scale_errors.append(float(np.mean(errs)) if errs else 0.0)

    axes[1].bar(methods, scale_errors, color=colors)
    axes[1].set_ylabel("Mean Scale Error (log)")
    axes[1].set_title("Scale Accuracy")
    axes[1].tick_params(axis="x", rotation=30)

    # Translation error
    trans_results = all_results.get("translation_sweep", {})
    trans_errors = []
    for mk in method_keys:
        method_data = trans_results.get(mk, {})
        errs = method_data.get("translation_error", [])
        trans_errors.append(float(np.mean(errs)) if errs else 0.0)

    axes[2].bar(methods, trans_errors, color=colors)
    axes[2].set_ylabel("Mean Translation Error (px)")
    axes[2].set_title("Translation Accuracy")
    axes[2].tick_params(axis="x", rotation=30)

    plt.suptitle("Method Comparison: Sim(2) Component Accuracy", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "method_comparison.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    path_png = os.path.join(output_dir, "method_comparison.png")
    fig.savefig(path_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_equivariance_summary(eq_metrics: Dict, output_dir: str):
    """Plot equivariance scores as a grouped bar chart."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    methods = ["Ours", "SIFT+RANSAC", "ORB+RANSAC", "ECC"]
    method_keys = ["ours", "sift_ransac", "orb_ransac", "ecc"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    x = np.arange(len(methods))
    width = 0.35

    # Rotation equivariance scores
    rot_scores = []
    for mk in method_keys:
        key = f"{mk}_rotation_equivariance"
        if key in eq_metrics:
            rot_scores.append(eq_metrics[key]["score"])
        else:
            rot_scores.append(0.0)

    # Scale equivariance scores
    scale_scores = []
    for mk in method_keys:
        key = f"{mk}_scale_equivariance"
        if key in eq_metrics:
            scale_scores.append(eq_metrics[key]["score"])
        else:
            scale_scores.append(0.0)

    bars1 = ax.bar(x - width / 2, rot_scores, width, label="Rotation Equivariance", color="#2196F3", alpha=0.8)
    bars2 = ax.bar(x + width / 2, scale_scores, width, label="Scale Equivariance", color="#FF9800", alpha=0.8)

    # Add threshold line
    ax.axhline(y=0.9, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Equivariance Threshold (0.9)")

    ax.set_ylabel("Equivariance Score (1 = perfectly flat)", fontsize=12)
    ax.set_title("Equivariance Scores by Method", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    path = os.path.join(output_dir, "equivariance_scores.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    path_png = os.path.join(output_dir, "equivariance_scores.png")
    fig.savefig(path_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_thermal_colormap_results(thermal_results: Dict, output_dir: str):
    """Plot thermal colormap invariance results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cmaps = list(thermal_results.keys())
    methods = ["ours", "sift", "orb"]
    method_labels = ["Ours", "SIFT", "ORB"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    # Rotation MAE
    x = np.arange(len(cmaps))
    width = 0.25
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        key = f"{method}_rotation_mae"
        vals = []
        for cmap in cmaps:
            v = thermal_results[cmap].get(key)
            vals.append(v if v is not None else 0.0)
        axes[0].bar(x + i * width, vals, width, label=label, color=color, alpha=0.8)

    axes[0].set_xlabel("Colormap")
    axes[0].set_ylabel("Rotation MAE (degrees)")
    axes[0].set_title("Rotation Error by Colormap")
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(cmaps)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Success rate
    for i, (method, label, color) in enumerate(zip(["sift", "orb"], ["SIFT", "ORB"], ["#FF9800", "#4CAF50"])):
        key = f"{method}_success_rate"
        vals = [thermal_results[cmap].get(key, 0.0) * 100 for cmap in cmaps]
        axes[1].bar(x + i * 0.35, vals, 0.35, label=label, color=color, alpha=0.8)

    axes[1].set_xlabel("Colormap")
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_title("Keypoint Matching Success Rate")
    axes[1].set_xticks(x + 0.175)
    axes[1].set_xticklabels(cmaps)
    axes[1].legend()
    axes[1].set_ylim(0, 110)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Thermal Image Colormap Invariance", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "thermal_colormap.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    path_png = os.path.join(output_dir, "thermal_colormap.png")
    fig.savefig(path_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# Print Tables
# =============================================================================

def print_summary_table(all_results: Dict):
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SIM(2) EVALUATION RESULTS")
    print("=" * 80)

    # Overall accuracy
    print("\n--- Per-Component Accuracy (mean across all parameters) ---")
    print(f"{'Method':<20} {'Rot MAE (deg)':<15} {'Scale Err (log)':<17} {'Trans Err (px)':<15}")
    print("-" * 67)

    methods = [
        ("Ours", "ours"),
        ("SIFT+RANSAC", "sift_ransac"),
        ("ORB+RANSAC", "orb_ransac"),
        ("ECC", "ecc"),
    ]

    for label, key in methods:
        rot_sweep = all_results.get("rotation_sweep", {})
        rot_errs = rot_sweep.get(key, {}).get("rotation_error", [])
        rot_mean = f"{np.mean(rot_errs):.2f}" if rot_errs else "N/A"

        scale_sweep = all_results.get("scale_sweep", {})
        scale_errs = scale_sweep.get(key, {}).get("scale_error", [])
        scale_mean = f"{np.mean(scale_errs):.4f}" if scale_errs else "N/A"

        trans_sweep = all_results.get("translation_sweep", {})
        trans_errs = trans_sweep.get(key, {}).get("translation_error", [])
        trans_mean = f"{np.mean(trans_errs):.2f}" if trans_errs else "N/A"

        print(f"{label:<20} {rot_mean:<15} {scale_mean:<17} {trans_mean:<15}")

    # Equivariance scores
    eq = all_results.get("equivariance_metrics", {})
    if eq:
        print("\n--- Equivariance Scores (higher = flatter error curve) ---")
        print(f"{'Method':<20} {'Rot Score':<12} {'Rot CV':<10} {'Rot Equiv?':<12} {'Scale Score':<12} {'Scale CV':<10} {'Scale Equiv?':<12}")
        print("-" * 88)

        for label, key in methods:
            rot_eq = eq.get(f"{key}_rotation_equivariance", {})
            scale_eq = eq.get(f"{key}_scale_equivariance", {})

            rot_score = f"{rot_eq.get('score', 0):.3f}" if rot_eq else "N/A"
            rot_cv = f"{rot_eq.get('cv', 0):.3f}" if rot_eq else "N/A"
            rot_is = "YES" if rot_eq.get("is_equivariant", False) else "NO"

            scale_score = f"{scale_eq.get('score', 0):.3f}" if scale_eq else "N/A"
            scale_cv = f"{scale_eq.get('cv', 0):.3f}" if scale_eq else "N/A"
            scale_is = "YES" if scale_eq.get("is_equivariant", False) else "NO"

            print(
                f"{label:<20} {rot_score:<12} {rot_cv:<10} {rot_is:<12} "
                f"{scale_score:<12} {scale_cv:<10} {scale_is:<12}"
            )

    # Thermal results
    thermal = all_results.get("thermal_colormap", {})
    if thermal:
        print("\n--- Thermal Colormap Invariance ---")
        print(f"{'Colormap':<15} {'Ours MAE':<12} {'SIFT MAE':<12} {'ORB MAE':<12} {'SIFT Success':<14} {'ORB Success':<12}")
        print("-" * 77)

        for cmap, data in thermal.items():
            ours_mae = f"{data.get('ours_rotation_mae', 0):.2f}" if data.get("ours_rotation_mae") is not None else "N/A"
            sift_mae = f"{data.get('sift_rotation_mae', 0):.2f}"
            orb_mae = f"{data.get('orb_rotation_mae', 0):.2f}"
            sift_sr = f"{data.get('sift_success_rate', 0) * 100:.0f}%"
            orb_sr = f"{data.get('orb_success_rate', 0) * 100:.0f}%"

            print(f"{cmap:<15} {ours_mae:<12} {sift_mae:<12} {orb_mae:<12} {sift_sr:<14} {orb_sr:<12}")

    print("\n" + "=" * 80)


# =============================================================================
# JSON serialization helpers
# =============================================================================

def make_json_serializable(obj):
    """Recursively convert numpy/torch types to Python native types for JSON."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, defaultdict):
        return make_json_serializable(dict(obj))
    return obj


# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Sim(2) Similarity Estimation Evaluation"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained model checkpoint (.pt file). "
             "If not provided, tests with untrained model (classical FMT baseline)."
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/similarity_evaluation",
        help="Directory to save results, plots, and JSON"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=50,
        help="Number of samples per parameter value"
    )
    parser.add_argument(
        "--pattern", type=str, default="asymmetric",
        choices=EVAL_PATTERNS,
        help="Pattern type for synthetic data"
    )
    parser.add_argument(
        "--skip-baselines", action="store_true",
        help="Skip classical baselines (faster, model-only evaluation)"
    )
    parser.add_argument(
        "--skip-thermal", action="store_true",
        help="Skip thermal-specific tests"
    )
    args = parser.parse_args()

    # Setup
    set_seed(SEED)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Sim(2) Similarity Estimation Evaluation")
    print("=" * 70)
    print(f"Device:       {device}")
    print(f"Checkpoint:   {args.checkpoint or 'None (untrained FMT baseline)'}")
    print(f"N-samples:    {args.n_samples}")
    print(f"Pattern:      {args.pattern}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Image size:   {IMAGE_SIZE}")
    print(f"Seed:         {SEED}")
    print("=" * 70)

    # ---- Load or create model ----
    print("\n--- Loading Model ---")
    model = create_log_polar_sim2_net(
        lp_angles=180,
        lp_radii=64,
        r_min=0.05,
        r_max=0.9,
        feature_channels=64,
        use_learned_features=True,
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        elif "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
        print("  Checkpoint loaded successfully.")
    else:
        if args.checkpoint:
            print(f"  WARNING: Checkpoint not found at {args.checkpoint}")
        print("  Using untrained model (classical FMT baseline mode)")

    model = model.to(device)
    model.eval()

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")

    all_results = {
        "config": {
            "checkpoint": args.checkpoint,
            "n_samples": args.n_samples,
            "pattern": args.pattern,
            "image_size": list(IMAGE_SIZE),
            "seed": SEED,
            "device": str(device),
            "n_params": n_params,
            "n_trainable": n_trainable,
            "rotation_angles": ROTATION_ANGLES,
            "scale_factors": SCALE_FACTORS,
            "translation_magnitudes": TRANSLATION_MAGNITUDES,
        }
    }

    t_start = time.time()

    # ---- 1. Rotation Sweep ----
    print("\n" + "=" * 50)
    print("1. ROTATION SWEEP")
    print("=" * 50)
    rotation_results = sweep_rotation(
        model=model,
        device=device,
        angles=ROTATION_ANGLES,
        n_samples=args.n_samples,
        pattern=args.pattern,
    )
    all_results["rotation_sweep"] = rotation_results

    # ---- 2. Scale Sweep ----
    print("\n" + "=" * 50)
    print("2. SCALE SWEEP")
    print("=" * 50)
    scale_results = sweep_scale(
        model=model,
        device=device,
        scales=SCALE_FACTORS,
        n_samples=args.n_samples,
        pattern=args.pattern,
    )
    all_results["scale_sweep"] = scale_results

    # ---- 3. Translation Sweep ----
    print("\n" + "=" * 50)
    print("3. TRANSLATION SWEEP")
    print("=" * 50)
    translation_results = sweep_translation(
        model=model,
        device=device,
        magnitudes=TRANSLATION_MAGNITUDES,
        n_samples=args.n_samples,
        pattern=args.pattern,
    )
    all_results["translation_sweep"] = translation_results

    # ---- 4. Equivariance Metrics ----
    print("\n" + "=" * 50)
    print("4. EQUIVARIANCE METRICS")
    print("=" * 50)
    eq_metrics = compute_equivariance_metrics(rotation_results, scale_results)
    all_results["equivariance_metrics"] = eq_metrics

    # ---- 5. Thermal Colormap Invariance ----
    if not args.skip_thermal:
        print("\n" + "=" * 50)
        print("5. THERMAL COLORMAP INVARIANCE")
        print("=" * 50)
        thermal_results = thermal_colormap_test(
            model=model,
            device=device,
            n_samples=args.n_samples,
        )
        all_results["thermal_colormap"] = thermal_results

    # ---- 6. Generalization Gap ----
    print("\n" + "=" * 50)
    print("6. GENERALIZATION GAP")
    print("=" * 50)
    print("  Testing in-distribution vs out-of-distribution performance...")

    # In-distribution: rotation in [-30, 30] (typical training range)
    in_dist_angles = [0, 15, 30]
    out_dist_angles = [90, 120, 150, 180, 210, 240, 270]

    rot_data = rotation_results.get("ours", {}).get("rotation_error", [])
    if rot_data:
        angle_to_error = dict(zip(ROTATION_ANGLES, rot_data))

        in_dist_errors = [angle_to_error.get(a, 0) for a in in_dist_angles if a in angle_to_error]
        out_dist_errors = [angle_to_error.get(a, 0) for a in out_dist_angles if a in angle_to_error]

        in_mean = float(np.mean(in_dist_errors)) if in_dist_errors else 0.0
        out_mean = float(np.mean(out_dist_errors)) if out_dist_errors else 0.0
        gen_gap = out_mean / (in_mean + 1e-10)

        all_results["generalization"] = {
            "in_distribution_angles": in_dist_angles,
            "out_distribution_angles": out_dist_angles,
            "in_distribution_error": in_mean,
            "out_distribution_error": out_mean,
            "generalization_gap": gen_gap,
            "is_generalizing": gen_gap < 2.0,
        }

        print(f"  In-distribution error:  {in_mean:.2f} deg (angles: {in_dist_angles})")
        print(f"  Out-distribution error: {out_mean:.2f} deg (angles: {out_dist_angles})")
        print(f"  Generalization gap:     {gen_gap:.2f}x")
        print(f"  Generalizes well:       {'YES' if gen_gap < 2.0 else 'NO'}")

    t_total = time.time() - t_start
    all_results["runtime_seconds"] = t_total

    # ---- Print Tables ----
    print_summary_table(all_results)

    # ---- Save JSON ----
    json_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(make_json_serializable(all_results), f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # ---- Generate Plots ----
    print("\n--- Generating Plots ---")
    plot_rotation_sweep(rotation_results, args.output_dir)
    plot_scale_sweep(scale_results, args.output_dir)
    plot_translation_sweep(translation_results, args.output_dir)
    plot_method_comparison_bar(all_results, args.output_dir)
    if eq_metrics:
        plot_equivariance_summary(eq_metrics, args.output_dir)
    if not args.skip_thermal and "thermal_colormap" in all_results:
        plot_thermal_colormap_results(all_results["thermal_colormap"], args.output_dir)

    print(f"\nTotal evaluation time: {t_total:.1f}s")
    print(f"All results saved to: {args.output_dir}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
