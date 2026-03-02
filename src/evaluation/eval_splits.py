"""
Evaluation on Different Data Splits

Implements evaluation protocols:
1. Temporal split: Train scenes 1-10 → Test scenes 11-12
2. Geometric split: Train rotations <30° → Test >45°
3. Colormap split: Train "hot" → Test "jet"
4. Adversarial split: Manually curated hard pairs
"""

import json
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import SyntheticThermalGenerator, ThermalPairDataset
from src.utils.homography import homography_matrix_to_vec
from src.training.metrics import (
    compute_all_metrics,
    corner_error,
    rotation_error,
    translation_error,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> dict[str, float]:
    """
    Evaluate model on a dataloader.

    Args:
        model: Model to evaluate
        dataloader: DataLoader with test data
        device: Device to use
        verbose: Whether to show progress bar

    Returns:
        Dictionary of metrics
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_gts = []

    iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader

    with torch.no_grad():
        for batch in iterator:
            image_src = batch["image_src"].to(device)
            image_tgt = batch["image_tgt"].to(device)
            H_gt = batch["homography_vec"]

            # Forward pass
            output = model(image_src, image_tgt)
            H_pred = output["homography"].cpu()

            all_preds.append(H_pred)
            all_gts.append(H_gt)

    # Concatenate all predictions
    H_pred = torch.cat(all_preds, dim=0)
    H_gt = torch.cat(all_gts, dim=0)

    # Compute metrics
    metrics = compute_all_metrics(H_pred, H_gt)

    return metrics


def evaluate_rotation_equivariance(
    model: torch.nn.Module,
    base_dataset,
    angles: list[float] | None = None,
    n_samples: int = 100,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    """
    Evaluate rotation equivariance.

    Tests model accuracy at different input rotations.
    For an equivariant model, accuracy should be constant.

    Args:
        model: Model to evaluate
        base_dataset: Base dataset (unrotated)
        angles: Rotation angles to test
        n_samples: Number of samples per angle
        device: Device to use

    Returns:
        Dictionary with angle-wise metrics
    """
    import cv2

    if angles is None:
        angles = [0, 15, 30, 45, 60, 90, 120, 150, 180]

    model.eval()
    model.to(device)

    results = {
        "angles": angles,
        "corner_error": [],
        "rotation_error": [],
        "translation_error": [],
    }

    for angle in tqdm(angles, desc="Testing rotation equivariance"):
        angle_rad = angle * np.pi / 180

        errors_corner = []
        errors_rot = []
        errors_trans = []

        for idx in range(min(n_samples, len(base_dataset))):
            sample = base_dataset[idx]

            # Get images and homography
            src_np = (sample["image_src"].squeeze().numpy() * 255).astype(np.uint8)
            tgt_np = (sample["image_tgt"].squeeze().numpy() * 255).astype(np.uint8)
            H_gt_orig = sample["homography"].numpy()

            # Build rotation matrix
            H, W = src_np.shape
            cx, cy = W / 2, H / 2
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

            T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
            R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
            T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32)
            rotation = T2 @ R @ T1

            # Rotate both images
            src_rotated = cv2.warpPerspective(src_np, rotation, (W, H))
            tgt_rotated = cv2.warpPerspective(tgt_np, rotation, (W, H))

            # Update homography: H_new = R @ H @ R^{-1}
            R_inv = np.linalg.inv(rotation)
            H_gt_new = rotation @ H_gt_orig @ R_inv

            # Convert to tensors
            src_tensor = torch.from_numpy(src_rotated).float().unsqueeze(0).unsqueeze(0) / 255.0
            tgt_tensor = torch.from_numpy(tgt_rotated).float().unsqueeze(0).unsqueeze(0) / 255.0

            # Predict
            with torch.no_grad():
                output = model(src_tensor.to(device), tgt_tensor.to(device))
                H_pred = output["homography"].cpu()

            # Convert GT to vec using canonical implementation
            H_gt_mat = torch.from_numpy(H_gt_new).float()
            H_gt_vec = homography_matrix_to_vec(H_gt_mat).unsqueeze(0)

            # Compute errors
            errors_corner.append(corner_error(H_pred, H_gt_vec).item())
            errors_rot.append(rotation_error(H_pred, H_gt_vec).item())
            errors_trans.append(translation_error(H_pred, H_gt_vec).item())

        results["corner_error"].append(np.mean(errors_corner))
        results["rotation_error"].append(np.mean(errors_rot))
        results["translation_error"].append(np.mean(errors_trans))

    return results


def evaluate_geometric_split(
    model: torch.nn.Module,
    dataset,
    train_angle_threshold: float = 30.0,
    test_angle_threshold: float = 45.0,
    device: torch.device = torch.device("cpu"),
) -> dict[str, dict[str, float]]:
    """
    Evaluate on geometric split (easy vs hard rotations).

    Args:
        model: Model to evaluate
        dataset: Full dataset with homographies
        train_angle_threshold: Max rotation in training
        test_angle_threshold: Min rotation for test set
        device: Device to use

    Returns:
        Metrics for 'easy' and 'hard' splits
    """
    model.eval()
    model.to(device)

    easy_preds, easy_gts = [], []
    hard_preds, hard_gts = [], []

    for idx in tqdm(range(len(dataset)), desc="Evaluating geometric split"):
        sample = dataset[idx]

        # Get rotation angle from homography
        H_gt = sample["homography"].numpy()
        angle = np.abs(np.arctan2(H_gt[1, 0], H_gt[0, 0]) * 180 / np.pi)

        # Get prediction
        with torch.no_grad():
            src = sample["image_src"].unsqueeze(0).to(device)
            tgt = sample["image_tgt"].unsqueeze(0).to(device)
            output = model(src, tgt)
            H_pred = output["homography"].cpu()

        H_gt_vec = sample["homography_vec"].unsqueeze(0)

        # Split by difficulty
        if angle < train_angle_threshold:
            easy_preds.append(H_pred)
            easy_gts.append(H_gt_vec)
        elif angle >= test_angle_threshold:
            hard_preds.append(H_pred)
            hard_gts.append(H_gt_vec)

    results = {}

    if easy_preds:
        easy_preds = torch.cat(easy_preds, dim=0)
        easy_gts = torch.cat(easy_gts, dim=0)
        results["easy"] = compute_all_metrics(easy_preds, easy_gts)

    if hard_preds:
        hard_preds = torch.cat(hard_preds, dim=0)
        hard_gts = torch.cat(hard_gts, dim=0)
        results["hard"] = compute_all_metrics(hard_preds, hard_gts)

    return results


def evaluate_all_splits(
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    """
    Run all evaluation splits.

    Args:
        model: Model to evaluate
        config: Configuration dictionary
        device: Device to use

    Returns:
        Comprehensive results dictionary
    """
    results = {}

    # 1. Standard test set evaluation
    if "test_data_root" in config:
        test_dataset = ThermalPairDataset(
            data_root=config["test_data_root"],
            split="test",
            image_size=config.get("image_size", (256, 256)),
        )
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        results["standard"] = evaluate_model(model, test_loader, device)

    # 2. Rotation equivariance evaluation
    synth_dataset = SyntheticThermalGenerator(n_samples=200, seed=42)
    results["rotation_equivariance"] = evaluate_rotation_equivariance(
        model, synth_dataset, device=device
    )

    # 3. Geometric split (if applicable)
    if "test_data_root" in config:
        results["geometric_split"] = evaluate_geometric_split(model, test_dataset, device=device)

    return results


def main():
    """Command-line evaluation entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate thermal homography model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--data-root", type=str, help="Test data root")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")

    args = parser.parse_args()

    # Load model
    from src.training.train import ThermalHomographyModule

    model = ThermalHomographyModule.load_from_checkpoint(args.checkpoint)
    device = torch.device(args.device)

    # Run evaluation
    config = {"test_data_root": args.data_root} if args.data_root else {}
    results = evaluate_all_splits(model, config, device)

    # Save results
    with open(args.output, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
            default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x,
        )

    logger.info(f"Results saved to {args.output}")

    # Print summary
    logger.info("=== Evaluation Summary ===")
    if "standard" in results:
        logger.info("Standard Test Set:")
        logger.info(f"  Corner Error: {results['standard']['corner_error_mean']:.2f} px")
        logger.info(f"  Recall@10px: {results['standard']['recall@10px']:.2%}")

    if "rotation_equivariance" in results:
        re = results["rotation_equivariance"]
        logger.info("Rotation Equivariance:")
        logger.info(
            f"  Corner Error Range: {min(re['corner_error']):.2f} - {max(re['corner_error']):.2f} px"
        )
        std = np.std(re["corner_error"])
        logger.info(f"  Corner Error Std: {std:.2f} px (lower = more equivariant)")
