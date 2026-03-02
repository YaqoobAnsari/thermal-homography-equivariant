#!/usr/bin/env python3
"""
MS-COCO Benchmark Evaluation Script

Evaluates homography estimation models on the Warped MS-COCO benchmark.
Supports:
- Standard evaluation (random 4-point perturbation)
- Rotation sweep evaluation (test at specific angles)
- Scale sweep evaluation (test at specific scales)
- Sim(2) grid evaluation (rotation × scale combinations)

Usage:
    # Standard evaluation
    python scripts/evaluate_mscoco.py --mode standard

    # Rotation equivariance test
    python scripts/evaluate_mscoco.py --mode rotation_sweep

    # Scale equivariance test
    python scripts/evaluate_mscoco.py --mode scale_sweep

    # Full Sim(2) grid
    python scripts/evaluate_mscoco.py --mode sim2_grid
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import WarpedMSCOCODataset, ControlledTransformDataset, compute_corner_error, compute_auc
from src.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def load_model(
    model_path: Optional[str] = None,
    model_type: str = "log_polar_sim2",
    device: str = "cuda",
):
    """Load a homography estimation model."""
    from src.models import create_log_polar_sim2_net

    if model_type == "log_polar_sim2":
        model = create_log_polar_sim2_net()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
        if "state_dict" in checkpoint:
            state_dict = {
                k.replace("model.", ""): v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("model.")
            }
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded checkpoint from {model_path}")

    model = model.to(device)
    model.eval()

    return model


def predict_homography(
    model: torch.nn.Module,
    image_src: torch.Tensor,
    image_tgt: torch.Tensor,
    device: str = "cuda",
) -> np.ndarray:
    """Predict homography for an image pair."""
    if image_src.dim() == 3:
        image_src = image_src.unsqueeze(0)
        image_tgt = image_tgt.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    image_src = image_src.to(device)
    image_tgt = image_tgt.to(device)

    with torch.no_grad():
        output = model(image_src, image_tgt)

    H_pred = output["homography"].cpu().numpy()

    if squeeze_output:
        H_pred = H_pred[0]

    return H_pred


def evaluate_standard(
    model: torch.nn.Module,
    coco_root: str,
    device: str = "cuda",
    n_samples: int = 1000,
    batch_size: int = 16,
    rho: int = 32,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Standard MS-COCO evaluation with random 4-point perturbation.

    Args:
        model: Homography estimation model
        coco_root: Path to COCO images
        device: Device for inference
        n_samples: Number of samples to evaluate
        batch_size: Batch size
        rho: Perturbation range (32 = small, 64 = large)
        seed: Random seed

    Returns:
        Dict with ACE, AUC metrics
    """
    logger.info(f"Standard evaluation: n_samples={n_samples}, rho={rho}")

    dataset = WarpedMSCOCODataset(
        coco_root=coco_root,
        mode="standard",
        rho=rho,
        n_samples=n_samples,
        seed=seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    errors = []
    patch_size = dataset.patch_size

    for batch in tqdm(dataloader, desc="Evaluating"):
        image_src = batch["image_src"]
        image_tgt = batch["image_tgt"]
        H_gt = batch["homography"].numpy()

        H_pred = predict_homography(model, image_src, image_tgt, device)

        for i in range(len(H_gt)):
            error = compute_corner_error(H_pred[i], H_gt[i], image_size=(patch_size, patch_size))
            errors.append(error)

    auc_results = compute_auc(errors, thresholds=[1, 3, 5, 10, 20])

    results = {
        "n_samples": len(errors),
        "rho": rho,
        "mean_corner_error": float(np.mean(errors)),
        "median_corner_error": float(np.median(errors)),
        "std_corner_error": float(np.std(errors)),
        **auc_results,
    }

    logger.info(
        f"Results: ACE={results['mean_corner_error']:.2f}px, "
        f"AUC@3={results['AUC@3']:.1f}%, AUC@5={results['AUC@5']:.1f}%"
    )

    return results


def evaluate_rotation_sweep(
    model: torch.nn.Module,
    coco_root: str,
    device: str = "cuda",
    rotation_angles: Optional[List[float]] = None,
    n_samples_per_angle: int = 100,
    seed: int = 42,
) -> Dict[str, any]:
    """
    Evaluate at specific rotation angles to test rotation equivariance.

    Args:
        model: Homography estimation model
        coco_root: Path to COCO images
        device: Device for inference
        rotation_angles: List of rotation angles in degrees
        n_samples_per_angle: Samples per angle
        seed: Random seed

    Returns:
        Dict with per-angle results and variance metrics
    """
    if rotation_angles is None:
        rotation_angles = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180]

    logger.info(f"Rotation sweep: angles={rotation_angles}, n_per_angle={n_samples_per_angle}")

    dataset = ControlledTransformDataset(
        coco_root=coco_root,
        rotation_angles=rotation_angles,
        scale_factors=[1.0],
        n_samples_per_config=n_samples_per_angle,
        seed=seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    errors_by_angle = {angle: [] for angle in rotation_angles}
    patch_size = dataset.patch_size

    for batch in tqdm(dataloader, desc="Rotation sweep"):
        image_src = batch["image_src"]
        image_tgt = batch["image_tgt"]
        H_gt = batch["homography"].numpy()
        angle = float(batch["rotation_config"][0])

        H_pred = predict_homography(model, image_src, image_tgt, device)
        error = compute_corner_error(H_pred[0], H_gt[0], image_size=(patch_size, patch_size))
        errors_by_angle[angle].append(error)

    # Compute statistics
    mean_errors = [np.mean(errors_by_angle[a]) for a in rotation_angles]
    std_errors = [np.std(errors_by_angle[a]) for a in rotation_angles]

    results = {
        "rotation_angles": rotation_angles,
        "mean_errors": mean_errors,
        "std_errors": std_errors,
        "per_angle": {
            angle: {
                "mean": float(np.mean(errors_by_angle[angle])),
                "std": float(np.std(errors_by_angle[angle])),
                "n_samples": len(errors_by_angle[angle]),
            }
            for angle in rotation_angles
        },
        # Equivariance metrics
        "overall_mean": float(np.mean(mean_errors)),
        "error_variance": float(np.var(mean_errors)),
        "error_std": float(np.std(mean_errors)),
        "coefficient_of_variation": float(np.std(mean_errors) / np.mean(mean_errors)) if np.mean(mean_errors) > 0 else 0,
        "max_min_ratio": float(max(mean_errors) / min(mean_errors)) if min(mean_errors) > 0 else float("inf"),
    }

    # Log summary
    logger.info(f"Rotation sweep results:")
    for angle in rotation_angles:
        logger.info(f"  {angle:>5}°: {results['per_angle'][angle]['mean']:.2f} ± {results['per_angle'][angle]['std']:.2f} px")
    logger.info(f"  Overall mean: {results['overall_mean']:.2f} px")
    logger.info(f"  CV (coefficient of variation): {results['coefficient_of_variation']:.2%}")

    return results


def evaluate_scale_sweep(
    model: torch.nn.Module,
    coco_root: str,
    device: str = "cuda",
    scale_factors: Optional[List[float]] = None,
    n_samples_per_scale: int = 100,
    seed: int = 42,
) -> Dict[str, any]:
    """Evaluate at specific scale factors to test scale equivariance."""
    if scale_factors is None:
        scale_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]

    logger.info(f"Scale sweep: scales={scale_factors}, n_per_scale={n_samples_per_scale}")

    dataset = ControlledTransformDataset(
        coco_root=coco_root,
        rotation_angles=[0],
        scale_factors=scale_factors,
        n_samples_per_config=n_samples_per_scale,
        seed=seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    errors_by_scale = {scale: [] for scale in scale_factors}
    patch_size = dataset.patch_size

    for batch in tqdm(dataloader, desc="Scale sweep"):
        image_src = batch["image_src"]
        image_tgt = batch["image_tgt"]
        H_gt = batch["homography"].numpy()
        scale = float(batch["scale_config"][0])

        H_pred = predict_homography(model, image_src, image_tgt, device)
        error = compute_corner_error(H_pred[0], H_gt[0], image_size=(patch_size, patch_size))
        errors_by_scale[scale].append(error)

    mean_errors = [np.mean(errors_by_scale[s]) for s in scale_factors]

    results = {
        "scale_factors": scale_factors,
        "mean_errors": mean_errors,
        "per_scale": {
            scale: {
                "mean": float(np.mean(errors_by_scale[scale])),
                "std": float(np.std(errors_by_scale[scale])),
            }
            for scale in scale_factors
        },
        "overall_mean": float(np.mean(mean_errors)),
        "error_variance": float(np.var(mean_errors)),
        "coefficient_of_variation": float(np.std(mean_errors) / np.mean(mean_errors)) if np.mean(mean_errors) > 0 else 0,
    }

    logger.info(f"Scale sweep results:")
    for scale in scale_factors:
        logger.info(f"  {scale:.1f}x: {results['per_scale'][scale]['mean']:.2f} px")
    logger.info(f"  CV: {results['coefficient_of_variation']:.2%}")

    return results


def evaluate_sim2_grid(
    model: torch.nn.Module,
    coco_root: str,
    device: str = "cuda",
    rotation_angles: Optional[List[float]] = None,
    scale_factors: Optional[List[float]] = None,
    n_samples_per_config: int = 50,
    seed: int = 42,
) -> Dict[str, any]:
    """Evaluate on full rotation × scale grid."""
    if rotation_angles is None:
        rotation_angles = list(range(-180, 181, 15))
    if scale_factors is None:
        scale_factors = [0.5, 0.7, 1.0, 1.3, 2.0]

    logger.info(f"Sim(2) grid: {len(rotation_angles)} rotations × {len(scale_factors)} scales")

    dataset = ControlledTransformDataset(
        coco_root=coco_root,
        rotation_angles=rotation_angles,
        scale_factors=scale_factors,
        n_samples_per_config=n_samples_per_config,
        seed=seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    errors_grid = {(r, s): [] for r in rotation_angles for s in scale_factors}
    patch_size = dataset.patch_size

    for batch in tqdm(dataloader, desc="Sim(2) grid"):
        image_src = batch["image_src"]
        image_tgt = batch["image_tgt"]
        H_gt = batch["homography"].numpy()
        rotation = float(batch["rotation_config"][0])
        scale = float(batch["scale_config"][0])

        H_pred = predict_homography(model, image_src, image_tgt, device)
        error = compute_corner_error(H_pred[0], H_gt[0], image_size=(patch_size, patch_size))
        errors_grid[(rotation, scale)].append(error)

    # Build 2D error matrix
    error_matrix = np.zeros((len(rotation_angles), len(scale_factors)))
    for i, rot in enumerate(rotation_angles):
        for j, scale in enumerate(scale_factors):
            error_matrix[i, j] = np.mean(errors_grid[(rot, scale)])

    results = {
        "rotation_angles": rotation_angles,
        "scale_factors": scale_factors,
        "error_matrix": error_matrix.tolist(),
        "overall_mean": float(np.mean(error_matrix)),
        "overall_std": float(np.std(error_matrix)),
        "row_variance": float(np.mean(np.var(error_matrix, axis=1))),  # Variance across scales
        "col_variance": float(np.mean(np.var(error_matrix, axis=0))),  # Variance across rotations
    }

    logger.info(f"Sim(2) grid results:")
    logger.info(f"  Overall mean: {results['overall_mean']:.2f} px")
    logger.info(f"  Overall std: {results['overall_std']:.2f} px")
    logger.info(f"  Variance across scales (mean): {results['row_variance']:.2f}")
    logger.info(f"  Variance across rotations (mean): {results['col_variance']:.2f}")

    return results


def plot_rotation_sweep(results: Dict, output_path: str) -> None:
    """Generate rotation sweep plot."""
    angles = results["rotation_angles"]
    means = results["mean_errors"]
    stds = results["std_errors"]

    plt.figure(figsize=(10, 6))
    plt.errorbar(angles, means, yerr=stds, fmt="o-", capsize=3, label="Our Method")
    plt.axhline(y=np.mean(means), color="r", linestyle="--", alpha=0.5, label=f"Mean: {np.mean(means):.2f} px")
    plt.xlabel("Rotation Angle (degrees)")
    plt.ylabel("Average Corner Error (pixels)")
    plt.title("Homography Estimation Error vs Rotation Angle")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Rotation sweep plot saved to {output_path}")


def plot_sim2_heatmap(results: Dict, output_path: str) -> None:
    """Generate Sim(2) error heatmap."""
    error_matrix = np.array(results["error_matrix"])
    rotation_angles = results["rotation_angles"]
    scale_factors = results["scale_factors"]

    plt.figure(figsize=(12, 8))
    plt.imshow(error_matrix.T, aspect="auto", cmap="viridis", origin="lower")
    plt.colorbar(label="Corner Error (pixels)")

    # Set ticks
    x_ticks = list(range(0, len(rotation_angles), max(1, len(rotation_angles) // 10)))
    plt.xticks(x_ticks, [rotation_angles[i] for i in x_ticks])
    plt.yticks(range(len(scale_factors)), [f"{s:.1f}x" for s in scale_factors])

    plt.xlabel("Rotation Angle (degrees)")
    plt.ylabel("Scale Factor")
    plt.title("Homography Estimation Error: Rotation × Scale Grid")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Sim(2) heatmap saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on MS-COCO benchmark")
    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["standard", "rotation_sweep", "scale_sweep", "sim2_grid", "all"],
        help="Evaluation mode",
    )
    parser.add_argument("--model-path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--model-type", type=str, default="log_polar_sim2", help="Model type")
    parser.add_argument("--coco-root", type=str, default="data/coco/val2017", help="Path to COCO images")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-samples", type=int, default=1000, help="Samples for standard eval")
    parser.add_argument("--n-samples-per-config", type=int, default=100, help="Samples per config for sweeps")
    parser.add_argument("--rho", type=int, default=32, help="Perturbation range for standard mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")

    args = parser.parse_args()

    setup_logging(level="INFO")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load model
    logger.info(f"Loading model (type={args.model_type})...")
    model = load_model(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
    )

    results = {}

    # Run evaluations based on mode
    if args.mode in ["standard", "all"]:
        results["standard"] = evaluate_standard(
            model, args.coco_root, args.device,
            n_samples=args.n_samples, rho=args.rho, seed=args.seed
        )

    if args.mode in ["rotation_sweep", "all"]:
        results["rotation_sweep"] = evaluate_rotation_sweep(
            model, args.coco_root, args.device,
            n_samples_per_angle=args.n_samples_per_config, seed=args.seed
        )
        plot_rotation_sweep(results["rotation_sweep"], str(output_dir / "rotation_sweep.png"))

    if args.mode in ["scale_sweep", "all"]:
        results["scale_sweep"] = evaluate_scale_sweep(
            model, args.coco_root, args.device,
            n_samples_per_scale=args.n_samples_per_config, seed=args.seed
        )

    if args.mode in ["sim2_grid", "all"]:
        results["sim2_grid"] = evaluate_sim2_grid(
            model, args.coco_root, args.device,
            n_samples_per_config=args.n_samples_per_config // 2, seed=args.seed
        )
        plot_sim2_heatmap(results["sim2_grid"], str(output_dir / "sim2_heatmap.png"))

    # Save results
    output_path = output_dir / f"mscoco_{args.mode}_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("MS-COCO Evaluation Summary")
    print("=" * 60)

    if "standard" in results:
        r = results["standard"]
        print(f"Standard (ρ={r['rho']}): ACE={r['mean_corner_error']:.2f}px, AUC@3={r['AUC@3']:.1f}%")

    if "rotation_sweep" in results:
        r = results["rotation_sweep"]
        print(f"Rotation sweep: Mean={r['overall_mean']:.2f}px, CV={r['coefficient_of_variation']:.2%}")

    if "scale_sweep" in results:
        r = results["scale_sweep"]
        print(f"Scale sweep: Mean={r['overall_mean']:.2f}px, CV={r['coefficient_of_variation']:.2%}")

    if "sim2_grid" in results:
        r = results["sim2_grid"]
        print(f"Sim(2) grid: Mean={r['overall_mean']:.2f}px, Std={r['overall_std']:.2f}px")

    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
