#!/usr/bin/env python3
"""
HPatches Benchmark Evaluation Script

Evaluates homography estimation models on the HPatches benchmark.
Reports AUC at multiple thresholds, split by viewpoint/illumination.

Usage:
    python scripts/evaluate_hpatches.py --model-path checkpoints/best.pt
    python scripts/evaluate_hpatches.py --model-type log_polar_sim2 --device cuda
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import HPatchesDataset, compute_corner_error, compute_auc
from src.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def load_model(
    model_path: Optional[str] = None,
    model_type: str = "log_polar_sim2",
    device: str = "cuda",
):
    """
    Load a homography estimation model.

    Args:
        model_path: Path to checkpoint file
        model_type: Type of model to create if no checkpoint
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    from src.models import create_log_polar_sim2_net

    # Create model
    if model_type == "log_polar_sim2":
        model = create_log_polar_sim2_net()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint if provided
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
        if "state_dict" in checkpoint:
            # Lightning checkpoint
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
    """
    Predict homography for an image pair.

    Args:
        model: Homography estimation model
        image_src: Source image [1, H, W] or [B, 1, H, W]
        image_tgt: Target image [1, H, W] or [B, 1, H, W]
        device: Device for inference

    Returns:
        Predicted homography [3, 3] or [B, 3, 3]
    """
    # Ensure batch dimension
    if image_src.dim() == 3:
        image_src = image_src.unsqueeze(0)
        image_tgt = image_tgt.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # Move to device
    image_src = image_src.to(device)
    image_tgt = image_tgt.to(device)

    # Predict
    with torch.no_grad():
        output = model(image_src, image_tgt)

    # Extract homography
    if "homography" in output:
        H_pred = output["homography"].cpu().numpy()
    else:
        # Reconstruct from components if needed
        raise ValueError("Model output does not contain homography")

    if squeeze_output:
        H_pred = H_pred[0]

    return H_pred


def evaluate_hpatches(
    model: torch.nn.Module,
    data_root: str,
    device: str = "cuda",
    batch_size: int = 1,
    resize_to: int = 480,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on HPatches benchmark.

    Args:
        model: Homography estimation model
        data_root: Path to hpatches-sequences-release
        device: Device for inference
        batch_size: Batch size for evaluation
        resize_to: Image resize target

    Returns:
        Dict with results for each split and overall
    """
    results = {}

    for split in ["all", "viewpoint", "illumination"]:
        logger.info(f"Evaluating on {split} split...")

        dataset = HPatchesDataset(
            data_root=data_root,
            split=split,
            resize_to=resize_to,
            return_metadata=True,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )

        errors = []
        errors_by_sequence = {}

        for batch in tqdm(dataloader, desc=f"{split}"):
            image_src = batch["image_src"]
            image_tgt = batch["image_tgt"]
            H_gt = batch["homography"].numpy()

            # Get image size for corner error calculation
            _, _, h, w = image_src.shape

            # Predict homography
            H_pred = predict_homography(model, image_src, image_tgt, device)

            # Compute corner error for each sample in batch
            for i in range(len(H_gt)):
                error = compute_corner_error(H_pred[i], H_gt[i], image_size=(h, w))
                errors.append(error)

                # Track by sequence
                seq = batch["sequence"][i]
                if seq not in errors_by_sequence:
                    errors_by_sequence[seq] = []
                errors_by_sequence[seq].append(error)

        # Compute AUC at multiple thresholds
        auc_results = compute_auc(errors, thresholds=[1, 3, 5, 10, 20])

        results[split] = {
            "n_samples": len(errors),
            "mean_corner_error": float(np.mean(errors)),
            "median_corner_error": float(np.median(errors)),
            "std_corner_error": float(np.std(errors)),
            **auc_results,
        }

        logger.info(
            f"  {split}: ACE={results[split]['mean_corner_error']:.2f}px, "
            f"AUC@3={results[split]['AUC@3']:.1f}%, "
            f"AUC@5={results[split]['AUC@5']:.1f}%"
        )

    return results


def print_results_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print results in a formatted table."""
    print("\n" + "=" * 80)
    print("HPatches Evaluation Results")
    print("=" * 80)

    # Header
    print(f"{'Split':<15} {'N':>6} {'ACE':>8} {'AUC@1':>8} {'AUC@3':>8} {'AUC@5':>8} {'AUC@10':>8}")
    print("-" * 80)

    # Results
    for split, metrics in results.items():
        print(
            f"{split:<15} "
            f"{metrics['n_samples']:>6} "
            f"{metrics['mean_corner_error']:>7.2f}px "
            f"{metrics['AUC@1']:>7.1f}% "
            f"{metrics['AUC@3']:>7.1f}% "
            f"{metrics['AUC@5']:>7.1f}% "
            f"{metrics['AUC@10']:>7.1f}%"
        )

    print("=" * 80)


def save_results(results: Dict[str, Dict[str, float]], output_path: str) -> None:
    """Save results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on HPatches benchmark")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="log_polar_sim2",
        choices=["log_polar_sim2"],
        help="Model type",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/hpatches/hpatches-sequences-release",
        help="Path to HPatches directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--resize-to",
        type=int,
        default=480,
        help="Resize images to this shorter edge",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    setup_logging(level="INFO")

    # Load model
    logger.info(f"Loading model (type={args.model_type})...")
    model = load_model(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
    )

    # Evaluate
    results = evaluate_hpatches(
        model=model,
        data_root=args.data_root,
        device=args.device,
        batch_size=args.batch_size,
        resize_to=args.resize_to,
    )

    # Print results
    print_results_table(results)

    # Save results
    if args.output:
        save_results(results, args.output)
    else:
        # Default output path
        output_dir = project_root / "results"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "hpatches_results.json"
        save_results(results, str(output_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
