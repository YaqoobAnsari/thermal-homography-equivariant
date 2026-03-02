"""
Evaluation Metrics for Homography Estimation

Implements standard metrics:
1. Corner reprojection error (primary metric)
2. Rotation error (degrees)
3. Translation error (pixels)
4. Registration recall @ threshold
"""

import math

import numpy as np
import torch
from torch import Tensor

from src.constants import DEFAULT_IMAGE_SIZE, DEFAULT_RECALL_THRESHOLDS, DEFAULT_ROTATION_BINS
from src.utils.homography import apply_homography, homography_vec_to_matrix
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def corner_error(
    H_pred: Tensor,
    H_gt: Tensor,
    image_size: tuple[int, int] = (256, 256),
) -> Tensor:
    """
    Compute mean corner reprojection error.

    Args:
        H_pred: [B, 3, 3] or [B, 8] predicted homography
        H_gt: [B, 3, 3] or [B, 8] ground truth homography
        image_size: (H, W) image dimensions

    Returns:
        [B] error for each sample in pixels
    """
    if H_pred.dim() == 2 and H_pred.shape[-1] == 8:
        H_pred = homography_vec_to_matrix(H_pred)
    if H_gt.dim() == 2 and H_gt.shape[-1] == 8:
        H_gt = homography_vec_to_matrix(H_gt)

    B = H_pred.shape[0]
    H, W = image_size
    device = H_pred.device

    # Define corners in image coordinates
    corners = torch.tensor(
        [
            [0, 0],
            [W, 0],
            [W, H],
            [0, H],
        ],
        device=device,
        dtype=H_pred.dtype,
    )
    corners = corners.unsqueeze(0).expand(B, -1, -1)

    # Transform corners
    corners_pred = apply_homography(H_pred, corners)
    corners_gt = apply_homography(H_gt, corners)

    # Compute per-corner error
    error = torch.norm(corners_pred - corners_gt, dim=-1)  # [B, 4]

    # Mean across corners
    mean_error = error.mean(dim=-1)  # [B]

    return mean_error


def rotation_error(H_pred: Tensor, H_gt: Tensor) -> Tensor:
    """
    Compute rotation error in degrees.

    Args:
        H_pred: [B, 3, 3] or [B, 8] predicted homography
        H_gt: [B, 3, 3] or [B, 8] ground truth homography

    Returns:
        [B] rotation error in degrees
    """
    if H_pred.dim() == 2 and H_pred.shape[-1] == 8:
        H_pred = homography_vec_to_matrix(H_pred)
    if H_gt.dim() == 2 and H_gt.shape[-1] == 8:
        H_gt = homography_vec_to_matrix(H_gt)

    # Extract rotation angles
    angle_pred = torch.atan2(H_pred[:, 1, 0], H_pred[:, 0, 0])
    angle_gt = torch.atan2(H_gt[:, 1, 0], H_gt[:, 0, 0])

    # Angle difference (handle wraparound)
    angle_diff = angle_pred - angle_gt
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

    # Convert to degrees
    error_deg = angle_diff.abs() * 180 / math.pi

    return error_deg


def translation_error(H_pred: Tensor, H_gt: Tensor) -> Tensor:
    """
    Compute translation error in pixels.

    Args:
        H_pred: [B, 3, 3] or [B, 8] predicted homography
        H_gt: [B, 3, 3] or [B, 8] ground truth homography

    Returns:
        [B] translation error in pixels
    """
    if H_pred.dim() == 2 and H_pred.shape[-1] == 8:
        H_pred = homography_vec_to_matrix(H_pred)
    if H_gt.dim() == 2 and H_gt.shape[-1] == 8:
        H_gt = homography_vec_to_matrix(H_gt)

    # Extract translation
    t_pred = H_pred[:, :2, 2]
    t_gt = H_gt[:, :2, 2]

    # L2 distance
    error = torch.norm(t_pred - t_gt, dim=-1)

    return error


def registration_recall(
    H_pred: Tensor,
    H_gt: Tensor,
    threshold: float = 10.0,
    image_size: tuple[int, int] = (256, 256),
) -> float:
    """
    Compute registration recall at given threshold.

    Args:
        H_pred: [B, 3, 3] or [B, 8] predicted homographies
        H_gt: [B, 3, 3] or [B, 8] ground truth homographies
        threshold: Error threshold in pixels
        image_size: Image dimensions

    Returns:
        Recall (fraction of samples with error < threshold)
    """
    errors = corner_error(H_pred, H_gt, image_size)
    recall = (errors < threshold).float().mean().item()
    return recall


def precision_recall_curve(
    H_pred: Tensor,
    H_gt: Tensor,
    thresholds: list[float] = None,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> dict[str, list[float]]:
    """
    Compute precision-recall at multiple thresholds.

    For homography estimation, we define:
    - Precision: fraction of predictions with error < threshold
    - Recall: same as precision (no false negatives in regression)

    This is useful for generating PR curves at varying error tolerances.

    Args:
        H_pred: Predicted homographies
        H_gt: Ground truth homographies
        thresholds: Error thresholds to evaluate
        image_size: Image dimensions

    Returns:
        Dictionary with 'thresholds', 'precision', 'recall' lists
    """
    if thresholds is None:
        thresholds = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]

    errors = corner_error(H_pred, H_gt, image_size)
    n_samples = errors.shape[0]

    precisions = []
    recalls = []

    for thresh in thresholds:
        n_correct = (errors < thresh).sum().item()
        precision = n_correct / n_samples if n_samples > 0 else 0.0
        recall = precision  # In regression, precision == recall
        precisions.append(precision)
        recalls.append(recall)

    return {
        "thresholds": thresholds,
        "precision": precisions,
        "recall": recalls,
    }


def compute_per_scene_metrics(
    H_pred: Tensor,
    H_gt: Tensor,
    scene_ids: list[str],
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> dict[str, dict[str, float]]:
    """
    Compute metrics broken down by scene.

    Args:
        H_pred: Predicted homographies
        H_gt: Ground truth homographies
        scene_ids: List of scene identifiers for each sample
        image_size: Image dimensions

    Returns:
        Dictionary mapping scene_id to metric dictionaries
    """
    unique_scenes = list(set(scene_ids))
    results = {}

    for scene in unique_scenes:
        mask = [i for i, s in enumerate(scene_ids) if s == scene]
        if not mask:
            continue

        H_pred_scene = H_pred[mask]
        H_gt_scene = H_gt[mask]

        ce = corner_error(H_pred_scene, H_gt_scene, image_size)
        re = rotation_error(H_pred_scene, H_gt_scene)
        te = translation_error(H_pred_scene, H_gt_scene)

        results[scene] = {
            "n_samples": len(mask),
            "corner_error_mean": ce.mean().item(),
            "corner_error_median": ce.median().item(),
            "rotation_error_mean": re.mean().item(),
            "translation_error_mean": te.mean().item(),
            "recall@10px": (ce < 10.0).float().mean().item(),
        }

    return results


def inference_benchmark(
    model,
    dataloader,
    device: torch.device = torch.device("cpu"),
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> dict[str, float]:
    """
    Benchmark inference speed (FPS) and memory usage.

    Args:
        model: The model to benchmark
        dataloader: DataLoader providing input samples
        device: Device to run on
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations

    Returns:
        Dictionary with 'fps', 'latency_ms', 'memory_mb' metrics
    """
    import time

    model.eval()
    model.to(device)

    # Get a sample batch
    batch = next(iter(dataloader))
    image_src = batch["image_src"].to(device)
    image_tgt = batch["image_tgt"].to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(image_src, image_tgt)

    # Synchronize if CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Memory before
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Timed iterations
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(image_src, image_tgt)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    # Memory after
    memory_mb = 0.0
    if device.type == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Compute stats
    avg_time = sum(times) / len(times)
    batch_size = image_src.shape[0]
    fps = batch_size / avg_time
    latency_ms = avg_time * 1000

    logger.info(
        f"Inference benchmark: {fps:.1f} FPS, {latency_ms:.2f} ms/batch, {memory_mb:.1f} MB"
    )

    return {
        "fps": fps,
        "latency_ms": latency_ms,
        "memory_mb": memory_mb,
        "batch_size": batch_size,
        "num_iterations": num_iterations,
    }


def decompose_homography(H: Tensor) -> dict[str, Tensor]:
    """
    SVD-based decomposition of homography into interpretable components.

    Decomposes the 3x3 homography matrix into:
    - Rotation angle (degrees)
    - Scale factors (x, y)
    - Shear
    - Translation

    Args:
        H: [B, 3, 3] or [B, 8] homography matrices

    Returns:
        Dictionary with decomposition components:
        - rotation_deg: [B] rotation angle in degrees
        - scale_x: [B] scale factor in x
        - scale_y: [B] scale factor in y
        - scale_ratio: [B] ratio scale_x / scale_y (aspect ratio change)
        - shear: [B] shear component
        - translation_x: [B] translation in x
        - translation_y: [B] translation in y
        - translation_norm: [B] L2 norm of translation
    """
    if H.dim() == 2 and H.shape[-1] == 8:
        H = homography_vec_to_matrix(H)

    # Extract 2x2 linear part (rotation, scale, shear)
    A = H[:, :2, :2]  # [B, 2, 2]

    # SVD decomposition: A = U @ S @ Vh
    # U, Vh are rotation matrices, S contains scales
    try:
        U, S, Vh = torch.linalg.svd(A)
    except RuntimeError:
        # Fallback for singular matrices
        B = H.shape[0]
        device = H.device
        return {
            "rotation_deg": torch.zeros(B, device=device),
            "scale_x": torch.ones(B, device=device),
            "scale_y": torch.ones(B, device=device),
            "scale_ratio": torch.ones(B, device=device),
            "shear": torch.zeros(B, device=device),
            "translation_x": H[:, 0, 2],
            "translation_y": H[:, 1, 2],
            "translation_norm": torch.norm(H[:, :2, 2], dim=-1),
        }

    # Rotation: R = U @ Vh
    R = torch.bmm(U, Vh)

    # Handle reflection (det(R) = -1)
    det = torch.det(R)
    # Flip sign if determinant is negative
    U_fixed = U.clone()
    U_fixed[:, :, 1] = U[:, :, 1] * det.unsqueeze(-1).sign()
    R = torch.bmm(U_fixed, Vh)

    # Extract rotation angle
    rotation_rad = torch.atan2(R[:, 1, 0], R[:, 0, 0])  # [B]
    rotation_deg = rotation_rad * 180.0 / math.pi

    # Scale factors from singular values
    scale_x = S[:, 0]  # [B]
    scale_y = S[:, 1]  # [B]
    scale_ratio = scale_x / (scale_y + 1e-8)

    # Compute shear (from the factorization A = R @ S @ Shear)
    # Simplified: measure deviation from pure rotation+scale
    R_inv = torch.linalg.inv(R)
    AS = torch.bmm(R_inv, A)  # Should be close to diagonal if no shear
    shear = AS[:, 0, 1]  # Off-diagonal element represents shear

    # Translation
    translation_x = H[:, 0, 2]
    translation_y = H[:, 1, 2]
    translation_norm = torch.norm(H[:, :2, 2], dim=-1)

    return {
        "rotation_deg": rotation_deg,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "scale_ratio": scale_ratio,
        "shear": shear,
        "translation_x": translation_x,
        "translation_y": translation_y,
        "translation_norm": translation_norm,
    }


def decomposition_error(
    H_pred: Tensor,
    H_gt: Tensor,
) -> dict[str, Tensor]:
    """
    Compute errors in decomposed homography components.

    Args:
        H_pred: [B, 3, 3] or [B, 8] predicted homographies
        H_gt: [B, 3, 3] or [B, 8] ground truth homographies

    Returns:
        Dictionary with component-wise errors:
        - rotation_error_deg: absolute rotation error in degrees
        - scale_x_error: absolute scale_x error
        - scale_y_error: absolute scale_y error
        - scale_ratio_error: absolute scale ratio error
        - shear_error: absolute shear error
        - translation_error: L2 translation error
    """
    d_pred = decompose_homography(H_pred)
    d_gt = decompose_homography(H_gt)

    # Rotation error (handle wraparound at 180/-180)
    rot_diff = d_pred["rotation_deg"] - d_gt["rotation_deg"]
    rot_diff = (
        torch.atan2(torch.sin(rot_diff * math.pi / 180), torch.cos(rot_diff * math.pi / 180))
        * 180
        / math.pi
    )
    rotation_error = rot_diff.abs()

    return {
        "rotation_error_deg": rotation_error,
        "scale_x_error": (d_pred["scale_x"] - d_gt["scale_x"]).abs(),
        "scale_y_error": (d_pred["scale_y"] - d_gt["scale_y"]).abs(),
        "scale_ratio_error": (d_pred["scale_ratio"] - d_gt["scale_ratio"]).abs(),
        "shear_error": (d_pred["shear"] - d_gt["shear"]).abs(),
        "translation_error": (
            (d_pred["translation_x"] - d_gt["translation_x"]).pow(2)
            + (d_pred["translation_y"] - d_gt["translation_y"]).pow(2)
        ).sqrt(),
    }


def compute_decomposition_metrics(
    H_pred: Tensor,
    H_gt: Tensor,
) -> dict[str, float]:
    """
    Compute aggregate decomposition metrics over a batch.

    Args:
        H_pred: Predicted homographies
        H_gt: Ground truth homographies

    Returns:
        Dictionary of mean decomposition errors
    """
    errors = decomposition_error(H_pred, H_gt)

    return {
        "decomp_rotation_error_mean": errors["rotation_error_deg"].mean().item(),
        "decomp_rotation_error_median": errors["rotation_error_deg"].median().item(),
        "decomp_scale_x_error_mean": errors["scale_x_error"].mean().item(),
        "decomp_scale_y_error_mean": errors["scale_y_error"].mean().item(),
        "decomp_scale_ratio_error_mean": errors["scale_ratio_error"].mean().item(),
        "decomp_shear_error_mean": errors["shear_error"].mean().item(),
        "decomp_translation_error_mean": errors["translation_error"].mean().item(),
    }


def evaluate_robustness(
    model,
    dataloader,
    perturbations: dict[str, list[float]],
    device: torch.device = torch.device("cpu"),
) -> dict[str, dict[str, float]]:
    """
    Evaluate model robustness to various perturbations.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation
        perturbations: Dictionary mapping perturbation types to levels
            E.g., {'gaussian_noise': [0.01, 0.05, 0.1],
                   'blur_sigma': [1, 3, 5]}
        device: Device to use

    Returns:
        Dictionary mapping perturbation settings to metrics
    """
    import cv2

    model.eval()
    results = {"clean": {}}

    # Evaluate on clean data first
    all_pred = []
    all_gt = []

    with torch.no_grad():
        for batch in dataloader:
            image_src = batch["image_src"].to(device)
            image_tgt = batch["image_tgt"].to(device)
            H_gt = batch["homography_vec"].to(device)

            output = model(image_src, image_tgt)
            H_pred = output["homography"]

            all_pred.append(H_pred.cpu())
            all_gt.append(H_gt.cpu())

    H_pred_all = torch.cat(all_pred, dim=0)
    H_gt_all = torch.cat(all_gt, dim=0)
    results["clean"] = compute_all_metrics(H_pred_all, H_gt_all)

    # Helper to apply perturbations
    def apply_perturbation(images: Tensor, pert_type: str, level: float) -> Tensor:
        """Apply perturbation to batch of images."""
        images_np = images.cpu().numpy()
        B, C, H, W = images_np.shape

        if pert_type == "gaussian_noise":
            noise = np.random.normal(0, level, images_np.shape).astype(np.float32)
            images_np = np.clip(images_np + noise, 0, 1)

        elif pert_type == "blur_sigma":
            for i in range(B):
                for c in range(C):
                    images_np[i, c] = cv2.GaussianBlur(images_np[i, c], (0, 0), sigmaX=level)

        elif pert_type == "brightness":
            images_np = np.clip(images_np * level, 0, 1)

        elif pert_type == "contrast":
            mean = images_np.mean(axis=(2, 3), keepdims=True)
            images_np = np.clip((images_np - mean) * level + mean, 0, 1)

        return torch.from_numpy(images_np).to(device)

    # Evaluate with perturbations
    for pert_type, levels in perturbations.items():
        for level in levels:
            key = f"{pert_type}_{level}"
            all_pred = []
            all_gt = []

            with torch.no_grad():
                for batch in dataloader:
                    image_src = batch["image_src"].to(device)
                    image_tgt = batch["image_tgt"].to(device)
                    H_gt = batch["homography_vec"].to(device)

                    # Apply perturbation to both images
                    image_src_pert = apply_perturbation(image_src, pert_type, level)
                    image_tgt_pert = apply_perturbation(image_tgt, pert_type, level)

                    output = model(image_src_pert, image_tgt_pert)
                    H_pred = output["homography"]

                    all_pred.append(H_pred.cpu())
                    all_gt.append(H_gt.cpu())

            H_pred_all = torch.cat(all_pred, dim=0)
            H_gt_all = torch.cat(all_gt, dim=0)
            results[key] = compute_all_metrics(H_pred_all, H_gt_all)

            logger.info(f"Robustness {key}: corner_error={results[key]['corner_error_mean']:.2f}")

    return results


def compute_all_metrics(
    H_pred: Tensor,
    H_gt: Tensor,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    thresholds: list[float] = None,
) -> dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        H_pred: Predicted homographies
        H_gt: Ground truth homographies
        image_size: Image dimensions
        thresholds: Recall thresholds to evaluate (default: from constants)

    Returns:
        Dictionary of metric values
    """
    if thresholds is None:
        thresholds = DEFAULT_RECALL_THRESHOLDS

    metrics = {}

    # Corner error
    ce = corner_error(H_pred, H_gt, image_size)
    metrics["corner_error_mean"] = ce.mean().item()
    metrics["corner_error_median"] = ce.median().item()
    metrics["corner_error_std"] = ce.std().item()

    # Rotation error
    re = rotation_error(H_pred, H_gt)
    metrics["rotation_error_mean"] = re.mean().item()
    metrics["rotation_error_median"] = re.median().item()

    # Translation error
    te = translation_error(H_pred, H_gt)
    metrics["translation_error_mean"] = te.mean().item()
    metrics["translation_error_median"] = te.median().item()

    # Registration recall at various thresholds
    for thresh in thresholds:
        recall = registration_recall(H_pred, H_gt, thresh, image_size)
        metrics[f"recall@{thresh}px"] = recall

    return metrics


class MetricTracker:
    """
    Track metrics over training/evaluation.

    Accumulates predictions and computes aggregate metrics.
    """

    def __init__(self, image_size: tuple[int, int] = (256, 256)):
        self.image_size = image_size
        self.reset()

    def reset(self):
        """Reset accumulated predictions."""
        self.predictions = []
        self.ground_truths = []
        self.metadata = []

    def update(
        self,
        H_pred: Tensor,
        H_gt: Tensor,
        metadata: dict | None = None,
    ):
        """Add batch of predictions."""
        self.predictions.append(H_pred.detach().cpu())
        self.ground_truths.append(H_gt.detach().cpu())
        if metadata:
            self.metadata.append(metadata)

    def compute(self) -> dict[str, float]:
        """Compute aggregate metrics."""
        if not self.predictions:
            return {}

        H_pred = torch.cat(self.predictions, dim=0)
        H_gt = torch.cat(self.ground_truths, dim=0)

        return compute_all_metrics(H_pred, H_gt, self.image_size)

    def compute_by_rotation(
        self,
        rotation_angles: Tensor,
        angle_bins: list[float] = None,
    ) -> dict[str, dict[str, float]]:
        """
        Compute metrics binned by input rotation.

        Used for rotation equivariance evaluation.

        Args:
            rotation_angles: Tensor of rotation angles for each sample
            angle_bins: List of bin boundaries (default: from constants)

        Returns:
            Dictionary mapping bin labels to metric dictionaries
        """
        if angle_bins is None:
            angle_bins = DEFAULT_ROTATION_BINS

        if not self.predictions:
            return {}

        H_pred = torch.cat(self.predictions, dim=0)
        H_gt = torch.cat(self.ground_truths, dim=0)

        results = {}

        for i in range(len(angle_bins) - 1):
            low, high = angle_bins[i], angle_bins[i + 1]
            mask = (rotation_angles >= low) & (rotation_angles < high)

            if mask.sum() == 0:
                continue

            H_pred_bin = H_pred[mask]
            H_gt_bin = H_gt[mask]

            metrics = compute_all_metrics(H_pred_bin, H_gt_bin, self.image_size)
            results[f"{low}-{high}deg"] = metrics

        return results

    def compute_precision_recall(
        self,
        thresholds: list[float] = None,
    ) -> dict[str, list[float]]:
        """
        Compute precision-recall curve from accumulated predictions.

        Args:
            thresholds: Error thresholds to evaluate

        Returns:
            Dictionary with precision-recall data
        """
        if not self.predictions:
            return {}

        H_pred = torch.cat(self.predictions, dim=0)
        H_gt = torch.cat(self.ground_truths, dim=0)

        return precision_recall_curve(H_pred, H_gt, thresholds, self.image_size)

    def compute_per_difficulty(
        self,
        difficulty_scores: Tensor,
        difficulty_bins: list[float] = None,
    ) -> dict[str, dict[str, float]]:
        """
        Compute metrics binned by sample difficulty.

        Difficulty can be measured by homography magnitude, scene complexity, etc.

        Args:
            difficulty_scores: Tensor of difficulty scores for each sample
            difficulty_bins: List of bin boundaries

        Returns:
            Dictionary mapping bin labels to metric dictionaries
        """
        if difficulty_bins is None:
            difficulty_bins = [0, 0.25, 0.5, 0.75, 1.0]

        if not self.predictions:
            return {}

        H_pred = torch.cat(self.predictions, dim=0)
        H_gt = torch.cat(self.ground_truths, dim=0)

        results = {}

        for i in range(len(difficulty_bins) - 1):
            low, high = difficulty_bins[i], difficulty_bins[i + 1]
            mask = (difficulty_scores >= low) & (difficulty_scores < high)

            if mask.sum() == 0:
                continue

            H_pred_bin = H_pred[mask]
            H_gt_bin = H_gt[mask]

            metrics = compute_all_metrics(H_pred_bin, H_gt_bin, self.image_size)
            results[f"difficulty_{low:.2f}-{high:.2f}"] = metrics

        return results


def evaluate_rotation_equivariance(
    model,
    dataset,
    angles: list[float] | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict[str, list[float]]:
    """
    Evaluate model's rotation equivariance.

    For an equivariant model, accuracy should be constant across angles.

    Args:
        model: The model to evaluate
        dataset: Dataset that returns rotated samples
        angles: Rotation angles to test
        device: Device to use

    Returns:
        Dictionary mapping metric names to lists of values (one per angle)
    """
    if angles is None:
        angles = [0, 30, 60, 90, 120, 150, 180]

    model.eval()
    results = {
        "angles": angles,
        "corner_error": [],
        "rotation_error": [],
        "translation_error": [],
    }

    for angle in angles:
        errors_corner = []
        errors_rotation = []
        errors_translation = []

        # Filter dataset for this angle
        for idx in range(len(dataset)):
            sample = dataset[idx]
            if sample.get("rotation_angle", 0) != angle:
                continue

            # Get prediction
            with torch.no_grad():
                image_src = sample["image_src"].unsqueeze(0).to(device)
                image_tgt = sample["image_tgt"].unsqueeze(0).to(device)
                output = model(image_src, image_tgt)
                H_pred = output["homography"]

            H_gt = sample["homography_vec"].unsqueeze(0)

            # Compute errors
            errors_corner.append(corner_error(H_pred.cpu(), H_gt).item())
            errors_rotation.append(rotation_error(H_pred.cpu(), H_gt).item())
            errors_translation.append(translation_error(H_pred.cpu(), H_gt).item())

        if errors_corner:
            results["corner_error"].append(np.mean(errors_corner))
            results["rotation_error"].append(np.mean(errors_rotation))
            results["translation_error"].append(np.mean(errors_translation))
        else:
            results["corner_error"].append(float("nan"))
            results["rotation_error"].append(float("nan"))
            results["translation_error"].append(float("nan"))

    return results
