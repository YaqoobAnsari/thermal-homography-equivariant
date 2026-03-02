"""
Visualization Tools for Thermal Homography

Generates figures for paper:
1. Homography visualization (warped images, correspondences)
2. Rotation equivariance plots
3. Similarity matrix visualization
4. Feature embeddings (t-SNE)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def visualize_homography(
    image_src: np.ndarray,
    image_tgt: np.ndarray,
    H_pred: np.ndarray,
    H_gt: np.ndarray | None = None,
    save_path: str | None = None,
    title: str = "",
) -> plt.Figure:
    """
    Visualize homography estimation result.

    Shows:
    - Source and target images
    - Warped source aligned with target
    - Corner correspondences

    Args:
        image_src: Source image [H, W] or [H, W, C]
        image_tgt: Target image [H, W] or [H, W, C]
        H_pred: Predicted homography [3, 3]
        H_gt: Ground truth homography [3, 3] (optional)
        save_path: Path to save figure
        title: Figure title

    Returns:
        Matplotlib figure
    """
    H, W = image_src.shape[:2]

    # Warp source to target
    warped_pred = cv2.warpPerspective(image_src, H_pred, (W, H))

    # Define corners
    corners = np.array(
        [
            [0, 0],
            [W, 0],
            [W, H],
            [0, H],
        ],
        dtype=np.float32,
    )

    # Transform corners
    corners_pred = cv2.perspectiveTransform(corners.reshape(1, -1, 2), H_pred).squeeze()
    if H_gt is not None:
        corners_gt = cv2.perspectiveTransform(corners.reshape(1, -1, 2), H_gt).squeeze()

    # Create figure
    n_cols = 4 if H_gt is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    # Source image with corners
    axes[0].imshow(image_src, cmap="hot" if image_src.ndim == 2 else None)
    corners_closed = np.vstack([corners, corners[0]])
    axes[0].plot(corners_closed[:, 0], corners_closed[:, 1], "g-", linewidth=2)
    axes[0].set_title("Source")
    axes[0].axis("off")

    # Target image with predicted corners
    axes[1].imshow(image_tgt, cmap="hot" if image_tgt.ndim == 2 else None)
    pred_closed = np.vstack([corners_pred, corners_pred[0]])
    axes[1].plot(pred_closed[:, 0], pred_closed[:, 1], "r-", linewidth=2, label="Predicted")
    if H_gt is not None:
        gt_closed = np.vstack([corners_gt, corners_gt[0]])
        axes[1].plot(gt_closed[:, 0], gt_closed[:, 1], "g--", linewidth=2, label="Ground Truth")
        axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title("Target + Corners")
    axes[1].axis("off")

    # Warped source
    axes[2].imshow(warped_pred, cmap="hot" if warped_pred.ndim == 2 else None)
    axes[2].set_title("Warped (Predicted)")
    axes[2].axis("off")

    # Alignment error (if GT available)
    if H_gt is not None:
        error = np.abs(warped_pred.astype(float) - image_tgt.astype(float))
        axes[3].imshow(error, cmap="viridis")
        axes[3].set_title(f"Alignment Error\n(Mean: {error.mean():.1f})")
        axes[3].axis("off")

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_rotation_equivariance(
    results: dict[str, list[float]],
    save_path: str | None = None,
    compare_baseline: dict[str, list[float]] | None = None,
) -> plt.Figure:
    """
    Plot rotation equivariance evaluation results.

    For an equivariant model, the plot should be flat.
    For a non-equivariant model, error increases with rotation.

    Args:
        results: Dictionary with 'angles' and metric lists
        save_path: Path to save figure
        compare_baseline: Optional baseline results to compare

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    angles = results["angles"]
    metrics = ["corner_error", "rotation_error", "translation_error"]
    titles = ["Corner Error (px)", "Rotation Error (deg)", "Translation Error (px)"]

    for ax, metric, title in zip(axes, metrics, titles):
        # Plot main results
        ax.plot(angles, results[metric], "b-o", linewidth=2, markersize=6, label="Ours (E2-GNN)")

        # Plot baseline if provided
        if compare_baseline and metric in compare_baseline:
            ax.plot(
                angles,
                compare_baseline[metric],
                "r--s",
                linewidth=2,
                markersize=6,
                label="Baseline (CNN)",
            )

        ax.set_xlabel("Input Rotation (degrees)", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

        # Add horizontal line at mean (for equivariant model)
        mean_val = np.mean(results[metric])
        ax.axhline(y=mean_val, color="b", linestyle=":", alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_similarity_matrix(
    similarity: np.ndarray,
    positions_src: np.ndarray | None = None,
    positions_tgt: np.ndarray | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualize similarity matrix from model.

    Args:
        similarity: [N, M] similarity matrix
        positions_src: [N, 2] source positions (optional)
        positions_tgt: [M, 2] target positions (optional)
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot similarity matrix as heatmap
    im = ax.imshow(similarity, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Similarity")

    ax.set_xlabel("Target Nodes")
    ax.set_ylabel("Source Nodes")
    ax.set_title("Feature Similarity Matrix")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_graph_features(
    features: np.ndarray,
    positions: np.ndarray,
    image: np.ndarray | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualize graph node features overlaid on image.

    Uses t-SNE to reduce features to 3D for RGB coloring.

    Args:
        features: [N, D] node features
        positions: [N, 2] node positions
        image: Background image (optional)
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import MinMaxScaler

    # Reduce features to 3D for RGB coloring
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(features) - 1))
    features_3d = tsne.fit_transform(features)

    # Scale to [0, 1] for RGB
    scaler = MinMaxScaler()
    colors = scaler.fit_transform(features_3d)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Show background image if provided
    if image is not None:
        ax.imshow(image, cmap="gray", alpha=0.5)

    # Scatter plot of nodes colored by features
    # Convert positions from normalized [-1, 1] to image coordinates
    if image is not None:
        H, W = image.shape[:2]
        pos_img = (positions + 1) * np.array([W, H]) / 2
    else:
        pos_img = positions

    ax.scatter(pos_img[:, 0], pos_img[:, 1], c=colors, s=50, alpha=0.8)

    ax.set_title("Graph Node Features (t-SNE colored)")
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_features_tsne(
    features: np.ndarray,
    labels: np.ndarray | None = None,
    save_path: str | None = None,
    perplexity: int = 30,
    title: str = "Feature Embeddings (t-SNE)",
) -> plt.Figure:
    """
    t-SNE visualization of learned features.

    Args:
        features: [N, D] feature array
        labels: [N] optional labels for coloring
        save_path: Path to save figure
        perplexity: t-SNE perplexity parameter
        title: Figure title

    Returns:
        Matplotlib figure
    """
    from sklearn.manifold import TSNE

    # Adjust perplexity for small datasets
    n_samples = features.shape[0]
    perplexity = min(perplexity, max(5, n_samples - 1))

    logger.info(f"Running t-SNE on {n_samples} samples with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 10))

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[colors[i]],
                label=str(label),
                alpha=0.7,
                s=20,
            )
        ax.legend(loc="best", fontsize=8)
    else:
        ax.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.7, s=20)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"t-SNE visualization saved to {save_path}")

    return fig


def visualize_features_umap(
    features: np.ndarray,
    labels: np.ndarray | None = None,
    save_path: str | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    title: str = "Feature Embeddings (UMAP)",
) -> plt.Figure:
    """
    UMAP visualization of learned features.

    UMAP often provides better global structure than t-SNE.

    Args:
        features: [N, D] feature array
        labels: [N] optional labels for coloring
        save_path: Path to save figure
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        title: Figure title

    Returns:
        Matplotlib figure
    """
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn not installed. Install with: pip install umap-learn")
        raise

    n_samples = features.shape[0]
    n_neighbors = min(n_neighbors, max(2, n_samples - 1))

    logger.info(f"Running UMAP on {n_samples} samples with n_neighbors={n_neighbors}...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    features_2d = reducer.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 10))

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[colors[i]],
                label=str(label),
                alpha=0.7,
                s=20,
            )
        ax.legend(loc="best", fontsize=8)
    else:
        ax.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.7, s=20)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"UMAP visualization saved to {save_path}")

    return fig


def visualize_attention_maps(
    model,
    image_src: Tensor,
    image_tgt: Tensor,
    save_path: str | None = None,
    layer_names: list[str] | None = None,
) -> plt.Figure:
    """
    Extract and visualize attention weights from E2 layers.

    Args:
        model: The model with attention layers
        image_src: [1, C, H, W] source image
        image_tgt: [1, C, H, W] target image
        save_path: Path to save figure
        layer_names: Specific layers to visualize

    Returns:
        Matplotlib figure
    """
    model.eval()
    attention_maps = {}

    # Hook to capture attention weights
    def get_attention_hook(name):
        def hook(module, input, output):
            if hasattr(module, "attention_weights"):
                attention_maps[name] = module.attention_weights.detach().cpu()
            elif isinstance(output, tuple) and len(output) > 1:
                # Some attention modules return (output, attention_weights)
                if output[1] is not None:
                    attention_maps[name] = output[1].detach().cpu()

        return hook

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if layer_names is None or name in layer_names:
            if "attention" in name.lower() or "attn" in name.lower():
                hooks.append(module.register_forward_hook(get_attention_hook(name)))

    # Forward pass
    with torch.no_grad():
        _ = model(image_src, image_tgt)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    if not attention_maps:
        logger.warning("No attention maps captured. Model may not have attention layers.")
        # Return empty figure
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No attention maps available", ha="center", va="center")
        return fig

    # Create visualization
    n_maps = len(attention_maps)
    n_cols = min(4, n_maps)
    n_rows = (n_maps + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_maps == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_maps > 1 else [axes]

    for ax, (name, attn) in zip(axes, attention_maps.items()):
        # Average over batch and heads if needed
        while attn.dim() > 2:
            attn = attn.mean(dim=0)

        im = ax.imshow(attn.numpy(), cmap="viridis", aspect="auto")
        ax.set_title(f"{name}\n{list(attn.shape)}", fontsize=8)
        plt.colorbar(im, ax=ax)

    # Hide unused axes
    for ax in axes[n_maps:]:
        ax.axis("off")

    plt.suptitle("Attention Maps", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Attention visualization saved to {save_path}")

    return fig


def create_failure_case_report(
    model,
    dataset,
    n_worst: int = 10,
    output_dir: str | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    """
    Generate report of worst-performing samples.

    Useful for debugging and understanding model limitations.

    Args:
        model: The model to evaluate
        dataset: Dataset to evaluate on
        n_worst: Number of worst cases to analyze
        output_dir: Directory to save visualizations
        device: Device to use

    Returns:
        Dictionary with failure analysis
    """
    from src.training.metrics import corner_error, homography_vec_to_matrix

    model.eval()
    model.to(device)

    # Collect all predictions and errors
    all_errors = []
    all_indices = []
    all_predictions = []

    logger.info(f"Evaluating {len(dataset)} samples for failure analysis...")

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            image_src = sample["image_src"].unsqueeze(0).to(device)
            image_tgt = sample["image_tgt"].unsqueeze(0).to(device)
            H_gt = sample["homography_vec"].unsqueeze(0)

            output = model(image_src, image_tgt)
            H_pred = output["homography"].cpu()

            error = corner_error(H_pred, H_gt).item()
            all_errors.append(error)
            all_indices.append(idx)
            all_predictions.append(H_pred.squeeze())

    # Sort by error (descending)
    sorted_indices = np.argsort(all_errors)[::-1]
    worst_indices = sorted_indices[:n_worst]

    # Analyze worst cases
    report = {
        "n_samples": len(dataset),
        "mean_error": np.mean(all_errors),
        "median_error": np.median(all_errors),
        "std_error": np.std(all_errors),
        "worst_cases": [],
    }

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    for rank, idx in enumerate(worst_indices):
        sample = dataset[all_indices[idx]]
        error = all_errors[idx]

        case_info = {
            "rank": rank + 1,
            "dataset_index": all_indices[idx],
            "corner_error": error,
        }

        # Visualize if output_dir provided
        if output_dir:
            src_np = (sample["image_src"].squeeze().numpy() * 255).astype(np.uint8)
            tgt_np = (sample["image_tgt"].squeeze().numpy() * 255).astype(np.uint8)

            H_pred_vec = all_predictions[idx].numpy()
            H_pred_mat = (
                homography_vec_to_matrix(torch.tensor(H_pred_vec).unsqueeze(0)).squeeze().numpy()
            )
            H_gt_mat = sample["homography"].numpy()

            fig = visualize_homography(
                src_np,
                tgt_np,
                H_pred_mat,
                H_gt_mat,
                title=f"Failure Case #{rank + 1} (Error: {error:.1f}px)",
            )
            save_path = output_path / f"failure_case_{rank + 1:02d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            case_info["visualization"] = str(save_path)

        report["worst_cases"].append(case_info)

    logger.info(f"Failure analysis complete. Worst error: {all_errors[worst_indices[0]]:.1f}px")

    if output_dir:
        # Save report as JSON
        import json

        report_path = output_path / "failure_report.json"

        # Convert numpy types to Python types for JSON serialization
        report_json = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in report.items()
        }
        with open(report_path, "w") as f:
            json.dump(report_json, f, indent=2, default=str)
        logger.info(f"Report saved to {report_path}")

    return report


def create_paper_figures(
    model,
    test_samples: list[dict],
    output_dir: str,
    device: torch.device = torch.device("cpu"),
):
    """
    Generate all figures for paper.

    Args:
        model: Trained model
        test_samples: List of test samples
        output_dir: Directory to save figures
        device: Device to use
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)

    logger.info("Generating paper figures...")

    # 1. Qualitative results
    logger.info("  Generating qualitative results...")
    for i, sample in enumerate(test_samples[:5]):
        with torch.no_grad():
            src = sample["image_src"].unsqueeze(0).to(device)
            tgt = sample["image_tgt"].unsqueeze(0).to(device)
            output = model(src, tgt)

        H_pred = output["homography"].cpu().numpy().squeeze()
        H_pred_mat = np.eye(3)
        H_pred_mat[0, 0] = H_pred[0]
        H_pred_mat[0, 1] = H_pred[1]
        H_pred_mat[0, 2] = H_pred[2]
        H_pred_mat[1, 0] = H_pred[3]
        H_pred_mat[1, 1] = H_pred[4]
        H_pred_mat[1, 2] = H_pred[5]
        H_pred_mat[2, 0] = H_pred[6]
        H_pred_mat[2, 1] = H_pred[7]

        src_np = (sample["image_src"].squeeze().numpy() * 255).astype(np.uint8)
        tgt_np = (sample["image_tgt"].squeeze().numpy() * 255).astype(np.uint8)
        H_gt = sample["homography"].numpy()

        visualize_homography(
            src_np,
            tgt_np,
            H_pred_mat,
            H_gt,
            save_path=str(output_path / f"qualitative_{i}.png"),
            title=f"Sample {i + 1}",
        )

    # 2. Similarity matrix visualization
    logger.info("  Generating similarity matrix...")
    if "similarity" in output:
        sim = output["similarity"].cpu().numpy().squeeze()
        visualize_similarity_matrix(sim, save_path=str(output_path / "similarity_matrix.png"))

    logger.info(f"Figures saved to {output_dir}")
