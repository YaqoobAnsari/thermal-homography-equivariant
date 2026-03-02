"""
Geometric Utilities

Common geometric operations for homography estimation.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def homography_matrix_to_vec_np(H: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 homography matrix to 8D vector (numpy version).

    Normalizes so H[2,2] = 1 and returns [h11, h12, h13, h21, h22, h23, h31, h32].

    This is the numpy equivalent of src.utils.homography.homography_matrix_to_vec
    (which operates on torch tensors and supports batched inputs).

    Args:
        H: [3, 3] homography matrix (numpy).

    Returns:
        [8] vector of homography parameters.
    """
    H = H / (H[2, 2] + 1e-8)
    return np.array([
        H[0, 0], H[0, 1], H[0, 2],
        H[1, 0], H[1, 1], H[1, 2],
        H[2, 0], H[2, 1],
    ], dtype=np.float32)


def compute_corner_error(
    H_pred: np.ndarray,
    H_gt: np.ndarray,
    image_size: tuple[int, int] = (256, 256),
) -> float:
    """
    Compute mean corner reprojection error (numpy version).

    Args:
        H_pred: [3, 3] predicted homography
        H_gt: [3, 3] ground truth homography
        image_size: (H, W) image dimensions

    Returns:
        Mean corner error in pixels
    """
    H, W = image_size

    # Define corners
    corners = np.array(
        [
            [0, 0],
            [W, 0],
            [W, H],
            [0, H],
        ],
        dtype=np.float32,
    ).reshape(1, -1, 2)

    # Transform corners
    corners_pred = cv2.perspectiveTransform(corners, H_pred).squeeze()
    corners_gt = cv2.perspectiveTransform(corners, H_gt).squeeze()

    # Compute error
    error = np.linalg.norm(corners_pred - corners_gt, axis=1)

    return error.mean()


def warp_image(
    image: np.ndarray,
    H: np.ndarray,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Warp image using homography.

    Args:
        image: Input image [H, W] or [H, W, C]
        H: [3, 3] homography matrix
        output_size: (W, H) output size (default: same as input)

    Returns:
        Warped image
    """
    if output_size is None:
        output_size = (image.shape[1], image.shape[0])

    return cv2.warpPerspective(image, H, output_size)


def warp_image_torch(
    image: Tensor,
    H: Tensor,
) -> Tensor:
    """
    Warp image using homography (differentiable PyTorch version).

    Args:
        image: [B, C, H, W] input images
        H: [B, 3, 3] homography matrices

    Returns:
        Warped images [B, C, H, W]
    """
    B, C, height, width = image.shape
    device = image.device

    # Create sampling grid
    y = torch.linspace(-1, 1, height, device=device)
    x = torch.linspace(-1, 1, width, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

    # Scale to image coordinates
    grid_scaled = grid * torch.tensor([width / 2, height / 2], device=device)
    grid_flat = grid_scaled.view(B, -1, 2)  # [B, H*W, 2]

    # Add homogeneous coordinate
    ones = torch.ones(B, height * width, 1, device=device)
    grid_h = torch.cat([grid_flat, ones], dim=-1)  # [B, H*W, 3]

    # Apply homography
    grid_transformed = torch.bmm(grid_h, H.transpose(-2, -1))  # [B, H*W, 3]

    # Convert back to Cartesian
    grid_transformed = grid_transformed[..., :2] / (grid_transformed[..., 2:3] + 1e-8)

    # Scale back to [-1, 1]
    grid_transformed = grid_transformed / torch.tensor([width / 2, height / 2], device=device)
    grid_transformed = grid_transformed.view(B, height, width, 2)

    # Sample
    warped = F.grid_sample(
        image,
        grid_transformed,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return warped


def create_grid_points(
    grid_size: int = 32,
    image_size: tuple[int, int] = (256, 256),
    normalized: bool = True,
) -> np.ndarray:
    """
    Create uniform grid of points.

    Args:
        grid_size: Number of points per side
        image_size: (H, W) image dimensions
        normalized: If True, return points in [-1, 1], else in pixels

    Returns:
        [N, 2] array of points where N = grid_size^2
    """
    H, W = image_size

    if normalized:
        y = np.linspace(-1, 1, grid_size)
        x = np.linspace(-1, 1, grid_size)
    else:
        y = np.linspace(0, H, grid_size)
        x = np.linspace(0, W, grid_size)

    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    points = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)

    return points


def build_knn_graph(
    positions: np.ndarray,
    k: int = 8,
) -> np.ndarray:
    """
    Build k-nearest neighbor graph from positions.

    Args:
        positions: [N, 2] node positions
        k: Number of neighbors

    Returns:
        [2, E] edge index array
    """
    from scipy.spatial import KDTree

    tree = KDTree(positions)
    distances, indices = tree.query(positions, k=k + 1)  # +1 because self is included

    # Build edge list (excluding self-loops)
    N = len(positions)
    edges = []
    for i in range(N):
        for j in indices[i, 1:]:  # Skip self
            edges.append([i, j])

    return np.array(edges).T


def decompose_homography_svd(H: np.ndarray) -> dict:
    """
    Decompose homography using SVD.

    Args:
        H: [3, 3] homography matrix

    Returns:
        Dictionary with decomposition components
    """
    # Normalize
    H = H / H[2, 2]

    # SVD of upper-left 2x2
    A = H[:2, :2]
    U, S, Vt = np.linalg.svd(A)

    # Extract components
    rotation = U @ Vt
    scale = S.mean()
    shear = S[0] / S[1] if S[1] > 0 else 1.0
    translation = H[:2, 2]
    perspective = H[2, :2]

    return {
        "rotation": rotation,
        "rotation_angle": np.arctan2(rotation[1, 0], rotation[0, 0]),
        "scale": scale,
        "shear": shear,
        "translation": translation,
        "perspective": perspective,
        "is_affine": np.allclose(perspective, 0),
    }


def normalize_homography(H: np.ndarray) -> np.ndarray:
    """
    Normalize homography so H[2,2] = 1.

    Args:
        H: [3, 3] homography

    Returns:
        Normalized homography
    """
    return H / (H[2, 2] + 1e-8)


def estimate_homography_dlt(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> np.ndarray:
    """
    Estimate homography using Direct Linear Transform (DLT).

    Args:
        src_points: [N, 2] source points
        dst_points: [N, 2] destination points

    Returns:
        [3, 3] homography matrix
    """
    # Use OpenCV for robust implementation
    H, _ = cv2.findHomography(src_points, dst_points, method=0)
    return normalize_homography(H)


def estimate_homography_ransac(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    threshold: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate homography using RANSAC.

    Args:
        src_points: [N, 2] source points
        dst_points: [N, 2] destination points
        threshold: RANSAC inlier threshold

    Returns:
        Tuple of (homography, inlier_mask)
    """
    H, mask = cv2.findHomography(
        src_points,
        dst_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold,
    )
    return normalize_homography(H), mask.ravel().astype(bool)
