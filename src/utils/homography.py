"""
Homography Manipulation Utilities

Functions for working with 8-DoF homography matrices.
"""

import math

import torch
from torch import Tensor

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def homography_vec_to_matrix(h_vec: Tensor) -> Tensor:
    """
    Convert 8D homography vector to 3x3 matrix.

    The 8D vector represents [h11, h12, h13, h21, h22, h23, h31, h32]
    with h33 = 1 (normalized homography).

    Args:
        h_vec: [B, 8] or [8] homography parameters

    Returns:
        H: [B, 3, 3] or [3, 3] homography matrices
    """
    if h_vec.dim() == 1:
        h_vec = h_vec.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    B = h_vec.shape[0]
    device = h_vec.device
    dtype = h_vec.dtype

    H = torch.zeros(B, 3, 3, device=device, dtype=dtype)
    H[:, 0, 0] = h_vec[:, 0]
    H[:, 0, 1] = h_vec[:, 1]
    H[:, 0, 2] = h_vec[:, 2]
    H[:, 1, 0] = h_vec[:, 3]
    H[:, 1, 1] = h_vec[:, 4]
    H[:, 1, 2] = h_vec[:, 5]
    H[:, 2, 0] = h_vec[:, 6]
    H[:, 2, 1] = h_vec[:, 7]
    H[:, 2, 2] = 1.0

    if squeeze:
        H = H.squeeze(0)

    return H


def homography_matrix_to_vec(H: Tensor) -> Tensor:
    """
    Convert 3x3 homography matrix to 8D vector.

    Args:
        H: [B, 3, 3] or [3, 3] homography matrices

    Returns:
        h_vec: [B, 8] or [8] homography parameters
    """
    if H.dim() == 2:
        H = H.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    # Normalize so H[2,2] = 1
    H = H / (H[:, 2:3, 2:3] + 1e-8)

    h_vec = torch.stack(
        [
            H[:, 0, 0],
            H[:, 0, 1],
            H[:, 0, 2],
            H[:, 1, 0],
            H[:, 1, 1],
            H[:, 1, 2],
            H[:, 2, 0],
            H[:, 2, 1],
        ],
        dim=-1,
    )

    if squeeze:
        h_vec = h_vec.squeeze(0)

    return h_vec


def decompose_homography(
    H: Tensor,
    return_type: str = "similarity_residual",
) -> dict:
    """
    Decompose homography into interpretable components.

    Options:
    - "similarity_residual": H = Similarity @ Residual
    - "rts": Rotation, Translation, Scale
    - "full": All components

    Args:
        H: [B, 3, 3] or [3, 3] homography matrices
        return_type: Type of decomposition

    Returns:
        Dictionary of components
    """
    if H.dim() == 2:
        H = H.unsqueeze(0)

    B = H.shape[0]
    device = H.device
    dtype = H.dtype

    # Normalize
    H = H / (H[:, 2:3, 2:3] + 1e-8)

    # Extract rotation angle from upper-left 2x2
    angle = torch.atan2(H[:, 1, 0], H[:, 0, 0])  # [B]

    # Extract scale (average of x and y scales)
    scale_x = torch.sqrt(H[:, 0, 0] ** 2 + H[:, 1, 0] ** 2)
    scale_y = torch.sqrt(H[:, 0, 1] ** 2 + H[:, 1, 1] ** 2)
    scale = (scale_x + scale_y) / 2  # [B]

    # Extract translation
    translation = H[:, :2, 2]  # [B, 2]

    # Build similarity transform
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    similarity = torch.zeros(B, 3, 3, device=device, dtype=dtype)
    similarity[:, 0, 0] = scale * cos_a
    similarity[:, 0, 1] = -scale * sin_a
    similarity[:, 1, 0] = scale * sin_a
    similarity[:, 1, 1] = scale * cos_a
    similarity[:, 0, 2] = translation[:, 0]
    similarity[:, 1, 2] = translation[:, 1]
    similarity[:, 2, 2] = 1.0

    # Compute residual
    similarity_inv = torch.linalg.inv(similarity)
    residual = torch.bmm(similarity_inv, H)

    result = {
        "angle": angle,  # [B] in radians
        "scale": scale,  # [B]
        "translation": translation,  # [B, 2]
        "similarity": similarity,  # [B, 3, 3]
        "residual": residual,  # [B, 3, 3]
    }

    # Compute how much of the transform is captured by similarity
    # (useful for justifying E(2) approximation)
    sim_norm = torch.linalg.matrix_norm(similarity - torch.eye(3, device=device))
    residual_norm = torch.linalg.matrix_norm(residual - torch.eye(3, device=device))
    result["similarity_fraction"] = sim_norm / (sim_norm + residual_norm + 1e-8)

    return result


def apply_homography(H: Tensor, points: Tensor) -> Tensor:
    """
    Apply homography transformation to 2D points.

    Args:
        H: [B, 3, 3] homography matrices
        points: [B, N, 2] 2D points

    Returns:
        Transformed points [B, N, 2]
    """
    B, N, _ = points.shape

    # Convert to homogeneous coordinates
    ones = torch.ones(B, N, 1, device=points.device, dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=-1)  # [B, N, 3]

    # Apply homography: p' = H @ p
    points_transformed = torch.bmm(points_h, H.transpose(-2, -1))  # [B, N, 3]

    # Convert back to Cartesian (divide by w)
    w = points_transformed[..., 2:3]
    points_cart = points_transformed[..., :2] / (w + 1e-8)

    return points_cart


def compose_homographies(H1: Tensor, H2: Tensor) -> Tensor:
    """
    Compose two homographies: H_combined = H2 @ H1

    This applies H1 first, then H2.

    Args:
        H1: [B, 3, 3] first homography
        H2: [B, 3, 3] second homography

    Returns:
        Combined homography [B, 3, 3]
    """
    return torch.bmm(H2, H1)


def invert_homography(H: Tensor) -> Tensor:
    """
    Invert homography matrix.

    Args:
        H: [B, 3, 3] or [3, 3] homography

    Returns:
        Inverse homography
    """
    return torch.linalg.inv(H)


def random_homography(
    batch_size: int = 1,
    rotation_range: tuple[float, float] = (-30, 30),
    translation_range: tuple[float, float] = (-30, 30),
    scale_range: tuple[float, float] = (0.9, 1.1),
    image_size: tuple[int, int] = (256, 256),
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Generate random homography matrices.

    Args:
        batch_size: Number of homographies
        rotation_range: Range of rotation in degrees
        translation_range: Range of translation in pixels
        scale_range: Range of scale factors
        image_size: Image dimensions for centering
        device: Target device

    Returns:
        [B, 3, 3] random homographies
    """
    H, W = image_size
    cx, cy = W / 2, H / 2

    # Random parameters
    angles = torch.empty(batch_size, device=device).uniform_(*rotation_range) * math.pi / 180
    tx = torch.empty(batch_size, device=device).uniform_(*translation_range)
    ty = torch.empty(batch_size, device=device).uniform_(*translation_range)
    scales = torch.empty(batch_size, device=device).uniform_(*scale_range)

    # Build homographies
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    H_mat = torch.zeros(batch_size, 3, 3, device=device)

    # Rotation around center + scale + translation
    H_mat[:, 0, 0] = scales * cos_a
    H_mat[:, 0, 1] = -scales * sin_a
    H_mat[:, 0, 2] = tx + cx - scales * (cx * cos_a - cy * sin_a)
    H_mat[:, 1, 0] = scales * sin_a
    H_mat[:, 1, 1] = scales * cos_a
    H_mat[:, 1, 2] = ty + cy - scales * (cx * sin_a + cy * cos_a)
    H_mat[:, 2, 2] = 1.0

    return H_mat


def four_point_to_homography(
    offsets: Tensor,
    image_size: tuple[int, int] = (256, 256),
) -> Tensor:
    """
    Convert 4-point corner offsets to 3×3 homography matrix via DLT.

    This is the inverse of what HomographyNet predicts: it takes the
    4 corner offsets and computes the corresponding homography matrix.

    Args:
        offsets: [B, 8] corner offsets (dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4)
        image_size: (H, W) image dimensions for defining corner positions

    Returns:
        H: [B, 3, 3] homography matrices
    """
    B = offsets.shape[0]
    device = offsets.device
    dtype = offsets.dtype
    H, W = image_size

    # Define source corners (image corners)
    corners_src = torch.tensor(
        [
            [0.0, 0.0],  # top-left
            [W, 0.0],  # top-right
            [W, H],  # bottom-right
            [0.0, H],  # bottom-left
        ],
        dtype=dtype,
        device=device,
    )
    corners_src = corners_src.unsqueeze(0).expand(B, -1, -1)  # [B, 4, 2]

    # Target corners = source corners + offsets
    offsets_reshaped = offsets.view(B, 4, 2)  # [B, 4, 2]
    corners_tgt = corners_src + offsets_reshaped

    # Compute homography using DLT (Direct Linear Transform)
    return _dlt_homography(corners_src, corners_tgt)


def _dlt_homography(pts_src: Tensor, pts_tgt: Tensor) -> Tensor:
    """
    Compute homography from 4+ point correspondences using DLT.

    Args:
        pts_src: [B, N, 2] source points (N >= 4)
        pts_tgt: [B, N, 2] target points

    Returns:
        H: [B, 3, 3] homography matrices
    """
    B, N, _ = pts_src.shape
    device = pts_src.device
    dtype = pts_src.dtype

    # Build DLT matrix
    x = pts_src[:, :, 0]  # [B, N]
    y = pts_src[:, :, 1]  # [B, N]
    xp = pts_tgt[:, :, 0]  # [B, N]
    yp = pts_tgt[:, :, 1]  # [B, N]

    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)

    # For each correspondence, we have two equations:
    # -x*h1 - y*h2 - h3 + x'*x*h7 + x'*y*h8 + x'*h9 = 0
    # -x*h4 - y*h5 - h6 + y'*x*h7 + y'*y*h8 + y'*h9 = 0

    # Build A matrix: [B, 2N, 9]
    A1 = torch.stack([-x, -y, -ones, zeros, zeros, zeros, x * xp, y * xp, xp], dim=-1)
    A2 = torch.stack([zeros, zeros, zeros, -x, -y, -ones, x * yp, y * yp, yp], dim=-1)

    A = torch.zeros(B, 2 * N, 9, device=device, dtype=dtype)
    A[:, 0::2, :] = A1
    A[:, 1::2, :] = A2

    # Solve using SVD
    try:
        _, _, Vh = torch.linalg.svd(A)
        h = Vh[:, -1, :]  # [B, 9] - last row of V^T (null space)
    except RuntimeError:
        # Fallback to identity
        h = torch.zeros(B, 9, device=device, dtype=dtype)
        h[:, 0] = 1.0  # h11
        h[:, 4] = 1.0  # h22
        h[:, 8] = 1.0  # h33

    # Reshape to 3x3 matrix
    H = h.view(B, 3, 3)

    # Normalize so H[2,2] = 1
    H = H / (H[:, 2:3, 2:3] + 1e-8)

    return H


def homography_to_four_point(
    H: Tensor,
    image_size: tuple[int, int] = (256, 256),
) -> Tensor:
    """
    Convert 3×3 homography matrix to 4-point corner offsets.

    This is what HomographyNet effectively does: the homography
    is represented as offsets of the four image corners.

    Args:
        H: [B, 3, 3] or [B, 8] homography
        image_size: (H, W) image dimensions

    Returns:
        offsets: [B, 8] corner offsets
    """
    if H.dim() == 2 and H.shape[-1] == 8:
        H = homography_vec_to_matrix(H)

    B = H.shape[0]
    device = H.device
    dtype = H.dtype
    Himg, Wimg = image_size

    # Define corners
    corners = torch.tensor(
        [
            [0.0, 0.0],
            [Wimg, 0.0],
            [Wimg, Himg],
            [0.0, Himg],
        ],
        dtype=dtype,
        device=device,
    )
    corners = corners.unsqueeze(0).expand(B, -1, -1)  # [B, 4, 2]

    # Apply homography to corners
    corners_transformed = apply_homography(H, corners)

    # Compute offsets
    offsets = corners_transformed - corners

    return offsets.view(B, 8)
