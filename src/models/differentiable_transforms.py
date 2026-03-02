"""
Differentiable Geometric Transforms for Sim(2) Equivariance

This module provides differentiable implementations of:
- Rotation
- Scaling
- Translation
- Combined Sim(2) transformations

All transforms support batched operations and gradient flow for
end-to-end training.

Key functions:
- rotate_image: Rotate image by angle (differentiable)
- scale_image: Scale image by factor (differentiable)
- translate_image: Translate image by offset (differentiable)
- apply_sim2: Apply full Sim(2) transformation
- build_sim2_homography: Build homography matrix from (θ, s, t)

Author: ECCV 2026 Submission
Date: 2026-01-30
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def rotation_matrix_2d(theta: Tensor) -> Tensor:
    """
    Build 2D rotation matrix from angle.

    Args:
        theta: [B] rotation angle in radians

    Returns:
        R: [B, 2, 2] rotation matrices
    """
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    R = torch.stack([
        torch.stack([cos_t, -sin_t], dim=-1),
        torch.stack([sin_t, cos_t], dim=-1),
    ], dim=-2)

    return R


def build_sim2_homography(
    rotation: Tensor,
    scale: Tensor,
    translation: Tensor,
    center: Tensor | None = None,
) -> Tensor:
    """
    Build 3x3 homography matrix from Sim(2) parameters.

    When center is None (default), the transformation is about the origin:
        H = T(t) @ S(s) @ R(θ)

    When center is provided, the transformation is about the center point:
        H = T(center) @ S(s) @ R(θ) @ T(-center) @ T(t)

    This is equivalent to:
        1. Translate point to origin (relative to center)
        2. Rotate and scale
        3. Translate back to center
        4. Apply additional translation

    The centered form expands to:
        H = [[s*cos(θ), -s*sin(θ), cx*(1 - s*cos(θ)) + cy*s*sin(θ) + tx],
             [s*sin(θ),  s*cos(θ), cy*(1 - s*cos(θ)) - cx*s*sin(θ) + ty],
             [0,         0,         1]]

    Args:
        rotation: [B] rotation angle in radians
        scale: [B] scale factor
        translation: [B, 2] translation vector (tx, ty) in pixels
        center: [B, 2] center point (cx, cy) for rotation/scaling, or None for origin

    Returns:
        H: [B, 3, 3] homography matrices
    """
    B = rotation.shape[0]
    device = rotation.device
    dtype = rotation.dtype

    cos_t = torch.cos(rotation)
    sin_t = torch.sin(rotation)
    s = scale

    tx = translation[:, 0]
    ty = translation[:, 1]

    H = torch.zeros(B, 3, 3, device=device, dtype=dtype)

    # Upper-left 2x2: s * R(θ)
    H[:, 0, 0] = s * cos_t
    H[:, 0, 1] = -s * sin_t
    H[:, 1, 0] = s * sin_t
    H[:, 1, 1] = s * cos_t

    if center is not None:
        # Centered transformation: rotate/scale about center point
        cx = center[:, 0]
        cy = center[:, 1]

        # Translation includes center offset compensation
        # H = T(center) @ S(s) @ R(θ) @ T(-center) @ T(t)
        H[:, 0, 2] = cx * (1 - s * cos_t) + cy * s * sin_t + tx
        H[:, 1, 2] = cy * (1 - s * cos_t) - cx * s * sin_t + ty
    else:
        # Origin-centered transformation (original behavior)
        H[:, 0, 2] = tx
        H[:, 1, 2] = ty

    # Bottom row
    H[:, 2, 2] = 1.0

    return H


def decompose_sim2_homography(H: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    Decompose homography matrix into Sim(2) parameters.

    Extracts rotation, scale, and translation from H = T @ S @ R.

    Args:
        H: [B, 3, 3] homography matrices

    Returns:
        rotation: [B] rotation angle in radians
        scale: [B] scale factor
        translation: [B, 2] translation vector
    """
    # Extract rotation from upper-left 2x2
    # H[:2,:2] = s * R, so R = H[:2,:2] / ||H[:2,:2]||
    h00, h01 = H[:, 0, 0], H[:, 0, 1]
    h10, h11 = H[:, 1, 0], H[:, 1, 1]

    # Scale is the norm of first column (or row)
    scale = torch.sqrt(h00**2 + h10**2)

    # Rotation angle
    rotation = torch.atan2(h10, h00)

    # Translation
    translation = H[:, :2, 2]

    return rotation, scale, translation


def get_rotation_grid(
    image: Tensor,
    theta: Tensor,
    center: Tensor | None = None,
) -> Tensor:
    """
    Get sampling grid for rotation transformation.

    Args:
        image: [B, C, H, W] input image (for shape info)
        theta: [B] rotation angle in radians
        center: [B, 2] rotation center in [-1, 1] coords, default (0, 0)

    Returns:
        grid: [B, H, W, 2] sampling grid for grid_sample
    """
    B, _, H, W = image.shape
    device = image.device
    dtype = image.dtype

    if center is None:
        center = torch.zeros(B, 2, device=device, dtype=dtype)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # Build affine matrix for rotation around center
    # First translate to center, then rotate, then translate back
    # M = T(center) @ R(θ) @ T(-center)
    # For grid_sample, we need the inverse transformation

    # Inverse rotation
    cos_t_inv = cos_t  # cos(-θ) = cos(θ)
    sin_t_inv = -sin_t  # sin(-θ) = -sin(θ)

    # Affine matrix (2x3) for grid_sample
    # [x'] = [cos -sin] [x - cx] + [cx]
    # [y']   [sin  cos] [y - cy]   [cy]
    affine = torch.zeros(B, 2, 3, device=device, dtype=dtype)
    affine[:, 0, 0] = cos_t_inv
    affine[:, 0, 1] = -sin_t_inv
    affine[:, 0, 2] = center[:, 0] - cos_t_inv * center[:, 0] + sin_t_inv * center[:, 1]
    affine[:, 1, 0] = sin_t_inv
    affine[:, 1, 1] = cos_t_inv
    affine[:, 1, 2] = center[:, 1] - sin_t_inv * center[:, 0] - cos_t_inv * center[:, 1]

    grid = F.affine_grid(affine, image.size(), align_corners=True)

    return grid


def rotate_image(
    image: Tensor,
    theta: Tensor,
    center: Tensor | None = None,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
) -> Tensor:
    """
    Rotate image by angle theta (differentiable).

    Args:
        image: [B, C, H, W] input image
        theta: [B] rotation angle in radians (positive = counterclockwise)
        center: [B, 2] rotation center in [-1, 1] coords, default (0, 0)
        mode: interpolation mode ('bilinear', 'nearest', 'bicubic')
        padding_mode: padding mode ('zeros', 'border', 'reflection')

    Returns:
        rotated: [B, C, H, W] rotated image
    """
    grid = get_rotation_grid(image, theta, center)
    rotated = F.grid_sample(
        image, grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True
    )
    return rotated


def get_scale_grid(
    image: Tensor,
    scale: Tensor,
    center: Tensor | None = None,
) -> Tensor:
    """
    Get sampling grid for scale transformation.

    Args:
        image: [B, C, H, W] input image (for shape info)
        scale: [B] scale factor (>1 = zoom in, <1 = zoom out)
        center: [B, 2] scale center in [-1, 1] coords, default (0, 0)

    Returns:
        grid: [B, H, W, 2] sampling grid for grid_sample
    """
    B, _, H, W = image.shape
    device = image.device
    dtype = image.dtype

    if center is None:
        center = torch.zeros(B, 2, device=device, dtype=dtype)

    # For grid_sample, we need inverse transformation
    # To scale up by s, we sample from 1/s positions
    inv_scale = 1.0 / scale

    # Affine matrix for scaling around center
    affine = torch.zeros(B, 2, 3, device=device, dtype=dtype)
    affine[:, 0, 0] = inv_scale
    affine[:, 0, 2] = center[:, 0] * (1 - inv_scale)
    affine[:, 1, 1] = inv_scale
    affine[:, 1, 2] = center[:, 1] * (1 - inv_scale)

    grid = F.affine_grid(affine, image.size(), align_corners=True)

    return grid


def scale_image(
    image: Tensor,
    scale: Tensor,
    center: Tensor | None = None,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
) -> Tensor:
    """
    Scale image by factor (differentiable).

    Args:
        image: [B, C, H, W] input image
        scale: [B] scale factor (>1 = zoom in, <1 = zoom out)
        center: [B, 2] scale center in [-1, 1] coords, default (0, 0)
        mode: interpolation mode
        padding_mode: padding mode

    Returns:
        scaled: [B, C, H, W] scaled image
    """
    grid = get_scale_grid(image, scale, center)
    scaled = F.grid_sample(
        image, grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True
    )
    return scaled


def get_translation_grid(
    image: Tensor,
    translation: Tensor,
) -> Tensor:
    """
    Get sampling grid for translation transformation.

    Args:
        image: [B, C, H, W] input image (for shape info)
        translation: [B, 2] translation in [-1, 1] coords

    Returns:
        grid: [B, H, W, 2] sampling grid for grid_sample
    """
    B, _, H, W = image.shape
    device = image.device
    dtype = image.dtype

    # For grid_sample, we need inverse transformation
    # To translate by t, we sample from positions shifted by -t
    inv_translation = -translation

    # Affine matrix for translation
    affine = torch.zeros(B, 2, 3, device=device, dtype=dtype)
    affine[:, 0, 0] = 1.0
    affine[:, 0, 2] = inv_translation[:, 0]
    affine[:, 1, 1] = 1.0
    affine[:, 1, 2] = inv_translation[:, 1]

    grid = F.affine_grid(affine, image.size(), align_corners=True)

    return grid


def translate_image(
    image: Tensor,
    translation: Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
) -> Tensor:
    """
    Translate image by offset (differentiable).

    Args:
        image: [B, C, H, W] input image
        translation: [B, 2] translation in [-1, 1] coords
        mode: interpolation mode
        padding_mode: padding mode

    Returns:
        translated: [B, C, H, W] translated image
    """
    grid = get_translation_grid(image, translation)
    translated = F.grid_sample(
        image, grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True
    )
    return translated


def get_sim2_grid(
    image: Tensor,
    rotation: Tensor,
    scale: Tensor,
    translation: Tensor,
    center: Tensor | None = None,
) -> Tensor:
    """
    Get sampling grid for full Sim(2) transformation.

    Transform order: first rotate, then scale, then translate.
    H = T(t) @ S(s) @ R(θ)

    For grid_sample, we need the inverse:
    H^{-1} = R(-θ) @ S(1/s) @ T(-t)

    Args:
        image: [B, C, H, W] input image (for shape info)
        rotation: [B] rotation angle in radians
        scale: [B] scale factor
        translation: [B, 2] translation in [-1, 1] coords
        center: [B, 2] transform center in [-1, 1] coords, default (0, 0)

    Returns:
        grid: [B, H, W, 2] sampling grid for grid_sample
    """
    B, _, H, W = image.shape
    device = image.device
    dtype = image.dtype

    if center is None:
        center = torch.zeros(B, 2, device=device, dtype=dtype)

    # Inverse parameters
    inv_rotation = -rotation
    inv_scale = 1.0 / scale
    inv_translation = -translation

    cos_t = torch.cos(inv_rotation)
    sin_t = torch.sin(inv_rotation)

    # Combined affine: R(-θ) @ S(1/s) @ T(-t)
    # But we also need to handle the center of rotation/scaling
    # Full transform around center c:
    #   T(c) @ R(θ) @ S(s) @ T(-c) @ T(t)
    # Inverse:
    #   T(-t) @ T(c) @ S(1/s) @ R(-θ) @ T(-c)

    # For simplicity, we'll compute the combined affine matrix
    s = inv_scale

    # Translation components
    tx = inv_translation[:, 0]
    ty = inv_translation[:, 1]
    cx = center[:, 0]
    cy = center[:, 1]

    # Combined affine matrix (2x3)
    # The full transformation in matrix form
    affine = torch.zeros(B, 2, 3, device=device, dtype=dtype)

    # Linear part: s * R(-θ)
    affine[:, 0, 0] = s * cos_t
    affine[:, 0, 1] = -s * sin_t
    affine[:, 1, 0] = s * sin_t
    affine[:, 1, 1] = s * cos_t

    # Translation part (combining all translations)
    # After rotation and scale around center, apply inverse translation
    affine[:, 0, 2] = tx + cx * (1 - s * cos_t) + cy * s * sin_t
    affine[:, 1, 2] = ty + cy * (1 - s * cos_t) - cx * s * sin_t

    grid = F.affine_grid(affine, image.size(), align_corners=True)

    return grid


def apply_sim2(
    image: Tensor,
    rotation: Tensor,
    scale: Tensor,
    translation: Tensor,
    center: Tensor | None = None,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
) -> Tensor:
    """
    Apply full Sim(2) transformation to image (differentiable).

    Transform order: rotate → scale → translate.

    Args:
        image: [B, C, H, W] input image
        rotation: [B] rotation angle in radians
        scale: [B] scale factor
        translation: [B, 2] translation in [-1, 1] coords
        center: [B, 2] transform center in [-1, 1] coords, default (0, 0)
        mode: interpolation mode
        padding_mode: padding mode

    Returns:
        transformed: [B, C, H, W] transformed image
    """
    grid = get_sim2_grid(image, rotation, scale, translation, center)
    transformed = F.grid_sample(
        image, grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True
    )
    return transformed


def apply_homography(
    image: Tensor,
    H: Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
) -> Tensor:
    """
    Apply homography transformation to image.

    For Sim(2) homographies, this is equivalent to apply_sim2,
    but works with the 3x3 matrix directly.

    Args:
        image: [B, C, H, W] input image
        H: [B, 3, 3] homography matrices
        mode: interpolation mode
        padding_mode: padding mode

    Returns:
        transformed: [B, C, H, W] transformed image
    """
    B, C, H_img, W_img = image.shape
    device = image.device
    dtype = image.dtype

    # Create coordinate grid
    y = torch.linspace(-1, 1, H_img, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, W_img, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

    # Homogeneous coordinates [x, y, 1]
    ones = torch.ones_like(grid_x)
    coords = torch.stack([grid_x, grid_y, ones], dim=-1)  # [H, W, 3]
    coords = coords.view(1, -1, 3).expand(B, -1, -1)  # [B, H*W, 3]

    # Apply inverse homography (for sampling)
    H_inv = torch.linalg.inv(H)

    # Transform coordinates
    coords_transformed = torch.bmm(coords, H_inv.transpose(-2, -1))  # [B, H*W, 3]

    # Normalize by homogeneous coordinate
    coords_transformed = coords_transformed[..., :2] / (coords_transformed[..., 2:3] + 1e-8)

    # Reshape to grid
    grid = coords_transformed.view(B, H_img, W_img, 2)

    # Sample
    transformed = F.grid_sample(
        image, grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True
    )

    return transformed


class DifferentiableSim2Transform(nn.Module):
    """
    Neural network module for differentiable Sim(2) transformations.

    Wraps the transform functions as a module for use in torch.nn.Sequential.
    """

    def __init__(
        self,
        mode: str = 'bilinear',
        padding_mode: str = 'zeros',
    ):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(
        self,
        image: Tensor,
        rotation: Tensor,
        scale: Tensor,
        translation: Tensor,
    ) -> Tensor:
        return apply_sim2(
            image, rotation, scale, translation,
            mode=self.mode, padding_mode=self.padding_mode
        )
