"""
Loss Functions for Homography Estimation

Implements multiple loss formulations:
1. Corner reprojection loss (primary)
2. Geodesic rotation loss (SO(2) geometry)
3. Translation loss (Euclidean)
4. Rank regularization loss (for similarity matrix)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.constants import DEFAULT_RANK_TARGET
from src.utils.homography import apply_homography, homography_vec_to_matrix
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def corner_loss(
    H_pred: Tensor,
    H_gt: Tensor,
    image_size: tuple[int, int] = (256, 256),
) -> Tensor:
    """
    Corner reprojection loss.

    Measures the average distance between predicted and ground truth
    corners after applying the homography.

    Args:
        H_pred: [B, 3, 3] or [B, 8] predicted homography
        H_gt: [B, 3, 3] or [B, 8] ground truth homography
        image_size: (H, W) for defining corners

    Returns:
        Scalar loss value
    """
    # Convert to matrix if needed
    if H_pred.shape[-1] == 8:
        H_pred = homography_vec_to_matrix(H_pred)
    if H_gt.shape[-1] == 8:
        H_gt = homography_vec_to_matrix(H_gt)

    B = H_pred.shape[0]
    H, W = image_size
    device = H_pred.device

    # Define corners (normalized to [-1, 1])
    corners = torch.tensor(
        [
            [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1],
        ],
        device=device,
        dtype=H_pred.dtype,
    )  # [4, 2]

    # Scale to image coordinates
    corners = corners * torch.tensor([W / 2, H / 2], device=device)
    corners = corners.unsqueeze(0).expand(B, -1, -1)  # [B, 4, 2]

    # Transform corners with predicted and GT homography
    corners_pred = apply_homography(H_pred, corners)
    corners_gt = apply_homography(H_gt, corners)

    # Compute L2 distance
    error = torch.norm(corners_pred - corners_gt, dim=-1)  # [B, 4]
    loss = error.mean()

    return loss


def geodesic_rotation_loss(H_pred: Tensor, H_gt: Tensor) -> Tensor:
    """
    Geodesic loss on the rotation component.

    For SO(2), the geodesic distance is simply the absolute angle difference.

    L = arccos((trace(R_pred^T @ R_gt) - 1) / 2)

    Note: For 2D rotations, this simplifies considerably.

    Args:
        H_pred: [B, 3, 3] or [B, 8] predicted homography
        H_gt: [B, 3, 3] or [B, 8] ground truth homography

    Returns:
        Scalar loss value (in radians)
    """
    if H_pred.shape[-1] == 8:
        H_pred = homography_vec_to_matrix(H_pred)
    if H_gt.shape[-1] == 8:
        H_gt = homography_vec_to_matrix(H_gt)

    # Extract rotation part (upper-left 2x2)
    R_pred = H_pred[:, :2, :2]
    R_gt = H_gt[:, :2, :2]

    # Compute rotation angle from each matrix
    # For a 2D rotation matrix [[cos, -sin], [sin, cos]], angle = atan2(sin, cos)
    angle_pred = torch.atan2(R_pred[:, 1, 0], R_pred[:, 0, 0])
    angle_gt = torch.atan2(R_gt[:, 1, 0], R_gt[:, 0, 0])

    # Angle difference (handle wraparound)
    angle_diff = angle_pred - angle_gt
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

    # Loss is absolute angle difference
    loss = angle_diff.abs().mean()

    return loss


def translation_loss(H_pred: Tensor, H_gt: Tensor) -> Tensor:
    """
    L2 loss on translation component.

    Args:
        H_pred: [B, 3, 3] or [B, 8] predicted homography
        H_gt: [B, 3, 3] or [B, 8] ground truth homography

    Returns:
        Scalar loss value
    """
    if H_pred.shape[-1] == 8:
        H_pred = homography_vec_to_matrix(H_pred)
    if H_gt.shape[-1] == 8:
        H_gt = homography_vec_to_matrix(H_gt)

    # Extract translation (third column, first two rows)
    t_pred = H_pred[:, :2, 2]
    t_gt = H_gt[:, :2, 2]

    # L2 loss
    loss = F.mse_loss(t_pred, t_gt)

    return loss


def photometric_loss(
    image_src: Tensor,
    image_tgt: Tensor,
    H_pred: Tensor,
) -> Tensor:
    """
    Photometric consistency loss.

    Warps source to target and compares pixel values.
    Useful as auxiliary loss but sensitive to colormap.

    Args:
        image_src: [B, C, H, W] source image
        image_tgt: [B, C, H, W] target image
        H_pred: [B, 3, 3] or [B, 8] predicted homography

    Returns:
        Scalar loss value
    """
    if H_pred.shape[-1] == 8:
        H_pred = homography_vec_to_matrix(H_pred)

    B, C, H, W = image_src.shape
    device = image_src.device

    # Create sampling grid
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

    # Scale to image coordinates
    grid_scaled = grid * torch.tensor([W / 2, H / 2], device=device)

    # Reshape for homography application
    grid_flat = grid_scaled.view(B, -1, 2)  # [B, H*W, 2]

    # Apply homography
    grid_transformed = apply_homography(H_pred, grid_flat)

    # Scale back to [-1, 1]
    grid_transformed = grid_transformed / torch.tensor([W / 2, H / 2], device=device)
    grid_transformed = grid_transformed.view(B, H, W, 2)

    # Warp source image
    warped = F.grid_sample(
        image_src,
        grid_transformed,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    # Compute loss (robust L1)
    loss = F.l1_loss(warped, image_tgt)

    return loss


class DynamicLossWeighting(nn.Module):
    """
    Uncertainty-weighted multi-task loss (Kendall et al., CVPR 2018).

    Learns task-specific uncertainty parameters to automatically balance
    multiple loss terms. Each task has a learnable log-variance parameter
    that determines its weight in the total loss.

    The loss for task i is: (1/2σ²ᵢ) * Lᵢ + log(σᵢ)

    This encourages the model to:
    - Decrease σ for tasks it can predict well
    - Increase σ for tasks with high inherent uncertainty

    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses
    for Scene Geometry and Semantics" (Kendall et al., 2018)
    """

    def __init__(self, num_tasks: int = 4, initial_log_var: float = 0.0):
        """
        Initialize dynamic loss weighting.

        Args:
            num_tasks: Number of loss terms to balance
            initial_log_var: Initial value for log-variance parameters
        """
        super().__init__()
        self.num_tasks = num_tasks
        # Learnable log-variance parameters (one per task)
        self.log_vars = nn.Parameter(torch.full((num_tasks,), initial_log_var))

    def forward(self, losses: list) -> Tensor:
        """
        Compute weighted sum of losses.

        Args:
            losses: List of loss tensors (one per task)

        Returns:
            Weighted total loss
        """
        if len(losses) != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} losses, got {len(losses)}")

        weighted_losses = []
        for i, loss in enumerate(losses):
            # Precision (inverse variance)
            precision = torch.exp(-self.log_vars[i])
            # Weighted loss + regularization term
            weighted = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted)

        return sum(weighted_losses)

    def get_weights(self) -> dict[str, float]:
        """
        Get current task weights (inverse variance).

        Returns:
            Dictionary mapping task index to weight
        """
        weights = {}
        for i in range(self.num_tasks):
            weights[f"task_{i}_weight"] = torch.exp(-self.log_vars[i]).item()
            weights[f"task_{i}_log_var"] = self.log_vars[i].item()
        return weights


class HomographyLoss(nn.Module):
    """
    Combined loss function for homography estimation.

    Combines:
    - Corner reprojection loss (primary)
    - Geodesic rotation loss (optional)
    - Translation loss (optional)
    - Rank regularization loss (optional)
    """

    def __init__(
        self,
        corner_weight: float = 1.0,
        rotation_weight: float = 0.1,
        translation_weight: float = 0.1,
        rank_weight: float = 0.01,
        image_size: tuple[int, int] = (256, 256),
        use_dynamic_weighting: bool = False,
    ):
        super().__init__()

        self.corner_weight = corner_weight
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
        self.rank_weight = rank_weight
        self.image_size = image_size
        self.use_dynamic_weighting = use_dynamic_weighting

        # Dynamic weighting module (learns weights automatically)
        if use_dynamic_weighting:
            self.dynamic_weighting = DynamicLossWeighting(num_tasks=4)
        else:
            self.dynamic_weighting = None

    def forward(
        self,
        H_pred: Tensor,
        H_gt: Tensor,
        similarity: Tensor | None = None,
    ) -> dict:
        """
        Compute combined loss.

        Args:
            H_pred: [B, 8] or [B, 3, 3] predicted homography
            H_gt: [B, 8] or [B, 3, 3] ground truth homography
            similarity: [B, N, N] similarity matrix (optional)

        Returns:
            Dictionary with loss values
        """
        losses = {}

        # Corner loss
        loss_corner = corner_loss(H_pred, H_gt, self.image_size)
        losses["corner"] = loss_corner

        # Rotation loss
        if self.rotation_weight > 0:
            loss_rotation = geodesic_rotation_loss(H_pred, H_gt)
            losses["rotation"] = loss_rotation
        else:
            loss_rotation = 0

        # Translation loss
        if self.translation_weight > 0:
            loss_translation = translation_loss(H_pred, H_gt)
            losses["translation"] = loss_translation
        else:
            loss_translation = 0

        # Rank regularization
        if self.rank_weight > 0 and similarity is not None:
            loss_rank = self._rank_loss(similarity)
            losses["rank"] = loss_rank
        else:
            loss_rank = 0

        # Total loss - either dynamic or fixed weighting
        if self.use_dynamic_weighting and self.dynamic_weighting is not None:
            # Collect non-zero losses for dynamic weighting
            active_losses = [loss_corner]
            if self.rotation_weight > 0:
                active_losses.append(loss_rotation)
            if self.translation_weight > 0:
                active_losses.append(loss_translation)
            if self.rank_weight > 0 and similarity is not None:
                active_losses.append(loss_rank)

            # Pad to 4 losses if needed (dynamic weighting expects fixed size)
            while len(active_losses) < 4:
                active_losses.append(torch.tensor(0.0, device=loss_corner.device))

            total = self.dynamic_weighting(active_losses)

            # Log learned weights
            weights = self.dynamic_weighting.get_weights()
            for k, v in weights.items():
                losses[f"dyn_{k}"] = v
        else:
            # Fixed weighting (original behavior)
            total = (
                self.corner_weight * loss_corner
                + self.rotation_weight * loss_rotation
                + self.translation_weight * loss_translation
                + self.rank_weight * loss_rank
            )

        losses["total"] = total

        return losses

    def _rank_loss(self, similarity: Tensor, target_rank: int = DEFAULT_RANK_TARGET) -> Tensor:
        """
        Rank regularization loss for similarity matrix.

        Encourages the similarity matrix to have approximately target_rank
        effective rank, which promotes unique correspondences.

        Args:
            similarity: Similarity matrix [B, N, M]
            target_rank: Desired effective rank

        Returns:
            Scalar rank loss
        """
        # Nuclear norm as proxy for rank
        nuclear_norm = torch.linalg.matrix_norm(similarity, ord="nuc")
        return (nuclear_norm - target_rank).abs().mean()
