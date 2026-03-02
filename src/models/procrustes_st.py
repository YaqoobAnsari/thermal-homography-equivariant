"""
Procrustes Scale and Translation Estimator

This module implements Procrustes analysis for estimating scale and
translation from point correspondences, assuming rotation has already
been detected and removed.

After de-rotation, the remaining transformation is:
    Q = s * P + t

where:
    - P: source points
    - Q: target points (matched)
    - s: scale factor
    - t: translation vector

The closed-form solution is:
    s = ||Q_centered||_F / ||P_centered||_F
    t = centroid(Q) - s * centroid(P)

This is a simplified version of full Procrustes that doesn't need to
estimate rotation (since it was already handled).

Author: Yaqoob Ansari
Date: 2026-01-30
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ProcrustesScaleTranslation(nn.Module):
    """
    Estimate scale and translation from point correspondences.

    This assumes rotation has already been estimated and removed,
    so the remaining transformation is just scale + translation.

    Mathematical foundation:
    Given correspondences P (source) → Q (target) with no rotation:
        Q = s * P + t

    Optimal solution:
        centroid_P = weighted_mean(P)
        centroid_Q = weighted_mean(Q)
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q
        s = ||Q_centered||_F / ||P_centered||_F
        t = centroid_Q - s * centroid_P

    Args:
        min_scale: Minimum allowed scale (for numerical stability)
        max_scale: Maximum allowed scale
        use_weighted: Use confidence weights for estimation
    """

    def __init__(
        self,
        min_scale: float = 0.1,
        max_scale: float = 10.0,
        use_weighted: bool = True,
    ):
        super().__init__()

        self.min_scale = min_scale
        self.max_scale = max_scale
        self.use_weighted = use_weighted

        logger.info(f"ProcrustesScaleTranslation: scale_range=[{min_scale}, {max_scale}]")

    def forward(
        self,
        positions_src: Tensor,
        positions_tgt: Tensor,
        confidence: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Estimate scale and translation from correspondences.

        Args:
            positions_src: [B, N, 2] source positions
            positions_tgt: [B, N, 2] matched target positions
            confidence: [B, N] optional confidence weights

        Returns:
            dict with:
                scale: [B] estimated scale factor
                translation: [B, 2] estimated translation vector
                residual: [B] average residual after transformation
        """
        B, N, _ = positions_src.shape
        device = positions_src.device

        # Confidence weights
        if confidence is not None and self.use_weighted:
            # Normalize weights to sum to 1
            weights = confidence / (confidence.sum(dim=-1, keepdim=True) + 1e-8)  # [B, N]
            weights = weights.unsqueeze(-1)  # [B, N, 1]
        else:
            weights = torch.ones(B, N, 1, device=device) / N

        # Weighted centroids
        centroid_src = (weights * positions_src).sum(dim=1)  # [B, 2]
        centroid_tgt = (weights * positions_tgt).sum(dim=1)  # [B, 2]

        # Center the point clouds
        P_centered = positions_src - centroid_src.unsqueeze(1)  # [B, N, 2]
        Q_centered = positions_tgt - centroid_tgt.unsqueeze(1)  # [B, N, 2]

        # Weighted centered points for scale estimation
        P_weighted = weights.sqrt() * P_centered
        Q_weighted = weights.sqrt() * Q_centered

        # Scale: ||Q||_F / ||P||_F
        P_norm = P_weighted.norm(dim=[1, 2]) + 1e-8  # [B]
        Q_norm = Q_weighted.norm(dim=[1, 2]) + 1e-8  # [B]

        scale = Q_norm / P_norm  # [B]
        scale = scale.clamp(self.min_scale, self.max_scale)

        # Translation: t = centroid_Q - s * centroid_P
        translation = centroid_tgt - scale.unsqueeze(-1) * centroid_src  # [B, 2]

        # Residual (how well the transformation fits)
        # Q_pred = s * P + t
        positions_pred = scale.unsqueeze(-1).unsqueeze(-1) * positions_src + translation.unsqueeze(1)
        residual_per_point = (positions_pred - positions_tgt).norm(dim=-1)  # [B, N]

        if confidence is not None:
            residual = (confidence * residual_per_point).sum(dim=-1) / (confidence.sum(dim=-1) + 1e-8)
        else:
            residual = residual_per_point.mean(dim=-1)  # [B]

        return {
            'scale': scale,
            'translation': translation,
            'residual': residual,
            'centroid_src': centroid_src,
            'centroid_tgt': centroid_tgt,
        }


class FullProcrustesEstimator(nn.Module):
    """
    Full Procrustes estimator for rotation, scale, and translation.

    This is the complete Procrustes analysis that estimates all three
    components simultaneously. Use this if you want to estimate rotation
    from correspondences rather than from cyclic correlation.

    Note: For the Sim(2) equivariant architecture, we prefer cyclic
    correlation for rotation (more robust) and use ProcrustesScaleTranslation
    for scale and translation after de-rotation.

    Mathematical foundation:
    Given correspondences P → Q:
        Q = s * R @ P + t

    Solution via SVD:
        1. Center: P_c = P - mean(P), Q_c = Q - mean(Q)
        2. Cross-covariance: H = P_c^T @ Q_c
        3. SVD: H = U @ Σ @ V^T
        4. Rotation: R = V @ U^T (fix sign if det < 0)
        5. Scale: s = trace(Σ) / ||P_c||²_F
        6. Translation: t = mean(Q) - s * R @ mean(P)
    """

    def __init__(
        self,
        min_scale: float = 0.1,
        max_scale: float = 10.0,
    ):
        super().__init__()

        self.min_scale = min_scale
        self.max_scale = max_scale

        logger.info(f"FullProcrustesEstimator: scale_range=[{min_scale}, {max_scale}]")

    def forward(
        self,
        positions_src: Tensor,
        positions_tgt: Tensor,
        confidence: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Full Procrustes estimation of R, s, t.

        Args:
            positions_src: [B, N, 2] source positions
            positions_tgt: [B, N, 2] matched target positions
            confidence: [B, N] optional confidence weights

        Returns:
            dict with rotation, scale, translation, R_matrix, residual
        """
        B, N, _ = positions_src.shape
        device = positions_src.device

        # Confidence weights
        if confidence is not None:
            weights = confidence / (confidence.sum(dim=-1, keepdim=True) + 1e-8)
            weights = weights.unsqueeze(-1)  # [B, N, 1]
        else:
            weights = torch.ones(B, N, 1, device=device) / N

        # Weighted centroids
        centroid_src = (weights * positions_src).sum(dim=1, keepdim=True)  # [B, 1, 2]
        centroid_tgt = (weights * positions_tgt).sum(dim=1, keepdim=True)  # [B, 1, 2]

        # Center the points
        P = positions_src - centroid_src  # [B, N, 2]
        Q = positions_tgt - centroid_tgt  # [B, N, 2]

        # Weighted centered points
        P_w = weights.sqrt() * P
        Q_w = weights.sqrt() * Q

        # Cross-covariance matrix H = P^T @ Q
        H = torch.bmm(P_w.transpose(-2, -1), Q_w)  # [B, 2, 2]

        # SVD
        U, S, Vh = torch.linalg.svd(H)

        # Rotation R = V @ U^T
        R = torch.bmm(Vh.transpose(-2, -1), U.transpose(-2, -1))  # [B, 2, 2]

        # Fix reflection (ensure det(R) = +1)
        det_R = R[:, 0, 0] * R[:, 1, 1] - R[:, 0, 1] * R[:, 1, 0]
        needs_flip = det_R < 0

        if needs_flip.any():
            Vh_fixed = Vh.clone()
            Vh_fixed[needs_flip, -1, :] *= -1
            R[needs_flip] = torch.bmm(
                Vh_fixed[needs_flip].transpose(-2, -1),
                U[needs_flip].transpose(-2, -1)
            )

        # Extract rotation angle
        rotation = torch.atan2(R[:, 1, 0], R[:, 0, 0])  # [B]

        # Scale: s = trace(Σ) / ||P_centered||²_F (using unweighted P)
        P_norm_sq = (P ** 2).sum(dim=[1, 2])  # [B]
        scale = S.sum(dim=1) / (P_norm_sq + 1e-8)  # [B]
        scale = scale.clamp(self.min_scale, self.max_scale)

        # Translation
        centroid_src_sq = centroid_src.squeeze(1)  # [B, 2]
        centroid_tgt_sq = centroid_tgt.squeeze(1)  # [B, 2]
        translation = centroid_tgt_sq - scale.unsqueeze(-1) * torch.bmm(
            R, centroid_src_sq.unsqueeze(-1)
        ).squeeze(-1)  # [B, 2]

        # Residual
        positions_pred = scale.unsqueeze(-1).unsqueeze(-1) * torch.bmm(
            positions_src.view(B * N, 1, 2),
            R.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, 2, 2).transpose(-2, -1)
        ).view(B, N, 2) + translation.unsqueeze(1)

        residual_per_point = (positions_pred - positions_tgt).norm(dim=-1)  # [B, N]
        residual = residual_per_point.mean(dim=-1)  # [B]

        return {
            'rotation': rotation,
            'scale': scale,
            'translation': translation,
            'R_matrix': R,
            'residual': residual,
        }


class RobustProcrustesEstimator(nn.Module):
    """
    Robust Procrustes with iterative reweighting.

    Uses iteratively reweighted least squares (IRLS) to be robust
    to outlier correspondences.

    The weight update is based on residuals:
        w_i = 1 / (residual_i + epsilon)

    This downweights points with large residuals.
    """

    def __init__(
        self,
        num_iterations: int = 3,
        min_scale: float = 0.1,
        max_scale: float = 10.0,
    ):
        super().__init__()

        self.num_iterations = num_iterations
        self.base_estimator = FullProcrustesEstimator(
            min_scale=min_scale,
            max_scale=max_scale,
        )

        logger.info(f"RobustProcrustesEstimator: {num_iterations} IRLS iterations")

    def forward(
        self,
        positions_src: Tensor,
        positions_tgt: Tensor,
        confidence: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Robust Procrustes with iterative reweighting.
        """
        B, N, _ = positions_src.shape
        device = positions_src.device

        # Initial weights
        if confidence is not None:
            weights = confidence.clone()
        else:
            weights = torch.ones(B, N, device=device)

        result = None

        for iteration in range(self.num_iterations):
            # Estimate with current weights
            result = self.base_estimator(positions_src, positions_tgt, weights)

            # Compute residuals
            R = result['R_matrix']
            scale = result['scale']
            translation = result['translation']

            # Transform source points
            positions_pred = scale.unsqueeze(-1).unsqueeze(-1) * torch.bmm(
                positions_src.view(B * N, 1, 2),
                R.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, 2, 2).transpose(-2, -1)
            ).view(B, N, 2) + translation.unsqueeze(1)

            residuals = (positions_pred - positions_tgt).norm(dim=-1)  # [B, N]

            # Update weights (downweight high-residual points)
            new_weights = 1.0 / (residuals + 0.01)

            # Combine with original confidence
            if confidence is not None:
                weights = confidence * new_weights
            else:
                weights = new_weights

            # Normalize
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        return result
