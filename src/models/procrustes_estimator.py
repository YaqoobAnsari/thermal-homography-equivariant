"""
Procrustes-Based Rotation and Scale Estimator

Uses soft correspondences from the feature similarity matrix to estimate
the geometric transformation (rotation, scale, translation) via Procrustes analysis.

Mathematical Foundation:
Given point correspondences P (source) and Q (target):
    Q ≈ s * R @ P + t

The optimal (R, s, t) minimizing ||Q - (s * R @ P + t)||²_F has a closed-form
solution via SVD:
    1. Center both point clouds: P_c = P - mean(P), Q_c = Q - mean(Q)
    2. Compute cross-covariance: H = P_c^T @ Q_c
    3. SVD: H = U @ Σ @ V^T
    4. Rotation: R = V @ U^T
    5. Scale: s = trace(Σ) / ||P_c||²_F
    6. Translation: t = mean(Q) - s * R @ mean(P)

For soft correspondences from similarity matrix:
    Q_soft[i] = Σ_j softmax(S[i,j]) * Q[j]

This is differentiable and works for any pattern because it operates on
geometry (correspondences), not image content directly.

References:
- Schönemann (1966): "A generalized solution of the orthogonal Procrustes problem"
- Arun et al. (1987): "Least-squares fitting of two 3D point sets"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ProcrustesRotationEstimator(nn.Module):
    """
    Estimate rotation (and optionally scale) from soft correspondences using Procrustes.

    This is a non-parametric, geometry-based approach that doesn't require
    learning and works for any pattern.

    Args:
        estimate_scale: Also estimate scale factor
        robust_weights: Use feature confidence as correspondence weights
        min_singular_value: Minimum singular value for numerical stability
    """

    def __init__(
        self,
        estimate_scale: bool = True,
        robust_weights: bool = True,
        min_singular_value: float = 1e-6,
    ):
        super().__init__()

        self.estimate_scale = estimate_scale
        self.robust_weights = robust_weights
        self.min_singular_value = min_singular_value

        logger.info(f"ProcrustesRotationEstimator: scale={estimate_scale}, robust={robust_weights}")

    def forward(
        self,
        positions_src: Tensor,
        positions_tgt: Tensor,
        similarity: Tensor,
        features_src: Tensor | None = None,
        features_tgt: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Estimate transformation from soft correspondences.

        Args:
            positions_src: [B, N, 2] source positions
            positions_tgt: [B, M, 2] target positions
            similarity: [B, N, M] similarity matrix (raw, pre-softmax)
            features_src: [B, N, D] optional source features for weighting
            features_tgt: [B, M, D] optional target features for weighting

        Returns:
            dict with:
                rotation: [B] rotation angle in radians
                scale: [B] scale factor (if estimate_scale=True)
                translation: [B, 2] translation vector
                R_matrix: [B, 2, 2] rotation matrix
        """
        B, N, _ = positions_src.shape
        M = positions_tgt.shape[1]

        # Soft correspondence matrix
        # correspondence[i] = softmax over targets for source i
        correspondence = F.softmax(similarity, dim=-1)  # [B, N, M]

        # Compute soft-matched target positions
        # Q_soft[i] = Σ_j correspondence[i,j] * positions_tgt[j]
        matched_tgt = torch.bmm(correspondence, positions_tgt)  # [B, N, 2]

        # Optional: weight correspondences by feature confidence
        if self.robust_weights and features_src is not None:
            # Weight by feature magnitude (more salient points weighted higher)
            weights = features_src.norm(dim=-1, keepdim=True)  # [B, N, 1]
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # Normalize
        else:
            weights = torch.ones(B, N, 1, device=positions_src.device) / N

        # Weighted centroids
        centroid_src = (weights * positions_src).sum(dim=1, keepdim=True)  # [B, 1, 2]
        centroid_tgt = (weights * matched_tgt).sum(dim=1, keepdim=True)  # [B, 1, 2]

        # Center the point clouds
        P = positions_src - centroid_src  # [B, N, 2]
        Q = matched_tgt - centroid_tgt    # [B, N, 2]

        # Apply weights
        P_weighted = weights.sqrt() * P  # [B, N, 2]
        Q_weighted = weights.sqrt() * Q  # [B, N, 2]

        # Cross-covariance matrix: H = P^T @ Q
        H = torch.bmm(P_weighted.transpose(-2, -1), Q_weighted)  # [B, 2, 2]

        # SVD
        U, S, Vh = torch.linalg.svd(H)

        # Rotation: R = V @ U^T
        R = torch.bmm(Vh.transpose(-2, -1), U.transpose(-2, -1))  # [B, 2, 2]

        # Handle reflection (ensure det(R) = +1)
        det_R = R[:, 0, 0] * R[:, 1, 1] - R[:, 0, 1] * R[:, 1, 0]
        needs_flip = det_R < 0

        if needs_flip.any():
            # Flip the last column of V to fix reflection
            Vh_fixed = Vh.clone()
            Vh_fixed[needs_flip, -1, :] *= -1
            R[needs_flip] = torch.bmm(
                Vh_fixed[needs_flip].transpose(-2, -1),
                U[needs_flip].transpose(-2, -1)
            )

        # Extract rotation angle
        rotation = torch.atan2(R[:, 1, 0], R[:, 0, 0])  # [B]

        # Scale
        if self.estimate_scale:
            # CRITICAL FIX: Use UNWEIGHTED centered points for scale
            # Correct formula: scale = trace(Σ) / ||P_centered||²_F
            # Previous bug: used weighted P (P_weighted) which biased scale
            P_norm_sq = (P ** 2).sum(dim=[1, 2])  # [B] - Unweighted!
            scale = S.sum(dim=1) / (P_norm_sq + 1e-8)  # [B]
            scale = scale.clamp(min=0.1, max=10.0)
        else:
            scale = torch.ones(B, device=positions_src.device)

        # Translation
        centroid_src_sq = centroid_src.squeeze(1)  # [B, 2]
        centroid_tgt_sq = centroid_tgt.squeeze(1)  # [B, 2]
        translation = centroid_tgt_sq - scale.unsqueeze(-1) * torch.bmm(
            R, centroid_src_sq.unsqueeze(-1)
        ).squeeze(-1)  # [B, 2]

        return {
            'rotation': rotation,
            'scale': scale,
            'translation': translation,
            'R_matrix': R,
        }


class DifferentiableProcrustesLayer(nn.Module):
    """
    Differentiable Procrustes alignment as a neural network layer.

    Can be inserted into the model to directly predict rotation/scale
    from learned correspondences.

    The key insight is that Procrustes is fully differentiable through SVD,
    allowing end-to-end training.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        temperature: float = 0.1,
    ):
        super().__init__()

        self.temperature = temperature

        # Optional: learned temperature
        self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))

        # Confidence prediction from features
        self.confidence_net = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features_src: Tensor,
        features_tgt: Tensor,
        positions_src: Tensor,
        positions_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Compute rotation and scale via differentiable Procrustes.

        Args:
            features_src: [B, N, D] source features
            features_tgt: [B, M, D] target features
            positions_src: [B, N, 2] source positions
            positions_tgt: [B, M, 2] target positions

        Returns:
            dict with rotation, scale, translation, similarity matrix
        """
        B, N, D = features_src.shape
        M = features_tgt.shape[1]

        # Compute similarity matrix
        features_src_norm = F.normalize(features_src, dim=-1)
        features_tgt_norm = F.normalize(features_tgt, dim=-1)

        similarity = torch.bmm(features_src_norm, features_tgt_norm.transpose(-2, -1))
        temp = self.log_temp.exp()
        similarity = similarity / temp

        # Soft correspondences
        correspondence = F.softmax(similarity, dim=-1)  # [B, N, M]

        # Soft matched positions
        matched_tgt = torch.bmm(correspondence, positions_tgt)  # [B, N, 2]

        # Correspondence confidence
        confidence = self.confidence_net(features_src)  # [B, N, 1]
        confidence = confidence / (confidence.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted centroids
        centroid_src = (confidence * positions_src).sum(dim=1, keepdim=True)
        centroid_tgt = (confidence * matched_tgt).sum(dim=1, keepdim=True)

        # Centered points
        P = positions_src - centroid_src
        Q = matched_tgt - centroid_tgt

        # Weighted points
        P_w = confidence.sqrt() * P
        Q_w = confidence.sqrt() * Q

        # Cross-covariance
        H = torch.bmm(P_w.transpose(-2, -1), Q_w)

        # SVD (differentiable in PyTorch)
        U, S, Vh = torch.linalg.svd(H)

        # Rotation
        R = torch.bmm(Vh.transpose(-2, -1), U.transpose(-2, -1))

        # Fix reflections
        det_R = torch.det(R)
        needs_flip = det_R < 0
        if needs_flip.any():
            flip_matrix = torch.eye(2, device=R.device).unsqueeze(0).expand(B, -1, -1).clone()
            flip_matrix[needs_flip, 1, 1] = -1
            R = torch.bmm(Vh.transpose(-2, -1), torch.bmm(flip_matrix, U.transpose(-2, -1)))

        rotation = torch.atan2(R[:, 1, 0], R[:, 0, 0])

        # Scale - CRITICAL FIX: Use UNWEIGHTED centered points
        # Correct formula: scale = trace(Σ) / ||P_centered||²_F
        P_norm_sq = (P ** 2).sum(dim=[1, 2])  # Unweighted
        scale = S.sum(dim=1) / (P_norm_sq + 1e-8)
        scale = scale.clamp(min=0.1, max=10.0)

        # Translation
        translation = centroid_tgt.squeeze(1) - scale.unsqueeze(-1) * torch.bmm(
            R, centroid_src.squeeze(1).unsqueeze(-1)
        ).squeeze(-1)

        return {
            'rotation': rotation,
            'scale': scale,
            'translation': translation,
            'R_matrix': R,
            'similarity': similarity,
            'correspondence': correspondence,
        }


class RANSACRobustProcrustes(nn.Module):
    """
    RANSAC-like robust Procrustes for outlier rejection.

    Uses multiple random subsets to find the best transformation,
    making it robust to incorrect correspondences.

    Note: This is less differentiable but more robust for inference.
    """

    def __init__(
        self,
        num_iterations: int = 100,
        subset_size: int = 4,
        inlier_threshold: float = 0.1,
    ):
        super().__init__()

        self.num_iterations = num_iterations
        self.subset_size = subset_size
        self.inlier_threshold = inlier_threshold

        logger.info(f"RANSACRobustProcrustes: {num_iterations} iters, subset={subset_size}")

    def forward(
        self,
        positions_src: Tensor,
        positions_tgt: Tensor,
        correspondence: Tensor,
    ) -> dict[str, Tensor]:
        """
        Robust Procrustes with RANSAC.

        Args:
            positions_src: [B, N, 2]
            positions_tgt: [B, M, 2]
            correspondence: [B, N, M] soft correspondence matrix

        Returns:
            dict with rotation, scale, translation, inlier_mask
        """
        B, N, _ = positions_src.shape
        device = positions_src.device

        # Get hard correspondences (argmax)
        hard_corr = correspondence.argmax(dim=-1)  # [B, N]
        matched_tgt = torch.gather(
            positions_tgt,
            1,
            hard_corr.unsqueeze(-1).expand(-1, -1, 2)
        )  # [B, N, 2]

        best_rotation = torch.zeros(B, device=device)
        best_scale = torch.ones(B, device=device)
        best_translation = torch.zeros(B, 2, device=device)
        best_inlier_count = torch.zeros(B, device=device)

        for _ in range(self.num_iterations):
            # Random subset
            indices = torch.randperm(N, device=device)[:self.subset_size]

            P_subset = positions_src[:, indices]  # [B, K, 2]
            Q_subset = matched_tgt[:, indices]    # [B, K, 2]

            # Procrustes on subset
            centroid_P = P_subset.mean(dim=1, keepdim=True)
            centroid_Q = Q_subset.mean(dim=1, keepdim=True)

            P_c = P_subset - centroid_P
            Q_c = Q_subset - centroid_Q

            H = torch.bmm(P_c.transpose(-2, -1), Q_c)
            U, S, Vh = torch.linalg.svd(H)
            R = torch.bmm(Vh.transpose(-2, -1), U.transpose(-2, -1))

            # Fix reflections
            det_R = torch.det(R)
            needs_flip = det_R < 0
            if needs_flip.any():
                Vh[needs_flip, -1, :] *= -1
                R[needs_flip] = torch.bmm(
                    Vh[needs_flip].transpose(-2, -1),
                    U[needs_flip].transpose(-2, -1)
                )

            rotation = torch.atan2(R[:, 1, 0], R[:, 0, 0])

            P_var = (P_c ** 2).sum(dim=[1, 2])
            scale = S.sum(dim=1) / (P_var + 1e-8)
            scale = scale.clamp(min=0.1, max=10.0)

            translation = centroid_Q.squeeze(1) - scale.unsqueeze(-1) * torch.bmm(
                R, centroid_P.squeeze(1).unsqueeze(-1)
            ).squeeze(-1)

            # Count inliers
            P_transformed = scale.unsqueeze(-1).unsqueeze(-1) * torch.bmm(
                R.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, 2, 2),
                positions_src.reshape(B * N, 2, 1)
            ).reshape(B, N, 2) + translation.unsqueeze(1)

            residuals = (P_transformed - matched_tgt).norm(dim=-1)  # [B, N]
            inliers = residuals < self.inlier_threshold
            inlier_count = inliers.float().sum(dim=1)  # [B]

            # Update best
            improved = inlier_count > best_inlier_count
            best_rotation = torch.where(improved, rotation, best_rotation)
            best_scale = torch.where(improved, scale, best_scale)
            best_translation = torch.where(
                improved.unsqueeze(-1),
                translation,
                best_translation
            )
            best_inlier_count = torch.maximum(best_inlier_count, inlier_count)

        return {
            'rotation': best_rotation,
            'scale': best_scale,
            'translation': best_translation,
            'inlier_count': best_inlier_count,
            'inlier_ratio': best_inlier_count / N,
        }
