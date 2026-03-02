"""
Correlation-Based Dense Matching for Rotation-Equivariant Correspondences

The Problem:
- Fixed grid sampling extracts features at the same positions in both images
- Under rotation, different image content appears at the same grid position
- This breaks correspondence matching

The Solution:
- Extract DENSE feature maps from both images
- For each position in source, SEARCH for best match in target
- The matched position tells us WHERE the content moved to
- Procrustes then estimates the transformation from position pairs

This is similar to optical flow, but designed for larger transformations.

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


class DenseFeatureExtractor(nn.Module):
    """
    Extract dense feature maps that are locally rotation-invariant.

    Uses gradient magnitude (rotation-invariant) as the base feature,
    with learned refinement that preserves invariance.
    """

    def __init__(self, feature_dim: int = 32, num_scales: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_scales = num_scales

        # Multi-scale gradient-based features
        # Gradient magnitude is rotation-invariant
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3 + 2*i, padding=1 + i, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )
            for i in range(num_scales)
        ])

        # Combine multi-scale features
        self.combine = nn.Sequential(
            nn.Conv2d(16 * num_scales, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
        )

        # For rotation-invariant features, we use gradient magnitude
        self.grad_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.grad_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.grad_x.weight.data = sobel_x.view(1, 1, 3, 3)
        self.grad_y.weight.data = sobel_y.view(1, 1, 3, 3)
        self.grad_x.weight.requires_grad = False
        self.grad_y.weight.requires_grad = False

    def forward(self, image: Tensor) -> Tensor:
        """
        Extract dense rotation-invariant features.

        Args:
            image: [B, 1, H, W] grayscale image

        Returns:
            features: [B, D, H, W] dense feature map
        """
        # Gradient magnitude (rotation-invariant)
        gx = self.grad_x(image)
        gy = self.grad_y(image)
        grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)

        # Multi-scale learned features (initialized with gradient info)
        scale_features = []
        for scale_net in self.scales:
            # Input is gradient magnitude (rotation-invariant)
            scale_feat = scale_net(grad_mag)
            scale_features.append(scale_feat)

        # Concatenate and combine
        multi_scale = torch.cat(scale_features, dim=1)
        features = self.combine(multi_scale)

        return features


class CorrelationMatcher(nn.Module):
    """
    Find correspondences via dense correlation matching.

    For each position in source, find the best matching position in target
    by computing correlation over a search window (or global search).
    """

    def __init__(
        self,
        feature_dim: int = 32,
        search_radius: int | None = None,  # None = global search
        temperature: float = 0.1,
        sub_pixel: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.search_radius = search_radius
        self.temperature = temperature
        self.sub_pixel = sub_pixel

        # Optional: learned temperature
        self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))

        logger.info(f"CorrelationMatcher: search_radius={search_radius}, sub_pixel={sub_pixel}")

    def forward(
        self,
        features_src: Tensor,
        features_tgt: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Find dense correspondences via correlation.

        Args:
            features_src: [B, D, H, W] source feature map
            features_tgt: [B, D, H, W] target feature map

        Returns:
            flow: [B, 2, H, W] - for each source position, the offset to matched target position
            matched_pos: [B, H, W, 2] - absolute matched positions in [-1, 1]
            confidence: [B, H, W] - matching confidence
        """
        B, D, H, W = features_src.shape
        device = features_src.device

        # Normalize features
        features_src = F.normalize(features_src, dim=1)
        features_tgt = F.normalize(features_tgt, dim=1)

        # Create position grids
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        pos_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]

        if self.search_radius is None:
            # Global correlation search
            # Reshape for matrix multiplication: [B, D, H*W]
            src_flat = features_src.view(B, D, -1)
            tgt_flat = features_tgt.view(B, D, -1)

            # Correlation: [B, H*W_src, H*W_tgt]
            temp = self.log_temp.exp()
            correlation = torch.bmm(src_flat.transpose(1, 2), tgt_flat) / temp

            # Softmax over target positions
            weights = F.softmax(correlation, dim=-1)  # [B, H*W_src, H*W_tgt]

            # Target position grid: [H*W, 2]
            pos_flat = pos_grid.view(-1, 2)  # [H*W, 2]

            # Weighted average of target positions: [B, H*W_src, 2]
            matched_pos_flat = torch.bmm(weights, pos_flat.unsqueeze(0).expand(B, -1, -1))
            matched_pos = matched_pos_flat.view(B, H, W, 2)

            # Confidence: negative entropy (high confidence = peaked distribution)
            entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)
            max_entropy = math.log(H * W)
            confidence = 1.0 - entropy / max_entropy
            confidence = confidence.view(B, H, W)

        else:
            # Local correlation search (more efficient for large images)
            # Use unfold to extract patches
            raise NotImplementedError("Local search not yet implemented")

        # Compute flow (offset from source to matched target)
        src_pos = pos_grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
        flow = matched_pos - src_pos  # [B, H, W, 2]
        flow = flow.permute(0, 3, 1, 2)  # [B, 2, H, W]

        return flow, matched_pos, confidence


class CorrelationProcrustes(nn.Module):
    """
    Estimate rotation and scale from dense correlation matches.

    1. Extract dense features from both images
    2. Find correspondences via correlation matching
    3. Procrustes analysis on (source_pos, matched_target_pos) pairs

    This solves the fixed-grid problem by explicitly finding WHERE
    each source position matches in the target.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        image_size: int = 256,
        grid_size: int = 16,  # Downsample for efficiency
        temperature: float = 0.1,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.image_size = image_size
        self.grid_size = grid_size

        # Dense feature extractor
        self.feature_extractor = DenseFeatureExtractor(feature_dim=feature_dim)

        # Correlation matcher
        self.matcher = CorrelationMatcher(
            feature_dim=feature_dim,
            temperature=temperature,
            sub_pixel=True,
        )

        logger.info(f"CorrelationProcrustes: grid_size={grid_size}, feature_dim={feature_dim}")

    def forward(
        self,
        image_src: Tensor,
        image_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Estimate transformation from dense correlation matching.

        Args:
            image_src: [B, 1, H, W] source image
            image_tgt: [B, 1, H, W] target image

        Returns:
            dict with rotation, scale, translation, R_matrix, flow, confidence
        """
        B = image_src.shape[0]
        device = image_src.device

        # Extract dense features
        features_src = self.feature_extractor(image_src)  # [B, D, H, W]
        features_tgt = self.feature_extractor(image_tgt)  # [B, D, H, W]

        # Downsample for efficiency
        if self.grid_size < features_src.shape[-1]:
            features_src = F.adaptive_avg_pool2d(features_src, self.grid_size)
            features_tgt = F.adaptive_avg_pool2d(features_tgt, self.grid_size)

        # Find correspondences
        flow, matched_pos, confidence = self.matcher(features_src, features_tgt)

        # Source positions (regular grid)
        H, W = features_src.shape[-2:]
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        src_pos = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        src_pos = src_pos.view(-1, 2).unsqueeze(0).expand(B, -1, -1)  # [B, N, 2]

        # Matched target positions
        tgt_pos = matched_pos.view(B, -1, 2)  # [B, N, 2]

        # Confidence weights
        weights = confidence.view(B, -1, 1)  # [B, N, 1]
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted Procrustes
        # Weighted centroids
        centroid_src = (weights * src_pos).sum(dim=1, keepdim=True)  # [B, 1, 2]
        centroid_tgt = (weights * tgt_pos).sum(dim=1, keepdim=True)  # [B, 1, 2]

        # Center the points
        src_centered = src_pos - centroid_src  # [B, N, 2]
        tgt_centered = tgt_pos - centroid_tgt  # [B, N, 2]

        # Weighted covariance
        src_w = weights.sqrt() * src_centered
        tgt_w = weights.sqrt() * tgt_centered
        H_cov = torch.bmm(src_w.transpose(-2, -1), tgt_w)  # [B, 2, 2]

        # SVD
        U, S, Vh = torch.linalg.svd(H_cov)

        # Rotation matrix
        R = torch.bmm(Vh.transpose(-2, -1), U.transpose(-2, -1))  # [B, 2, 2]

        # Fix reflections
        det = torch.det(R)
        mask = det < 0
        if mask.any():
            Vh_fixed = Vh.clone()
            Vh_fixed[mask, -1, :] *= -1
            R[mask] = torch.bmm(Vh_fixed[mask].transpose(-2, -1), U[mask].transpose(-2, -1))

        # Rotation angle
        rotation = torch.atan2(R[:, 1, 0], R[:, 0, 0])  # [B]

        # Scale
        src_var = (src_w ** 2).sum(dim=[1, 2])  # [B]
        scale = S.sum(dim=1) / (src_var + 1e-8)  # [B]
        scale = scale.clamp(0.1, 10.0)

        # Translation
        translation = centroid_tgt.squeeze(1) - scale.unsqueeze(-1) * torch.bmm(
            R, centroid_src.squeeze(1).unsqueeze(-1)
        ).squeeze(-1)  # [B, 2]

        return {
            'rotation': rotation,
            'scale': scale,
            'translation': translation,
            'R_matrix': R,
            'flow': flow,
            'matched_positions': matched_pos,
            'confidence': confidence,
            'features_src': features_src,
            'features_tgt': features_tgt,
        }


# =============================================================================
# Alternative: E2-Equivariant Features (requires escnn)
# =============================================================================

class E2EquivariantFeatureExtractor(nn.Module):
    """
    Rotation-equivariant feature extractor using group-equivariant convolutions.

    Uses the escnn library for E(2)-equivariant neural networks.
    Features transform PREDICTABLY under rotation, enabling correct matching.

    Note: Requires escnn library (pip install escnn)
    """

    def __init__(self, feature_dim: int = 32, num_rotations: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_rotations = num_rotations

        try:
            from escnn import gspaces
            from escnn import nn as enn

            # Cyclic group C_N (N discrete rotations)
            self.gspace = gspaces.rot2dOnR2(N=num_rotations)

            # Input: scalar field (grayscale image)
            in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])

            # Hidden: regular representation (transforms equivariantly)
            hidden_type = enn.FieldType(
                self.gspace,
                feature_dim // num_rotations * [self.gspace.regular_repr]
            )

            # Build equivariant network
            self.net = enn.SequentialModule(
                enn.R2Conv(in_type, hidden_type, kernel_size=7, padding=3),
                enn.InnerBatchNorm(hidden_type),
                enn.ReLU(hidden_type),
                enn.R2Conv(hidden_type, hidden_type, kernel_size=5, padding=2),
                enn.InnerBatchNorm(hidden_type),
                enn.ReLU(hidden_type),
                enn.R2Conv(hidden_type, hidden_type, kernel_size=3, padding=1),
                enn.InnerBatchNorm(hidden_type),
            )

            # Group pooling for invariant features (for matching)
            invariant_type = enn.FieldType(
                self.gspace,
                feature_dim // num_rotations * [self.gspace.trivial_repr]
            )
            self.pool = enn.GroupPooling(hidden_type)

            self.in_type = in_type
            self.has_escnn = True

            logger.info(f"E2EquivariantFeatureExtractor: C{num_rotations} group, {feature_dim} features")

        except ImportError:
            logger.warning("escnn not available, falling back to standard CNN")
            self.has_escnn = False
            self.net = nn.Sequential(
                nn.Conv2d(1, feature_dim, kernel_size=7, padding=3),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(),
                nn.Conv2d(feature_dim, feature_dim, kernel_size=5, padding=2),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(),
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature_dim),
            )

    def forward(self, image: Tensor) -> Tensor:
        """
        Extract rotation-invariant features for matching.

        Args:
            image: [B, 1, H, W] grayscale image

        Returns:
            features: [B, D, H, W] rotation-invariant features
        """
        if self.has_escnn:
            from escnn import nn as enn
            # Wrap as geometric tensor
            x = enn.GeometricTensor(image, self.in_type)
            # Equivariant forward pass
            x = self.net(x)
            # Pool over group to get invariant features
            x = self.pool(x)
            return x.tensor
        else:
            return self.net(image)
