"""
E2-Equivariant Feature Extractors

This module provides two types of E2-equivariant feature extractors:

1. E2InvariantFeatureExtractor: Produces rotation-INVARIANT features
   - Uses GroupPooling to pool over rotation group
   - Good for: Classification, when you DON'T want to detect rotation
   - BAD for: Geometric estimation (can't detect rotation!)

2. E2EquivariantEncoder: Produces rotation-EQUIVARIANT features
   - NO GroupPooling - features SHIFT under rotation
   - Good for: Rotation DETECTION via cyclic correlation
   - This is what Sim2EquivariantNet uses

Key insight (2026-01-30):
- INVARIANT features are useful for classification but HARMFUL for geometric estimation
- To DETECT rotation, we need EQUIVARIANT features that transform predictably

Using the escnn library for mathematically-guaranteed equivariance.

Author: ECCV 2026 Submission
Date: 2026-01-30
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# E2-equivariant neural networks
from escnn import gspaces
from escnn import nn as enn

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class E2InvariantFeatureExtractor(nn.Module):
    """
    Extract rotation-INVARIANT features using E2-equivariant networks.

    Architecture:
    1. Input: Scalar field (grayscale image)
    2. Hidden layers: Regular representation (transforms equivariantly)
    3. Output: Group pooling → rotation-INVARIANT features

    Why this works for matching:
    - The same image patch will produce the SAME feature regardless of rotation
    - This enables matching: patch at (x,y) in src matches patch at (x',y') in tgt
      if they have similar invariant features
    """

    def __init__(
        self,
        out_channels: int = 32,
        num_rotations: int = 8,  # C_8 = 8 discrete rotations (45° apart)
        hidden_channels: int = 16,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.num_rotations = num_rotations

        # Define the symmetry group: cyclic group C_N
        # C_8 means 8 rotations: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
        self.gspace = gspaces.rot2dOnR2(N=num_rotations)

        # Input type: scalar field (trivial representation)
        # A grayscale image is invariant under rotation of the "internal" space
        self.in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])

        # Hidden type: regular representation
        # Features in this space rotate WITH the image
        # Each "channel" is actually N channels (one per rotation)
        self.hidden_type = enn.FieldType(
            self.gspace,
            hidden_channels * [self.gspace.regular_repr]
        )

        # Output type for pooled (invariant) features
        self.out_type = enn.FieldType(
            self.gspace,
            out_channels * [self.gspace.trivial_repr]
        )

        # Build the equivariant network
        self.conv1 = enn.R2Conv(self.in_type, self.hidden_type, kernel_size=7, padding=3)
        self.bn1 = enn.InnerBatchNorm(self.hidden_type)
        self.relu1 = enn.ReLU(self.hidden_type)

        self.conv2 = enn.R2Conv(self.hidden_type, self.hidden_type, kernel_size=5, padding=2)
        self.bn2 = enn.InnerBatchNorm(self.hidden_type)
        self.relu2 = enn.ReLU(self.hidden_type)

        self.conv3 = enn.R2Conv(self.hidden_type, self.hidden_type, kernel_size=3, padding=1)
        self.bn3 = enn.InnerBatchNorm(self.hidden_type)
        self.relu3 = enn.ReLU(self.hidden_type)

        # Group pooling: max over rotation group → rotation-INVARIANT features
        # This is the key: same content produces same feature regardless of rotation
        self.pool = enn.GroupPooling(self.hidden_type)

        # Final projection to desired output channels
        pooled_type = self.pool.out_type
        self.proj = enn.R2Conv(pooled_type, self.out_type, kernel_size=1)

        logger.info(f"E2InvariantFeatureExtractor: C{num_rotations}, "
                   f"hidden={hidden_channels}×{num_rotations}, out={out_channels}")

    def forward(self, image: Tensor) -> Tensor:
        """
        Extract rotation-invariant features.

        Args:
            image: [B, 1, H, W] grayscale image

        Returns:
            features: [B, out_channels, H, W] rotation-invariant features
        """
        # Wrap input as geometric tensor
        x = enn.GeometricTensor(image, self.in_type)

        # Equivariant forward pass
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))

        # Pool over rotation group → invariant features
        x = self.pool(x)

        # Project to output channels
        x = self.proj(x)

        return x.tensor


class E2EquivariantEncoder(nn.Module):
    """
    E2-Equivariant Feature Encoder using Regular Representation.

    CRITICAL: This encoder does NOT use GroupPooling!

    Unlike E2InvariantFeatureExtractor which produces rotation-invariant
    features via GroupPooling, this encoder produces equivariant features
    that SHIFT under rotation.

    This is essential for rotation DETECTION:
    - For regular representation with C_N group, rotation by k × (360°/N)
      cyclically shifts feature channels by k positions
    - By finding this shift via cross-correlation, we can detect the rotation

    Output shape: [B, C, N_rot, H, W]
    - C: number of feature channels
    - N_rot: number of rotation channels (= N in C_N group)

    Args:
        num_rotations: Number of discrete rotations (N in C_N group)
        feature_channels: Number of output feature channels
        in_channels: Number of input channels (1 for grayscale)
    """

    def __init__(
        self,
        num_rotations: int = 16,
        feature_channels: int = 32,
        in_channels: int = 1,
    ):
        super().__init__()

        self.num_rotations = num_rotations
        self.feature_channels = feature_channels

        # Cyclic group C_N
        self.gspace = gspaces.rot2dOnR2(N=num_rotations)

        # Input: scalar field (grayscale image)
        self.in_type = enn.FieldType(
            self.gspace,
            in_channels * [self.gspace.trivial_repr]
        )

        # Hidden: regular representation - features SHIFT under rotation
        self.hidden_type = enn.FieldType(
            self.gspace,
            feature_channels * [self.gspace.regular_repr]
        )

        # Equivariant encoder - NO GroupPooling!
        self.encoder = enn.SequentialModule(
            enn.R2Conv(self.in_type, self.hidden_type, kernel_size=7, padding=3),
            enn.InnerBatchNorm(self.hidden_type),
            enn.ReLU(self.hidden_type),
            enn.R2Conv(self.hidden_type, self.hidden_type, kernel_size=5, padding=2),
            enn.InnerBatchNorm(self.hidden_type),
            enn.ReLU(self.hidden_type),
            enn.R2Conv(self.hidden_type, self.hidden_type, kernel_size=3, padding=1),
            enn.InnerBatchNorm(self.hidden_type),
            enn.ReLU(self.hidden_type),
        )

        logger.info(f"E2EquivariantEncoder: C{num_rotations}, {feature_channels} channels")
        logger.info(f"  Output: [B, {feature_channels}, {num_rotations}, H, W]")
        logger.info(f"  NO GroupPooling - features SHIFT under rotation")

    def forward(self, image: Tensor) -> Tensor:
        """
        Extract equivariant features (NOT invariant!).

        Args:
            image: [B, C, H, W] input image

        Returns:
            features: [B, feature_channels, N_rot, H, W] equivariant features
                      These features cyclically shift when the image is rotated!
        """
        # Handle multi-channel input
        if image.shape[1] == 3:
            image = image.mean(dim=1, keepdim=True)

        x = enn.GeometricTensor(image, self.in_type)
        x = self.encoder(x)

        # Reshape from [B, C*N_rot, H, W] to [B, C, N_rot, H, W]
        B = image.shape[0]
        H, W = x.tensor.shape[2:]
        features = x.tensor.view(B, self.feature_channels, self.num_rotations, H, W)

        return features


class E2CorrelationMatcher(nn.Module):
    """
    Match images using E2-invariant features and correlation.

    The E2-invariant features ensure that the same image content
    produces the same feature vector regardless of rotation.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        num_rotations: int = 8,
        temperature: float = 0.1,
    ):
        super().__init__()

        self.feature_extractor = E2InvariantFeatureExtractor(
            out_channels=feature_dim,
            num_rotations=num_rotations,
        )

        self.temperature = temperature
        self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))

    def forward(
        self,
        image_src: Tensor,
        image_tgt: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Find correspondences using E2-invariant features.

        Args:
            image_src: [B, 1, H, W] source image
            image_tgt: [B, 1, H, W] target image

        Returns:
            matched_positions: [B, H, W, 2] - matched target positions
            flow: [B, 2, H, W] - flow from source to target
            confidence: [B, H, W] - matching confidence
        """
        B, _, H, W = image_src.shape
        device = image_src.device

        # Extract invariant features
        features_src = self.feature_extractor(image_src)  # [B, D, H, W]
        features_tgt = self.feature_extractor(image_tgt)  # [B, D, H, W]

        D = features_src.shape[1]

        # NOTE: Removed F.normalize() - it destroys scale information
        # Instead, compute similarity with proper temperature scaling

        # Reshape for correlation: [B, H*W, D]
        src_flat = features_src.view(B, D, -1).permute(0, 2, 1)  # [B, H*W, D]
        tgt_flat = features_tgt.view(B, D, -1).permute(0, 2, 1)  # [B, H*W, D]

        # Correlation matrix with dimension-aware temperature scaling
        # This avoids destroying scale info while maintaining stable softmax
        temp = self.log_temp.exp()
        correlation = torch.bmm(src_flat, tgt_flat.transpose(1, 2))
        correlation = correlation / (temp * (D ** 0.5))  # Scale by sqrt(dim)

        # Softmax over target positions
        weights = F.softmax(correlation, dim=-1)  # [B, H*W_src, H*W_tgt]

        # Create target position grid
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        pos_grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # [H*W, 2]

        # Weighted average of target positions
        matched_pos_flat = torch.bmm(weights, pos_grid.unsqueeze(0).expand(B, -1, -1))
        matched_positions = matched_pos_flat.view(B, H, W, 2)

        # Source positions
        src_pos = pos_grid.view(H, W, 2).unsqueeze(0).expand(B, -1, -1, -1)

        # Flow
        flow = (matched_positions - src_pos).permute(0, 3, 1, 2)  # [B, 2, H, W]

        # Confidence: negative entropy
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)
        max_entropy = math.log(H * W)
        confidence = (1.0 - entropy / max_entropy).view(B, H, W)

        return matched_positions, flow, confidence


class E2Procrustes(nn.Module):
    """
    Full E2-equivariant Procrustes estimation.

    1. Extract E2-invariant features
    2. Match via correlation
    3. Procrustes on matched positions
    """

    def __init__(
        self,
        feature_dim: int = 32,
        num_rotations: int = 8,
        grid_size: int = 16,
        temperature: float = 0.1,
    ):
        super().__init__()

        self.grid_size = grid_size

        self.matcher = E2CorrelationMatcher(
            feature_dim=feature_dim,
            num_rotations=num_rotations,
            temperature=temperature,
        )

        logger.info(f"E2Procrustes: grid_size={grid_size}, C{num_rotations}")

    def forward(
        self,
        image_src: Tensor,
        image_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Estimate transformation using E2-equivariant features.
        """
        B = image_src.shape[0]
        device = image_src.device

        # Downsample for efficiency
        if self.grid_size < image_src.shape[-1]:
            image_src_ds = F.adaptive_avg_pool2d(image_src, self.grid_size)
            image_tgt_ds = F.adaptive_avg_pool2d(image_tgt, self.grid_size)
        else:
            image_src_ds = image_src
            image_tgt_ds = image_tgt

        # Get matches
        matched_positions, flow, confidence = self.matcher(image_src_ds, image_tgt_ds)

        H, W = matched_positions.shape[1:3]

        # Source position grid
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        src_pos = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        src_pos = src_pos.view(-1, 2).unsqueeze(0).expand(B, -1, -1)  # [B, N, 2]

        # Matched target positions
        tgt_pos = matched_positions.view(B, -1, 2)  # [B, N, 2]

        # Confidence weights
        weights = confidence.view(B, -1, 1)  # [B, N, 1]
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted Procrustes
        centroid_src = (weights * src_pos).sum(dim=1, keepdim=True)
        centroid_tgt = (weights * tgt_pos).sum(dim=1, keepdim=True)

        src_centered = src_pos - centroid_src
        tgt_centered = tgt_pos - centroid_tgt

        src_w = weights.sqrt() * src_centered
        tgt_w = weights.sqrt() * tgt_centered

        H_cov = torch.bmm(src_w.transpose(-2, -1), tgt_w)

        U, S, Vh = torch.linalg.svd(H_cov)

        R = torch.bmm(Vh.transpose(-2, -1), U.transpose(-2, -1))

        # Fix reflections
        det = torch.det(R)
        mask = det < 0
        if mask.any():
            Vh_fixed = Vh.clone()
            Vh_fixed[mask, -1, :] *= -1
            R[mask] = torch.bmm(Vh_fixed[mask].transpose(-2, -1), U[mask].transpose(-2, -1))

        rotation = torch.atan2(R[:, 1, 0], R[:, 0, 0])

        # CRITICAL FIX: Scale formula must use UNWEIGHTED centered points
        # Correct formula: scale = trace(Σ) / ||P_centered||²_F
        # Previous bug: used weighted points (src_w) which biased scale estimation
        P_norm_sq = (src_centered ** 2).sum(dim=[1, 2])  # Unweighted
        scale = S.sum(dim=1) / (P_norm_sq + 1e-8)
        scale = scale.clamp(0.1, 10.0)

        translation = centroid_tgt.squeeze(1) - scale.unsqueeze(-1) * torch.bmm(
            R, centroid_src.squeeze(1).unsqueeze(-1)
        ).squeeze(-1)

        return {
            'rotation': rotation,
            'scale': scale,
            'translation': translation,
            'R_matrix': R,
            'flow': flow,
            'matched_positions': matched_positions,
            'confidence': confidence,
        }
