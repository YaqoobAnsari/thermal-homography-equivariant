"""
Dense Spatial Matcher for Correspondence Finding

This module implements dense spatial matching between rotation-aligned
images. After rotation has been detected and removed, we need to find
spatial correspondences for scale and translation estimation.

Key insight: After de-rotation, the remaining transformation is just
scale + translation (a similarity without rotation). Standard correlation
matching can find these correspondences.

Architecture:
1. Extract invariant features from both images
2. Compute dense correlation volume
3. Soft-argmax to get sub-pixel correspondences
4. Return matched positions and confidence

Author: Yaqoob Ansari
Date: 2026-01-30
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DenseSpatialMatcher(nn.Module):
    """
    Dense spatial correspondence matching using correlation.

    Given rotation-aligned source and target images, finds spatial
    correspondences via correlation matching.

    The output is soft correspondences: for each source position,
    we get a weighted average of target positions based on feature
    similarity.

    Args:
        feature_dim: Feature dimension for matching
        temperature: Temperature for soft-argmax (lower = sharper)
        learnable_temperature: Whether to learn the temperature
        grid_size: Number of grid positions per side for output
    """

    def __init__(
        self,
        feature_dim: int = 32,
        temperature: float = 0.1,
        learnable_temperature: bool = True,
        grid_size: int = 16,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.grid_size = grid_size

        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer('log_temperature', torch.log(torch.tensor(temperature)))

        # Pre-compute grid positions
        y = torch.linspace(-1, 1, grid_size)
        x = torch.linspace(-1, 1, grid_size)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        positions = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        self.register_buffer('grid_positions', positions.view(-1, 2))  # [N, 2]

        logger.info(f"DenseSpatialMatcher: grid_size={grid_size}, feature_dim={feature_dim}")

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp()

    def forward(
        self,
        features_src: Tensor,
        features_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Find spatial correspondences via correlation matching.

        Args:
            features_src: [B, D, H, W] source features
            features_tgt: [B, D, H, W] target features

        Returns:
            dict with:
                src_positions: [B, N, 2] source grid positions
                matched_positions: [B, N, 2] matched target positions
                confidence: [B, N] matching confidence
                correlation: [B, N_src, N_tgt] correlation matrix
        """
        B, D, H_src, W_src = features_src.shape
        _, _, H_tgt, W_tgt = features_tgt.shape
        device = features_src.device

        # Reshape features for correlation: [B, D, H*W] -> [B, H*W, D]
        src_flat = features_src.view(B, D, -1).permute(0, 2, 1)  # [B, N_src, D]
        tgt_flat = features_tgt.view(B, D, -1).permute(0, 2, 1)  # [B, N_tgt, D]

        N_src = src_flat.shape[1]
        N_tgt = tgt_flat.shape[1]

        # Normalize features
        src_norm = F.normalize(src_flat, dim=-1)
        tgt_norm = F.normalize(tgt_flat, dim=-1)

        # Correlation matrix
        correlation = torch.bmm(src_norm, tgt_norm.transpose(1, 2))  # [B, N_src, N_tgt]
        correlation = correlation / self.temperature

        # Soft correspondences (softmax over target positions)
        weights = F.softmax(correlation, dim=-1)  # [B, N_src, N_tgt]

        # Target position grid
        y_tgt = torch.linspace(-1, 1, H_tgt, device=device)
        x_tgt = torch.linspace(-1, 1, W_tgt, device=device)
        grid_y_tgt, grid_x_tgt = torch.meshgrid(y_tgt, x_tgt, indexing='ij')
        tgt_positions = torch.stack([grid_x_tgt, grid_y_tgt], dim=-1).view(-1, 2)  # [N_tgt, 2]

        # Weighted average of target positions
        matched_positions = torch.bmm(weights, tgt_positions.unsqueeze(0).expand(B, -1, -1))  # [B, N_src, 2]

        # Source position grid
        y_src = torch.linspace(-1, 1, H_src, device=device)
        x_src = torch.linspace(-1, 1, W_src, device=device)
        grid_y_src, grid_x_src = torch.meshgrid(y_src, x_src, indexing='ij')
        src_positions = torch.stack([grid_x_src, grid_y_src], dim=-1).view(-1, 2)  # [N_src, 2]
        src_positions = src_positions.unsqueeze(0).expand(B, -1, -1)  # [B, N_src, 2]

        # Confidence: negative entropy (higher = more certain)
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)  # [B, N_src]
        max_entropy = math.log(N_tgt)
        confidence = (1.0 - entropy / max_entropy)  # [B, N_src]

        return {
            'src_positions': src_positions,
            'matched_positions': matched_positions,
            'confidence': confidence,
            'correlation': correlation,
        }


class AlignedDenseMatcher(nn.Module):
    """
    Dense matcher for rotation-aligned images.

    This module is designed to work after rotation detection:
    1. Detect rotation via cyclic correlation
    2. De-rotate target image
    3. Use this matcher to find scale/translation correspondences

    Includes a lightweight CNN feature extractor that can be shared
    or separate from the rotation detection encoder.

    Args:
        in_channels: Number of input channels (1 for grayscale)
        feature_dim: Feature dimension for matching
        temperature: Temperature for soft-argmax
        downsample: Downsample factor for efficiency
    """

    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 32,
        temperature: float = 0.1,
        downsample: int = 4,
    ):
        super().__init__()

        self.downsample = downsample

        # Lightweight feature extractor
        # No rotation equivariance needed since images are aligned
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
        )

        # Spatial matcher
        self.matcher = DenseSpatialMatcher(
            feature_dim=feature_dim,
            temperature=temperature,
        )

        logger.info(f"AlignedDenseMatcher: feature_dim={feature_dim}, downsample={downsample}")

    def forward(
        self,
        image_src: Tensor,
        image_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Find spatial correspondences between aligned images.

        Args:
            image_src: [B, C, H, W] source image
            image_tgt: [B, C, H, W] target image (should be de-rotated)

        Returns:
            dict with src_positions, matched_positions, confidence
        """
        # Downsample for efficiency
        if self.downsample > 1:
            image_src_ds = F.avg_pool2d(image_src, self.downsample)
            image_tgt_ds = F.avg_pool2d(image_tgt, self.downsample)
        else:
            image_src_ds = image_src
            image_tgt_ds = image_tgt

        # Extract features
        features_src = self.encoder(image_src_ds)
        features_tgt = self.encoder(image_tgt_ds)

        # Match
        result = self.matcher(features_src, features_tgt)

        return result


class MultiScaleDenseMatcher(nn.Module):
    """
    Multi-scale dense matcher for robustness to scale variations.

    Extracts features at multiple scales and combines them for
    more robust correspondence finding, especially when there's
    significant scale difference between images.

    Args:
        in_channels: Number of input channels
        feature_dim: Feature dimension for matching
        scales: List of scale factors to use
        temperature: Temperature for soft-argmax
    """

    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 32,
        scales: list[float] = [1.0, 0.5, 0.25],
        temperature: float = 0.1,
    ):
        super().__init__()

        self.scales = scales

        # Shared feature encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, feature_dim, 3, padding=1),
        )

        # Scale-wise matchers
        self.matchers = nn.ModuleList([
            DenseSpatialMatcher(feature_dim=feature_dim, temperature=temperature)
            for _ in scales
        ])

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(len(scales) * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        logger.info(f"MultiScaleDenseMatcher: scales={scales}")

    def forward(
        self,
        image_src: Tensor,
        image_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Multi-scale correspondence finding.
        """
        B = image_src.shape[0]
        device = image_src.device

        all_matched = []
        all_confidence = []

        for i, scale in enumerate(self.scales):
            # Resize images
            if scale != 1.0:
                size = (int(image_src.shape[2] * scale), int(image_src.shape[3] * scale))
                src_scaled = F.interpolate(image_src, size=size, mode='bilinear', align_corners=True)
                tgt_scaled = F.interpolate(image_tgt, size=size, mode='bilinear', align_corners=True)
            else:
                src_scaled = image_src
                tgt_scaled = image_tgt

            # Extract features
            feat_src = self.encoder(src_scaled)
            feat_tgt = self.encoder(tgt_scaled)

            # Match
            result = self.matchers[i](feat_src, feat_tgt)
            all_matched.append(result['matched_positions'])
            all_confidence.append(result['confidence'])

        # Combine predictions (weighted average by confidence)
        all_matched = torch.stack(all_matched, dim=-1)  # [B, N, 2, S]
        all_confidence = torch.stack(all_confidence, dim=-1)  # [B, N, S]

        # Normalize confidence weights
        weights = F.softmax(all_confidence, dim=-1)  # [B, N, S]

        # Weighted average of matched positions
        matched_positions = (all_matched * weights.unsqueeze(2)).sum(dim=-1)  # [B, N, 2]

        # Combined confidence
        confidence = all_confidence.max(dim=-1).values  # [B, N]

        # Source positions from first scale
        H, W = int(image_src.shape[2] * self.scales[0]), int(image_src.shape[3] * self.scales[0])
        y = torch.linspace(-1, 1, H // 4, device=device)  # Assuming encoder downsamples by 4
        x = torch.linspace(-1, 1, W // 4, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        src_positions = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
        src_positions = src_positions.unsqueeze(0).expand(B, -1, -1)

        return {
            'src_positions': src_positions,
            'matched_positions': matched_positions,
            'confidence': confidence,
        }
