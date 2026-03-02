"""
Cyclic Rotation Estimator using FFT-based Cross-Correlation

This module implements rotation detection via cyclic cross-correlation
of equivariant features. The key insight is that ESCNN regular representation
features cyclically shift under rotation, and we can recover the rotation
angle by finding this shift.

For C_N equivariant features:
- Rotation by k × (360°/N) shifts feature channels by k positions
- FFT-based cross-correlation efficiently finds the shift
- Peak of correlation → rotation angle

This is the DETECTION approach (not invariance). We use equivariant
(not invariant) features specifically so we can detect the transformation.

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


class CyclicRotationEstimator(nn.Module):
    """
    Estimate rotation via FFT-based cyclic cross-correlation.

    Given equivariant features from source and target images, computes
    the rotation between them by finding the cyclic shift that maximizes
    cross-correlation.

    Mathematical foundation:
    For regular representation features f with N rotation channels:
        f(R_k · img)[c, r] = f(img)[c, (r - k) mod N]

    The cyclic cross-correlation:
        corr[k] = Σ_{c,r} F_src[c, r] · F_tgt[c, (r + k) mod N]

    has its peak at k where rotation = k × (360°/N).

    FFT implementation (O(N log N)):
        corr = IFFT(FFT(F_src) ⊙ conj(FFT(F_tgt)))

    Args:
        num_rotations: Number of discrete rotations (N in C_N group)
        temperature: Temperature for soft-argmax (higher = sharper)
        learnable_temperature: Whether to learn the temperature
    """

    def __init__(
        self,
        num_rotations: int = 16,
        temperature: float = 10.0,
        learnable_temperature: bool = True,
    ):
        super().__init__()

        self.num_rotations = num_rotations
        self.angle_resolution = 2 * math.pi / num_rotations  # radians per index

        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer('log_temperature', torch.log(torch.tensor(temperature)))

        logger.info(f"CyclicRotationEstimator: C{num_rotations}, angle_res={math.degrees(self.angle_resolution):.2f}°")

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp()

    def fft_cyclic_correlation(
        self,
        feat_src: Tensor,
        feat_tgt: Tensor,
    ) -> Tensor:
        """
        Compute cyclic cross-correlation using FFT.

        Args:
            feat_src: [B, C, N_rot] source features (spatially pooled)
            feat_tgt: [B, C, N_rot] target features (spatially pooled)

        Returns:
            correlation: [B, N_rot] correlation at each rotation index
        """
        # FFT along rotation dimension
        F_src = torch.fft.fft(feat_src, dim=-1)  # [B, C, N_rot]
        F_tgt = torch.fft.fft(feat_tgt, dim=-1)  # [B, C, N_rot]

        # Cross-correlation in frequency domain: F_src ⊙ conj(F_tgt)
        cross_spectrum = F_src * F_tgt.conj()

        # Inverse FFT to get correlation
        correlation = torch.fft.ifft(cross_spectrum, dim=-1).real  # [B, C, N_rot]

        # Sum over channels
        correlation = correlation.sum(dim=1)  # [B, N_rot]

        return correlation

    def soft_argmax(self, correlation: Tensor) -> Tensor:
        """
        Differentiable soft-argmax to get sub-index precision.

        Args:
            correlation: [B, N_rot] correlation values

        Returns:
            rotation_idx: [B] fractional rotation index
        """
        N = correlation.shape[-1]
        device = correlation.device

        # Softmax weights
        weights = F.softmax(correlation * self.temperature, dim=-1)  # [B, N_rot]

        # Index values
        indices = torch.arange(N, device=device, dtype=correlation.dtype)  # [N_rot]

        # Weighted sum of indices
        rotation_idx = (weights * indices).sum(dim=-1)  # [B]

        return rotation_idx

    def forward(
        self,
        feat_src: Tensor,
        feat_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Estimate rotation between source and target features.

        Args:
            feat_src: [B, C, N_rot, H, W] equivariant source features
            feat_tgt: [B, C, N_rot, H, W] equivariant target features

        Returns:
            dict with:
                rotation: [B] rotation angle in radians
                rotation_deg: [B] rotation angle in degrees
                rotation_idx: [B] rotation index (fractional)
                correlation: [B, N_rot] correlation at each rotation index
                confidence: [B] peak sharpness (confidence measure)
        """
        # Global spatial pooling: [B, C, N_rot, H, W] → [B, C, N_rot]
        feat_src_global = feat_src.mean(dim=[3, 4])
        feat_tgt_global = feat_tgt.mean(dim=[3, 4])

        # Normalize features (improves numerical stability)
        feat_src_global = F.normalize(feat_src_global, dim=1)
        feat_tgt_global = F.normalize(feat_tgt_global, dim=1)

        # FFT-based cyclic cross-correlation
        correlation = self.fft_cyclic_correlation(feat_src_global, feat_tgt_global)

        # Soft-argmax for differentiable peak finding
        rotation_idx = self.soft_argmax(correlation)

        # Convert index to angle
        rotation = rotation_idx * self.angle_resolution  # radians
        rotation_deg = rotation * 180.0 / math.pi

        # Confidence: ratio of peak to mean (higher = more certain)
        peak_val = correlation.max(dim=-1).values
        mean_val = correlation.mean(dim=-1)
        confidence = (peak_val - mean_val) / (mean_val.abs() + 1e-8)

        return {
            'rotation': rotation,
            'rotation_deg': rotation_deg,
            'rotation_idx': rotation_idx,
            'correlation': correlation,
            'confidence': confidence,
        }


class CyclicRotationEstimatorWithEncoder(nn.Module):
    """
    Complete rotation estimator with built-in equivariant encoder.

    This combines the E2-equivariant feature extraction with cyclic
    cross-correlation for end-to-end rotation detection.

    Architecture:
    1. ESCNN encoder with regular representation (NO GroupPooling)
    2. Global spatial pooling
    3. FFT-based cyclic cross-correlation
    4. Soft-argmax for differentiable peak finding
    """

    def __init__(
        self,
        num_rotations: int = 16,
        hidden_channels: int = 32,
        temperature: float = 10.0,
    ):
        super().__init__()

        from escnn import gspaces
        from escnn import nn as enn

        self.num_rotations = num_rotations
        self.hidden_channels = hidden_channels

        # Cyclic group C_N
        self.gspace = gspaces.rot2dOnR2(N=num_rotations)

        # Input: scalar field (grayscale image)
        self.in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])

        # Hidden: regular representation - features SHIFT under rotation
        self.hidden_type = enn.FieldType(
            self.gspace,
            hidden_channels * [self.gspace.regular_repr]
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

        # Rotation estimator
        self.rotation_estimator = CyclicRotationEstimator(
            num_rotations=num_rotations,
            temperature=temperature,
        )

        logger.info(f"CyclicRotationEstimatorWithEncoder: C{num_rotations}, {hidden_channels} channels")

    def extract_equivariant_features(self, image: Tensor) -> Tensor:
        """
        Extract equivariant features (NOT invariant!).

        Args:
            image: [B, 1, H, W] grayscale image

        Returns:
            features: [B, C, N_rot, H, W] equivariant features
        """
        from escnn import nn as enn

        x = enn.GeometricTensor(image, self.in_type)
        x = self.encoder(x)

        # Reshape from [B, C*N_rot, H, W] to [B, C, N_rot, H, W]
        B = image.shape[0]
        H, W = x.tensor.shape[2:]
        features = x.tensor.view(B, self.hidden_channels, self.num_rotations, H, W)

        return features

    def forward(
        self,
        image_src: Tensor,
        image_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Estimate rotation between two images.

        Args:
            image_src: [B, 1, H, W] source image
            image_tgt: [B, 1, H, W] target image

        Returns:
            dict with rotation, rotation_deg, correlation, confidence, features
        """
        # Extract equivariant features
        feat_src = self.extract_equivariant_features(image_src)
        feat_tgt = self.extract_equivariant_features(image_tgt)

        # Estimate rotation via cyclic correlation
        result = self.rotation_estimator(feat_src, feat_tgt)

        # Add features to result
        result['features_src'] = feat_src
        result['features_tgt'] = feat_tgt

        return result
