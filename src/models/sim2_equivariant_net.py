"""
Sim(2) Equivariant Homography Estimation Network

This is the main model implementing the Cyclic Correlation + Procrustes
architecture for true Sim(2) equivariance.

Key Architecture:
1. E2EquivariantEncoder: Extract EQUIVARIANT features (NOT invariant!)
   - Uses regular representation
   - NO GroupPooling (preserves geometric information)
   - Output: [B, C, N_rot, H, W] - features SHIFT under rotation

2. CyclicRotationEstimator: Detect rotation via FFT cross-correlation
   - Exploits cyclic shift property of regular representation
   - Peak of correlation → rotation angle

3. De-rotate target image: Align images using detected rotation

4. AlignedDenseMatcher: Find spatial correspondences after alignment

5. ProcrustesScaleTranslation: Extract scale and translation

6. Build homography: H = T(t) @ S(s) @ R(θ)

Why this works (unlike the broken version):
- Previous: GroupPooling → INVARIANT features → Can't detect rotation
- New: NO GroupPooling → EQUIVARIANT features → Cyclic shift encodes rotation

Author: Yaqoob Ansari
Date: 2026-01-30
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from escnn import gspaces
from escnn import nn as enn

from src.utils.logging_config import get_logger
from .cyclic_rotation_estimator import CyclicRotationEstimator
from .dense_spatial_matcher import AlignedDenseMatcher
from .procrustes_st import ProcrustesScaleTranslation
from .differentiable_transforms import rotate_image, build_sim2_homography

logger = get_logger(__name__)


class E2EquivariantEncoder(nn.Module):
    """
    E2-Equivariant Feature Encoder using Regular Representation.

    CRITICAL: This encoder does NOT use GroupPooling!

    The regular representation has the property that rotation by
    k × (360°/N) cyclically shifts feature channels by k positions.
    This is what enables rotation DETECTION via cyclic correlation.

    GroupPooling would make features INVARIANT to rotation, which
    destroys the geometric information we need for detection.

    Output shape: [B, C, N_rot, H, W]
    - C: number of feature channels
    - N_rot: number of rotation channels (discrete rotations)
    - H, W: spatial dimensions

    Under rotation by k × (360°/N):
        output[:, :, r, :, :] → output[:, :, (r-k) mod N, :, :]
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

        logger.info(f"E2EquivariantEncoder: C{num_rotations}, {feature_channels} feature channels")
        logger.info(f"  Output shape: [B, {feature_channels}, {num_rotations}, H, W]")
        logger.info(f"  NO GroupPooling - features are EQUIVARIANT (shift under rotation)")

    def forward(self, image: Tensor) -> Tensor:
        """
        Extract equivariant features.

        Args:
            image: [B, C, H, W] input image (C=1 for grayscale)

        Returns:
            features: [B, feature_channels, N_rot, H, W] equivariant features
        """
        # Handle grayscale
        if image.shape[1] == 3:
            image = image.mean(dim=1, keepdim=True)

        x = enn.GeometricTensor(image, self.in_type)
        x = self.encoder(x)

        # Reshape from [B, C*N_rot, H, W] to [B, C, N_rot, H, W]
        B = image.shape[0]
        H, W = x.tensor.shape[2:]
        features = x.tensor.view(B, self.feature_channels, self.num_rotations, H, W)

        return features


class InvariantFeatureEncoder(nn.Module):
    """
    Standard CNN feature encoder for use after rotation alignment.

    After de-rotating the target image, we don't need rotation
    equivariance for spatial matching. A simple CNN suffices.
    """

    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 32,
    ):
        super().__init__()

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

    def forward(self, image: Tensor) -> Tensor:
        """Extract features from image."""
        if image.shape[1] == 3:
            image = image.mean(dim=1, keepdim=True)
        return self.encoder(image)


class Sim2EquivariantNet(nn.Module):
    """
    Sim(2) Equivariant Homography Estimation Network.

    This is the main model that correctly handles Sim(2) transformations
    using the Cyclic Correlation + Procrustes approach.

    Pipeline:
    1. Extract EQUIVARIANT features (not invariant!)
    2. Detect rotation via cyclic cross-correlation
    3. De-rotate target image
    4. Extract spatial features from aligned images
    5. Find correspondences via correlation matching
    6. Procrustes for scale and translation
    7. Build homography from (θ, s, t)

    Args:
        num_rotations: Number of discrete rotations for C_N group
        feature_channels: Number of feature channels for equivariant encoder
        spatial_feature_dim: Feature dimension for spatial matching
        rotation_temperature: Temperature for rotation soft-argmax
        matching_temperature: Temperature for correspondence softmax
        min_scale: Minimum allowed scale
        max_scale: Maximum allowed scale
    """

    def __init__(
        self,
        num_rotations: int = 16,
        feature_channels: int = 32,
        spatial_feature_dim: int = 32,
        rotation_temperature: float = 10.0,
        matching_temperature: float = 0.1,
        min_scale: float = 0.1,
        max_scale: float = 10.0,
    ):
        super().__init__()

        self.num_rotations = num_rotations

        # 1. Equivariant encoder for rotation detection
        self.equivariant_encoder = E2EquivariantEncoder(
            num_rotations=num_rotations,
            feature_channels=feature_channels,
        )

        # 2. Rotation estimator via cyclic correlation
        self.rotation_estimator = CyclicRotationEstimator(
            num_rotations=num_rotations,
            temperature=rotation_temperature,
        )

        # 3. Spatial matcher for correspondences (after alignment)
        self.spatial_matcher = AlignedDenseMatcher(
            in_channels=1,
            feature_dim=spatial_feature_dim,
            temperature=matching_temperature,
            downsample=4,
        )

        # 4. Procrustes for scale and translation
        self.procrustes = ProcrustesScaleTranslation(
            min_scale=min_scale,
            max_scale=max_scale,
        )

        logger.info("Sim2EquivariantNet initialized:")
        logger.info(f"  - Rotation: Cyclic correlation (C{num_rotations})")
        logger.info(f"  - Scale/Translation: Procrustes after de-rotation")
        logger.info(f"  - NO GroupPooling - true Sim(2) detection capability")

    def forward(
        self,
        img_src: Tensor,
        img_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Estimate Sim(2) transformation between images.

        Args:
            img_src: [B, C, H, W] source image
            img_tgt: [B, C, H, W] target image

        Returns:
            dict with:
                homography: [B, 3, 3] estimated homography matrix
                rotation: [B] rotation angle in radians
                rotation_deg: [B] rotation angle in degrees
                scale: [B] scale factor
                translation: [B, 2] translation vector
                confidence: dict with rotation and matching confidence
        """
        B = img_src.shape[0]
        device = img_src.device

        # Handle grayscale conversion
        if img_src.shape[1] == 3:
            img_src = img_src.mean(dim=1, keepdim=True)
        if img_tgt.shape[1] == 3:
            img_tgt = img_tgt.mean(dim=1, keepdim=True)

        # ========================================
        # Step 1: Extract equivariant features
        # ========================================
        feat_equiv_src = self.equivariant_encoder(img_src)  # [B, C, N_rot, H, W]
        feat_equiv_tgt = self.equivariant_encoder(img_tgt)  # [B, C, N_rot, H, W]

        # ========================================
        # Step 2: Detect rotation via cyclic correlation
        # ========================================
        rotation_result = self.rotation_estimator(feat_equiv_src, feat_equiv_tgt)
        rotation = rotation_result['rotation']  # [B] in radians
        rotation_deg = rotation_result['rotation_deg']  # [B] in degrees
        rotation_confidence = rotation_result['confidence']

        # ========================================
        # Step 3: De-rotate target image
        # ========================================
        # Rotate target by -θ to align with source
        img_tgt_aligned = rotate_image(img_tgt, -rotation)

        # ========================================
        # Step 4: Find spatial correspondences
        # ========================================
        match_result = self.spatial_matcher(img_src, img_tgt_aligned)
        src_positions = match_result['src_positions']  # [B, N, 2]
        matched_positions = match_result['matched_positions']  # [B, N, 2]
        match_confidence = match_result['confidence']  # [B, N]

        # ========================================
        # Step 5: Procrustes for scale and translation
        # ========================================
        procrustes_result = self.procrustes(
            src_positions,
            matched_positions,
            match_confidence
        )
        scale = procrustes_result['scale']  # [B]
        translation = procrustes_result['translation']  # [B, 2]

        # ========================================
        # Step 6: Build homography matrix
        # ========================================
        homography = build_sim2_homography(rotation, scale, translation)

        return {
            'homography': homography,
            'rotation': rotation,
            'rotation_deg': rotation_deg,
            'scale': scale,
            'translation': translation,
            'confidence': {
                'rotation': rotation_confidence,
                'matching': match_confidence.mean(dim=-1),
            },
            'correlation': rotation_result['correlation'],
            'src_positions': src_positions,
            'matched_positions': matched_positions,
        }


class Sim2EquivariantNetLite(nn.Module):
    """
    Lightweight version of Sim2EquivariantNet.

    Uses fewer rotations and smaller feature dimensions for faster
    inference, suitable for real-time applications.
    """

    def __init__(
        self,
        num_rotations: int = 8,
        feature_channels: int = 16,
        spatial_feature_dim: int = 16,
    ):
        super().__init__()

        self.model = Sim2EquivariantNet(
            num_rotations=num_rotations,
            feature_channels=feature_channels,
            spatial_feature_dim=spatial_feature_dim,
        )

        logger.info("Sim2EquivariantNetLite: Lightweight configuration")

    def forward(self, img_src: Tensor, img_tgt: Tensor) -> dict[str, Tensor]:
        return self.model(img_src, img_tgt)


def create_sim2_equivariant_net(
    config: str = 'default',
    **kwargs,
) -> Sim2EquivariantNet:
    """
    Factory function to create Sim2EquivariantNet with preset configurations.

    Args:
        config: Configuration preset ('default', 'lite', 'high_precision')
        **kwargs: Override any configuration parameters

    Returns:
        Sim2EquivariantNet instance
    """
    configs = {
        'default': {
            'num_rotations': 16,
            'feature_channels': 32,
            'spatial_feature_dim': 32,
        },
        'lite': {
            'num_rotations': 8,
            'feature_channels': 16,
            'spatial_feature_dim': 16,
        },
        'high_precision': {
            'num_rotations': 32,
            'feature_channels': 64,
            'spatial_feature_dim': 64,
        },
    }

    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Available: {list(configs.keys())}")

    params = configs[config].copy()
    params.update(kwargs)

    return Sim2EquivariantNet(**params)

