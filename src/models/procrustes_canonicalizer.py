"""
Procrustes-Based Canonicalizer for Sim(2) Equivariance

This module replaces the ImageGradientOrientationEstimator approach with
a geometrically principled Procrustes-based method.

Key advantages over structure tensor (ImageGradientOrientationEstimator):
1. No 90-degree ambiguity - SVD gives unique rotation
2. No X/Y directional bias - operates on point geometry
3. Differentiable via SVD for end-to-end training
4. Works on correspondences, not image content directly

The approach:
1. Extract rotation-INVARIANT features using ESCNN (E2InvariantFeatureExtractor)
2. Match features via correlation to get soft correspondences
3. Use Procrustes SVD to estimate rotation, scale, translation from correspondences
4. This gives the transformation from source to target directly

References:
- Schonemann (1966): Orthogonal Procrustes problem
- E2-CNN (Weiler & Cesa, 2019): Steerable CNNs for equivariance
- Thermal Homography with Sim(2) Equivariance

Author: Yaqoob Ansari
Date: 2026-01-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils.logging_config import get_logger

from .e2_feature_extractor import E2InvariantFeatureExtractor, E2Procrustes
from .procrustes_estimator import DifferentiableProcrustesLayer

logger = get_logger(__name__)


class ProcrustesCanonicalizer(nn.Module):
    """
    Procrustes-based Sim(2) canonicalizer using ESCNN features.

    This replaces the Sim2Canonicalizer which used ImageGradientOrientationEstimator
    (structure tensor based, has 90-degree ambiguity).

    The Procrustes approach:
    1. Extracts rotation-INVARIANT features using E2-equivariant CNNs
    2. Computes soft correspondences via feature similarity
    3. Estimates rotation, scale, translation via Procrustes SVD

    This guarantees:
    - Same image content produces same features (rotation-invariant)
    - Unique rotation estimation (no 90-degree ambiguity)
    - Differentiable for end-to-end training
    """

    def __init__(
        self,
        feature_dim: int = 32,
        num_rotations: int = 8,  # C8 group (45 degree spacing)
        grid_size: int = 16,
        temperature: float = 0.1,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_rotations = num_rotations
        self.grid_size = grid_size

        # E2Procrustes does everything we need:
        # - ESCNN feature extraction (rotation-invariant)
        # - Correlation matching
        # - Procrustes SVD for rotation/scale/translation
        self.e2_procrustes = E2Procrustes(
            feature_dim=feature_dim,
            num_rotations=num_rotations,
            grid_size=grid_size,
            temperature=temperature,
        )

        logger.info(f"ProcrustesCanonicalizer: C{num_rotations}, grid={grid_size}, "
                   f"features={feature_dim}")
        logger.info("  -> Uses ESCNN for rotation-invariant features")
        logger.info("  -> Uses Procrustes SVD for transformation estimation")
        logger.info("  -> No 90-degree ambiguity (unlike structure tensor)")

    def forward(
        self,
        image_src: Tensor,
        image_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Estimate Sim(2) transformation from source to target.

        Args:
            image_src: [B, C, H, W] source image
            image_tgt: [B, C, H, W] target image

        Returns:
            dict with:
                rotation: [B] rotation angle in radians (src -> tgt)
                scale: [B] scale factor (src -> tgt)
                translation: [B, 2] translation vector (src -> tgt)
                R_matrix: [B, 2, 2] rotation matrix
                confidence: [B, H, W] matching confidence map
        """
        # Ensure single channel
        if image_src.shape[1] == 3:
            image_src = image_src.mean(dim=1, keepdim=True)
        if image_tgt.shape[1] == 3:
            image_tgt = image_tgt.mean(dim=1, keepdim=True)

        # Get transformation via E2Procrustes
        result = self.e2_procrustes(image_src, image_tgt)

        return {
            'rotation': result['rotation'],
            'scale': result['scale'],
            'translation': result['translation'],
            'R_matrix': result['R_matrix'],
            'confidence': result['confidence'],
            'flow': result['flow'],
        }

    def build_homography(
        self,
        rotation: Tensor,
        scale: Tensor,
        translation: Tensor,
    ) -> Tensor:
        """
        Build 3x3 homography matrix from Sim(2) parameters.

        H = [[s*cos(r), -s*sin(r), tx],
             [s*sin(r),  s*cos(r), ty],
             [0,         0,        1 ]]

        Args:
            rotation: [B] rotation angle in radians
            scale: [B] scale factor
            translation: [B, 2] translation (tx, ty)

        Returns:
            H: [B, 3, 3] homography matrix
        """
        B = rotation.shape[0]
        device = rotation.device
        dtype = rotation.dtype

        cos_r = torch.cos(rotation)
        sin_r = torch.sin(rotation)

        H = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        H[:, 0, 0] = scale * cos_r
        H[:, 0, 1] = -scale * sin_r
        H[:, 0, 2] = translation[:, 0]
        H[:, 1, 0] = scale * sin_r
        H[:, 1, 1] = scale * cos_r
        H[:, 1, 2] = translation[:, 1]
        H[:, 2, 2] = 1.0

        return H


class ProcrustesHomographyEstimator(nn.Module):
    """
    Full homography estimator using Procrustes for Sim(2) component.

    This can be used as a standalone homography estimator or as part of
    ThermalHomographyNet.

    Architecture:
    1. ProcrustesCanonicalizer estimates rotation, scale, translation
    2. Optional: residual refinement network for perspective distortion
    """

    def __init__(
        self,
        feature_dim: int = 32,
        num_rotations: int = 8,
        grid_size: int = 16,
        temperature: float = 0.1,
        use_perspective_refinement: bool = False,
    ):
        super().__init__()

        self.canonicalizer = ProcrustesCanonicalizer(
            feature_dim=feature_dim,
            num_rotations=num_rotations,
            grid_size=grid_size,
            temperature=temperature,
        )

        self.use_perspective_refinement = use_perspective_refinement

        if use_perspective_refinement:
            # Small network to predict perspective parameters h31, h32
            # These are typically small for near-planar scenes
            self.perspective_net = nn.Sequential(
                nn.Linear(feature_dim * 2, 64),
                nn.SiLU(),
                nn.Linear(64, 32),
                nn.SiLU(),
                nn.Linear(32, 2),  # h31, h32
            )
            # Initialize to near-zero (perspective is small)
            nn.init.zeros_(self.perspective_net[-1].weight)
            nn.init.zeros_(self.perspective_net[-1].bias)

    def forward(
        self,
        image_src: Tensor,
        image_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Estimate homography from image pair.

        Args:
            image_src: [B, C, H, W] source image
            image_tgt: [B, C, H, W] target image

        Returns:
            dict with:
                homography: [B, 3, 3] homography matrix
                rotation: [B] estimated rotation
                scale: [B] estimated scale
                translation: [B, 2] estimated translation
                confidence: [B, H, W] matching confidence
        """
        # Get Sim(2) parameters from Procrustes
        canon_result = self.canonicalizer(image_src, image_tgt)

        rotation = canon_result['rotation']
        scale = canon_result['scale']
        translation = canon_result['translation']

        # Build Sim(2) homography
        H = self.canonicalizer.build_homography(rotation, scale, translation)

        # Optional perspective refinement
        if self.use_perspective_refinement:
            # TODO: Add perspective refinement using features
            pass

        return {
            'homography': H,
            'rotation': rotation,
            'scale': scale,
            'translation': translation,
            'confidence': canon_result['confidence'],
        }
