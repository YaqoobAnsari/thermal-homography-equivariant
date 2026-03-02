"""
Data Augmentation for Thermal Homography

Implements geometric and photometric augmentations that:
1. Are colormap-invariant (work on any thermal colormap)
2. Properly update ground truth homographies
3. Support the evaluation of rotation equivariance
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ThermalAugmentation:
    """
    Augmentation pipeline for thermal image pairs.

    Applies synchronized transforms to both images and updates homography.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        rotation_range: float = 45.0,
        translation_range: float = 0.1,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        apply_noise: bool = True,
        noise_std: float = 0.02,
    ):
        self.image_size = image_size
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.apply_noise = apply_noise
        self.noise_std = noise_std

    def __call__(
        self,
        image_src: np.ndarray,
        image_tgt: np.ndarray,
        homography: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Apply augmentation to image pair.

        Args:
            image_src: Source image [H, W] or [H, W, C]
            image_tgt: Target image [H, W] or [H, W, C]
            homography: Ground truth homography [3, 3]

        Returns:
            Dictionary with augmented images and updated homography
        """
        H, W = self.image_size

        # Random geometric augmentation for source
        aug_src = self._random_geometric_transform(H, W)
        src_warped = cv2.warpPerspective(image_src, aug_src, (W, H))

        # Random geometric augmentation for target
        aug_tgt = self._random_geometric_transform(H, W)
        tgt_warped = cv2.warpPerspective(image_tgt, aug_tgt, (W, H))

        # Update homography: H_new = aug_tgt @ H @ aug_src^{-1}
        homography_new = aug_tgt @ homography @ np.linalg.inv(aug_src)

        # Apply noise
        if self.apply_noise:
            src_warped = self._add_noise(src_warped)
            tgt_warped = self._add_noise(tgt_warped)

        return {
            "image_src": src_warped,
            "image_tgt": tgt_warped,
            "homography": homography_new,
            "aug_src": aug_src,
            "aug_tgt": aug_tgt,
        }

    def _random_geometric_transform(self, H: int, W: int) -> np.ndarray:
        """Generate random geometric transformation matrix."""
        # Random rotation
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        angle_rad = angle * np.pi / 180

        # Random scale
        scale = np.random.uniform(*self.scale_range)

        # Random translation
        tx = np.random.uniform(-self.translation_range, self.translation_range) * W
        ty = np.random.uniform(-self.translation_range, self.translation_range) * H

        # Build transformation matrix
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Center the rotation
        cx, cy = W / 2, H / 2

        # Rotation around center + scale + translation
        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
        R = np.array([
            [scale * cos_a, -scale * sin_a, 0],
            [scale * sin_a, scale * cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        T2 = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], dtype=np.float32)

        return T2 @ R @ T1

    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to image."""
        noise = np.random.randn(*image.shape) * self.noise_std * 255
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)


class SyntheticWarpAugmentation:
    """
    Generate training pairs by synthetically warping a single image.

    This provides infinite training data with perfect ground truth.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        # Rotation parameters
        rotation_range: Tuple[float, float] = (-180, 180),
        # Translation parameters (as fraction of image size)
        translation_range: Tuple[float, float] = (-0.2, 0.2),
        # Scale parameters
        scale_range: Tuple[float, float] = (0.8, 1.2),
        # Perspective distortion strength
        perspective_strength: float = 0.0001,
        # Whether to include perspective distortion
        use_perspective: bool = True,
    ):
        self.image_size = image_size
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.perspective_strength = perspective_strength
        self.use_perspective = use_perspective

    def __call__(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Generate image pair from single image.

        Args:
            image: Input image [H, W] or [H, W, C]

        Returns:
            Dictionary with source, target images and homography
        """
        H, W = self.image_size

        # Resize if needed
        if image.shape[:2] != (H, W):
            image = cv2.resize(image, (W, H))

        # Generate random homography
        homography = self._random_homography(H, W)

        # Warp image
        warped = cv2.warpPerspective(image, homography, (W, H))

        return {
            "image_src": image,
            "image_tgt": warped,
            "homography": homography,
        }

    def _random_homography(self, H: int, W: int) -> np.ndarray:
        """Generate random homography matrix."""
        # Random rotation
        angle = np.random.uniform(*self.rotation_range) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Random scale
        scale = np.random.uniform(*self.scale_range)

        # Random translation
        tx = np.random.uniform(*self.translation_range) * W
        ty = np.random.uniform(*self.translation_range) * H

        # Build similarity transform
        cx, cy = W / 2, H / 2
        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
        R = np.array([
            [scale * cos_a, -scale * sin_a, 0],
            [scale * sin_a, scale * cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        T2 = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], dtype=np.float32)

        homography = T2 @ R @ T1

        # Add perspective distortion if enabled
        if self.use_perspective:
            perspective = np.eye(3, dtype=np.float32)
            perspective[2, 0] = np.random.uniform(-1, 1) * self.perspective_strength
            perspective[2, 1] = np.random.uniform(-1, 1) * self.perspective_strength
            homography = homography @ perspective

        return homography


def get_train_transforms(
    image_size: Tuple[int, int] = (256, 256),
    rotation_range: float = 45.0,
) -> A.Compose:
    """
    Get training augmentation pipeline.

    Uses Albumentations for efficient augmentation.
    """
    return A.Compose([
        # Geometric transforms
        A.Rotate(limit=rotation_range, p=0.5, border_mode=cv2.BORDER_REPLICATE),
        A.RandomScale(scale_limit=0.1, p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=0,
            p=0.5,
            border_mode=cv2.BORDER_REPLICATE,
        ),

        # Photometric transforms (colormap-invariant)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),

        # Resize to target size
        A.Resize(height=image_size[0], width=image_size[1]),
    ])


def get_val_transforms(
    image_size: Tuple[int, int] = (256, 256),
) -> A.Compose:
    """Get validation/test transforms (just resize)."""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
    ])


class RotationEquivarianceTest:
    """
    Test transform for evaluating rotation equivariance.

    Rotates images by known angles to verify model generalizes.
    """

    def __init__(
        self,
        angles: List[float] = [0, 30, 45, 60, 90, 120, 150, 180],
        image_size: Tuple[int, int] = (256, 256),
    ):
        self.angles = angles
        self.image_size = image_size

    def __call__(
        self,
        image_src: np.ndarray,
        image_tgt: np.ndarray,
        homography: np.ndarray,
        angle_idx: int,
    ) -> Dict[str, Any]:
        """
        Rotate image pair by specified angle.

        Args:
            image_src: Source image
            image_tgt: Target image
            homography: Ground truth homography
            angle_idx: Index into angles list

        Returns:
            Rotated images and updated homography
        """
        angle = self.angles[angle_idx]
        H, W = self.image_size

        # Build rotation matrix
        angle_rad = angle * np.pi / 180
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        cx, cy = W / 2, H / 2

        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
        R = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32)

        rotation_matrix = T2 @ R @ T1

        # Rotate both images
        src_rotated = cv2.warpPerspective(image_src, rotation_matrix, (W, H))
        tgt_rotated = cv2.warpPerspective(image_tgt, rotation_matrix, (W, H))

        # Update homography: H_new = R @ H @ R^{-1}
        R_inv = np.linalg.inv(rotation_matrix)
        homography_new = rotation_matrix @ homography @ R_inv

        return {
            "image_src": src_rotated,
            "image_tgt": tgt_rotated,
            "homography": homography_new,
            "rotation_angle": angle,
        }
