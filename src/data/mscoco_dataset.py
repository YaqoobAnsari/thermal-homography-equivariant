"""
Warped MS-COCO Dataset

Implements the standard deep homography benchmark following DeTone et al. 2016.
Generates synthetic image pairs by:
1. Sampling random patches from COCO images
2. Applying random 4-point perturbations to create homographies
3. Warping patches to create training pairs

Extended mode supports controlled rotation/scale for equivariance testing.

References:
- DeTone et al., "Deep Image Homography Estimation", 2016
- https://arxiv.org/abs/1606.03798
"""

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class WarpedMSCOCODataset(Dataset):
    """
    MS-COCO with synthetic homography generation.

    Standard protocol (DeTone et al. 2016):
    1. Sample random patch of size (patch_size × patch_size) from image
    2. Define 4 corners of the patch
    3. Perturb each corner by random offset in [-rho, rho]
    4. Compute homography H from original to perturbed corners
    5. Warp patch using H^{-1} to create target image

    Extended mode for equivariance testing:
    - Explicit rotation control (instead of random 4-point)
    - Explicit scale control
    - Combined rotation + scale + translation
    """

    def __init__(
        self,
        coco_root: str,
        patch_size: int = 128,
        rho: int = 32,  # perturbation range in pixels
        mode: str = "standard",  # "standard", "rotation", "scale", "sim2"
        rotation_range: Optional[Tuple[float, float]] = None,  # degrees
        scale_range: Optional[Tuple[float, float]] = None,
        translation_range: Optional[Tuple[float, float]] = None,  # pixels
        n_samples: Optional[int] = None,  # limit number of samples
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
        grayscale: bool = True,
    ):
        """
        Initialize Warped MS-COCO dataset.

        Args:
            coco_root: Path to COCO images directory (train2017 or val2017)
            patch_size: Size of extracted patches
            rho: Max corner perturbation for standard mode
            mode: Generation mode:
                - "standard": Random 4-point perturbation (DeTone protocol)
                - "rotation": Controlled rotation only
                - "scale": Controlled scale only
                - "sim2": Controlled rotation + scale + translation
            rotation_range: (min, max) rotation in degrees for controlled modes
            scale_range: (min, max) scale factor for controlled modes
            translation_range: (min, max) translation in pixels
            n_samples: Number of samples to generate. None uses all images once.
            transform: Optional transform for images
            seed: Random seed for reproducibility
            grayscale: Convert images to grayscale
        """
        super().__init__()

        self.coco_root = Path(coco_root)
        self.patch_size = patch_size
        self.rho = rho
        self.mode = mode
        self.rotation_range = rotation_range or (-30, 30)
        self.scale_range = scale_range or (0.9, 1.1)
        self.translation_range = translation_range or (-20, 20)
        self.n_samples = n_samples
        self.transform = transform
        self.grayscale = grayscale

        # Validate root directory
        if not self.coco_root.exists():
            raise FileNotFoundError(f"COCO root not found: {self.coco_root}")

        # Find all images
        self.image_paths = self._find_images()
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.coco_root}")

        # Set up random generator
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

        # Determine actual sample count
        if self.n_samples is None:
            self._actual_samples = len(self.image_paths)
        else:
            self._actual_samples = min(self.n_samples, len(self.image_paths) * 10)

        logger.info(
            f"Initialized WarpedMSCOCO: {len(self.image_paths)} images, "
            f"{self._actual_samples} samples, mode={self.mode}"
        )

    def _find_images(self) -> List[Path]:
        """Find all image files in the COCO directory."""
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        images = []

        for ext in extensions:
            images.extend(self.coco_root.glob(ext))

        return sorted(images)

    def __len__(self) -> int:
        return self._actual_samples

    def _load_image(self, path: Path) -> np.ndarray:
        """Load image and convert to grayscale if needed."""
        if self.grayscale:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(str(path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is None:
            raise FileNotFoundError(f"Could not load image: {path}")

        return image

    def _sample_patch_location(
        self, image_h: int, image_w: int
    ) -> Tuple[int, int]:
        """Sample random patch location ensuring enough margin for warping."""
        # Need extra margin for warping
        margin = self.rho + 10

        # Ensure image is large enough
        min_size = self.patch_size + 2 * margin
        if image_h < min_size or image_w < min_size:
            return None, None

        # Sample top-left corner
        y = self.rng.randint(margin, image_h - self.patch_size - margin)
        x = self.rng.randint(margin, image_w - self.patch_size - margin)

        return y, x

    def _generate_standard_homography(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate homography via 4-point perturbation (DeTone protocol).

        Returns:
            (H, corners_offset): Homography matrix and 4-point offset
        """
        # Original corners (patch coordinates)
        corners = np.array([
            [0, 0],
            [self.patch_size, 0],
            [self.patch_size, self.patch_size],
            [0, self.patch_size],
        ], dtype=np.float32)

        # Random perturbation for each corner
        offsets = self.rng.uniform(-self.rho, self.rho, (4, 2)).astype(np.float32)
        perturbed_corners = corners + offsets

        # Compute homography from original to perturbed
        H, _ = cv2.findHomography(corners, perturbed_corners)
        H = H.astype(np.float32)

        return H, offsets

    def _generate_rotation_homography(self) -> np.ndarray:
        """Generate homography with controlled rotation."""
        angle_deg = self.rng.uniform(self.rotation_range[0], self.rotation_range[1])
        angle_rad = np.radians(angle_deg)

        # Center of patch
        cx, cy = self.patch_size / 2, self.patch_size / 2

        # Rotation matrix around center
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        H = np.array([
            [cos_a, -sin_a, cx - cos_a * cx + sin_a * cy],
            [sin_a, cos_a, cy - sin_a * cx - cos_a * cy],
            [0, 0, 1],
        ], dtype=np.float32)

        return H

    def _generate_scale_homography(self) -> np.ndarray:
        """Generate homography with controlled scale."""
        scale = self.rng.uniform(self.scale_range[0], self.scale_range[1])

        # Center of patch
        cx, cy = self.patch_size / 2, self.patch_size / 2

        # Scale around center
        H = np.array([
            [scale, 0, cx * (1 - scale)],
            [0, scale, cy * (1 - scale)],
            [0, 0, 1],
        ], dtype=np.float32)

        return H

    def _generate_sim2_homography(self) -> np.ndarray:
        """Generate homography with controlled rotation + scale + translation."""
        angle_deg = self.rng.uniform(self.rotation_range[0], self.rotation_range[1])
        angle_rad = np.radians(angle_deg)
        scale = self.rng.uniform(self.scale_range[0], self.scale_range[1])
        tx = self.rng.uniform(self.translation_range[0], self.translation_range[1])
        ty = self.rng.uniform(self.translation_range[0], self.translation_range[1])

        # Center of patch
        cx, cy = self.patch_size / 2, self.patch_size / 2

        # Sim(2) transformation: rotate + scale around center, then translate
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        H = np.array([
            [scale * cos_a, -scale * sin_a, cx - scale * (cos_a * cx - sin_a * cy) + tx],
            [scale * sin_a, scale * cos_a, cy - scale * (sin_a * cx + cos_a * cy) + ty],
            [0, 0, 1],
        ], dtype=np.float32)

        return H

    def _generate_homography(self) -> np.ndarray:
        """Generate homography based on mode."""
        if self.mode == "standard":
            H, _ = self._generate_standard_homography()
            return H
        elif self.mode == "rotation":
            return self._generate_rotation_homography()
        elif self.mode == "scale":
            return self._generate_scale_homography()
        elif self.mode == "sim2":
            return self._generate_sim2_homography()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _homography_to_vec(self, H: np.ndarray) -> np.ndarray:
        """Convert 3x3 homography to 8D vector."""
        H = H / (H[2, 2] + 1e-8)
        return np.array([
            H[0, 0], H[0, 1], H[0, 2],
            H[1, 0], H[1, 1], H[1, 2],
            H[2, 0], H[2, 1],
        ], dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Generate a warped image pair.

        Returns dict with:
            - image_src: [1, H, W] source patch tensor
            - image_tgt: [1, H, W] warped target patch tensor
            - homography: [3, 3] homography matrix
            - homography_vec: [8] homography as 8D vector
        """
        # Select image (cycle through if n_samples > n_images)
        image_idx = idx % len(self.image_paths)
        image_path = self.image_paths[image_idx]

        # Try to load and sample from this image, retry if needed
        max_retries = 5
        for _ in range(max_retries):
            try:
                image = self._load_image(image_path)
                h, w = image.shape[:2]

                # Sample patch location
                y, x = self._sample_patch_location(h, w)
                if y is None:
                    # Image too small, try another
                    image_idx = self.rng.randint(0, len(self.image_paths))
                    image_path = self.image_paths[image_idx]
                    continue

                # Extract source patch
                patch_src = image[y:y+self.patch_size, x:x+self.patch_size].copy()

                # Generate homography
                H = self._generate_homography()

                # For warping, we need to use a larger region
                margin = self.rho + 10
                y_ext = max(0, y - margin)
                x_ext = max(0, x - margin)
                y_end = min(h, y + self.patch_size + margin)
                x_end = min(w, x + self.patch_size + margin)

                extended_region = image[y_ext:y_end, x_ext:x_end].copy()

                # Adjust homography for extended region coordinates
                # Offset to move patch corner to origin of extended region
                offset_x = x - x_ext
                offset_y = y - y_ext

                T_offset = np.array([
                    [1, 0, -offset_x],
                    [0, 1, -offset_y],
                    [0, 0, 1],
                ], dtype=np.float32)

                T_offset_inv = np.array([
                    [1, 0, offset_x],
                    [0, 1, offset_y],
                    [0, 0, 1],
                ], dtype=np.float32)

                # H_ext works on extended region: T_offset @ H @ T_offset_inv
                H_ext = T_offset_inv @ H @ T_offset

                # Warp extended region
                warped_region = cv2.warpPerspective(
                    extended_region, H_ext,
                    (extended_region.shape[1], extended_region.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )

                # Extract target patch from warped region
                patch_tgt = warped_region[offset_y:offset_y+self.patch_size,
                                          offset_x:offset_x+self.patch_size].copy()

                break

            except Exception as e:
                logger.warning(f"Error processing image {image_path}: {e}")
                image_idx = self.rng.randint(0, len(self.image_paths))
                image_path = self.image_paths[image_idx]
        else:
            # All retries failed, return zero patches
            logger.error(f"Failed to generate sample after {max_retries} retries")
            patch_src = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
            patch_tgt = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
            H = np.eye(3, dtype=np.float32)

        # Apply transforms if specified
        if self.transform is not None:
            transformed = self.transform(image=patch_src, image2=patch_tgt)
            patch_src = transformed["image"]
            patch_tgt = transformed["image2"]

        # Convert to tensors and normalize
        src_tensor = torch.from_numpy(patch_src.astype(np.float32)) / 255.0
        tgt_tensor = torch.from_numpy(patch_tgt.astype(np.float32)) / 255.0

        # Add channel dimension
        if src_tensor.dim() == 2:
            src_tensor = src_tensor.unsqueeze(0)
            tgt_tensor = tgt_tensor.unsqueeze(0)

        return {
            "image_src": src_tensor,
            "image_tgt": tgt_tensor,
            "homography": torch.from_numpy(H).float(),
            "homography_vec": torch.from_numpy(self._homography_to_vec(H)).float(),
        }

    def set_rotation_range(self, min_deg: float, max_deg: float) -> None:
        """Update rotation range (for controlled testing)."""
        self.rotation_range = (min_deg, max_deg)

    def set_scale_range(self, min_scale: float, max_scale: float) -> None:
        """Update scale range (for controlled testing)."""
        self.scale_range = (min_scale, max_scale)

    def set_seed(self, seed: int) -> None:
        """Reset random seed for reproducibility."""
        self.rng = np.random.RandomState(seed)


class ControlledTransformDataset(WarpedMSCOCODataset):
    """
    Dataset for testing at specific transformation values.

    Use this for generating equivariance test plots where you want
    to evaluate at exact rotation angles or scale factors.
    """

    def __init__(
        self,
        coco_root: str,
        rotation_angles: Optional[List[float]] = None,  # specific angles in degrees
        scale_factors: Optional[List[float]] = None,  # specific scale factors
        n_samples_per_config: int = 100,
        **kwargs
    ):
        """
        Initialize controlled transform dataset.

        Args:
            coco_root: Path to COCO images
            rotation_angles: List of specific rotation angles to test
            scale_factors: List of specific scale factors to test
            n_samples_per_config: Samples per (rotation, scale) configuration
        """
        self.rotation_angles = rotation_angles or [0]
        self.scale_factors = scale_factors or [1.0]
        self.n_samples_per_config = n_samples_per_config

        # Calculate total samples
        n_configs = len(self.rotation_angles) * len(self.scale_factors)
        total_samples = n_configs * n_samples_per_config

        # Initialize parent with sim2 mode
        super().__init__(
            coco_root=coco_root,
            mode="sim2",
            n_samples=total_samples,
            **kwargs
        )

        # Build configuration index
        self._build_config_index()

    def _build_config_index(self):
        """Build index mapping sample idx to (rotation, scale) config."""
        self.config_index = []

        for rot in self.rotation_angles:
            for scale in self.scale_factors:
                for _ in range(self.n_samples_per_config):
                    self.config_index.append((rot, scale))

    def __len__(self) -> int:
        return len(self.config_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample with specific rotation and scale."""
        rotation, scale = self.config_index[idx]

        # Temporarily set ranges to exact values
        orig_rot = self.rotation_range
        orig_scale = self.scale_range

        self.rotation_range = (rotation, rotation)
        self.scale_range = (scale, scale)

        # Generate sample
        sample = super().__getitem__(idx)

        # Add config info
        sample["rotation_config"] = rotation
        sample["scale_config"] = scale

        # Restore ranges
        self.rotation_range = orig_rot
        self.scale_range = orig_scale

        return sample
