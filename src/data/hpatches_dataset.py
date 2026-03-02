"""
HPatches Dataset Loader

HPatches is the standard benchmark for evaluating local feature descriptors
and homography estimation methods. It contains 116 sequences with 5 image
pairs each (580 pairs total), split into:
- 59 viewpoint sequences (v_*): Large viewpoint changes
- 57 illumination sequences (i_*): Illumination changes only

Reference:
- Balntas et al., "HPatches: A Benchmark and Evaluation of Handcrafted and
  Learned Local Descriptors", CVPR 2017
- https://github.com/hpatches/hpatches-dataset
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.utils.geometry import homography_matrix_to_vec_np
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class HPatchesDataset(Dataset):
    """
    HPatches benchmark dataset loader.

    Directory structure expected:
    hpatches-sequences-release/
    ├── i_ajuntament/          # illumination sequence
    │   ├── 1.ppm              # reference image
    │   ├── 2.ppm ... 6.ppm    # target images
    │   ├── H_1_2 ... H_1_6    # homographies (3x3, space-separated)
    ├── v_adam/                # viewpoint sequence
    │   └── ...
    └── ...

    Each sequence has 6 images: 1 reference + 5 targets.
    Homographies transform from image 1 (reference) to images 2-6.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "all",  # "all", "viewpoint", "illumination"
        resize_to: Optional[int] = 480,  # shorter edge, None for original size
        transform: Optional[Callable] = None,
        return_metadata: bool = False,
    ):
        """
        Initialize HPatches dataset.

        Args:
            data_root: Path to hpatches-sequences-release directory
            split: Which sequences to include:
                - "all": All 116 sequences (580 pairs)
                - "viewpoint": Only v_* sequences (59 sequences, 295 pairs)
                - "illumination": Only i_* sequences (57 sequences, 285 pairs)
            resize_to: Resize images so shorter edge equals this value.
                       Homography is scaled accordingly. None keeps original size.
            transform: Optional transform to apply to images.
            return_metadata: Whether to return sequence info in samples.
        """
        super().__init__()

        self.data_root = Path(data_root)
        self.split = split
        self.resize_to = resize_to
        self.transform = transform
        self.return_metadata = return_metadata

        # Validate data root
        if not self.data_root.exists():
            raise FileNotFoundError(f"HPatches data root not found: {self.data_root}")

        # Load sequence list and build sample index
        self.samples = self._load_samples()

        logger.info(
            f"Loaded HPatches dataset: {len(self.samples)} pairs "
            f"from {len(self._get_sequences())} sequences ({self.split})"
        )

    def _get_sequences(self) -> List[Path]:
        """Get list of sequence directories based on split."""
        all_sequences = sorted([
            d for d in self.data_root.iterdir()
            if d.is_dir() and (d.name.startswith("v_") or d.name.startswith("i_"))
        ])

        if self.split == "all":
            return all_sequences
        elif self.split == "viewpoint":
            return [s for s in all_sequences if s.name.startswith("v_")]
        elif self.split == "illumination":
            return [s for s in all_sequences if s.name.startswith("i_")]
        else:
            raise ValueError(f"Unknown split: {self.split}. Use 'all', 'viewpoint', or 'illumination'")

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Build sample list: (sequence, ref_image, target_image, homography)."""
        samples = []
        sequences = self._get_sequences()

        for seq_dir in sequences:
            seq_name = seq_dir.name
            seq_type = "viewpoint" if seq_name.startswith("v_") else "illumination"

            # Reference image is always 1.ppm
            ref_path = seq_dir / "1.ppm"
            if not ref_path.exists():
                logger.warning(f"Missing reference image: {ref_path}")
                continue

            # Each sequence has 5 target images (2-6)
            for target_idx in range(2, 7):
                target_path = seq_dir / f"{target_idx}.ppm"
                h_file = seq_dir / f"H_1_{target_idx}"

                if not target_path.exists():
                    logger.warning(f"Missing target image: {target_path}")
                    continue

                if not h_file.exists():
                    logger.warning(f"Missing homography file: {h_file}")
                    continue

                samples.append({
                    "sequence": seq_name,
                    "seq_type": seq_type,
                    "ref_path": str(ref_path),
                    "target_path": str(target_path),
                    "h_file": str(h_file),
                    "target_idx": target_idx,
                })

        return samples

    def _load_homography(self, h_file: str) -> np.ndarray:
        """
        Load homography from HPatches format file.

        Format: 3x3 matrix with space-separated values, one row per line.
        """
        H = np.loadtxt(h_file, dtype=np.float32)
        if H.shape != (3, 3):
            raise ValueError(f"Invalid homography shape {H.shape} in {h_file}")
        return H

    def _load_and_resize_image(
        self, path: str
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Load image and resize if needed.

        Returns:
            Tuple of (image, original_size, new_size) where sizes are (H, W).
        """
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {path}")

        orig_h, orig_w = image.shape[:2]
        original_size = (orig_h, orig_w)

        if self.resize_to is not None:
            # Resize to fixed square dimensions for batching compatibility
            # This ensures all images have identical dimensions regardless of aspect ratio
            new_h = self.resize_to
            new_w = self.resize_to
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            new_size = (new_h, new_w)
        else:
            new_size = original_size

        return image, original_size, new_size

    def _scale_homography(
        self,
        H: np.ndarray,
        src_orig_size: Tuple[int, int],
        src_new_size: Tuple[int, int],
        tgt_orig_size: Tuple[int, int],
        tgt_new_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Scale homography when images are resized.

        The homography H maps points from image 1 to image 2:
            p2 = H @ p1

        When both images are resized, we need:
            H_scaled = S2 @ H @ S1_inv

        Where:
            S1 = scaling from original image 1 to resized image 1
            S2 = scaling from original image 2 to resized image 2

        Args:
            H: Original 3x3 homography matrix
            src_orig_size: (H, W) of original source/reference image
            src_new_size: (H, W) of resized source/reference image
            tgt_orig_size: (H, W) of original target image
            tgt_new_size: (H, W) of resized target image

        Returns:
            Scaled 3x3 homography matrix
        """
        # Source scaling (image 1 -> reference)
        src_scale_y = src_new_size[0] / src_orig_size[0]
        src_scale_x = src_new_size[1] / src_orig_size[1]

        # Target scaling (image 2-6)
        tgt_scale_y = tgt_new_size[0] / tgt_orig_size[0]
        tgt_scale_x = tgt_new_size[1] / tgt_orig_size[1]

        # Source scaling matrix inverse (maps resized coords to original)
        S1_inv = np.array([
            [1/src_scale_x, 0, 0],
            [0, 1/src_scale_y, 0],
            [0, 0, 1],
        ], dtype=np.float32)

        # Target scaling matrix (maps original coords to resized)
        S2 = np.array([
            [tgt_scale_x, 0, 0],
            [0, tgt_scale_y, 0],
            [0, 0, 1],
        ], dtype=np.float32)

        # H_scaled = S2 @ H @ S1_inv
        H_scaled = S2 @ H @ S1_inv

        return H_scaled.astype(np.float32)

    def _homography_to_vec(self, H: np.ndarray) -> np.ndarray:
        """
        Convert 3x3 homography to 8D vector representation.

        Delegates to the canonical numpy implementation in src.utils.geometry.
        See also: src.utils.homography.homography_matrix_to_vec (torch version).
        """
        return homography_matrix_to_vec_np(H)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample.

        Returns dict with:
            - image_src: [1, H, W] source/reference image tensor
            - image_tgt: [1, H, W] target image tensor
            - homography: [3, 3] homography matrix
            - homography_vec: [8] homography as 8D vector
            - (optional) sequence, seq_type, target_idx if return_metadata=True
        """
        sample = self.samples[idx]

        # Load images with resizing
        ref_image, ref_orig, ref_new = self._load_and_resize_image(sample["ref_path"])
        tgt_image, tgt_orig, tgt_new = self._load_and_resize_image(sample["target_path"])

        # Load and scale homography
        H = self._load_homography(sample["h_file"])
        H = self._scale_homography(H, ref_orig, ref_new, tgt_orig, tgt_new)

        # Apply transforms if specified
        if self.transform is not None:
            transformed = self.transform(image=ref_image, image2=tgt_image)
            ref_image = transformed["image"]
            tgt_image = transformed["image2"]

        # Convert to tensors and normalize to [0, 1]
        ref_tensor = torch.from_numpy(ref_image).float() / 255.0
        tgt_tensor = torch.from_numpy(tgt_image).float() / 255.0

        # Add channel dimension
        if ref_tensor.dim() == 2:
            ref_tensor = ref_tensor.unsqueeze(0)
            tgt_tensor = tgt_tensor.unsqueeze(0)

        # Prepare output
        output = {
            "image_src": ref_tensor,
            "image_tgt": tgt_tensor,
            "homography": torch.from_numpy(H).float(),
            "homography_vec": torch.from_numpy(self._homography_to_vec(H)).float(),
        }

        if self.return_metadata:
            output["sequence"] = sample["sequence"]
            output["seq_type"] = sample["seq_type"]
            output["target_idx"] = sample["target_idx"]

        return output

    def get_sequence_names(self) -> List[str]:
        """Get list of unique sequence names."""
        return sorted(set(s["sequence"] for s in self.samples))

    def get_samples_for_sequence(self, sequence_name: str) -> List[int]:
        """Get sample indices for a specific sequence."""
        return [i for i, s in enumerate(self.samples) if s["sequence"] == sequence_name]


def compute_corner_error(
    H_pred: np.ndarray,
    H_gt: np.ndarray,
    image_size: Tuple[int, int] = (480, 640),
) -> float:
    """
    Compute average corner error in pixels.

    Note: A canonical numpy version also exists in src.utils.geometry.compute_corner_error
    (uses cv2.perspectiveTransform, defaults to 256x256). A batched torch version exists in
    src.training.metrics.corner_error. This local version is kept because it uses a different
    default image_size (480x640) matching the HPatches evaluation convention and implements
    the transform manually without cv2.

    Args:
        H_pred: Predicted homography [3, 3]
        H_gt: Ground truth homography [3, 3]
        image_size: (H, W) of the image

    Returns:
        Mean corner error in pixels
    """
    h, w = image_size

    # Define 4 corners
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ], dtype=np.float32).T  # [3, 4]

    # Warp corners with predicted homography
    pred_corners = H_pred @ corners
    pred_corners = pred_corners[:2] / (pred_corners[2:3] + 1e-8)  # Normalize

    # Warp corners with ground truth homography
    gt_corners = H_gt @ corners
    gt_corners = gt_corners[:2] / (gt_corners[2:3] + 1e-8)  # Normalize

    # Compute L2 distance
    error = np.linalg.norm(pred_corners - gt_corners, axis=0)

    return float(error.mean())


def compute_auc(errors: List[float], thresholds: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
    """
    Compute AUC (percentage of samples with error below threshold).

    Args:
        errors: List of corner errors
        thresholds: Pixel thresholds for AUC computation

    Returns:
        Dict mapping "AUC@X" to percentage value
    """
    errors = np.array(errors)
    results = {}

    for tau in thresholds:
        accuracy = (errors < tau).mean() * 100
        results[f"AUC@{tau}"] = accuracy

    return results
