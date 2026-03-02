"""
Thermal Image Pair Dataset

Handles loading of thermal image pairs with ground truth homographies.
Supports multiple data formats and evaluation splits.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from src.utils.geometry import homography_matrix_to_vec_np
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ThermalPairDataset(Dataset):
    """
    Dataset for thermal image pairs with ground truth homographies.

    Expected directory structure:
    data_root/
    ├── pairs/
    │   ├── 000000_src.png
    │   ├── 000000_tgt.png
    │   └── ...
    ├── homographies.json  # {"000000": [[h11, h12, ...], ...], ...}
    └── metadata.json      # Optional: scene info, colormap, etc.

    OR for RoadScene/OTCBVS style:
    data_root/
    ├── ir/
    │   ├── 000000.png
    │   └── ...
    ├── vis/
    │   ├── 000000.png
    │   └── ...
    └── homographies/ (optional)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (256, 256),
        return_metadata: bool = False,
        data_format: str = "pairs",  # "pairs", "roadscene", "otcbvs"
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.return_metadata = return_metadata
        self.data_format = data_format

        # Load data index
        self.samples = self._load_samples()

        logger.info(f"Loaded {len(self.samples)} samples for {split} split")

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load sample list based on data format."""
        samples = []

        if self.data_format == "pairs":
            samples = self._load_pairs_format()
        elif self.data_format == "roadscene":
            samples = self._load_roadscene_format()
        elif self.data_format == "synthetic":
            # Synthetic data is generated on-the-fly
            samples = [{"idx": i} for i in range(1000)]
        else:
            raise ValueError(f"Unknown data format: {self.data_format}")

        return samples

    def _load_pairs_format(self) -> List[Dict[str, Any]]:
        """Load data in pairs format."""
        samples = []

        pairs_dir = self.data_root / "pairs"
        if not pairs_dir.exists():
            pairs_dir = self.data_root

        # Find all source images
        src_images = sorted(pairs_dir.glob("*_src.*")) + sorted(pairs_dir.glob("*_1.*"))

        # Load homographies if available
        homographies = {}
        homo_file = self.data_root / "homographies.json"
        if homo_file.exists():
            try:
                with open(homo_file) as f:
                    homographies = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse homographies.json: {e}")
                raise ValueError(f"Invalid JSON in homographies file: {homo_file}") from e
            except IOError as e:
                logger.warning(f"Could not read homographies file: {e}")
                homographies = {}

        for src_path in src_images:
            # Find corresponding target image
            stem = src_path.stem.replace("_src", "").replace("_1", "")
            tgt_candidates = [
                src_path.parent / f"{stem}_tgt{src_path.suffix}",
                src_path.parent / f"{stem}_2{src_path.suffix}",
            ]

            tgt_path = None
            for candidate in tgt_candidates:
                if candidate.exists():
                    tgt_path = candidate
                    break

            if tgt_path is None:
                continue

            # Get homography if available
            H = homographies.get(stem, np.eye(3).tolist())

            samples.append({
                "src_path": str(src_path),
                "tgt_path": str(tgt_path),
                "homography": np.array(H, dtype=np.float32),
                "id": stem,
            })

        return samples

    def _load_roadscene_format(self) -> List[Dict[str, Any]]:
        """Load RoadScene or similar IR/VIS dataset format."""
        samples = []

        ir_dir = self.data_root / "ir"
        vis_dir = self.data_root / "vis"

        if not ir_dir.exists():
            ir_dir = self.data_root / "IR"
        if not vis_dir.exists():
            vis_dir = self.data_root / "VIS"

        # Find all IR images
        ir_images = sorted(ir_dir.glob("*.*"))

        for ir_path in ir_images:
            # Find corresponding visible image
            stem = ir_path.stem
            vis_candidates = list(vis_dir.glob(f"{stem}.*"))

            if not vis_candidates:
                continue

            vis_path = vis_candidates[0]

            # For this format, we don't have ground truth homographies
            # They need to be computed or the dataset provides them
            samples.append({
                "src_path": str(ir_path),
                "tgt_path": str(vis_path),
                "homography": np.eye(3, dtype=np.float32),
                "id": stem,
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Get a sample."""
        sample = self.samples[idx]

        # Load images with error handling
        original_size = None
        try:
            src_image, src_orig_size = self._load_image(sample["src_path"])
            tgt_image, tgt_orig_size = self._load_image(sample["tgt_path"])
            # Use source image's original size for homography scaling
            original_size = src_orig_size
        except FileNotFoundError as e:
            logger.warning(f"Sample {idx}: {e}. Returning zero images.")
            src_image = np.zeros(self.image_size, dtype=np.uint8)
            tgt_image = np.zeros(self.image_size, dtype=np.uint8)
            original_size = self.image_size

        # Get homography and scale it to match the resized images
        homography = sample["homography"].copy()
        homography = self._scale_homography(homography, original_size, self.image_size)

        # Apply transforms if specified
        if self.transform is not None:
            transformed = self.transform(
                image=src_image,
                image2=tgt_image,
                keypoints=[],  # Placeholder for homography adjustment
            )
            src_image = transformed["image"]
            tgt_image = transformed["image2"]

        # Convert to tensors
        src_tensor = torch.from_numpy(src_image).float()
        tgt_tensor = torch.from_numpy(tgt_image).float()

        # Normalize to [0, 1]
        src_tensor = src_tensor / 255.0 if src_tensor.max() > 1 else src_tensor
        tgt_tensor = tgt_tensor / 255.0 if tgt_tensor.max() > 1 else tgt_tensor

        # Add channel dimension if needed
        if src_tensor.dim() == 2:
            src_tensor = src_tensor.unsqueeze(0)
            tgt_tensor = tgt_tensor.unsqueeze(0)

        # Convert homography to 8D vector (normalized)
        homography_vec = self._homography_to_vec(homography)

        output = {
            "image_src": src_tensor,
            "image_tgt": tgt_tensor,
            "homography": torch.from_numpy(homography).float(),
            "homography_vec": torch.from_numpy(homography_vec).float(),
        }

        if self.return_metadata:
            output["id"] = sample["id"]
            output["src_path"] = sample["src_path"]
            output["tgt_path"] = sample["tgt_path"]

        return output

    def _load_image(self, path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Load and preprocess an image.

        Args:
            path: Path to the image file.

        Returns:
            Tuple of (preprocessed grayscale image, original size (H, W)).

        Raises:
            FileNotFoundError: If image cannot be loaded.
        """
        try:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        except cv2.error as e:
            logger.error(f"OpenCV error loading image {path}: {e}")
            raise FileNotFoundError(f"Could not load image: {path}") from e

        if image is None:
            raise FileNotFoundError(f"Could not load image: {path}")

        # Store original size for homography scaling
        original_size = (image.shape[0], image.shape[1])  # (H, W)

        # Resize
        try:
            image = cv2.resize(image, self.image_size)
        except cv2.error as e:
            logger.error(f"OpenCV error resizing image {path}: {e}")
            raise ValueError(f"Could not resize image: {path}") from e

        return image, original_size

    def _scale_homography(
        self,
        H: np.ndarray,
        original_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Scale homography matrix when images are resized.

        When resizing from (H_orig, W_orig) to (H_new, W_new), the homography
        must be transformed as: H_scaled = S @ H @ S^{-1}
        where S = diag(scale_x, scale_y, 1).

        This ensures that points transformed in the original coordinate system
        map correctly in the resized coordinate system.

        Args:
            H: Original 3x3 homography matrix.
            original_size: (H, W) of original images.
            target_size: (H, W) of resized images.

        Returns:
            Scaled 3x3 homography matrix.
        """
        H_orig, W_orig = original_size
        H_new, W_new = target_size

        # Scale factors
        scale_x = W_new / W_orig
        scale_y = H_new / H_orig

        # Scaling matrix: maps original coords to new coords
        S = np.array([
            [scale_x, 0, 0],
            [0, scale_y, 0],
            [0, 0, 1],
        ], dtype=np.float32)

        # Inverse scaling matrix
        S_inv = np.array([
            [1/scale_x, 0, 0],
            [0, 1/scale_y, 0],
            [0, 0, 1],
        ], dtype=np.float32)

        # Transform homography: H_scaled = S @ H @ S_inv
        # This makes H work in the resized coordinate system
        H_scaled = S @ H @ S_inv

        return H_scaled.astype(np.float32)

    def _homography_to_vec(self, H: np.ndarray) -> np.ndarray:
        """
        Convert 3x3 homography matrix to 8D vector representation.

        Delegates to the canonical numpy implementation in src.utils.geometry.
        See also: src.utils.homography.homography_matrix_to_vec (torch version).
        """
        return homography_matrix_to_vec_np(H)

    def validate_dataset(self, check_files: bool = True) -> Dict[str, Any]:
        """
        Validate the dataset for common issues.

        Args:
            check_files: Whether to check if all image files exist.

        Returns:
            Dictionary with validation results including:
            - n_samples: Total number of samples
            - n_valid: Number of valid samples with existing files
            - n_missing: Number of samples with missing files
            - missing_files: List of missing file paths
        """
        results = {
            'n_samples': len(self.samples),
            'n_valid': 0,
            'n_missing': 0,
            'missing_files': [],
        }

        if not check_files:
            results['n_valid'] = len(self.samples)
            return results

        for sample in self.samples:
            src_exists = Path(sample['src_path']).exists()
            tgt_exists = Path(sample['tgt_path']).exists()

            if src_exists and tgt_exists:
                results['n_valid'] += 1
            else:
                results['n_missing'] += 1
                if not src_exists:
                    results['missing_files'].append(sample['src_path'])
                if not tgt_exists:
                    results['missing_files'].append(sample['tgt_path'])

        if results['n_missing'] > 0:
            logger.warning(
                f"Dataset validation: {results['n_missing']}/{results['n_samples']} "
                f"samples have missing files"
            )
        else:
            logger.info(
                f"Dataset validation passed: all {results['n_samples']} samples valid"
            )

        return results


class ThermalDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for thermal homography.

    Handles train/val/test splits and data loading.
    """

    def __init__(
        self,
        data_root: str,
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (256, 256),
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        data_format: str = "pairs",
        train_split: float = 0.8,
        seed: int = 42,
    ):
        super().__init__()

        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.data_format = data_format
        self.train_split = train_split
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Set up datasets."""
        # Create full dataset
        full_dataset = ThermalPairDataset(
            data_root=self.data_root,
            split="all",
            transform=None,
            image_size=self.image_size,
            data_format=self.data_format,
        )

        # Split into train/val/test
        n_samples = len(full_dataset)
        n_train = int(n_samples * self.train_split)
        n_val = (n_samples - n_train) // 2
        n_test = n_samples - n_train - n_val

        # Use generator for reproducibility
        generator = torch.Generator().manual_seed(self.seed)

        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=generator,
        )

        # Apply transforms (wrap datasets)
        if self.train_transform:
            self.train_dataset.dataset.transform = self.train_transform
        if self.val_transform:
            self.val_dataset.dataset.transform = self.val_transform
            self.test_dataset.dataset.transform = self.val_transform

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def create_dummy_dataset(
    output_dir: str,
    n_samples: int = 100,
    image_size: Tuple[int, int] = (256, 256),
) -> None:
    """
    Create a dummy dataset for testing.

    Generates random thermal-like images with known homographies.
    """
    output_path = Path(output_dir)
    pairs_dir = output_path / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    homographies = {}

    for i in range(n_samples):
        # Generate random source image
        src = np.random.randint(100, 200, size=image_size, dtype=np.uint8)

        # Add some structure (blobs)
        for _ in range(5):
            cx, cy = np.random.randint(50, image_size[0] - 50, size=2)
            radius = np.random.randint(20, 50)
            cv2.circle(src, (cx, cy), radius, int(np.random.randint(50, 255)), -1)

        # Apply Gaussian blur
        src = cv2.GaussianBlur(src, (5, 5), 1.5)

        # Generate random homography
        # Small rotation + translation
        angle = np.random.uniform(-30, 30) * np.pi / 180
        tx = np.random.uniform(-20, 20)
        ty = np.random.uniform(-20, 20)
        scale = np.random.uniform(0.9, 1.1)

        cos_a, sin_a = np.cos(angle), np.sin(angle)
        H = np.array([
            [scale * cos_a, -scale * sin_a, tx],
            [scale * sin_a, scale * cos_a, ty],
            [0, 0, 1],
        ], dtype=np.float32)

        # Warp source to create target
        tgt = cv2.warpPerspective(src, H, image_size)

        # Save images
        sample_id = f"{i:06d}"
        cv2.imwrite(str(pairs_dir / f"{sample_id}_src.png"), src)
        cv2.imwrite(str(pairs_dir / f"{sample_id}_tgt.png"), tgt)

        # Store homography (inverse: from target to source)
        H_inv = np.linalg.inv(H)
        homographies[sample_id] = H_inv.tolist()

    # Save homographies
    with open(output_path / "homographies.json", "w") as f:
        json.dump(homographies, f, indent=2)

    logger.info(f"Created dummy dataset with {n_samples} samples at {output_path}")
