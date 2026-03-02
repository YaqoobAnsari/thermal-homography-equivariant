"""
Data loading and augmentation for thermal homography.

Modules:
- thermal_dataset: PyTorch dataset for thermal image pairs
- augmentation: Geometric and photometric augmentations
- synthetic_generator: Generate synthetic training data
- hpatches_dataset: HPatches benchmark dataset
- mscoco_dataset: Warped MS-COCO dataset
"""

from .thermal_dataset import ThermalPairDataset, ThermalDataModule
from .augmentation import ThermalAugmentation, get_train_transforms, get_val_transforms
from .synthetic_generator import SyntheticThermalGenerator, generate_checkerboard_pair
from .hpatches_dataset import HPatchesDataset, compute_corner_error, compute_auc
from .mscoco_dataset import WarpedMSCOCODataset, ControlledTransformDataset

__all__ = [
    # Thermal datasets
    "ThermalPairDataset",
    "ThermalDataModule",
    # Augmentation
    "ThermalAugmentation",
    "get_train_transforms",
    "get_val_transforms",
    # Synthetic
    "SyntheticThermalGenerator",
    "generate_checkerboard_pair",
    # HPatches benchmark
    "HPatchesDataset",
    "compute_corner_error",
    "compute_auc",
    # MS-COCO benchmark
    "WarpedMSCOCODataset",
    "ControlledTransformDataset",
]
