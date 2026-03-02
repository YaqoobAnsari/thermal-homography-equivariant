"""
Training components for thermal homography.

Modules:
- train: Main training loop and Lightning module
- losses: Homography loss functions
- metrics: Evaluation metrics
"""

from .losses import (
    HomographyLoss,
    corner_loss,
    geodesic_rotation_loss,
    translation_loss,
)
from .metrics import (
    corner_error,
    registration_recall,
    rotation_error,
    translation_error,
)

__all__ = [
    "HomographyLoss",
    "corner_loss",
    "geodesic_rotation_loss",
    "translation_loss",
    "corner_error",
    "rotation_error",
    "translation_error",
    "registration_recall",
]
