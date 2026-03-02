"""
Training components for thermal similarity estimation.

Modules:
- train: Main training loop and Lightning module
- losses: Similarity/homography loss functions
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
    # Sim(2)-specific metrics
    scale_error,
    sim2_component_errors,
    equivariance_score,
    compute_sim2_metrics,
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
    # Sim(2)-specific metrics
    "scale_error",
    "sim2_component_errors",
    "equivariance_score",
    "compute_sim2_metrics",
]
