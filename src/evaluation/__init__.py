"""
Evaluation utilities for thermal homography.

Modules:
- eval_splits: Evaluation on different data splits
- visualize: Visualization tools
- ablations: Ablation study utilities
"""

from .eval_splits import evaluate_all_splits, evaluate_model
from .visualize import plot_rotation_equivariance, visualize_homography

__all__ = [
    "evaluate_model",
    "evaluate_all_splits",
    "visualize_homography",
    "plot_rotation_equivariance",
]
