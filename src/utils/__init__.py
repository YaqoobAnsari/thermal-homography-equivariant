"""
Utility functions for thermal homography.

Modules:
- homography: Homography manipulation utilities
- equivariance_tests: Numerical equivariance verification
- geometry: Geometric utilities
- logging_config: Logging infrastructure
"""

from .equivariance_tests import (
    test_equivariance_numerical,
    test_model_equivariance,
)
from .geometry import (
    compute_corner_error,
    warp_image,
)
from .homography import (
    apply_homography,
    decompose_homography,
    homography_matrix_to_vec,
    homography_vec_to_matrix,
)
from .logging_config import (
    MetricLogger,
    get_logger,
    log_to_wandb,
    setup_logging,
)

__all__ = [
    "homography_vec_to_matrix",
    "homography_matrix_to_vec",
    "decompose_homography",
    "apply_homography",
    "test_equivariance_numerical",
    "test_model_equivariance",
    "compute_corner_error",
    "warp_image",
    "get_logger",
    "setup_logging",
    "log_to_wandb",
    "MetricLogger",
]
