"""
Constants Module for Thermal Homography

Centralizes magic numbers and default values used throughout the codebase.
This improves maintainability and makes configuration more explicit.
"""

import math

# =============================================================================
# Image Constants
# =============================================================================

DEFAULT_IMAGE_SIZE: tuple[int, int] = (256, 256)
DEFAULT_GRID_SIZE: int = 32
DEFAULT_FEATURE_DIM: int = 32
DEFAULT_PATCH_SIZE: int = 7

# =============================================================================
# Training Constants
# =============================================================================

DEFAULT_BATCH_SIZE: int = 8
DEFAULT_LEARNING_RATE: float = 1e-4
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_NUM_WORKERS: int = 4
DEFAULT_MAX_EPOCHS: int = 100
DEFAULT_WARMUP_EPOCHS: int = 5
DEFAULT_GRADIENT_CLIP: float = 1.0
DEFAULT_EARLY_STOPPING_PATIENCE: int = 20
DEFAULT_EARLY_STOPPING_MIN_DELTA: float = 0.001
DEFAULT_EARLY_STOPPING_WARMUP: int = 5

# Early stopping patience by epoch tier
EARLY_STOPPING_PATIENCE_SHORT: int = 10   # For <=30 epochs
EARLY_STOPPING_PATIENCE_MEDIUM: int = 20  # For <=100 epochs
EARLY_STOPPING_PATIENCE_LONG: int = 30    # For >100 epochs

# =============================================================================
# Loss Constants
# =============================================================================

DEFAULT_CORNER_WEIGHT: float = 1.0
DEFAULT_ROTATION_WEIGHT: float = 0.1
DEFAULT_TRANSLATION_WEIGHT: float = 0.1
DEFAULT_RANK_WEIGHT: float = 0.01
DEFAULT_RANK_TARGET: int = 32

# =============================================================================
# Metric Constants
# =============================================================================

DEFAULT_RECALL_THRESHOLDS: list[float] = [3.0, 5.0, 10.0, 20.0]
DEFAULT_ROTATION_BINS: list[float] = [0, 15, 30, 45, 60, 90, 180]

# =============================================================================
# Numerical Constants
# =============================================================================

EPSILON: float = 1e-8
PI: float = math.pi

# =============================================================================
# GNN Constants
# =============================================================================

DEFAULT_GNN_HIDDEN_DIM: int = 64
DEFAULT_GNN_NUM_LAYERS: int = 4
DEFAULT_K_NEIGHBORS: int = 8

# =============================================================================
# LRFT Constants
# =============================================================================

DEFAULT_LRFT_OUT_NODES: int = 128
DEFAULT_LRFT_RANK: int = 32

# =============================================================================
# Synthetic Data Constants
# =============================================================================

DEFAULT_ROTATION_RANGE: tuple[float, float] = (-30.0, 30.0)
DEFAULT_TRANSLATION_RANGE: tuple[float, float] = (-30.0, 30.0)
DEFAULT_SCALE_RANGE: tuple[float, float] = (0.9, 1.1)
DEFAULT_NOISE_STD: float = 10.0

# =============================================================================
# Logging Constants
# =============================================================================

DEFAULT_METRIC_LOG_FREQUENCY: int = 100
DEFAULT_SAVE_TOP_K: int = 3
DEFAULT_LOG_EVERY_N_STEPS: int = 10

# =============================================================================
# Equivariance Test Constants
# =============================================================================

DEFAULT_EQUIVARIANCE_TOLERANCE: float = 1e-4
DEFAULT_TEST_ANGLES: list[float] = [0, 15, 30, 45, 60, 90, 120, 150, 180]

# =============================================================================
# Color/Visualization Constants
# =============================================================================

DEFAULT_COLORMAP: str = "hot"
DEFAULT_FIGURE_DPI: int = 150
