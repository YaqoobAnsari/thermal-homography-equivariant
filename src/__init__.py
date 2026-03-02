"""
Thermal Homography with E(2)-Equivariant Graph Neural Networks

A keypoint-free approach to thermal image alignment using geometric deep learning.
"""

__version__ = "0.1.0"
__author__ = "Yaqoob Ansari"

# Use centralized config for path management (lazy initialization)
# Explicit re-exports for public API
from src.config import CHECKPOINTS_DIR as CHECKPOINTS_DIR
from src.config import CONFIGS_DIR as CONFIGS_DIR
from src.config import DATA_DIR as DATA_DIR
from src.config import PROJECT_ROOT as PROJECT_ROOT
from src.config import get_checkpoints_dir as get_checkpoints_dir
from src.config import get_data_dir as get_data_dir
from src.config import get_output_path as get_output_path

# Note: Directories are now created lazily when first accessed via get_*_dir() functions
# This improves portability and avoids side effects on import
