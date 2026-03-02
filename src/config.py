"""
Centralized Configuration Module for Thermal Homography

Provides platform-independent path management and configuration utilities.
Supports environment variables for customization in different deployment environments.

Environment Variables:
    THERMAL_DATA_DIR: Root directory for datasets (default: PROJECT_ROOT/data)
    THERMAL_OUTPUT_DIR: Output directory for results (default: system temp)
    THERMAL_CHECKPOINTS_DIR: Checkpoint directory (default: PROJECT_ROOT/checkpoints)
    THERMAL_LOGS_DIR: Logs directory (default: PROJECT_ROOT/logs)
    THERMAL_LOG_LEVEL: Logging level (default: INFO)
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

# Auto-detect project root (directory containing src/)
_current_file = Path(__file__).resolve()
PROJECT_ROOT = _current_file.parent.parent

# Configurable directories via environment variables
DATA_DIR = Path(os.environ.get("THERMAL_DATA_DIR", PROJECT_ROOT / "data"))
CHECKPOINTS_DIR = Path(os.environ.get("THERMAL_CHECKPOINTS_DIR", PROJECT_ROOT / "checkpoints"))
LOGS_DIR = Path(os.environ.get("THERMAL_LOGS_DIR", PROJECT_ROOT / "logs"))
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Output directory - uses system temp by default for portability
_default_output = Path(tempfile.gettempdir()) / "thermal_homography"
OUTPUT_DIR = Path(os.environ.get("THERMAL_OUTPUT_DIR", _default_output))

# Log level from environment
LOG_LEVEL = os.environ.get("THERMAL_LOG_LEVEL", "INFO").upper()


def _ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if needed."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir() -> Path:
    """Get the data directory, creating if needed."""
    return _ensure_dir(DATA_DIR)


def get_checkpoints_dir() -> Path:
    """Get the checkpoints directory, creating if needed."""
    return _ensure_dir(CHECKPOINTS_DIR)


def get_logs_dir() -> Path:
    """Get the logs directory, creating if needed."""
    return _ensure_dir(LOGS_DIR)


def get_output_dir() -> Path:
    """Get the output directory, creating if needed."""
    return _ensure_dir(OUTPUT_DIR)


def get_output_path(filename: str, subdir: Optional[str] = None) -> Path:
    """
    Get a platform-safe output path for temporary files.

    Args:
        filename: Name of the output file
        subdir: Optional subdirectory within output dir

    Returns:
        Full path to the output file

    Example:
        >>> path = get_output_path('synthetic_test.png')
        >>> path = get_output_path('result.png', subdir='visualizations')
    """
    output_dir = get_output_dir()
    if subdir:
        output_dir = _ensure_dir(output_dir / subdir)
    return output_dir / filename


def get_config_path(config_name: str) -> Path:
    """
    Get path to a configuration file.

    Args:
        config_name: Name of config file (with or without .yaml extension)

    Returns:
        Full path to the config file

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if not config_name.endswith(".yaml"):
        config_name = f"{config_name}.yaml"

    path = CONFIGS_DIR / config_name
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path


def resolve_path(path: Union[str, Path], base: Optional[Path] = None) -> Path:
    """
    Resolve a path, making it absolute if relative.

    Args:
        path: Path to resolve (can be string or Path)
        base: Base directory for relative paths (default: PROJECT_ROOT)

    Returns:
        Absolute path
    """
    path = Path(path)
    if path.is_absolute():
        return path
    base = base or PROJECT_ROOT
    return (base / path).resolve()


class Config:
    """
    Configuration container with lazy directory initialization.

    Usage:
        config = Config()
        config.get_checkpoint_path('model_best.ckpt')
    """

    def __init__(self):
        self._initialized = False

    def _lazy_init(self):
        """Initialize directories on first access."""
        if not self._initialized:
            # Only create directories when actually needed
            self._initialized = True

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        self._lazy_init()
        return get_data_dir()

    @property
    def checkpoints_dir(self) -> Path:
        self._lazy_init()
        return get_checkpoints_dir()

    @property
    def logs_dir(self) -> Path:
        self._lazy_init()
        return get_logs_dir()

    @property
    def output_dir(self) -> Path:
        self._lazy_init()
        return get_output_dir()

    @property
    def configs_dir(self) -> Path:
        return CONFIGS_DIR

    def get_checkpoint_path(self, name: str) -> Path:
        """Get path to a checkpoint file."""
        return self.checkpoints_dir / name

    def get_log_path(self, name: str) -> Path:
        """Get path to a log file."""
        return self.logs_dir / name

    def get_output_path(self, filename: str, subdir: Optional[str] = None) -> Path:
        """Get platform-safe output path."""
        return get_output_path(filename, subdir)


# Global config instance
config = Config()


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


def validate_config(config_dict: dict[str, Any]) -> dict[str, list[str]]:
    """
    Validate configuration dictionary against schema.

    Performs validation for:
    - Required fields
    - Type checking
    - Value constraints (ranges, allowed values)
    - Logical consistency

    Args:
        config_dict: Configuration dictionary to validate

    Returns:
        Dictionary with 'errors' and 'warnings' lists

    Example:
        >>> result = validate_config({'model': {'type': 'e2_gnn'}})
        >>> if result['errors']:
        ...     raise ConfigValidationError(result['errors'])
    """
    errors = []
    warnings = []

    # Model validation
    if "model" in config_dict:
        model = config_dict["model"]

        # Valid model types
        VALID_MODEL_TYPES = [
            "e2_gnn",
            "resnet",
            "unet",
            "correlation",
            "homographynet",
            "lucas_kanade",
            "baseshomo",
            "ihn",
        ]

        if "type" not in model:
            errors.append("model.type is required")
        elif model["type"] not in VALID_MODEL_TYPES:
            errors.append(
                f"model.type must be one of: {', '.join(VALID_MODEL_TYPES)} (got: {model['type']})"
            )

        if "feature_dim" in model:
            if not isinstance(model["feature_dim"], int):
                errors.append("model.feature_dim must be an integer")
            elif model["feature_dim"] <= 0:
                errors.append("model.feature_dim must be positive")
            elif model["feature_dim"] > 512:
                warnings.append(f"model.feature_dim={model['feature_dim']} is unusually large")

        if "grid_size" in model:
            if not isinstance(model["grid_size"], int):
                errors.append("model.grid_size must be an integer")
            elif model["grid_size"] <= 0:
                errors.append("model.grid_size must be positive")

        if "gnn_num_layers" in model:
            if not isinstance(model["gnn_num_layers"], int):
                errors.append("model.gnn_num_layers must be an integer")
            elif model["gnn_num_layers"] < 1:
                errors.append("model.gnn_num_layers must be at least 1")
            elif model["gnn_num_layers"] > 10:
                warnings.append(f"model.gnn_num_layers={model['gnn_num_layers']} may be too deep")

    # Training validation
    if "training" in config_dict:
        training = config_dict["training"]

        if "learning_rate" in training:
            lr = training["learning_rate"]
            if not isinstance(lr, (int, float)):
                errors.append("training.learning_rate must be a number")
            elif lr <= 0:
                errors.append("training.learning_rate must be positive")
            elif lr > 1.0:
                warnings.append(f"training.learning_rate={lr} is unusually high")

        if "batch_size" in training:
            bs = training["batch_size"]
            if not isinstance(bs, int):
                errors.append("training.batch_size must be an integer")
            elif bs <= 0:
                errors.append("training.batch_size must be positive")

        if "max_epochs" in training:
            epochs = training["max_epochs"]
            if not isinstance(epochs, int):
                errors.append("training.max_epochs must be an integer")
            elif epochs <= 0:
                errors.append("training.max_epochs must be positive")

        if "precision" in training:
            precision = training["precision"]
            if precision not in [16, 32, 64, "16", "32", "64", "bf16", "bf16-mixed", "16-mixed"]:
                errors.append(
                    f"training.precision must be 16, 32, 64, bf16, or mixed (got: {precision})"
                )

        if "gradient_clip" in training:
            clip = training["gradient_clip"]
            if clip is not None and (not isinstance(clip, (int, float)) or clip <= 0):
                errors.append("training.gradient_clip must be a positive number or None")

        if "warmup_epochs" in training and "max_epochs" in training:
            if training["warmup_epochs"] >= training["max_epochs"]:
                errors.append("training.warmup_epochs must be less than max_epochs")

    # Loss validation
    if "loss" in config_dict:
        loss = config_dict["loss"]

        for weight_name in [
            "corner_weight",
            "rotation_weight",
            "translation_weight",
            "rank_weight",
        ]:
            if weight_name in loss:
                w = loss[weight_name]
                if not isinstance(w, (int, float)):
                    errors.append(f"loss.{weight_name} must be a number")
                elif w < 0:
                    errors.append(f"loss.{weight_name} must be non-negative")

    # Data validation
    if "data" in config_dict:
        data = config_dict["data"]

        if "image_size" in data:
            size = data["image_size"]
            if not isinstance(size, (list, tuple)) or len(size) != 2:
                errors.append("data.image_size must be a list/tuple of 2 integers")
            elif not all(isinstance(s, int) and s > 0 for s in size):
                errors.append("data.image_size values must be positive integers")

        if not data.get("synthetic", True) and not data.get("data_root"):
            errors.append("data.data_root is required when synthetic=False")

    # Seed validation
    if "seed" in config_dict:
        seed = config_dict["seed"]
        if not isinstance(seed, int):
            errors.append("seed must be an integer")
        elif seed < 0:
            warnings.append("seed should typically be non-negative")

    return {
        "errors": errors,
        "warnings": warnings,
        "valid": len(errors) == 0,
    }


def validate_config_strict(config_dict: dict[str, Any]) -> None:
    """
    Validate configuration and raise error if invalid.

    Args:
        config_dict: Configuration dictionary to validate

    Raises:
        ConfigValidationError: If validation fails
    """
    result = validate_config(config_dict)

    if result["warnings"]:
        import logging

        logger = logging.getLogger(__name__)
        for warning in result["warnings"]:
            logger.warning(f"Config warning: {warning}")

    if not result["valid"]:
        raise ConfigValidationError(
            "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in result["errors"])
        )


def print_config():
    """Print current configuration for debugging."""
    print("Thermal Homography Configuration:")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"  DATA_DIR: {DATA_DIR}")
    print(f"  CHECKPOINTS_DIR: {CHECKPOINTS_DIR}")
    print(f"  LOGS_DIR: {LOGS_DIR}")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"  LOG_LEVEL: {LOG_LEVEL}")
