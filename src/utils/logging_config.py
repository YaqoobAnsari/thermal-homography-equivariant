"""
Logging Configuration for Thermal Homography

Provides centralized logging infrastructure with:
- Configurable log levels
- File rotation (10MB max, 5 backup files)
- Optional colored console output
- Optional JSON formatting for structured logging
- Integration hooks for wandb

Usage:
    from src.utils.logging_config import get_logger, setup_logging

    # Setup logging (call once at program start)
    setup_logging(level='INFO', log_file='training.log')

    # Get logger for a module
    logger = get_logger(__name__)
    logger.info("Training started")
    logger.debug("Batch 0 processed")
    logger.warning("Learning rate is very small")
    logger.error("Failed to load checkpoint", exc_info=True)
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import LOG_LEVEL, get_logs_dir

# Track if logging has been setup
_logging_initialized = False

# Store reference to file handler for log file path retrieval
_file_handler: logging.Handler | None = None


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter with colored output for terminals.

    Colors:
        DEBUG: Cyan
        INFO: Green
        WARNING: Yellow
        ERROR: Red
        CRITICAL: Bold Red
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str | None = None, datefmt: str | None = None, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        # Save original levelname
        original_levelname = record.levelname

        if self.use_colors and record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"

        result = super().format(record)

        # Restore original levelname
        record.levelname = original_levelname

        return result


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Useful for log aggregation systems and parsing.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data

        return json.dumps(log_data)


def setup_logging(
    level: str | int = None,
    log_file: str | None = None,
    log_dir: Path | None = None,
    json_format: bool = False,
    colored: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    propagate: bool = False,
) -> logging.Logger:
    """
    Setup logging configuration for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               Default: from THERMAL_LOG_LEVEL env var or INFO
        log_file: Optional log file name (will be created in log_dir)
        log_dir: Directory for log files (default: from config)
        json_format: Use JSON formatting for file output
        colored: Use colored output for console
        max_bytes: Max size per log file before rotation
        backup_count: Number of backup files to keep
        propagate: Whether child loggers should propagate to root

    Returns:
        Root logger instance

    Example:
        # Basic setup
        setup_logging()

        # With file logging
        setup_logging(level='DEBUG', log_file='training.log')

        # JSON formatted for production
        setup_logging(log_file='app.log', json_format=True)
    """
    global _logging_initialized, _file_handler

    # Use env var or default if level not specified
    if level is None:
        level = LOG_LEVEL

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers if re-initializing
    if _logging_initialized:
        root_logger.handlers.clear()

    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    console_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    console_formatter = ColoredFormatter(console_format, datefmt="%H:%M:%S", use_colors=colored)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation (optional)
    if log_file:
        log_dir = log_dir or get_logs_dir()
        log_path = log_dir / log_file

        _file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        _file_handler.setLevel(level)

        if json_format:
            file_formatter = JSONFormatter()
        else:
            file_format = (
                "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
            )
            file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")

        _file_handler.setFormatter(file_formatter)
        root_logger.addHandler(_file_handler)

    # Configure library loggers to be less verbose
    for lib_logger in ["matplotlib", "PIL", "urllib3", "h5py"]:
        logging.getLogger(lib_logger).setLevel(logging.WARNING)

    _logging_initialized = True

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.

    If logging hasn't been initialized, performs basic setup.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Processing batch %d", batch_idx)
    """
    global _logging_initialized

    if not _logging_initialized:
        setup_logging()

    return logging.getLogger(name)


def get_log_file_path() -> Path | None:
    """Get the current log file path if file logging is enabled."""
    if _file_handler and hasattr(_file_handler, "baseFilename"):
        return Path(_file_handler.baseFilename)
    return None


class LogContext:
    """
    Context manager for temporary log level changes.

    Example:
        with LogContext(level='DEBUG'):
            logger.debug("This will be shown")
        # Level restored after context
    """

    def __init__(self, level: str | int, logger_name: str | None = None):
        self.level = level if isinstance(level, int) else getattr(logging, level.upper())
        self.logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
        self.old_level = None

    def __enter__(self):
        self.old_level = self.logger.level
        self.logger.setLevel(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)
        return False


def log_to_wandb(metrics: dict[str, Any], step: int | None = None):
    """
    Log metrics to Weights & Biases if available.

    This is a convenience wrapper that handles the case where
    wandb is not initialized.

    Args:
        metrics: Dictionary of metrics to log
        step: Optional step number
    """
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass
    except Exception as e:
        logger = get_logger(__name__)
        logger.debug(f"Failed to log to wandb: {e}")


def log_config_to_wandb(config: dict[str, Any]):
    """
    Log configuration to Weights & Biases if available.

    Args:
        config: Configuration dictionary
    """
    try:
        import wandb

        if wandb.run is not None:
            wandb.config.update(config)
    except ImportError:
        pass
    except Exception as e:
        logger = get_logger(__name__)
        logger.debug(f"Failed to log config to wandb: {e}")


class MetricLogger:
    """
    Helper class for logging metrics during training.

    Buffers metrics and logs them at specified intervals.
    Also supports wandb integration.

    Example:
        metric_logger = MetricLogger(log_interval=100)

        for batch_idx, batch in enumerate(dataloader):
            loss = train_step(batch)
            metric_logger.update(loss=loss.item(), batch_idx=batch_idx)
            metric_logger.log_if_needed(batch_idx)
    """

    def __init__(
        self,
        logger_name: str = "metrics",
        log_interval: int = 100,
        use_wandb: bool = True,
    ):
        self.logger = get_logger(logger_name)
        self.log_interval = log_interval
        self.use_wandb = use_wandb
        self.metrics: dict[str, list] = {}
        self.step = 0

    def update(self, **kwargs):
        """Add metrics for current step."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def log_if_needed(self, step: int):
        """Log metrics if at log interval."""
        if step > 0 and step % self.log_interval == 0:
            self.log(step)

    def log(self, step: int | None = None):
        """Log current metrics and reset."""
        if not self.metrics:
            return

        step = step or self.step

        # Compute averages
        avg_metrics = {}
        for key, values in self.metrics.items():
            if values:
                avg_metrics[key] = sum(values) / len(values)

        # Log to console
        metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in avg_metrics.items())
        self.logger.info(f"Step {step} | {metric_str}")

        # Log to wandb
        if self.use_wandb:
            log_to_wandb(avg_metrics, step=step)

        # Reset
        self.metrics.clear()
        self.step = step

    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
