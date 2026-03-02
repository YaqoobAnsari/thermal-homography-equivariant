"""
Training Script for Thermal Homography

Implements PyTorch Lightning training with:
- Multi-GPU support
- Wandb logging
- Checkpoint management
- Learning rate scheduling
"""

from collections import deque
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch import Tensor
from torch.utils.data import DataLoader

from src.constants import (
    DEFAULT_METRIC_LOG_FREQUENCY,
    DEFAULT_SAVE_TOP_K,
)
from src.data import SyntheticThermalGenerator, ThermalPairDataset
from src.models import (
    BasesHomoBaseline,
    CorrelationBaseline,
    HomographyNet,
    IterativeHomographyNetwork,
    LucasKanadeBaseline,
    ResNetBaseline,
    ThermalHomographyNet,
    UNetBaseline,
)
from src.training.losses import HomographyLoss
from src.training.metrics import (
    MetricTracker,
    corner_error,
    rotation_error,
)
from src.training.optimizers import create_optimizer, create_scheduler
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# Model Registry - maps model type strings to model classes
MODEL_REGISTRY = {
    "e2_gnn": ThermalHomographyNet,
    "resnet": ResNetBaseline,
    "unet": UNetBaseline,
    "correlation": CorrelationBaseline,
    "homographynet": HomographyNet,
    "lucas_kanade": LucasKanadeBaseline,
    "baseshomo": BasesHomoBaseline,
    "ihn": IterativeHomographyNetwork,
}


def find_latest_checkpoint(log_dir: str) -> str | None:
    """
    Find the latest checkpoint in a log directory.

    Args:
        log_dir: Directory to search for checkpoints

    Returns:
        Path to latest checkpoint, or None if not found
    """
    log_path = Path(log_dir)
    checkpoint_dir = log_path / "checkpoints"

    if not checkpoint_dir.exists():
        return None

    # Find all .ckpt files
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        return None

    # Sort by modification time and return latest
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    logger.info(f"Found latest checkpoint: {latest}")
    return str(latest)


class NaNDetectionCallback(pl.Callback):
    """
    Callback to detect NaN/Inf in training and halt gracefully.

    This provides an additional safety net beyond the in-module check.
    """

    def __init__(self, check_weights: bool = True, check_gradients: bool = True):
        super().__init__()
        self.check_weights = check_weights
        self.check_gradients = check_gradients

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Check for NaN after each training batch."""
        # Check loss
        if outputs is None:
            return

        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
            logger.error(f"NaN/Inf loss detected at batch {batch_idx}")
            trainer.should_stop = True
            return

        # Check model weights
        if self.check_weights:
            for name, param in pl_module.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    logger.error(f"NaN/Inf detected in parameter: {name}")
                    trainer.should_stop = True
                    return

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Check gradients before optimizer step."""
        if not self.check_gradients:
            return

        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    logger.error(f"NaN/Inf detected in gradient: {name}")
                    trainer.should_stop = True
                    return


class GradientMonitorCallback(pl.Callback):
    """
    Callback to monitor and log gradient statistics.

    Useful for debugging training instability.
    """

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Log gradient statistics periodically."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        grad_norms = {}
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.norm().item()

        if grad_norms:
            max_norm = max(grad_norms.values())
            min_norm = min(grad_norms.values())
            mean_norm = sum(grad_norms.values()) / len(grad_norms)

            pl_module.log("grad/max_norm", max_norm)
            pl_module.log("grad/min_norm", min_norm)
            pl_module.log("grad/mean_norm", mean_norm)


class TrainingVisualizationCallback(pl.Callback):
    """
    Callback to generate live training plots.

    Saves loss curves and metric visualizations during training.
    """

    def __init__(self, output_dir: str, plot_frequency: int = 100):
        """
        Initialize training visualization callback.

        Args:
            output_dir: Directory to save plots
            plot_frequency: Generate plots every N batches
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.plot_frequency = plot_frequency
        self.train_losses = deque(maxlen=1000)
        self.val_losses = deque(maxlen=1000)
        self.val_corner_errors = deque(maxlen=1000)
        self.epochs = deque(maxlen=1000)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Record training loss."""
        if outputs is not None:
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            if loss is not None:
                self.train_losses.append(loss.item())

        # Generate plots periodically
        if batch_idx > 0 and batch_idx % self.plot_frequency == 0:
            self._save_loss_plot(trainer.global_step)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Record validation metrics."""
        # Get logged metrics
        metrics = trainer.callback_metrics
        if "val/loss_total" in metrics:
            self.val_losses.append(metrics["val/loss_total"].item())
        if "val/corner_error_mean" in metrics:
            self.val_corner_errors.append(metrics["val/corner_error_mean"].item())
        self.epochs.append(trainer.current_epoch)

        # Generate epoch summary plot
        self._save_epoch_plot(trainer.current_epoch)

    def _save_loss_plot(self, step: int):
        """Save training loss curve."""
        import matplotlib.pyplot as plt

        self.output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Smooth the training loss with a moving average
        window = min(50, len(self.train_losses) // 10 + 1)
        if len(self.train_losses) > window:
            import numpy as np

            smoothed = np.convolve(self.train_losses, np.ones(window) / window, mode="valid")
            ax.plot(smoothed, "b-", alpha=0.8, label="Train Loss (smoothed)")
        ax.plot(self.train_losses, "b-", alpha=0.2, label="Train Loss (raw)")

        ax.set_xlabel("Batch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss (Step {step})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "train_loss.png", dpi=100)
        plt.close(fig)

    def _save_epoch_plot(self, epoch: int):
        """Save epoch summary plot."""
        import matplotlib.pyplot as plt

        if not self.val_losses:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Validation loss
        axes[0].plot(self.epochs, self.val_losses, "r-o", label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Corner error
        if self.val_corner_errors:
            axes[1].plot(self.epochs, self.val_corner_errors, "g-o", label="Corner Error")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Error (px)")
            axes[1].set_title("Validation Corner Error")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"Training Progress (Epoch {epoch})", fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / "epoch_summary.png", dpi=100)
        plt.close(fig)


class ConfigSavingCheckpoint(ModelCheckpoint):
    """
    Extended ModelCheckpoint that saves config alongside checkpoints.

    Ensures experiment reproducibility by bundling config with weights.
    """

    def __init__(self, config=None, **kwargs):
        """
        Initialize checkpoint with config.

        Args:
            config: OmegaConf config to save
            **kwargs: Arguments passed to ModelCheckpoint
        """
        super().__init__(**kwargs)
        self.config = config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Save config when checkpoint is saved."""
        super().on_save_checkpoint(trainer, pl_module, checkpoint)

        if self.config is not None and self.best_model_path:
            config_path = Path(self.best_model_path).with_suffix(".yaml")
            OmegaConf.save(self.config, config_path)
            logger.info(f"Config saved to {config_path}")


class ThermalHomographyModule(pl.LightningModule):
    """
    PyTorch Lightning module for thermal homography training.
    """

    def __init__(
        self,
        # Model config
        model_type: str = "e2_gnn",
        feature_dim: int = 32,
        grid_size: int = 32,
        gnn_num_layers: int = 4,
        use_lrft: bool = True,
        # Loss config
        corner_weight: float = 1.0,
        rotation_weight: float = 0.1,
        translation_weight: float = 0.1,
        rank_weight: float = 0.01,
        # Training config
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        optimizer_type: str = "adamw",
        scheduler_type: str = "cosine",
        # Image config
        image_size: tuple = (256, 256),
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build model using registry
        if model_type not in MODEL_REGISTRY:
            available = ", ".join(MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")

        model_class = MODEL_REGISTRY[model_type]

        # Different models have different constructor signatures
        if model_type == "e2_gnn":
            self.model = model_class(
                feature_dim=feature_dim,
                grid_size=grid_size,
                gnn_num_layers=gnn_num_layers,
                use_lrft=use_lrft,
            )
        elif model_type in ["resnet", "unet", "homographynet", "lucas_kanade"]:
            self.model = model_class()
        elif model_type == "correlation":
            self.model = model_class(
                feature_dim=128,
                hidden_dim=128,
                num_iterations=3,
            )
        elif model_type == "baseshomo":
            self.model = model_class(
                num_bases=8,
                feature_dim=256,
            )
        elif model_type == "ihn":
            self.model = model_class(
                num_iterations=3,
            )
        else:
            # Generic fallback
            self.model = model_class()

        # Loss function
        self.loss_fn = HomographyLoss(
            corner_weight=corner_weight,
            rotation_weight=rotation_weight,
            translation_weight=translation_weight,
            rank_weight=rank_weight,
            image_size=image_size,
        )

        # Metric trackers
        self.train_metrics = MetricTracker(image_size)
        self.val_metrics = MetricTracker(image_size)

    def forward(self, image_src: Tensor, image_tgt: Tensor) -> dict[str, Tensor]:
        return self.model(image_src, image_tgt)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor | None:
        image_src = batch["image_src"]
        image_tgt = batch["image_tgt"]
        H_gt = batch["homography_vec"]

        # Forward pass
        output = self(image_src, image_tgt)
        H_pred = output["homography"]

        # Compute loss
        similarity = output.get("similarity", None)
        losses = self.loss_fn(H_pred, H_gt, similarity)

        # NaN detection - halt training if loss becomes NaN
        if torch.isnan(losses["total"]) or torch.isinf(losses["total"]):
            logger.error(
                f"NaN/Inf detected in loss at batch {batch_idx}! "
                f"Losses: {', '.join(f'{k}={v.item() if torch.is_tensor(v) else v}' for k, v in losses.items())}"
            )
            self.trainer.should_stop = True
            return None

        # Log losses
        for name, value in losses.items():
            self.log(
                f"train/loss_{name}", value, on_step=True, on_epoch=True, prog_bar=(name == "total")
            )

        # Log metrics (less frequently to save compute)
        if batch_idx % DEFAULT_METRIC_LOG_FREQUENCY == 0:
            with torch.no_grad():
                ce = corner_error(H_pred, H_gt).mean()
                re = rotation_error(H_pred, H_gt).mean()
                self.log("train/corner_error", ce, on_step=True, on_epoch=False)
                self.log("train/rotation_error", re, on_step=True, on_epoch=False)

        return losses["total"]

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms for monitoring training stability."""
        # Compute total gradient norm
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        self.log("train/grad_norm", total_norm, on_step=True, on_epoch=False)

        # Warn if gradient norm is suspiciously large
        if total_norm > 100.0:
            logger.warning(f"Large gradient norm detected: {total_norm:.2f}")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Tensor]:
        image_src = batch["image_src"]
        image_tgt = batch["image_tgt"]
        H_gt = batch["homography_vec"]

        # Forward pass
        output = self(image_src, image_tgt)
        H_pred = output["homography"]

        # Compute loss
        similarity = output.get("similarity", None)
        losses = self.loss_fn(H_pred, H_gt, similarity)

        # Track metrics
        self.val_metrics.update(H_pred, H_gt)

        # Log losses
        for name, value in losses.items():
            self.log(f"val/loss_{name}", value, on_epoch=True, prog_bar=(name == "total"))

        return {"loss": losses["total"], "H_pred": H_pred, "H_gt": H_gt}

    def on_validation_epoch_end(self):
        # Compute and log aggregate metrics
        metrics = self.val_metrics.compute()

        for name, value in metrics.items():
            self.log(f"val/{name}", value)

        # Reset for next epoch
        self.val_metrics.reset()

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Tensor]:
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        # Create optimizer using factory
        optimizer_type = getattr(self.hparams, "optimizer_type", "adamw")
        optimizer = create_optimizer(
            self,
            optimizer_type=optimizer_type,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        # Create scheduler using factory
        scheduler_type = getattr(self.hparams, "scheduler_type", "cosine")

        # Get max_epochs from trainer if available
        max_epochs = self.trainer.max_epochs if self.trainer else 100

        scheduler = create_scheduler(
            optimizer,
            scheduler_type=scheduler_type,
            T_max=max_epochs,
            warmup_epochs=self.hparams.warmup_epochs,
        )

        # Handle ReduceLROnPlateau differently (needs monitor)
        if scheduler_type == "plateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss_total",
                    "interval": "epoch",
                },
            }

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def train(config: DictConfig, resume_from: str | None = None):
    """
    Main training function.

    Args:
        config: Hydra configuration
        resume_from: Path to checkpoint to resume from (optional)
    """
    # Set seed for reproducibility
    pl.seed_everything(config.seed, workers=True)

    # Handle checkpoint resume
    ckpt_path = resume_from
    if ckpt_path is None and config.training.get("auto_resume", False):
        ckpt_path = find_latest_checkpoint(config.logging.log_dir)
        if ckpt_path:
            logger.info(f"Auto-resuming from: {ckpt_path}")

    # Create model
    model = ThermalHomographyModule(
        model_type=config.model.type,
        feature_dim=config.model.get("feature_dim", 32),
        grid_size=config.model.get("grid_size", 32),
        gnn_num_layers=config.model.get("gnn_num_layers", 4),
        use_lrft=config.model.get("use_lrft", True),
        corner_weight=config.loss.corner_weight,
        rotation_weight=config.loss.rotation_weight,
        translation_weight=config.loss.translation_weight,
        rank_weight=config.loss.rank_weight,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_epochs=config.training.warmup_epochs,
        optimizer_type=config.training.get("optimizer_type", "adamw"),
        scheduler_type=config.training.get("scheduler_type", "cosine"),
        image_size=tuple(config.data.image_size),
    )

    # Create data loaders
    if config.data.synthetic:
        train_dataset = SyntheticThermalGenerator(
            n_samples=config.data.n_train_samples,
            image_size=tuple(config.data.image_size),
            seed=config.seed,
        )
        val_dataset = SyntheticThermalGenerator(
            n_samples=config.data.n_val_samples,
            image_size=tuple(config.data.image_size),
            seed=config.seed + 1,
        )
    else:
        train_dataset = ThermalPairDataset(
            data_root=config.data.data_root,
            split="train",
            image_size=tuple(config.data.image_size),
        )
        val_dataset = ThermalPairDataset(
            data_root=config.data.data_root,
            split="val",
            image_size=tuple(config.data.image_size),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    # Setup logging
    loggers = []
    if config.logging.use_wandb:
        wandb_logger = WandbLogger(
            project=config.logging.wandb_project,
            name=config.logging.experiment_name,
            config=OmegaConf.to_container(config),
        )
        loggers.append(wandb_logger)

    tb_logger = TensorBoardLogger(
        save_dir=config.logging.log_dir,
        name=config.logging.experiment_name,
    )
    loggers.append(tb_logger)

    # Callbacks
    callbacks = [
        ConfigSavingCheckpoint(
            config=config,
            dirpath=Path(config.logging.log_dir) / "checkpoints",
            filename="{epoch}-{val/loss_total:.4f}",
            monitor="val/loss_total",
            mode="min",
            save_top_k=DEFAULT_SAVE_TOP_K,
        ),
        EarlyStopping(
            monitor="val/loss_total",
            patience=config.training.early_stopping_patience,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
        NaNDetectionCallback(check_weights=True, check_gradients=True),
        GradientMonitorCallback(log_every_n_steps=100),
        TrainingVisualizationCallback(
            output_dir=Path(config.logging.log_dir) / "plots",
            plot_frequency=100,
        ),
    ]

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=config.training.precision,
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=config.training.gradient_clip,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        log_every_n_steps=config.logging.log_every_n_steps,
        val_check_interval=config.training.val_check_interval,
        deterministic=True,
    )

    # Train (with optional resume)
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    # Test with best checkpoint
    if config.training.run_test:
        trainer.test(model, val_loader, ckpt_path="best")

    return model


def main():
    """Entry point for command-line training."""
    # Default config for quick testing
    default_config = {
        "seed": 42,
        "model": {
            "type": "e2_gnn",
            "feature_dim": 32,
            "grid_size": 16,
            "gnn_num_layers": 2,
            "use_lrft": True,
        },
        "loss": {
            "corner_weight": 1.0,
            "rotation_weight": 0.1,
            "translation_weight": 0.1,
            "rank_weight": 0.01,
        },
        "data": {
            "synthetic": True,
            "n_train_samples": 1000,
            "n_val_samples": 200,
            "image_size": [256, 256],
            "data_root": "",
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "warmup_epochs": 5,
            "max_epochs": 100,
            "num_workers": 0,
            "precision": 32,
            "gradient_clip": 1.0,
            "accumulate_grad_batches": 1,
            "val_check_interval": 1.0,
            "early_stopping_patience": 20,
            "run_test": True,
        },
        "logging": {
            "use_wandb": False,
            "wandb_project": "thermal-homography",
            "experiment_name": "test_run",
            "log_dir": "logs",
            "log_every_n_steps": 10,
        },
    }

    config = OmegaConf.create(default_config)
    train(config)


if __name__ == "__main__":
    main()
