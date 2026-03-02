"""
Optimizer and Scheduler Factory

Provides flexible creation of optimizers and learning rate schedulers
for training thermal homography models.

Usage:
    optimizer = create_optimizer(model, optimizer_type='adamw', lr=1e-4)
    scheduler = create_scheduler(optimizer, scheduler_type='cosine', T_max=100)
"""

import math
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
    _LRScheduler,
)

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class CosineAnnealingWithWarmup(_LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.

    Learning rate starts from warmup_start_lr, linearly increases to
    base_lr during warmup_epochs, then follows cosine annealing.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        warmup_epochs: int = 5,
        warmup_start_lr: float = 1e-7,
        eta_min: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.T_max - self.warmup_epochs
            )
            progress = min(1.0, progress)
            return [
                self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class LinearWarmupScheduler(_LRScheduler):
    """Simple linear warmup followed by constant learning rate."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 5,
        warmup_start_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        return self.base_lrs


# Optimizer Registry
OPTIMIZER_REGISTRY = {
    "adamw": lambda params, lr, weight_decay, **kwargs: torch.optim.AdamW(
        params, lr=lr, weight_decay=weight_decay, **kwargs
    ),
    "adam": lambda params, lr, weight_decay, **kwargs: torch.optim.Adam(
        params, lr=lr, weight_decay=weight_decay, **kwargs
    ),
    "sgd": lambda params, lr, weight_decay, **kwargs: torch.optim.SGD(
        params, lr=lr, weight_decay=weight_decay, momentum=kwargs.pop("momentum", 0.9), **kwargs
    ),
    "rmsprop": lambda params, lr, weight_decay, **kwargs: torch.optim.RMSprop(
        params, lr=lr, weight_decay=weight_decay, **kwargs
    ),
    "adagrad": lambda params, lr, weight_decay, **kwargs: torch.optim.Adagrad(
        params, lr=lr, weight_decay=weight_decay, **kwargs
    ),
}

# Scheduler Registry
SCHEDULER_REGISTRY = {
    "cosine": lambda opt, **kw: CosineAnnealingWithWarmup(
        opt,
        T_max=kw.get("T_max", 100),
        warmup_epochs=kw.get("warmup_epochs", 5),
        warmup_start_lr=kw.get("warmup_start_lr", 1e-7),
        eta_min=kw.get("eta_min", 1e-7),
    ),
    "cosine_no_warmup": lambda opt, **kw: CosineAnnealingLR(
        opt, T_max=kw.get("T_max", 100), eta_min=kw.get("eta_min", 1e-7)
    ),
    "step": lambda opt, **kw: StepLR(
        opt, step_size=kw.get("step_size", 30), gamma=kw.get("gamma", 0.1)
    ),
    "plateau": lambda opt, **kw: ReduceLROnPlateau(
        opt,
        mode=kw.get("mode", "min"),
        factor=kw.get("factor", 0.1),
        patience=kw.get("patience", 10),
        min_lr=kw.get("min_lr", 1e-7),
    ),
    "onecycle": lambda opt, **kw: OneCycleLR(
        opt,
        max_lr=kw.get("max_lr", kw.get("lr", 1e-3)),
        total_steps=kw.get("total_steps", 1000),
        pct_start=kw.get("pct_start", 0.3),
        anneal_strategy=kw.get("anneal_strategy", "cos"),
    ),
    "linear_warmup": lambda opt, **kw: LinearWarmupScheduler(
        opt,
        warmup_epochs=kw.get("warmup_epochs", 5),
        warmup_start_lr=kw.get("warmup_start_lr", 1e-7),
    ),
    "none": lambda opt, **kw: LambdaLR(opt, lr_lambda=lambda epoch: 1.0),
}


def create_optimizer(
    model: nn.Module | Iterator[nn.Parameter],
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    **kwargs,
) -> Optimizer:
    """
    Create optimizer from registry.

    Args:
        model: Model or parameter iterator
        optimizer_type: Type of optimizer (adamw, adam, sgd, rmsprop, adagrad)
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Configured optimizer

    Raises:
        ValueError: If optimizer_type is not in registry
    """
    if optimizer_type not in OPTIMIZER_REGISTRY:
        available = ", ".join(OPTIMIZER_REGISTRY.keys())
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Available: {available}")

    # Get parameters
    if isinstance(model, nn.Module):
        params = model.parameters()
    else:
        params = model

    optimizer = OPTIMIZER_REGISTRY[optimizer_type](
        params, lr=lr, weight_decay=weight_decay, **kwargs
    )

    logger.info(f"Created {optimizer_type} optimizer with lr={lr}, weight_decay={weight_decay}")

    return optimizer


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    **kwargs,
) -> _LRScheduler:
    """
    Create learning rate scheduler from registry.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler (cosine, step, plateau, onecycle, etc.)
        **kwargs: Scheduler-specific arguments

    Returns:
        Configured scheduler

    Raises:
        ValueError: If scheduler_type is not in registry
    """
    if scheduler_type not in SCHEDULER_REGISTRY:
        available = ", ".join(SCHEDULER_REGISTRY.keys())
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Available: {available}")

    scheduler = SCHEDULER_REGISTRY[scheduler_type](optimizer, **kwargs)

    logger.info(f"Created {scheduler_type} scheduler")

    return scheduler


def create_optimizer_with_scheduler(
    model: nn.Module | Iterator[nn.Parameter],
    optimizer_type: str = "adamw",
    scheduler_type: str = "cosine",
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    optimizer_kwargs: dict[str, Any] | None = None,
    scheduler_kwargs: dict[str, Any] | None = None,
) -> tuple:
    """
    Create both optimizer and scheduler together.

    Args:
        model: Model or parameter iterator
        optimizer_type: Type of optimizer
        scheduler_type: Type of scheduler
        lr: Learning rate
        weight_decay: Weight decay
        optimizer_kwargs: Additional optimizer arguments
        scheduler_kwargs: Additional scheduler arguments

    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer_kwargs = optimizer_kwargs or {}
    scheduler_kwargs = scheduler_kwargs or {}

    optimizer = create_optimizer(
        model,
        optimizer_type=optimizer_type,
        lr=lr,
        weight_decay=weight_decay,
        **optimizer_kwargs,
    )

    # Pass lr to scheduler for OneCycleLR
    scheduler_kwargs.setdefault("lr", lr)

    scheduler = create_scheduler(
        optimizer,
        scheduler_type=scheduler_type,
        **scheduler_kwargs,
    )

    return optimizer, scheduler


def get_parameter_groups(
    model: nn.Module,
    base_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    no_decay_keywords: tuple = ("bias", "LayerNorm", "BatchNorm"),
    lr_multipliers: dict[str, float] | None = None,
) -> list:
    """
    Create parameter groups with different learning rates and weight decay.

    Useful for fine-tuning or when different parts of the model need
    different learning rates.

    Args:
        model: Model to create parameter groups for
        base_lr: Base learning rate
        weight_decay: Default weight decay
        no_decay_keywords: Parameters matching these keywords get no weight decay
        lr_multipliers: Dict mapping parameter name patterns to LR multipliers

    Returns:
        List of parameter group dictionaries
    """
    lr_multipliers = lr_multipliers or {}

    # Separate parameters by weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter should have no weight decay
        no_decay = any(kw in name for kw in no_decay_keywords)

        # Find LR multiplier
        multiplier = 1.0
        for pattern, mult in lr_multipliers.items():
            if pattern in name:
                multiplier = mult
                break

        param_group = {
            "params": [param],
            "lr": base_lr * multiplier,
            "weight_decay": 0.0 if no_decay else weight_decay,
            "name": name,
        }

        if no_decay:
            no_decay_params.append(param_group)
        else:
            decay_params.append(param_group)

    logger.info(
        f"Created {len(decay_params)} params with decay, " f"{len(no_decay_params)} without decay"
    )

    return decay_params + no_decay_params
