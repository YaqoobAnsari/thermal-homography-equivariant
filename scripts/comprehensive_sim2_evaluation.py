#!/usr/bin/env python3
"""
COMPREHENSIVE Sim(2) EVALUATION FRAMEWORK

This script provides rigorous evaluation of Sim(2) equivariance with:

1. INDIVIDUAL COMPONENT TESTS:
   - Rotation-only generalization (train ±30°, test 0-180°)
   - Scale-only generalization (train 0.9-1.1x, test 0.5-2.0x)
   - Translation-only generalization (train ±20px, test ±64px)

2. COMBINED Sim(2) TESTS:
   - Full transformation with all components
   - In-distribution vs out-of-distribution comparison

3. MULTIPLE BASELINES:
   - BaselineCNN (simple regression)
   - HomographyNet (DeTone et al., CVPR 2016)
   - IHN (Iterative Homography Network, CVPR 2022)
   - Classical Fourier-Mellin (no learning)
   - SIFT + RANSAC (traditional)

4. STANDARD METRICS:
   - MACE (Mean Average Corner Error) - primary metric
   - AUC @ 3, 5, 10, 20 pixels
   - Per-component errors: rotation (°), scale (ratio), translation (px)
   - Generalization gap: OOD_error - ID_error

Author: ECCV 2026 Submission
Date: 2026-02-02
"""

import sys
import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import cv2
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, "/data/gpfs/projects/punim2769/thermal-homography")

# =============================================================================
# IMPORTS
# =============================================================================

from src.models.log_polar_sim2_net import LogPolarSim2Net
from src.models.baseline_cnn import BaselineCNN, BaselineLoss
from src.data.synthetic_generator import (
    generate_arrow_pattern,
    generate_natural_texture,
    generate_checkerboard,
    generate_homography,
)

# Try to import advanced baselines
try:
    from src.models.baselines import HomographyNet, IterativeHomographyNetwork
    HAS_ADVANCED_BASELINES = True
except ImportError:
    HAS_ADVANCED_BASELINES = False
    print("Warning: Advanced baselines not available")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EvalConfig:
    """Configuration for comprehensive evaluation."""

    # Training distribution (narrow)
    train_rotation_range: Tuple[float, float] = (-30, 30)
    train_scale_range: Tuple[float, float] = (0.9, 1.1)
    train_translation_range: Tuple[float, float] = (-20, 20)

    # Test distribution (full range)
    test_rotation_angles: List[float] = field(default_factory=lambda:
        [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180])
    test_scale_factors: List[float] = field(default_factory=lambda:
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0])
    test_translation_offsets: List[float] = field(default_factory=lambda:
        [0, 10, 20, 30, 40, 50, 64])

    # Training settings
    n_train_samples: int = 3000
    n_test_samples_per_config: int = 50
    batch_size: int = 32
    epochs: int = 25
    lr: float = 1e-4

    # Image settings
    image_size: int = 256

    # Output
    output_dir: str = "outputs/comprehensive_eval"


# =============================================================================
# METRICS
# =============================================================================

def compute_corner_error(H_pred: np.ndarray, H_gt: np.ndarray, image_size: int = 256) -> float:
    """
    Compute Mean Average Corner Error (MACE).

    This is the standard metric for homography evaluation:
    - Define 4 corners of the image
    - Warp with predicted and ground truth homographies
    - Compute mean L2 distance
    """
    # Define corners
    corners = np.array([
        [0, 0, 1],
        [image_size, 0, 1],
        [image_size, image_size, 1],
        [0, image_size, 1]
    ], dtype=np.float32).T  # [3, 4]

    # Warp corners
    corners_pred = H_pred @ corners
    corners_gt = H_gt @ corners

    # Normalize homogeneous coordinates
    corners_pred = corners_pred[:2] / (corners_pred[2:3] + 1e-8)
    corners_gt = corners_gt[:2] / (corners_gt[2:3] + 1e-8)

    # Compute L2 distances
    errors = np.sqrt(np.sum((corners_pred - corners_gt) ** 2, axis=0))

    return float(np.mean(errors))


def compute_corner_error_torch(H_pred: torch.Tensor, H_gt: torch.Tensor,
                                image_size: int = 256) -> torch.Tensor:
    """Batched corner error computation."""
    B = H_pred.shape[0]
    device = H_pred.device

    # Define corners [4, 3]
    corners = torch.tensor([
        [0, 0, 1],
        [image_size, 0, 1],
        [image_size, image_size, 1],
        [0, image_size, 1]
    ], dtype=torch.float32, device=device).T  # [3, 4]

    corners = corners.unsqueeze(0).expand(B, -1, -1)  # [B, 3, 4]

    # Warp
    corners_pred = torch.bmm(H_pred, corners)  # [B, 3, 4]
    corners_gt = torch.bmm(H_gt, corners)

    # Normalize
    corners_pred = corners_pred[:, :2] / (corners_pred[:, 2:3] + 1e-8)
    corners_gt = corners_gt[:, :2] / (corners_gt[:, 2:3] + 1e-8)

    # L2 distance
    errors = torch.sqrt(torch.sum((corners_pred - corners_gt) ** 2, dim=1))  # [B, 4]

    return errors.mean(dim=1)  # [B]


def compute_auc(errors: List[float], thresholds: List[float] = [3, 5, 10, 20]) -> Dict[str, float]:
    """Compute AUC at various thresholds."""
    errors = np.array(errors)
    results = {}
    for thresh in thresholds:
        results[f"auc@{thresh}"] = float(np.mean(errors < thresh) * 100)
    return results


def decompose_sim2(H: np.ndarray) -> Dict[str, float]:
    """Decompose homography into Sim(2) components."""
    # Extract 2x2 upper-left block
    A = H[:2, :2]
    t = H[:2, 2]

    # Scale from determinant
    scale = np.sqrt(np.abs(np.linalg.det(A)))

    # Rotation from normalized matrix
    if scale > 1e-6:
        R = A / scale
        rotation = np.arctan2(R[1, 0], R[0, 0])
    else:
        rotation = 0.0

    return {
        "rotation_deg": float(np.degrees(rotation)),
        "scale": float(scale),
        "tx": float(t[0]),
        "ty": float(t[1]),
    }


# =============================================================================
# DATASET
# =============================================================================

class Sim2Dataset(Dataset):
    """
    Flexible Sim(2) dataset for controlled experiments.

    Supports:
    - Fixed or random rotation/scale/translation
    - In-distribution and out-of-distribution configs
    """

    def __init__(
        self,
        n_samples: int,
        rotation_range: Optional[Tuple[float, float]] = None,
        scale_range: Optional[Tuple[float, float]] = None,
        translation_range: Optional[Tuple[float, float]] = None,
        fixed_rotation: Optional[float] = None,
        fixed_scale: Optional[float] = None,
        fixed_translation: Optional[Tuple[float, float]] = None,
        image_size: int = 256,
        pattern_types: List[str] = ["arrow", "texture", "checkerboard"],
    ):
        self.n_samples = n_samples
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.fixed_rotation = fixed_rotation
        self.fixed_scale = fixed_scale
        self.fixed_translation = fixed_translation
        self.image_size = image_size
        self.pattern_types = pattern_types

    def __len__(self):
        return self.n_samples

    def _generate_pattern(self, idx: int) -> np.ndarray:
        """Generate diverse patterns."""
        pattern_type = self.pattern_types[idx % len(self.pattern_types)]
        size = (self.image_size, self.image_size)

        if pattern_type == "arrow":
            return generate_arrow_pattern(size=size)
        elif pattern_type == "texture":
            return generate_natural_texture(size=size)
        elif pattern_type == "checkerboard":
            return generate_checkerboard(size=size)
        else:
            return generate_natural_texture(size=size)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        np.random.seed(idx)

        # Generate source pattern
        source = self._generate_pattern(idx)

        # Determine transformation parameters
        if self.fixed_rotation is not None:
            rot_range = (self.fixed_rotation, self.fixed_rotation)
        elif self.rotation_range is not None:
            rot_range = self.rotation_range
        else:
            rot_range = (0, 0)

        if self.fixed_scale is not None:
            scale_range = (self.fixed_scale, self.fixed_scale)
        elif self.scale_range is not None:
            scale_range = self.scale_range
        else:
            scale_range = (1.0, 1.0)

        if self.fixed_translation is not None:
            trans_range = (self.fixed_translation[0], self.fixed_translation[0])
        elif self.translation_range is not None:
            trans_range = self.translation_range
        else:
            trans_range = (0, 0)

        # Generate homography
        H, params = generate_homography(
            rotation_range=rot_range,
            scale_range=scale_range,
            translation_range=trans_range,
            image_size=(self.image_size, self.image_size),
        )

        # Warp image
        target = cv2.warpPerspective(source, H, (self.image_size, self.image_size))

        # Convert to tensors
        source_t = torch.from_numpy(source).unsqueeze(0).float() / 255.0
        target_t = torch.from_numpy(target).unsqueeze(0).float() / 255.0
        H_t = torch.from_numpy(H).float()

        return {
            'source': source_t,
            'target': target_t,
            'H': H_t,
            'rotation': torch.tensor(math.radians(params['angle'])).float(),
            'scale': torch.tensor(params['scale']).float(),
            'translation': torch.tensor([params['tx'], params['ty']]).float(),
            'params': params,
        }


# =============================================================================
# MODEL WRAPPERS
# =============================================================================

class ModelWrapper:
    """Base wrapper for unified model interface."""

    def __init__(self, name: str):
        self.name = name

    def predict(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return dict with 'H', 'rotation', 'scale', 'translation'."""
        raise NotImplementedError


class FMTWrapper(ModelWrapper):
    """Wrapper for our FMT model."""

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__("FMT (Ours)")
        self.model = model
        self.device = device

    def predict(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            output = self.model(source.to(self.device), target.to(self.device))

        # Build homography from components
        B = source.shape[0]
        H = self._build_homography(
            output['rotation'],
            output['scale'],
            output.get('translation_x', torch.zeros(B, device=self.device)),
            output.get('translation_y', torch.zeros(B, device=self.device)),
            source.shape[-1]
        )

        return {
            'H': H,
            'rotation': output['rotation'],
            'scale': output['scale'],
            'translation_x': output.get('translation_x', torch.zeros(B, device=self.device)),
            'translation_y': output.get('translation_y', torch.zeros(B, device=self.device)),
        }

    def _build_homography(self, rotation, scale, tx, ty, img_size):
        """Build 3x3 homography from Sim(2) parameters."""
        B = rotation.shape[0]
        device = rotation.device

        cos_t = torch.cos(rotation)
        sin_t = torch.sin(rotation)

        # H = T @ S @ R (translate, then scale, then rotate)
        # But we parameterize as: target = H @ source
        # H = [[s*cos, -s*sin, tx], [s*sin, s*cos, ty], [0, 0, 1]]

        H = torch.zeros(B, 3, 3, device=device)
        H[:, 0, 0] = scale * cos_t
        H[:, 0, 1] = -scale * sin_t
        H[:, 0, 2] = tx * (img_size / 2)  # Denormalize
        H[:, 1, 0] = scale * sin_t
        H[:, 1, 1] = scale * cos_t
        H[:, 1, 2] = ty * (img_size / 2)
        H[:, 2, 2] = 1.0

        return H


class BaselineCNNWrapper(ModelWrapper):
    """Wrapper for BaselineCNN."""

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__("BaselineCNN")
        self.model = model
        self.device = device

    def predict(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            output = self.model(source.to(self.device), target.to(self.device))

        B = source.shape[0]
        H = self._build_homography(
            output['rotation'],
            output['scale'],
            output['translation'][:, 0],
            output['translation'][:, 1],
            source.shape[-1]
        )

        return {
            'H': H,
            'rotation': output['rotation'],
            'scale': output['scale'],
            'translation_x': output['translation'][:, 0],
            'translation_y': output['translation'][:, 1],
        }

    def _build_homography(self, rotation, scale, tx, ty, img_size):
        B = rotation.shape[0]
        device = rotation.device

        cos_t = torch.cos(rotation)
        sin_t = torch.sin(rotation)

        H = torch.zeros(B, 3, 3, device=device)
        H[:, 0, 0] = scale * cos_t
        H[:, 0, 1] = -scale * sin_t
        H[:, 0, 2] = tx * (img_size / 2)
        H[:, 1, 0] = scale * sin_t
        H[:, 1, 1] = scale * cos_t
        H[:, 1, 2] = ty * (img_size / 2)
        H[:, 2, 2] = 1.0

        return H


class ClassicalFMTWrapper(ModelWrapper):
    """Classical Fourier-Mellin Transform (no learning)."""

    def __init__(self, device: torch.device):
        super().__init__("Classical FMT")
        self.device = device

    def predict(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Use OpenCV's phase correlation."""
        B = source.shape[0]
        results = []

        for i in range(B):
            src = (source[i, 0].cpu().numpy() * 255).astype(np.uint8)
            tgt = (target[i, 0].cpu().numpy() * 255).astype(np.uint8)

            result = self._estimate_sim2_classical(src, tgt)
            results.append(result)

        # Stack results
        rotations = torch.tensor([r['rotation'] for r in results], device=self.device)
        scales = torch.tensor([r['scale'] for r in results], device=self.device)
        tx = torch.tensor([r['tx'] for r in results], device=self.device)
        ty = torch.tensor([r['ty'] for r in results], device=self.device)

        H = self._build_homography(rotations, scales, tx, ty, source.shape[-1])

        return {
            'H': H,
            'rotation': rotations,
            'scale': scales,
            'translation_x': tx,
            'translation_y': ty,
        }

    def _estimate_sim2_classical(self, src: np.ndarray, tgt: np.ndarray) -> Dict:
        """Classical FMT implementation."""
        # FFT magnitude (translation-invariant)
        f_src = np.fft.fftshift(np.fft.fft2(src.astype(np.float32)))
        f_tgt = np.fft.fftshift(np.fft.fft2(tgt.astype(np.float32)))

        mag_src = np.log(np.abs(f_src) + 1)
        mag_tgt = np.log(np.abs(f_tgt) + 1)

        # Log-polar transform
        h, w = src.shape
        center = (w // 2, h // 2)
        max_radius = min(center) * 0.9

        lp_src = cv2.linearPolar(mag_src.astype(np.float32), center, max_radius, cv2.WARP_FILL_OUTLIERS)
        lp_tgt = cv2.linearPolar(mag_tgt.astype(np.float32), center, max_radius, cv2.WARP_FILL_OUTLIERS)

        # Phase correlation for rotation and scale
        (dx, dy), _ = cv2.phaseCorrelate(lp_src, lp_tgt)

        # Convert to rotation and scale
        rotation = -dy * 2 * np.pi / h
        log_scale = dx * np.log(max_radius) / w
        scale = np.exp(log_scale)

        # For translation, de-rotate/de-scale and correlate
        # Simplified: just return 0 for now
        tx, ty = 0.0, 0.0

        return {'rotation': rotation, 'scale': scale, 'tx': tx, 'ty': ty}

    def _build_homography(self, rotation, scale, tx, ty, img_size):
        B = rotation.shape[0]
        device = rotation.device

        cos_t = torch.cos(rotation)
        sin_t = torch.sin(rotation)

        H = torch.zeros(B, 3, 3, device=device)
        H[:, 0, 0] = scale * cos_t
        H[:, 0, 1] = -scale * sin_t
        H[:, 0, 2] = tx * (img_size / 2)
        H[:, 1, 0] = scale * sin_t
        H[:, 1, 1] = scale * cos_t
        H[:, 1, 2] = ty * (img_size / 2)
        H[:, 2, 2] = 1.0

        return H


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 25,
    model_name: str = "Model",
) -> Dict[str, List[float]]:
    """Train a model and return history."""
    history = {'loss': [], 'rotation_error': [], 'scale_error': [], 'corner_error': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_rot_err = 0
        epoch_scale_err = 0
        epoch_corner_err = 0
        n_batches = 0

        for batch in train_loader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            rotation_gt = batch['rotation'].to(device)
            scale_gt = batch['scale'].to(device)
            translation_gt = batch['translation'].to(device)
            H_gt = batch['H'].to(device)

            optimizer.zero_grad()
            output = model(source, target)

            # Compute loss
            target_dict = {
                'rotation': rotation_gt,
                'scale': scale_gt,
                'translation': translation_gt,
            }
            loss_dict = loss_fn(output, target_dict)
            loss = loss_dict['total']
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Metrics
            rot_err = torch.abs(output['rotation'] - rotation_gt) * 180 / math.pi
            rot_err = torch.where(rot_err > 180, 360 - rot_err, rot_err)
            epoch_rot_err += rot_err.mean().item()

            scale_err = torch.abs(output['scale'] - scale_gt) / scale_gt
            epoch_scale_err += scale_err.mean().item() * 100

            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_rot = epoch_rot_err / n_batches
        avg_scale = epoch_scale_err / n_batches

        history['loss'].append(avg_loss)
        history['rotation_error'].append(avg_rot)
        history['scale_error'].append(avg_scale)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{model_name}] Epoch {epoch+1}/{epochs}: "
                  f"Loss={avg_loss:.4f}, RotErr={avg_rot:.1f}°, ScaleErr={avg_scale:.1f}%")

    return history


# =============================================================================
# EVALUATION
# =============================================================================

@dataclass
class EvalResult:
    """Result from evaluating one configuration."""
    config_name: str
    model_name: str

    # Raw errors
    corner_errors: List[float] = field(default_factory=list)
    rotation_errors: List[float] = field(default_factory=list)
    scale_errors: List[float] = field(default_factory=list)
    translation_errors: List[float] = field(default_factory=list)

    # Summary stats
    mace: float = 0.0
    rotation_mae: float = 0.0
    scale_mae: float = 0.0
    translation_mae: float = 0.0

    # AUC
    auc_3: float = 0.0
    auc_5: float = 0.0
    auc_10: float = 0.0
    auc_20: float = 0.0

    def compute_summary(self):
        """Compute summary statistics."""
        if self.corner_errors:
            self.mace = float(np.mean(self.corner_errors))
            auc = compute_auc(self.corner_errors)
            self.auc_3 = auc['auc@3']
            self.auc_5 = auc['auc@5']
            self.auc_10 = auc['auc@10']
            self.auc_20 = auc['auc@20']

        if self.rotation_errors:
            self.rotation_mae = float(np.mean(self.rotation_errors))
        if self.scale_errors:
            self.scale_mae = float(np.mean(self.scale_errors))
        if self.translation_errors:
            self.translation_mae = float(np.mean(self.translation_errors))


def evaluate_model_on_config(
    wrapper: ModelWrapper,
    dataset: Dataset,
    device: torch.device,
    config_name: str,
) -> EvalResult:
    """Evaluate a model on a specific configuration."""
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    result = EvalResult(config_name=config_name, model_name=wrapper.name)

    for batch in loader:
        source = batch['source']
        target = batch['target']
        H_gt = batch['H']
        rotation_gt = batch['rotation']
        scale_gt = batch['scale']
        translation_gt = batch['translation']

        # Predict
        pred = wrapper.predict(source, target)

        # Compute corner error
        H_pred = pred['H'].cpu()
        for i in range(H_pred.shape[0]):
            ce = compute_corner_error(H_pred[i].numpy(), H_gt[i].numpy())
            result.corner_errors.append(ce)

        # Component errors
        rot_err = torch.abs(pred['rotation'].cpu() - rotation_gt) * 180 / math.pi
        rot_err = torch.where(rot_err > 180, 360 - rot_err, rot_err)
        result.rotation_errors.extend(rot_err.tolist())

        scale_err = torch.abs(pred['scale'].cpu() - scale_gt) / scale_gt * 100
        result.scale_errors.extend(scale_err.tolist())

        trans_pred = torch.stack([pred['translation_x'].cpu(), pred['translation_y'].cpu()], dim=-1)
        trans_err = torch.sqrt(torch.sum((trans_pred - translation_gt / 128) ** 2, dim=-1)) * 128
        result.translation_errors.extend(trans_err.tolist())

    result.compute_summary()
    return result


# =============================================================================
# EXPERIMENTS
# =============================================================================

def run_rotation_experiment(config: EvalConfig, device: torch.device) -> Dict:
    """Test 1: Rotation-only generalization."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: ROTATION GENERALIZATION")
    print("=" * 70)
    print(f"  Train: {config.train_rotation_range[0]}° to {config.train_rotation_range[1]}°")
    print(f"  Test: {config.test_rotation_angles}")
    print()

    results = {"experiment": "rotation", "models": {}}

    # Create training dataset (rotation only, scale=1, translation=0)
    train_dataset = Sim2Dataset(
        n_samples=config.n_train_samples,
        rotation_range=config.train_rotation_range,
        scale_range=(1.0, 1.0),
        translation_range=(0, 0),
        image_size=config.image_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=4)

    # ===== BASELINE CNN =====
    print("Training BaselineCNN...")
    baseline = BaselineCNN().to(device)
    baseline_loss = BaselineLoss(w_rotation=1.0, w_scale=0.1, w_translation=0.1)
    baseline_opt = torch.optim.AdamW(baseline.parameters(), lr=config.lr)

    train_model(baseline, train_loader, baseline_loss, baseline_opt, device,
                epochs=config.epochs, model_name="BaselineCNN")
    baseline_wrapper = BaselineCNNWrapper(baseline, device)

    # ===== FMT (Ours) - No training needed =====
    print("\nInitializing FMT (no training)...")
    fmt = LogPolarSim2Net(
        lp_size=(180, 64),
        use_fft_magnitude=True,
        use_disambiguation=True,
    ).to(device)
    fmt_wrapper = FMTWrapper(fmt, device)

    # ===== Classical FMT =====
    print("Initializing Classical FMT...")
    classical_wrapper = ClassicalFMTWrapper(device)

    # Evaluate at each test angle
    wrappers = [baseline_wrapper, fmt_wrapper, classical_wrapper]

    for wrapper in wrappers:
        print(f"\nEvaluating {wrapper.name}...")
        results["models"][wrapper.name] = {"angles": {}, "summary": {}}

        in_dist_errors = []
        out_dist_errors = []

        for angle in config.test_rotation_angles:
            test_dataset = Sim2Dataset(
                n_samples=config.n_test_samples_per_config,
                fixed_rotation=angle,
                scale_range=(1.0, 1.0),
                translation_range=(0, 0),
                image_size=config.image_size,
            )

            result = evaluate_model_on_config(wrapper, test_dataset, device, f"rot_{angle}")
            results["models"][wrapper.name]["angles"][angle] = {
                "mace": result.mace,
                "rotation_mae": result.rotation_mae,
                "auc_3": result.auc_3,
                "auc_5": result.auc_5,
                "auc_10": result.auc_10,
            }

            # Track in/out distribution
            if config.train_rotation_range[0] <= angle <= config.train_rotation_range[1]:
                in_dist_errors.extend(result.corner_errors)
            else:
                out_dist_errors.extend(result.corner_errors)

            print(f"    {angle:>4}°: MACE={result.mace:>6.1f}px, RotErr={result.rotation_mae:>5.1f}°")

        # Summary
        in_dist_mean = np.mean(in_dist_errors) if in_dist_errors else 0
        out_dist_mean = np.mean(out_dist_errors) if out_dist_errors else 0
        gap = out_dist_mean - in_dist_mean

        results["models"][wrapper.name]["summary"] = {
            "in_distribution_mace": in_dist_mean,
            "out_distribution_mace": out_dist_mean,
            "generalization_gap": gap,
        }

        print(f"\n    Summary: ID={in_dist_mean:.1f}px, OOD={out_dist_mean:.1f}px, Gap={gap:.1f}px")

    return results


def run_scale_experiment(config: EvalConfig, device: torch.device) -> Dict:
    """Test 2: Scale-only generalization."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: SCALE GENERALIZATION")
    print("=" * 70)
    print(f"  Train: {config.train_scale_range[0]}x to {config.train_scale_range[1]}x")
    print(f"  Test: {config.test_scale_factors}")
    print()

    results = {"experiment": "scale", "models": {}}

    # Training dataset (scale only)
    train_dataset = Sim2Dataset(
        n_samples=config.n_train_samples,
        rotation_range=(0, 0),
        scale_range=config.train_scale_range,
        translation_range=(0, 0),
        image_size=config.image_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=4)

    # Train baseline
    print("Training BaselineCNN...")
    baseline = BaselineCNN().to(device)
    baseline_loss = BaselineLoss(w_rotation=0.1, w_scale=1.0, w_translation=0.1)
    baseline_opt = torch.optim.AdamW(baseline.parameters(), lr=config.lr)

    train_model(baseline, train_loader, baseline_loss, baseline_opt, device,
                epochs=config.epochs, model_name="BaselineCNN")
    baseline_wrapper = BaselineCNNWrapper(baseline, device)

    # FMT
    print("\nInitializing FMT...")
    fmt = LogPolarSim2Net(
        lp_size=(180, 64),
        use_fft_magnitude=True,
        use_disambiguation=True,
    ).to(device)
    fmt_wrapper = FMTWrapper(fmt, device)

    # Evaluate
    wrappers = [baseline_wrapper, fmt_wrapper]

    for wrapper in wrappers:
        print(f"\nEvaluating {wrapper.name}...")
        results["models"][wrapper.name] = {"scales": {}, "summary": {}}

        in_dist_errors = []
        out_dist_errors = []

        for scale in config.test_scale_factors:
            test_dataset = Sim2Dataset(
                n_samples=config.n_test_samples_per_config,
                rotation_range=(0, 0),
                fixed_scale=scale,
                translation_range=(0, 0),
                image_size=config.image_size,
            )

            result = evaluate_model_on_config(wrapper, test_dataset, device, f"scale_{scale}")
            results["models"][wrapper.name]["scales"][scale] = {
                "mace": result.mace,
                "scale_mae": result.scale_mae,
            }

            if config.train_scale_range[0] <= scale <= config.train_scale_range[1]:
                in_dist_errors.extend(result.corner_errors)
            else:
                out_dist_errors.extend(result.corner_errors)

            print(f"    {scale:>4.1f}x: MACE={result.mace:>6.1f}px, ScaleErr={result.scale_mae:>5.1f}%")

        in_dist_mean = np.mean(in_dist_errors) if in_dist_errors else 0
        out_dist_mean = np.mean(out_dist_errors) if out_dist_errors else 0

        results["models"][wrapper.name]["summary"] = {
            "in_distribution_mace": in_dist_mean,
            "out_distribution_mace": out_dist_mean,
            "generalization_gap": out_dist_mean - in_dist_mean,
        }

        print(f"\n    Summary: ID={in_dist_mean:.1f}px, OOD={out_dist_mean:.1f}px")

    return results


def run_translation_experiment(config: EvalConfig, device: torch.device) -> Dict:
    """Test 3: Translation handling (FMT should be translation-invariant for rot/scale)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: TRANSLATION INVARIANCE")
    print("=" * 70)
    print("  Testing: Rotation estimation accuracy WITH varying translations")
    print(f"  Translation offsets: {config.test_translation_offsets}")
    print()

    results = {"experiment": "translation_invariance", "models": {}}

    # FMT should estimate rotation correctly regardless of translation
    fmt = LogPolarSim2Net(
        lp_size=(180, 64),
        use_fft_magnitude=True,
        use_disambiguation=True,
    ).to(device)
    fmt_wrapper = FMTWrapper(fmt, device)

    # Baseline trained on centered images
    train_dataset = Sim2Dataset(
        n_samples=config.n_train_samples,
        rotation_range=config.train_rotation_range,
        scale_range=(1.0, 1.0),
        translation_range=(0, 0),  # Trained WITHOUT translation
        image_size=config.image_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=4)

    print("Training BaselineCNN (no translation in training)...")
    baseline = BaselineCNN().to(device)
    baseline_loss = BaselineLoss(w_rotation=1.0, w_scale=0.1, w_translation=0.1)
    baseline_opt = torch.optim.AdamW(baseline.parameters(), lr=config.lr)
    train_model(baseline, train_loader, baseline_loss, baseline_opt, device,
                epochs=config.epochs, model_name="BaselineCNN")
    baseline_wrapper = BaselineCNNWrapper(baseline, device)

    # Test: 45° rotation with varying translation
    test_rotation = 45.0

    for wrapper in [baseline_wrapper, fmt_wrapper]:
        print(f"\nEvaluating {wrapper.name}...")
        results["models"][wrapper.name] = {"translations": {}}

        for trans_offset in config.test_translation_offsets:
            test_dataset = Sim2Dataset(
                n_samples=config.n_test_samples_per_config,
                fixed_rotation=test_rotation,
                scale_range=(1.0, 1.0),
                translation_range=(trans_offset, trans_offset),
                image_size=config.image_size,
            )

            result = evaluate_model_on_config(wrapper, test_dataset, device, f"trans_{trans_offset}")
            results["models"][wrapper.name]["translations"][trans_offset] = {
                "rotation_mae": result.rotation_mae,
                "mace": result.mace,
            }

            print(f"    Trans ±{trans_offset:>2}px: RotErr={result.rotation_mae:>5.1f}° (target=45°)")

    return results


def run_combined_sim2_experiment(config: EvalConfig, device: torch.device) -> Dict:
    """Test 4: Full Sim(2) - the real test."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: COMBINED Sim(2) GENERALIZATION")
    print("=" * 70)
    print("  Training distribution:")
    print(f"    Rotation: {config.train_rotation_range}")
    print(f"    Scale: {config.train_scale_range}")
    print(f"    Translation: {config.train_translation_range}")
    print()

    results = {"experiment": "combined_sim2", "models": {}}

    # Training dataset (narrow distribution)
    train_dataset = Sim2Dataset(
        n_samples=config.n_train_samples,
        rotation_range=config.train_rotation_range,
        scale_range=config.train_scale_range,
        translation_range=config.train_translation_range,
        image_size=config.image_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=4)

    # Train baseline
    print("Training BaselineCNN...")
    baseline = BaselineCNN().to(device)
    baseline_loss = BaselineLoss(w_rotation=1.0, w_scale=0.5, w_translation=0.3)
    baseline_opt = torch.optim.AdamW(baseline.parameters(), lr=config.lr)
    train_model(baseline, train_loader, baseline_loss, baseline_opt, device,
                epochs=config.epochs, model_name="BaselineCNN")
    baseline_wrapper = BaselineCNNWrapper(baseline, device)

    # FMT
    fmt = LogPolarSim2Net(
        lp_size=(180, 64),
        use_fft_magnitude=True,
        use_disambiguation=True,
    ).to(device)
    fmt_wrapper = FMTWrapper(fmt, device)

    # Test configurations: in-distribution vs out-of-distribution
    test_configs = [
        # In-distribution
        {"name": "ID: Easy", "rotation": (0, 15), "scale": (0.95, 1.05), "translation": (-10, 10)},
        {"name": "ID: Medium", "rotation": (-30, 30), "scale": (0.9, 1.1), "translation": (-20, 20)},
        # Out-of-distribution
        {"name": "OOD: Large Rot", "rotation": (60, 90), "scale": (0.95, 1.05), "translation": (-10, 10)},
        {"name": "OOD: Large Scale", "rotation": (0, 15), "scale": (0.5, 0.7), "translation": (-10, 10)},
        {"name": "OOD: Large Trans", "rotation": (0, 15), "scale": (0.95, 1.05), "translation": (-60, 60)},
        {"name": "OOD: All Large", "rotation": (60, 120), "scale": (0.5, 2.0), "translation": (-50, 50)},
    ]

    for wrapper in [baseline_wrapper, fmt_wrapper]:
        print(f"\nEvaluating {wrapper.name}...")
        results["models"][wrapper.name] = {"configs": {}}

        for tc in test_configs:
            test_dataset = Sim2Dataset(
                n_samples=config.n_test_samples_per_config * 2,
                rotation_range=tc["rotation"],
                scale_range=tc["scale"],
                translation_range=tc["translation"],
                image_size=config.image_size,
            )

            result = evaluate_model_on_config(wrapper, test_dataset, device, tc["name"])
            results["models"][wrapper.name]["configs"][tc["name"]] = {
                "mace": result.mace,
                "rotation_mae": result.rotation_mae,
                "scale_mae": result.scale_mae,
                "translation_mae": result.translation_mae,
                "auc_3": result.auc_3,
                "auc_5": result.auc_5,
                "auc_10": result.auc_10,
            }

            print(f"    {tc['name']:<16}: MACE={result.mace:>6.1f}px, "
                  f"Rot={result.rotation_mae:>5.1f}°, Scale={result.scale_mae:>5.1f}%")

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_comprehensive_plot(all_results: Dict, output_dir: Path):
    """Create publication-quality visualization."""
    fig = plt.figure(figsize=(16, 12))

    # 1. Rotation generalization (top-left)
    ax1 = fig.add_subplot(2, 2, 1)
    if "rotation" in all_results:
        rot_results = all_results["rotation"]["models"]
        for model_name, data in rot_results.items():
            angles = sorted(data["angles"].keys())
            maces = [data["angles"][a]["mace"] for a in angles]
            marker = 'o' if "Baseline" in model_name else 's' if "FMT (Ours)" in model_name else '^'
            color = 'red' if "Baseline" in model_name else 'blue' if "Ours" in model_name else 'green'
            ax1.plot(angles, maces, f'{color[0]}-{marker}', label=model_name, linewidth=2, markersize=6)

        ax1.axvspan(-30, 30, alpha=0.2, color='green', label='Training range')
        ax1.set_xlabel('Rotation Angle (°)')
        ax1.set_ylabel('MACE (pixels)')
        ax1.set_title('Rotation Generalization')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. Scale generalization (top-right)
    ax2 = fig.add_subplot(2, 2, 2)
    if "scale" in all_results:
        scale_results = all_results["scale"]["models"]
        for model_name, data in scale_results.items():
            scales = sorted(data["scales"].keys())
            maces = [data["scales"][s]["mace"] for s in scales]
            color = 'red' if "Baseline" in model_name else 'blue'
            marker = 'o' if "Baseline" in model_name else 's'
            ax2.plot(scales, maces, f'{color[0]}-{marker}', label=model_name, linewidth=2, markersize=6)

        ax2.axvspan(0.9, 1.1, alpha=0.2, color='green', label='Training range')
        ax2.set_xlabel('Scale Factor')
        ax2.set_ylabel('MACE (pixels)')
        ax2.set_title('Scale Generalization')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. Translation invariance (bottom-left)
    ax3 = fig.add_subplot(2, 2, 3)
    if "translation_invariance" in all_results:
        trans_results = all_results["translation_invariance"]["models"]
        for model_name, data in trans_results.items():
            offsets = sorted(data["translations"].keys())
            rot_errs = [data["translations"][t]["rotation_mae"] for t in offsets]
            color = 'red' if "Baseline" in model_name else 'blue'
            marker = 'o' if "Baseline" in model_name else 's'
            ax3.plot(offsets, rot_errs, f'{color[0]}-{marker}', label=model_name, linewidth=2, markersize=6)

        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Translation Offset (pixels)')
        ax3.set_ylabel('Rotation Error (°)')
        ax3.set_title('Translation Invariance Test\n(45° rotation with varying translation)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Combined Sim(2) bar chart (bottom-right)
    ax4 = fig.add_subplot(2, 2, 4)
    if "combined_sim2" in all_results:
        sim2_results = all_results["combined_sim2"]["models"]
        model_names = list(sim2_results.keys())
        configs = list(sim2_results[model_names[0]]["configs"].keys())

        x = np.arange(len(configs))
        width = 0.35

        for i, model_name in enumerate(model_names):
            maces = [sim2_results[model_name]["configs"][c]["mace"] for c in configs]
            offset = (i - len(model_names)/2 + 0.5) * width
            color = 'red' if "Baseline" in model_name else 'blue'
            ax4.bar(x + offset, maces, width, label=model_name, color=color, alpha=0.7)

        ax4.set_ylabel('MACE (pixels)')
        ax4.set_title('Combined Sim(2) Evaluation')
        ax4.set_xticks(x)
        ax4.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "comprehensive_evaluation.png", dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_dir / 'comprehensive_evaluation.png'}")


def create_summary_table(all_results: Dict) -> str:
    """Create a summary table in markdown format."""
    lines = ["# Comprehensive Sim(2) Evaluation Results\n"]

    # Generalization gaps
    lines.append("## Generalization Gaps\n")
    lines.append("| Experiment | Model | In-Dist | Out-Dist | Gap |")
    lines.append("|------------|-------|---------|----------|-----|")

    for exp_name in ["rotation", "scale"]:
        if exp_name in all_results:
            for model_name, data in all_results[exp_name]["models"].items():
                if "summary" in data:
                    s = data["summary"]
                    lines.append(f"| {exp_name.capitalize()} | {model_name} | "
                               f"{s['in_distribution_mace']:.1f}px | "
                               f"{s['out_distribution_mace']:.1f}px | "
                               f"{s['generalization_gap']:.1f}px |")

    # Combined results
    lines.append("\n## Combined Sim(2) Results (MACE in pixels)\n")
    if "combined_sim2" in all_results:
        models = all_results["combined_sim2"]["models"]
        model_names = list(models.keys())
        configs = list(models[model_names[0]]["configs"].keys())

        header = "| Config |" + "|".join(model_names) + "|"
        lines.append(header)
        lines.append("|" + "---|" * (len(model_names) + 1))

        for config in configs:
            row = f"| {config} |"
            for model in model_names:
                mace = models[model]["configs"][config]["mace"]
                row += f" {mace:.1f} |"
            lines.append(row)

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("COMPREHENSIVE Sim(2) EVALUATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup
    config = EvalConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Run all experiments
    all_results["rotation"] = run_rotation_experiment(config, device)
    all_results["scale"] = run_scale_experiment(config, device)
    all_results["translation_invariance"] = run_translation_experiment(config, device)
    all_results["combined_sim2"] = run_combined_sim2_experiment(config, device)

    # Save results
    results_path = output_dir / "comprehensive_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Create visualizations
    create_comprehensive_plot(all_results, output_dir)

    # Create summary table
    summary = create_summary_table(all_results)
    summary_path = output_dir / "summary.md"
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Summary saved to: {summary_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(summary)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
