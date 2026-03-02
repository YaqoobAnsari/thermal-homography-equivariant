#!/usr/bin/env python3
"""
Extended Equivariant Networks Comparison for Rotation Generalization

This script compares multiple rotation-equivariant architectures against SIGMA:

1. BaselineCNN - Non-equivariant (control)
2. E2CNN_N8 - Discrete C8 rotation equivariance
3. E2CNN_N16 - Discrete C16 rotation equivariance
4. E2CNN_N32 - Discrete C32 rotation equivariance
5. E2CNN_SO2 - Continuous SO(2) rotation equivariance (irreps)
6. E2CNN_D8 - Dihedral D8 (rotation + flip) equivariance
7. SteerableCNN - Steerable filters with irreducible representations
8. SIGMA - Fourier-Mellin continuous equivariance

Key insight: Discrete rotation equivariance (C_N) does NOT generalize to arbitrary rotations
because it only respects rotations that are multiples of 360/N degrees.
Only continuous equivariance (SO(2) or Fourier-Mellin) can generalize to all rotations.

References:
- Cohen & Welling, "Group Equivariant Convolutional Networks", ICML 2016
- Weiler & Cesa, "General E(2)-Equivariant Steerable CNNs", NeurIPS 2019
- Worrall et al., "Harmonic Networks: Deep Translation and Rotation Equivariance", CVPR 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import e2cnn
try:
    from e2cnn import gspaces
    from e2cnn import nn as e2nn
    E2CNN_AVAILABLE = True
except ImportError:
    E2CNN_AVAILABLE = False
    print("WARNING: e2cnn not installed. Install with: pip install e2cnn")

from src.data.synthetic_generator import SyntheticThermalGenerator
from src.models.log_polar_sim2_net import LogPolarSim2Net


# =============================================================================
# EQUIVARIANT ENCODERS
# =============================================================================

class E2CNNEncoder_Cyclic(nn.Module):
    """
    Rotation-equivariant encoder using cyclic group C_N.
    Uses regular representation (feature maps transform under group action).
    """

    def __init__(self, N: int = 8, in_channels: int = 2, hidden_dim: int = 64):
        super().__init__()

        if not E2CNN_AVAILABLE:
            raise ImportError("e2cnn is required")

        self.N = N
        self.gspace = gspaces.Rot2dOnR2(N=N)

        self.in_type = e2nn.FieldType(self.gspace, in_channels * [self.gspace.trivial_repr])

        hidden_type1 = e2nn.FieldType(self.gspace, hidden_dim * [self.gspace.regular_repr])
        hidden_type2 = e2nn.FieldType(self.gspace, hidden_dim * [self.gspace.regular_repr])
        hidden_type3 = e2nn.FieldType(self.gspace, hidden_dim * [self.gspace.regular_repr])

        self.encoder = e2nn.SequentialModule(
            e2nn.R2Conv(self.in_type, hidden_type1, kernel_size=7, padding=3, bias=False),
            e2nn.InnerBatchNorm(hidden_type1),
            e2nn.ReLU(hidden_type1, inplace=True),
            e2nn.PointwiseMaxPool(hidden_type1, 2),

            e2nn.R2Conv(hidden_type1, hidden_type2, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(hidden_type2),
            e2nn.ReLU(hidden_type2, inplace=True),
            e2nn.PointwiseMaxPool(hidden_type2, 2),

            e2nn.R2Conv(hidden_type2, hidden_type3, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(hidden_type3),
            e2nn.ReLU(hidden_type3, inplace=True),
            e2nn.PointwiseMaxPool(hidden_type3, 2),
        )

        self.out_type = hidden_type3
        self.feature_dim = hidden_dim * N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = e2nn.GeometricTensor(x, self.in_type)
        out = self.encoder(x)
        return out.tensor


class E2CNNEncoder_Dihedral(nn.Module):
    """
    Rotation + Flip equivariant encoder using dihedral group D_N.
    Includes reflection symmetry in addition to rotations.
    """

    def __init__(self, N: int = 8, in_channels: int = 2, hidden_dim: int = 64):
        super().__init__()

        if not E2CNN_AVAILABLE:
            raise ImportError("e2cnn is required")

        self.N = N
        # Dihedral group: rotations + reflections
        self.gspace = gspaces.FlipRot2dOnR2(N=N)

        self.in_type = e2nn.FieldType(self.gspace, in_channels * [self.gspace.trivial_repr])

        hidden_type1 = e2nn.FieldType(self.gspace, hidden_dim * [self.gspace.regular_repr])
        hidden_type2 = e2nn.FieldType(self.gspace, hidden_dim * [self.gspace.regular_repr])
        hidden_type3 = e2nn.FieldType(self.gspace, hidden_dim * [self.gspace.regular_repr])

        self.encoder = e2nn.SequentialModule(
            e2nn.R2Conv(self.in_type, hidden_type1, kernel_size=7, padding=3, bias=False),
            e2nn.InnerBatchNorm(hidden_type1),
            e2nn.ReLU(hidden_type1, inplace=True),
            e2nn.PointwiseMaxPool(hidden_type1, 2),

            e2nn.R2Conv(hidden_type1, hidden_type2, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(hidden_type2),
            e2nn.ReLU(hidden_type2, inplace=True),
            e2nn.PointwiseMaxPool(hidden_type2, 2),

            e2nn.R2Conv(hidden_type2, hidden_type3, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(hidden_type3),
            e2nn.ReLU(hidden_type3, inplace=True),
            e2nn.PointwiseMaxPool(hidden_type3, 2),
        )

        self.out_type = hidden_type3
        # D_N has 2N elements (N rotations x 2 for flip)
        self.feature_dim = hidden_dim * 2 * N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = e2nn.GeometricTensor(x, self.in_type)
        out = self.encoder(x)
        return out.tensor


class E2CNNEncoder_SO2(nn.Module):
    """
    Continuous SO(2) rotation-equivariant encoder using irreducible representations.

    This uses the continuous rotation group SO(2) with a finite set of irreps
    (frequency bands) to approximate continuous equivariance.

    Key difference from C_N:
    - C_N: Only equivariant to rotations of 360/N degrees
    - SO(2) with irreps: Equivariant to any rotation (within the frequency band limit)
    """

    def __init__(self, L_max: int = 4, in_channels: int = 2, hidden_dim: int = 64):
        """
        Args:
            L_max: Maximum frequency (number of irreps). Higher = more rotation accuracy.
            in_channels: Number of input channels
            hidden_dim: Hidden dimension
        """
        super().__init__()

        if not E2CNN_AVAILABLE:
            raise ImportError("e2cnn is required")

        self.L_max = L_max

        # Continuous rotation group SO(2)
        # N=-1 means continuous (uses irreps instead of regular repr)
        self.gspace = gspaces.Rot2dOnR2(N=-1, maximum_frequency=L_max)

        # Input: trivial representation (scalar field)
        self.in_type = e2nn.FieldType(self.gspace, in_channels * [self.gspace.trivial_repr])

        # Build irrep list: include frequencies 0, 1, ..., L_max
        # Each frequency l contributes a 2D irrep (except l=0 which is 1D)
        irreps = [self.gspace.irrep(l) for l in range(L_max + 1)]

        hidden_type1 = e2nn.FieldType(self.gspace, hidden_dim * irreps)
        hidden_type2 = e2nn.FieldType(self.gspace, hidden_dim * irreps)
        hidden_type3 = e2nn.FieldType(self.gspace, hidden_dim * irreps)

        self.encoder = e2nn.SequentialModule(
            e2nn.R2Conv(self.in_type, hidden_type1, kernel_size=7, padding=3, bias=False),
            e2nn.InnerBatchNorm(hidden_type1),
            e2nn.NormNonLinearity(hidden_type1),  # Use norm nonlinearity for irreps
            e2nn.PointwiseMaxPool(hidden_type1, 2),

            e2nn.R2Conv(hidden_type1, hidden_type2, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(hidden_type2),
            e2nn.NormNonLinearity(hidden_type2),
            e2nn.PointwiseMaxPool(hidden_type2, 2),

            e2nn.R2Conv(hidden_type2, hidden_type3, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(hidden_type3),
            e2nn.NormNonLinearity(hidden_type3),
            e2nn.PointwiseMaxPool(hidden_type3, 2),
        )

        self.out_type = hidden_type3
        # Feature dim: 1 + 2*L_max (1 for freq 0, 2 for each other freq)
        self.feature_dim = hidden_dim * (1 + 2 * L_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = e2nn.GeometricTensor(x, self.in_type)
        out = self.encoder(x)
        return out.tensor


class SteerableCNNEncoder(nn.Module):
    """
    Steerable CNN encoder using circular harmonics (Worrall et al., 2017 style).

    Uses irreducible representations for rotation equivariance with steerable filters.
    This is similar to Harmonic Networks but using the e2cnn framework.
    """

    def __init__(self, L_max: int = 3, in_channels: int = 2, hidden_dim: int = 64):
        super().__init__()

        if not E2CNN_AVAILABLE:
            raise ImportError("e2cnn is required")

        self.L_max = L_max

        # Use continuous rotation with irreps
        self.gspace = gspaces.Rot2dOnR2(N=-1, maximum_frequency=L_max)

        self.in_type = e2nn.FieldType(self.gspace, in_channels * [self.gspace.trivial_repr])

        # For steerable CNN, we use specific irreps at each layer
        # Lower frequencies at early layers, higher at later layers
        irreps_l1 = [self.gspace.irrep(l) for l in range(min(2, L_max) + 1)]
        irreps_l2 = [self.gspace.irrep(l) for l in range(min(3, L_max) + 1)]
        irreps_l3 = [self.gspace.irrep(l) for l in range(L_max + 1)]

        hidden_type1 = e2nn.FieldType(self.gspace, hidden_dim * irreps_l1)
        hidden_type2 = e2nn.FieldType(self.gspace, hidden_dim * irreps_l2)
        hidden_type3 = e2nn.FieldType(self.gspace, hidden_dim * irreps_l3)

        self.encoder = e2nn.SequentialModule(
            e2nn.R2Conv(self.in_type, hidden_type1, kernel_size=7, padding=3, bias=False),
            e2nn.InnerBatchNorm(hidden_type1),
            e2nn.NormNonLinearity(hidden_type1),
            e2nn.PointwiseMaxPool(hidden_type1, 2),

            e2nn.R2Conv(hidden_type1, hidden_type2, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(hidden_type2),
            e2nn.NormNonLinearity(hidden_type2),
            e2nn.PointwiseMaxPool(hidden_type2, 2),

            e2nn.R2Conv(hidden_type2, hidden_type3, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(hidden_type3),
            e2nn.NormNonLinearity(hidden_type3),
            e2nn.PointwiseMaxPool(hidden_type3, 2),
        )

        self.out_type = hidden_type3
        self.feature_dim = hidden_dim * (1 + 2 * L_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = e2nn.GeometricTensor(x, self.in_type)
        out = self.encoder(x)
        return out.tensor


# =============================================================================
# FULL NETWORKS
# =============================================================================

class EquivariantHomographyNet(nn.Module):
    """
    Generic wrapper for equivariant encoders to regress Sim(2) parameters.
    """

    def __init__(self, encoder, name="EquivariantNet"):
        super().__init__()
        self.name = name
        self.encoder = encoder

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)  # (theta, log_scale, tx, ty)
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> dict:
        x = torch.cat([source, target], dim=1)
        features = self.encoder(x)
        params = self.regressor(features)

        theta = params[:, 0] * 180
        log_scale = params[:, 1]
        tx = params[:, 2] * 128
        ty = params[:, 3] * 128

        return {
            'rotation': theta,
            'scale': torch.exp(log_scale),
            'translation': torch.stack([tx, ty], dim=1)
        }


class BaselineCNN(nn.Module):
    """Simple CNN baseline (non-equivariant) for comparison."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 7, padding=3),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> dict:
        x = torch.cat([source, target], dim=1)
        features = self.encoder(x)
        params = self.regressor(features)

        theta = params[:, 0] * 180
        log_scale = params[:, 1]
        tx = params[:, 2] * 128
        ty = params[:, 3] * 128

        return {
            'rotation': theta,
            'scale': torch.exp(log_scale),
            'translation': torch.stack([tx, ty], dim=1)
        }


# =============================================================================
# MACE COMPUTATION
# =============================================================================

def compute_mace(pred_rotation, pred_scale, pred_tx, pred_ty,
                 gt_rotation, gt_scale, gt_tx, gt_ty, image_size=256):
    """
    Compute Mean Average Corner Error (MACE) in pixels.

    Args:
        pred_rotation: Predicted rotation in degrees
        pred_scale: Predicted scale factor
        pred_tx, pred_ty: Predicted translation in pixels
        gt_rotation: Ground truth rotation in degrees
        gt_scale: Ground truth scale factor
        gt_tx, gt_ty: Ground truth translation in pixels
        image_size: Image size (assumed square)

    Returns:
        MACE in pixels
    """
    # Define corners (centered coordinates)
    half = image_size / 2
    corners = np.array([
        [-half, -half],
        [half, -half],
        [half, half],
        [-half, half]
    ])

    def apply_sim2(corners, rotation_deg, scale, tx, ty):
        """Apply Sim(2) transformation to corners."""
        theta = np.radians(rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        transformed = scale * (corners @ R.T) + np.array([tx, ty])
        return transformed

    # Apply transformations
    pred_corners = apply_sim2(corners, pred_rotation, pred_scale, pred_tx, pred_ty)
    gt_corners = apply_sim2(corners, gt_rotation, gt_scale, gt_tx, gt_ty)

    # Compute MACE
    mace = np.mean(np.linalg.norm(pred_corners - gt_corners, axis=1))
    return mace


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_model(model, train_loader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_rot_error = 0
    n_batches = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
        source = batch['source'].to(device)
        target = batch['target'].to(device)
        gt_rotation = batch['rotation'].to(device)
        gt_scale = batch['scale'].to(device)

        optimizer.zero_grad()

        output = model(source, target)

        rot_loss = 1 - torch.cos((output['rotation'] - gt_rotation) * np.pi / 180)
        scale_loss = torch.abs(torch.log(output['scale'] / gt_scale))

        loss = rot_loss.mean() + scale_loss.mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        rot_error = torch.abs(output['rotation'] - gt_rotation)
        rot_error = torch.min(rot_error, 360 - rot_error)
        total_rot_error += rot_error.mean().item()

        n_batches += 1

    return total_loss / n_batches, total_rot_error / n_batches


def evaluate_model(model, test_angles, pattern_types, device, n_samples=100):
    """Evaluate model at specific test angles with rotation error AND MACE."""
    model.eval()

    results = {}

    with torch.no_grad():
        for angle in test_angles:
            rot_errors = []
            mace_errors = []

            for sample_idx in range(n_samples):
                pattern = pattern_types[sample_idx % len(pattern_types)]
                gen = SyntheticThermalGenerator(
                    n_samples=1,
                    image_size=(256, 256),
                    pattern_type=pattern,
                    rotation_range=(angle, angle),
                    scale_range=(1.0, 1.0),
                    translation_range=(0, 0),
                    seed=hash((angle, sample_idx)) % (2**32)
                )
                sample = gen[0]

                source = sample['image_src'].unsqueeze(0).to(device)
                target = sample['image_tgt'].unsqueeze(0).to(device)
                gt_rotation = sample['rotation'].item() * 180 / np.pi
                gt_scale = sample['scale'].item()

                output = model(source, target)
                pred_rotation = output['rotation'].item()
                pred_scale = output['scale'].item()

                # Get translation if available
                if 'translation' in output:
                    pred_tx = output['translation'][0, 0].item()
                    pred_ty = output['translation'][0, 1].item()
                else:
                    pred_tx, pred_ty = 0.0, 0.0

                # Rotation error
                rot_error = abs(pred_rotation - gt_rotation)
                if rot_error > 180:
                    rot_error = 360 - rot_error
                rot_errors.append(rot_error)

                # MACE (ground truth translation is 0 for this test)
                mace = compute_mace(
                    pred_rotation, pred_scale, pred_tx, pred_ty,
                    gt_rotation, gt_scale, 0.0, 0.0,
                    image_size=256
                )
                mace_errors.append(mace)

            results[angle] = {
                'rotation_mean': float(np.mean(rot_errors)),
                'rotation_std': float(np.std(rot_errors)),
                'rotation_median': float(np.median(rot_errors)),
                'mace_mean': float(np.mean(mace_errors)),
                'mace_std': float(np.std(mace_errors)),
                'mace_median': float(np.median(mace_errors)),
                'pass_3deg': float(np.mean([e < 3 for e in rot_errors]) * 100),
                'pass_5deg': float(np.mean([e < 5 for e in rot_errors]) * 100),
                'pass_3px': float(np.mean([e < 3 for e in mace_errors]) * 100),
                'pass_5px': float(np.mean([e < 5 for e in mace_errors]) * 100),
            }

    return results


class TrainingDataset(torch.utils.data.Dataset):
    """Dataset wrapper for training."""

    def __init__(self, rotation_range, n_samples, pattern_types=['arrow', 'asymmetric', 'L_shape', 'T_shape']):
        self.rotation_range = rotation_range
        self.n_samples = n_samples
        self.pattern_types = pattern_types

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        pattern = self.pattern_types[idx % len(self.pattern_types)]
        gen = SyntheticThermalGenerator(
            n_samples=1,
            image_size=(256, 256),
            pattern_type=pattern,
            rotation_range=self.rotation_range,
            scale_range=(0.95, 1.05),
            translation_range=(0, 0),
            seed=idx
        )
        sample = gen[0]
        return {
            'source': sample['image_src'],
            'target': sample['image_tgt'],
            'rotation': sample['rotation'] * 180 / np.pi,
            'scale': sample['scale']
        }


def create_model(model_type: str, device: torch.device):
    """Factory function to create models."""

    if model_type == 'BaselineCNN':
        return BaselineCNN(hidden_dim=64).to(device)

    elif model_type.startswith('E2CNN_N'):
        # E2CNN_N8 -> 8, E2CNN_N16 -> 16, etc.
        N = int(model_type.split('_N')[1])
        encoder = E2CNNEncoder_Cyclic(N=N, in_channels=2, hidden_dim=32)
        return EquivariantHomographyNet(encoder, name=model_type).to(device)

    elif model_type.startswith('E2CNN_D'):
        # E2CNN_D8 -> 8
        N = int(model_type.split('_D')[1])
        encoder = E2CNNEncoder_Dihedral(N=N, in_channels=2, hidden_dim=32)
        return EquivariantHomographyNet(encoder, name=model_type).to(device)

    elif model_type == 'E2CNN_SO2':
        encoder = E2CNNEncoder_SO2(L_max=4, in_channels=2, hidden_dim=32)
        return EquivariantHomographyNet(encoder, name=model_type).to(device)

    elif model_type == 'E2CNN_SO2_L8':
        encoder = E2CNNEncoder_SO2(L_max=8, in_channels=2, hidden_dim=32)
        return EquivariantHomographyNet(encoder, name=model_type).to(device)

    elif model_type == 'SteerableCNN':
        encoder = SteerableCNNEncoder(L_max=3, in_channels=2, hidden_dim=32)
        return EquivariantHomographyNet(encoder, name=model_type).to(device)

    elif model_type == 'SteerableCNN_L6':
        encoder = SteerableCNNEncoder(L_max=6, in_channels=2, hidden_dim=32)
        return EquivariantHomographyNet(encoder, name=model_type).to(device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_experiment(args):
    """Run the full comparison experiment."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    pattern_types = ['arrow', 'asymmetric', 'L_shape', 'T_shape']
    test_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    train_rotation_range = (-30, 30)

    results = {
        'config': {
            'train_rotation_range': train_rotation_range,
            'test_angles': test_angles,
            'train_samples': args.train_samples,
            'train_epochs': args.epochs,
            'batch_size': args.batch_size,
            'device': str(device)
        },
        'models': {}
    }

    # Define models to test
    model_types = [
        'BaselineCNN',           # Non-equivariant baseline
        'E2CNN_N8',              # Discrete C8 (45° steps)
        'E2CNN_N16',             # Discrete C16 (22.5° steps)
        'E2CNN_N32',             # Discrete C32 (11.25° steps)
        'E2CNN_D8',              # Dihedral D8 (rotation + flip)
        'E2CNN_SO2',             # Continuous SO(2) with L_max=4
        'E2CNN_SO2_L8',          # Continuous SO(2) with L_max=8
        'SteerableCNN',          # Steerable with L_max=3
        'SteerableCNN_L6',       # Steerable with L_max=6
    ]

    # Filter based on availability
    if not E2CNN_AVAILABLE:
        model_types = ['BaselineCNN']
        print("WARNING: Only BaselineCNN will be tested (e2cnn not available)")

    # Train and evaluate each model
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Model: {model_type}")
        print(f"{'='*60}")

        try:
            model = create_model(model_type, device)
        except Exception as e:
            print(f"  Failed to create model: {e}")
            continue

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

        # Create dataset and loader
        train_dataset = TrainingDataset(train_rotation_range, args.train_samples, pattern_types)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        # Training loop
        best_error = float('inf')
        for epoch in range(1, args.epochs + 1):
            try:
                loss, rot_error = train_model(model, train_loader, optimizer, device, epoch)
                scheduler.step()

                if epoch % 5 == 0 or epoch == args.epochs:
                    print(f"  Epoch {epoch}: loss={loss:.4f}, rot_error={rot_error:.2f}°")

                if rot_error < best_error:
                    best_error = rot_error
            except Exception as e:
                print(f"  Training failed at epoch {epoch}: {e}")
                break

        print(f"  Best training error: {best_error:.2f}°")

        # Evaluate
        print(f"  Evaluating on test angles...")
        try:
            eval_results = evaluate_model(model, test_angles, pattern_types, device, args.test_samples)
        except Exception as e:
            print(f"  Evaluation failed: {e}")
            continue

        # Compute summary (rotation)
        in_dist_rot = [eval_results[a]['rotation_mean'] for a in [0, 15, 30]]
        out_dist_rot = [eval_results[a]['rotation_mean'] for a in test_angles if a > 30]
        in_dist_rot_mean = np.mean(in_dist_rot)
        out_dist_rot_mean = np.mean(out_dist_rot)
        rot_gap = out_dist_rot_mean - in_dist_rot_mean
        overall_rot_mean = np.mean([eval_results[a]['rotation_mean'] for a in test_angles])

        # Compute summary (MACE)
        in_dist_mace = [eval_results[a]['mace_mean'] for a in [0, 15, 30]]
        out_dist_mace = [eval_results[a]['mace_mean'] for a in test_angles if a > 30]
        in_dist_mace_mean = np.mean(in_dist_mace)
        out_dist_mace_mean = np.mean(out_dist_mace)
        mace_gap = out_dist_mace_mean - in_dist_mace_mean
        overall_mace_mean = np.mean([eval_results[a]['mace_mean'] for a in test_angles])

        results['models'][model_type] = {
            'per_angle': eval_results,
            'rotation_in_dist': float(in_dist_rot_mean),
            'rotation_out_dist': float(out_dist_rot_mean),
            'rotation_gap': float(rot_gap),
            'rotation_overall': float(overall_rot_mean),
            'mace_in_dist': float(in_dist_mace_mean),
            'mace_out_dist': float(out_dist_mace_mean),
            'mace_gap': float(mace_gap),
            'mace_overall': float(overall_mace_mean),
            'n_params': n_params
        }

        print(f"  Rotation - In-dist: {in_dist_rot_mean:.2f}°, Out-dist: {out_dist_rot_mean:.2f}°, Gap: {rot_gap:.2f}°")
        print(f"  MACE     - In-dist: {in_dist_mace_mean:.1f}px, Out-dist: {out_dist_mace_mean:.1f}px, Gap: {mace_gap:.1f}px")

    # Evaluate SIGMA (no training needed)
    print(f"\n{'='*60}")
    print("Evaluating SIGMA (Fourier-Mellin)")
    print(f"{'='*60}")

    sigma = LogPolarSim2Net(
        lp_size=(180, 64),
        r_min=0.05,
        r_max=0.9,
        use_fft_magnitude=True,
        use_disambiguation=True
    ).to(device)
    sigma.eval()

    n_params_sigma = sum(p.numel() for p in sigma.parameters())
    print(f"  Parameters: {n_params_sigma:,}")

    sigma_results = {}
    with torch.no_grad():
        for angle in tqdm(test_angles, desc="SIGMA evaluation"):
            rot_errors = []
            mace_errors = []
            for sample_idx in range(args.test_samples):
                pattern = pattern_types[sample_idx % len(pattern_types)]
                gen = SyntheticThermalGenerator(
                    n_samples=1,
                    image_size=(256, 256),
                    pattern_type=pattern,
                    rotation_range=(angle, angle),
                    scale_range=(1.0, 1.0),
                    translation_range=(0, 0),
                    seed=hash(('sigma', angle, sample_idx)) % (2**32)
                )
                sample = gen[0]

                source = sample['image_src'].unsqueeze(0).to(device)
                target = sample['image_tgt'].unsqueeze(0).to(device)
                gt_rotation = sample['rotation'].item() * 180 / np.pi
                gt_scale = sample['scale'].item()

                output = sigma(source, target)
                pred_rotation = output['rotation_deg'].item()
                pred_scale = output['scale'].item()

                # Translation from SIGMA
                if 'translation' in output:
                    pred_tx = output['translation'][0, 0].item()
                    pred_ty = output['translation'][0, 1].item()
                else:
                    pred_tx, pred_ty = 0.0, 0.0

                # Rotation error
                rot_error = abs(pred_rotation - gt_rotation)
                if rot_error > 180:
                    rot_error = 360 - rot_error
                rot_errors.append(rot_error)

                # MACE
                mace = compute_mace(
                    pred_rotation, pred_scale, pred_tx, pred_ty,
                    gt_rotation, gt_scale, 0.0, 0.0,
                    image_size=256
                )
                mace_errors.append(mace)

            sigma_results[angle] = {
                'rotation_mean': float(np.mean(rot_errors)),
                'rotation_std': float(np.std(rot_errors)),
                'rotation_median': float(np.median(rot_errors)),
                'mace_mean': float(np.mean(mace_errors)),
                'mace_std': float(np.std(mace_errors)),
                'mace_median': float(np.median(mace_errors)),
                'pass_3deg': float(np.mean([e < 3 for e in rot_errors]) * 100),
                'pass_5deg': float(np.mean([e < 5 for e in rot_errors]) * 100),
                'pass_3px': float(np.mean([e < 3 for e in mace_errors]) * 100),
                'pass_5px': float(np.mean([e < 5 for e in mace_errors]) * 100),
            }

    # Compute summary (rotation)
    in_dist_rot = [sigma_results[a]['rotation_mean'] for a in [0, 15, 30]]
    out_dist_rot = [sigma_results[a]['rotation_mean'] for a in test_angles if a > 30]
    in_dist_rot_mean = np.mean(in_dist_rot)
    out_dist_rot_mean = np.mean(out_dist_rot)
    rot_gap = out_dist_rot_mean - in_dist_rot_mean
    overall_rot_mean = np.mean([sigma_results[a]['rotation_mean'] for a in test_angles])

    # Compute summary (MACE)
    in_dist_mace = [sigma_results[a]['mace_mean'] for a in [0, 15, 30]]
    out_dist_mace = [sigma_results[a]['mace_mean'] for a in test_angles if a > 30]
    in_dist_mace_mean = np.mean(in_dist_mace)
    out_dist_mace_mean = np.mean(out_dist_mace)
    mace_gap = out_dist_mace_mean - in_dist_mace_mean
    overall_mace_mean = np.mean([sigma_results[a]['mace_mean'] for a in test_angles])

    results['models']['SIGMA'] = {
        'per_angle': sigma_results,
        'rotation_in_dist': float(in_dist_rot_mean),
        'rotation_out_dist': float(out_dist_rot_mean),
        'rotation_gap': float(rot_gap),
        'rotation_overall': float(overall_rot_mean),
        'mace_in_dist': float(in_dist_mace_mean),
        'mace_out_dist': float(out_dist_mace_mean),
        'mace_gap': float(mace_gap),
        'mace_overall': float(overall_mace_mean),
        'n_params': n_params_sigma
    }

    print(f"  Rotation - In-dist: {in_dist_rot_mean:.2f}°, Out-dist: {out_dist_rot_mean:.2f}°, Gap: {rot_gap:.2f}°")
    print(f"  MACE     - In-dist: {in_dist_mace_mean:.1f}px, Out-dist: {out_dist_mace_mean:.1f}px, Gap: {mace_gap:.1f}px")

    # Print final comparison - ROTATION
    print(f"\n{'='*90}")
    print("FINAL COMPARISON - ROTATION ERROR (degrees)")
    print(f"{'='*90}")
    print(f"{'Model':<20} {'Params':>10} {'In-Dist':>12} {'Out-Dist':>12} {'Gap':>12} {'Overall':>12}")
    print("-" * 90)

    # Sort by rotation gap
    sorted_models = sorted(results['models'].items(), key=lambda x: x[1]['rotation_gap'])

    for model_name, m in sorted_models:
        params_str = f"{m.get('n_params', 0) / 1000:.0f}K"
        print(f"{model_name:<20} {params_str:>10} {m['rotation_in_dist']:>12.2f}° {m['rotation_out_dist']:>12.2f}° "
              f"{m['rotation_gap']:>12.2f}° {m['rotation_overall']:>12.2f}°")

    # Print final comparison - MACE
    print(f"\n{'='*90}")
    print("FINAL COMPARISON - MACE (pixels)")
    print(f"{'='*90}")
    print(f"{'Model':<20} {'Params':>10} {'In-Dist':>12} {'Out-Dist':>12} {'Gap':>12} {'Overall':>12}")
    print("-" * 90)

    # Sort by MACE gap
    sorted_models = sorted(results['models'].items(), key=lambda x: x[1]['mace_gap'])

    for model_name, m in sorted_models:
        params_str = f"{m.get('n_params', 0) / 1000:.0f}K"
        print(f"{model_name:<20} {params_str:>10} {m['mace_in_dist']:>12.1f}px {m['mace_out_dist']:>12.1f}px "
              f"{m['mace_gap']:>12.1f}px {m['mace_overall']:>12.1f}px")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results['timestamp'] = datetime.now().isoformat()

    output_file = output_dir / 'extended_equivariant_comparison.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Extended Equivariant Networks Comparison')
    parser.add_argument('--train_samples', type=int, default=5000,
                        help='Number of training samples')
    parser.add_argument('--test_samples', type=int, default=100,
                        help='Number of test samples per angle')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='outputs/e2cnn_baseline',
                        help='Output directory')

    args = parser.parse_args()

    print("="*70)
    print("Extended Equivariant Networks Comparison Experiment")
    print("="*70)
    print(f"Train samples: {args.train_samples}")
    print(f"Test samples per angle: {args.test_samples}")
    print(f"Epochs: {args.epochs}")
    print(f"e2cnn available: {E2CNN_AVAILABLE}")
    print()
    print("Models to test:")
    print("  - BaselineCNN: Non-equivariant baseline")
    print("  - E2CNN_N8/N16/N32: Discrete cyclic group C_N")
    print("  - E2CNN_D8: Dihedral group (rotation + flip)")
    print("  - E2CNN_SO2/SO2_L8: Continuous SO(2) with irreps")
    print("  - SteerableCNN/SteerableCNN_L6: Steerable filters")
    print("  - SIGMA: Fourier-Mellin continuous equivariance")

    run_experiment(args)


if __name__ == '__main__':
    main()
