#!/usr/bin/env python3
"""
E2-CNN Baseline for Rotation-Equivariant Homography Estimation

This implements a rotation-equivariant CNN using the e2cnn library
to compare against SIGMA's Fourier-based equivariance.

Key difference:
- E2-CNN: Discrete rotation equivariance (N=8 or N=16)
- SIGMA: Continuous rotation equivariance via Fourier-Mellin

Expected outcome:
- E2-CNN should generalize better than vanilla CNN
- SIGMA should still outperform because of continuous equivariance

Reference:
- Cohen & Welling, "Group Equivariant Convolutional Networks", ICML 2016
- Weiler & Cesa, "General E(2)-Equivariant Steerable CNNs", NeurIPS 2019
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


class E2CNNEncoder(nn.Module):
    """
    Rotation-equivariant encoder using E2-CNN.

    Uses the cyclic group C_N for discrete rotation equivariance.
    """

    def __init__(self, N: int = 8, in_channels: int = 2, hidden_dim: int = 64):
        """
        Args:
            N: Order of cyclic group (number of discrete rotations)
            in_channels: Number of input channels (2 for source+target)
            hidden_dim: Hidden dimension
        """
        super().__init__()

        if not E2CNN_AVAILABLE:
            raise ImportError("e2cnn is required. Install with: pip install e2cnn")

        self.N = N

        # Define the group space: rotations by multiples of 2π/N
        self.gspace = gspaces.Rot2dOnR2(N=N)

        # Input type: scalar field (trivial representation)
        self.in_type = e2nn.FieldType(self.gspace, in_channels * [self.gspace.trivial_repr])

        # Hidden types: regular representation (transforms under rotations)
        hidden_type1 = e2nn.FieldType(self.gspace, hidden_dim * [self.gspace.regular_repr])
        hidden_type2 = e2nn.FieldType(self.gspace, hidden_dim * [self.gspace.regular_repr])
        hidden_type3 = e2nn.FieldType(self.gspace, hidden_dim * [self.gspace.regular_repr])

        # Build equivariant encoder
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

        # Output type after encoder
        self.out_type = hidden_type3

        # Compute output size
        # After 3x max pool of 2: 256 -> 128 -> 64 -> 32
        self.spatial_size = 32
        self.feature_dim = hidden_dim * N  # regular repr has N dimensions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Equivariant features [B, feature_dim, H', W']
        """
        # Wrap input as geometric tensor
        x = e2nn.GeometricTensor(x, self.in_type)

        # Apply equivariant encoder
        out = self.encoder(x)

        # Return tensor
        return out.tensor


class E2CNNHomographyNet(nn.Module):
    """
    E2-CNN based network for Sim(2) parameter estimation.

    Architecture:
    1. Concatenate source and target images
    2. Apply rotation-equivariant encoder
    3. Pool to rotation-invariant features
    4. Regress (θ, s, tx, ty)
    """

    def __init__(self, N: int = 8, hidden_dim: int = 64):
        """
        Args:
            N: Order of cyclic group
            hidden_dim: Hidden dimension for encoder
        """
        super().__init__()

        self.N = N

        # Equivariant encoder
        self.encoder = E2CNNEncoder(N=N, in_channels=2, hidden_dim=hidden_dim)

        # Invariant pooling: average over rotations to get rotation-invariant features
        # This is key - we need invariant features to regress absolute rotation
        feature_dim = hidden_dim  # After averaging over N rotation channels

        # Regression head
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.encoder.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)  # (theta, log_scale, tx, ty)
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Args:
            source: Source image [B, 1, H, W]
            target: Target image [B, 1, H, W]

        Returns:
            Dictionary with 'rotation', 'scale', 'translation'
        """
        # Concatenate source and target
        x = torch.cat([source, target], dim=1)  # [B, 2, H, W]

        # Equivariant features
        features = self.encoder(x)  # [B, hidden_dim * N, H', W']

        # Regress parameters
        params = self.regressor(features)  # [B, 4]

        # Parse outputs
        theta = params[:, 0] * 180  # Scale to degrees
        log_scale = params[:, 1]
        tx = params[:, 2] * 128  # Scale to pixels
        ty = params[:, 3] * 128

        return {
            'rotation': theta,
            'scale': torch.exp(log_scale),
            'translation': torch.stack([tx, ty], dim=1)
        }


class BaselineCNN(nn.Module):
    """
    Simple CNN baseline (non-equivariant) for comparison.
    """

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

        # Rotation loss (geodesic on circle)
        rot_loss = 1 - torch.cos((output['rotation'] - gt_rotation) * np.pi / 180)

        # Scale loss (log space)
        scale_loss = torch.abs(torch.log(output['scale'] / gt_scale))

        loss = rot_loss.mean() + scale_loss.mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute rotation error
        rot_error = torch.abs(output['rotation'] - gt_rotation)
        rot_error = torch.min(rot_error, 360 - rot_error)
        total_rot_error += rot_error.mean().item()

        n_batches += 1

    return total_loss / n_batches, total_rot_error / n_batches


def evaluate_model(model, test_angles, pattern_types, device, n_samples=100):
    """Evaluate model at specific test angles."""
    model.eval()

    results = {}

    with torch.no_grad():
        for angle in test_angles:
            errors = []

            for sample_idx in range(n_samples):
                # Create generator for this specific angle
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
                gt_rotation = sample['rotation'].item() * 180 / np.pi  # Convert radians to degrees

                output = model(source, target)
                pred_rotation = output['rotation'].item()

                error = abs(pred_rotation - gt_rotation)
                if error > 180:
                    error = 360 - error

                errors.append(error)

            results[angle] = {
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'median': float(np.median(errors))
            }

    return results


class TrainingDataset(torch.utils.data.Dataset):
    """Dataset wrapper for training that returns data in the expected format."""

    def __init__(self, rotation_range, n_samples, pattern_types=['arrow', 'asymmetric', 'L_shape', 'T_shape']):
        self.rotation_range = rotation_range
        self.n_samples = n_samples
        self.pattern_types = pattern_types

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Create a generator for this specific sample
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
            'rotation': sample['rotation'] * 180 / np.pi,  # Convert radians to degrees
            'scale': sample['scale']
        }


def run_experiment(args):
    """Run the full E2-CNN comparison experiment."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Pattern types for asymmetric testing
    pattern_types = ['arrow', 'asymmetric', 'L_shape', 'T_shape']

    # Test angles
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

    # Models to test
    models_to_test = []

    # 1. Baseline CNN
    print("\n" + "="*60)
    print("Training BaselineCNN")
    print("="*60)

    baseline = BaselineCNN(hidden_dim=64).to(device)
    models_to_test.append(('BaselineCNN', baseline))

    # 2. E2-CNN variants
    if E2CNN_AVAILABLE:
        for N in [8, 16]:
            print(f"\n" + "="*60)
            print(f"Training E2-CNN (N={N})")
            print("="*60)

            e2cnn_model = E2CNNHomographyNet(N=N, hidden_dim=32).to(device)
            models_to_test.append((f'E2CNN_N{N}', e2cnn_model))
    else:
        print("\nSkipping E2-CNN (not installed)")

    # 3. SIGMA (for comparison)
    print("\n" + "="*60)
    print("Loading SIGMA (no training needed)")
    print("="*60)

    sigma = LogPolarSim2Net(
        lp_size=(180, 64),
        r_min=0.05,
        r_max=0.9,
        use_fft_magnitude=True,
        use_disambiguation=True
    ).to(device)
    sigma.eval()

    # Train each model
    for model_name, model in models_to_test:
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")

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
            loss, rot_error = train_model(model, train_loader, optimizer, device, epoch)
            scheduler.step()

            if epoch % 5 == 0 or epoch == args.epochs:
                print(f"  Epoch {epoch}: loss={loss:.4f}, rot_error={rot_error:.2f}°")

            if rot_error < best_error:
                best_error = rot_error

        print(f"  Best training error: {best_error:.2f}°")

        # Evaluate
        print(f"  Evaluating on test angles...")
        eval_results = evaluate_model(model, test_angles, pattern_types, device, args.test_samples)

        # Compute summary
        in_dist_errors = [eval_results[a]['mean'] for a in [0, 15, 30]]
        out_dist_errors = [eval_results[a]['mean'] for a in test_angles if a > 30]

        in_dist_mean = np.mean(in_dist_errors)
        out_dist_mean = np.mean(out_dist_errors)
        gap = out_dist_mean - in_dist_mean
        overall_mean = np.mean([eval_results[a]['mean'] for a in test_angles])

        results['models'][model_name] = {
            'per_angle': eval_results,
            'in_dist_mean': float(in_dist_mean),
            'out_dist_mean': float(out_dist_mean),
            'gap': float(gap),
            'overall_mean': float(overall_mean)
        }

        print(f"  In-dist: {in_dist_mean:.2f}°, Out-dist: {out_dist_mean:.2f}°, Gap: {gap:.2f}°")

    # Evaluate SIGMA (no training)
    print(f"\n{'='*60}")
    print("Evaluating SIGMA")
    print(f"{'='*60}")

    sigma_results = {}
    with torch.no_grad():
        for angle in tqdm(test_angles, desc="SIGMA evaluation"):
            errors = []
            for sample_idx in range(args.test_samples):
                # Create generator for this specific angle
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
                gt_rotation = sample['rotation'].item() * 180 / np.pi  # Convert radians to degrees

                output = sigma(source, target)
                pred_rotation = output['rotation_deg'].item()  # Use degrees output

                error = abs(pred_rotation - gt_rotation)
                if error > 180:
                    error = 360 - error
                errors.append(error)

            sigma_results[angle] = {
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'median': float(np.median(errors))
            }

    in_dist_errors = [sigma_results[a]['mean'] for a in [0, 15, 30]]
    out_dist_errors = [sigma_results[a]['mean'] for a in test_angles if a > 30]

    in_dist_mean = np.mean(in_dist_errors)
    out_dist_mean = np.mean(out_dist_errors)
    gap = out_dist_mean - in_dist_mean
    overall_mean = np.mean([sigma_results[a]['mean'] for a in test_angles])

    results['models']['SIGMA'] = {
        'per_angle': sigma_results,
        'in_dist_mean': float(in_dist_mean),
        'out_dist_mean': float(out_dist_mean),
        'gap': float(gap),
        'overall_mean': float(overall_mean)
    }

    print(f"  In-dist: {in_dist_mean:.2f}°, Out-dist: {out_dist_mean:.2f}°, Gap: {gap:.2f}°")

    # Print final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'In-Dist':>10} {'Out-Dist':>10} {'Gap':>10} {'Overall':>10}")
    print("-" * 60)

    for model_name in results['models']:
        m = results['models'][model_name]
        print(f"{model_name:<20} {m['in_dist_mean']:>10.2f}° {m['out_dist_mean']:>10.2f}° "
              f"{m['gap']:>10.2f}° {m['overall_mean']:>10.2f}°")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results['timestamp'] = datetime.now().isoformat()

    output_file = output_dir / 'e2cnn_comparison.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='E2-CNN Baseline Comparison')
    parser.add_argument('--train_samples', type=int, default=10000,
                        help='Number of training samples')
    parser.add_argument('--test_samples', type=int, default=100,
                        help='Number of test samples per angle')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='outputs/e2cnn_comparison',
                        help='Output directory')

    args = parser.parse_args()

    print("="*60)
    print("E2-CNN Baseline Comparison Experiment")
    print("="*60)
    print(f"Train samples: {args.train_samples}")
    print(f"Test samples per angle: {args.test_samples}")
    print(f"Epochs: {args.epochs}")
    print(f"e2cnn available: {E2CNN_AVAILABLE}")

    run_experiment(args)


if __name__ == '__main__':
    main()
