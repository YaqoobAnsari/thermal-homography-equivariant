#!/usr/bin/env python3
"""
Baseline Comparison Script for ECCV Paper

Compares FMT (ours) against implemented baselines:
1. BaselineCNN - direct regression
2. HomographyNet - DeTone et al. 2016
3. IHN - Cao et al. 2022 (if available)

Uses same evaluation protocol as generalization_experiment.py (verified working).

Author: ECCV 2026 Submission
"""

import sys
import os
import json
import math
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import cv2

sys.path.insert(0, "/data/gpfs/projects/punim2769/thermal-homography")

from src.models.log_polar_sim2_net import LogPolarSim2Net
from src.models.baseline_cnn import BaselineCNN, BaselineLoss
from src.data.synthetic_generator import (
    generate_arrow_pattern,
    generate_natural_texture,
    generate_checkerboard,
    generate_homography,  # Use verified function
)

# Check for optional HomographyNet
try:
    from src.models.homography_net import HomographyNet
    HAS_HOMOGRAPHYNET = True
except ImportError:
    HAS_HOMOGRAPHYNET = False


class RotationDataset(Dataset):
    """Dataset with specific rotation angles for generalization testing.

    Uses same data generation as verified generalization_experiment.py.
    """

    def __init__(
        self,
        angles: List[float],
        samples_per_angle: int = 50,
        image_size: int = 256,
    ):
        self.angles = angles
        self.samples_per_angle = samples_per_angle
        self.image_size = image_size
        self.total_samples = len(angles) * samples_per_angle

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        angle_idx = idx // self.samples_per_angle
        sample_idx = idx % self.samples_per_angle

        angle = self.angles[angle_idx]

        # Generate pattern (deterministic based on sample_idx)
        np.random.seed(sample_idx * 1000 + angle_idx)
        pattern_type = sample_idx % 3
        size = (self.image_size, self.image_size)

        if pattern_type == 0:
            source = generate_arrow_pattern(size=size)
        elif pattern_type == 1:
            source = generate_checkerboard(size=size, squares=8)
        else:
            source = generate_natural_texture(size=size)

        # Use verified generate_homography with fixed rotation, scale=1, no translation
        H, params = generate_homography(
            rotation_range=(angle, angle),  # Fixed rotation
            scale_range=(1.0, 1.0),          # No scale
            translation_range=(0, 0),        # No translation
            image_size=size,
        )

        # Warp
        target = cv2.warpPerspective(source, H, size)

        # To tensors
        source_t = torch.from_numpy(source).unsqueeze(0).float() / 255.0
        target_t = torch.from_numpy(target).unsqueeze(0).float() / 255.0

        return {
            'source': source_t,
            'target': target_t,
            'angle': angle,
            'angle_rad': params['angle'],  # Already in radians from generate_homography
            'scale': params['scale'],
        }


class TrainDataset(Dataset):
    """Training dataset with random rotations in a given range.

    Uses same data generation as verified generalization_experiment.py.
    """

    def __init__(
        self,
        n_samples: int,
        rotation_range: Tuple[float, float] = (-30, 30),
        scale_range: Tuple[float, float] = (0.95, 1.05),
        translation_range: Tuple[float, float] = (-10, 10),
        image_size: int = 256,
    ):
        self.n_samples = n_samples
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.image_size = image_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        np.random.seed(idx)

        # Generate pattern
        pattern_type = idx % 3
        size = (self.image_size, self.image_size)

        if pattern_type == 0:
            source = generate_arrow_pattern(size=size)
        elif pattern_type == 1:
            source = generate_checkerboard(size=size, squares=8)
        else:
            source = generate_natural_texture(size=size)

        # Use verified generate_homography
        H, params = generate_homography(
            rotation_range=self.rotation_range,
            scale_range=self.scale_range,
            translation_range=self.translation_range,
            image_size=size,
        )

        target = cv2.warpPerspective(source, H, size)

        source_t = torch.from_numpy(source).unsqueeze(0).float() / 255.0
        target_t = torch.from_numpy(target).unsqueeze(0).float() / 255.0

        return {
            'source': source_t,
            'target': target_t,
            'rotation': torch.tensor(params['angle']).float(),  # Already radians
            'scale': torch.tensor(params['scale']).float(),
            'translation': torch.tensor([params['tx'] / 128, params['ty'] / 128]).float(),
        }


def train_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 25,
    lr: float = 1e-4,
    name: str = "Model",
) -> Dict:
    """Train a baseline model."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = BaselineLoss()

    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_rot_err = 0
        n_batches = 0

        for batch in train_loader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            rot_gt = batch['rotation'].to(device)
            scale_gt = batch['scale'].to(device)
            trans_gt = batch['translation'].to(device)

            optimizer.zero_grad()
            output = model(source, target)

            loss_dict = loss_fn(output, {
                'rotation': rot_gt,
                'scale': scale_gt,
                'translation': trans_gt,
            })
            loss = loss_dict['total']
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Rotation error
            rot_err = torch.abs(output['rotation'] - rot_gt) * 180 / math.pi
            rot_err = torch.where(rot_err > 180, 360 - rot_err, rot_err)
            epoch_rot_err += rot_err.mean().item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_rot = epoch_rot_err / n_batches

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{name}] Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, RotErr={avg_rot:.1f}°")

        history.append({'loss': avg_loss, 'rot_err': avg_rot})

    return {'history': history}


def evaluate_rotation_generalization(
    model: nn.Module,
    test_angles: List[float],
    device: torch.device,
    samples_per_angle: int = 100,
    name: str = "Model",
    use_fmt: bool = False,
) -> Dict:
    """Evaluate rotation generalization."""
    model.eval()

    dataset = RotationDataset(
        angles=test_angles,
        samples_per_angle=samples_per_angle,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Collect errors per angle
    errors_by_angle = {a: [] for a in test_angles}

    with torch.no_grad():
        for batch in loader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            angles = batch['angle'].numpy()
            angles_rad = batch['angle_rad'].to(device)

            output = model(source, target)

            # Rotation error
            rot_pred = output['rotation']
            rot_err = torch.abs(rot_pred - angles_rad) * 180 / math.pi
            rot_err = torch.where(rot_err > 180, 360 - rot_err, rot_err)

            for i, angle in enumerate(angles):
                errors_by_angle[angle].append(rot_err[i].item())

    # Summarize
    results = {'per_angle': {}, 'errors': {}}
    for angle in test_angles:
        errs = errors_by_angle[angle]
        results['per_angle'][angle] = {
            'mean': np.mean(errs),
            'std': np.std(errs),
            'n': len(errs),
        }
        results['errors'][str(int(angle))] = np.mean(errs)
        print(f"  [{name}] {angle:>4}°: {np.mean(errs):.2f}° ± {np.std(errs):.2f}°")

    # Overall mean
    all_errs = [e for errs in errors_by_angle.values() for e in errs]
    results['mean'] = np.mean(all_errs)
    results['std'] = np.std(all_errs)

    # Generalization gap (in-dist: 0-30, out-dist: >30)
    in_dist = [e for a, errs in errors_by_angle.items() for e in errs if abs(a) <= 30]
    out_dist = [e for a, errs in errors_by_angle.items() for e in errs if abs(a) > 30]

    results['in_dist_mean'] = np.mean(in_dist) if in_dist else 0
    results['out_dist_mean'] = np.mean(out_dist) if out_dist else 0
    results['gap'] = results['out_dist_mean'] - results['in_dist_mean']

    print(f"  [{name}] Mean: {results['mean']:.2f}°, Gap: {results['gap']:.2f}°")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--train-epochs', type=int, default=25)
    parser.add_argument('--train-samples', type=int, default=5000)
    parser.add_argument('--test-samples', type=int, default=100)
    parser.add_argument('--output-dir', default='outputs/baseline_comparison')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test angles
    test_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]

    # Training data
    train_dataset = TrainDataset(
        n_samples=args.train_samples,
        rotation_range=(-30, 30),
        scale_range=(0.95, 1.05),
        translation_range=(-10, 10),
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    results = {
        'config': {
            'train_rotation_range': [-30, 30],
            'test_angles': test_angles,
            'train_samples': args.train_samples,
            'test_samples_per_angle': args.test_samples,
        },
        'models': {},
    }

    # =========================================================================
    # 1. FMT (Ours) - No training needed
    # =========================================================================
    print("\n" + "="*60)
    print("Evaluating FMT (Ours) - No training")
    print("="*60)

    fmt = LogPolarSim2Net(
        lp_size=(180, 64),
        use_fft_magnitude=True,
        use_disambiguation=True,
    ).to(device)

    fmt_results = evaluate_rotation_generalization(
        fmt, test_angles, device, args.test_samples, "FMT"
    )
    results['models']['FMT'] = fmt_results

    # =========================================================================
    # 2. BaselineCNN
    # =========================================================================
    print("\n" + "="*60)
    print("Training BaselineCNN")
    print("="*60)

    baseline = BaselineCNN().to(device)
    train_baseline(baseline, train_loader, device, args.train_epochs, name="BaselineCNN")

    baseline_results = evaluate_rotation_generalization(
        baseline, test_angles, device, args.test_samples, "BaselineCNN"
    )
    results['models']['BaselineCNN'] = baseline_results

    # =========================================================================
    # 3. HomographyNet (if available)
    # =========================================================================
    if HAS_HOMOGRAPHYNET:
        print("\n" + "="*60)
        print("Training HomographyNet")
        print("="*60)

        try:
            homographynet = HomographyNet().to(device)
            train_baseline(homographynet, train_loader, device, args.train_epochs, name="HomographyNet")

            hnet_results = evaluate_rotation_generalization(
                homographynet, test_angles, device, args.test_samples, "HomographyNet"
            )
            results['models']['HomographyNet'] = hnet_results
        except Exception as e:
            print(f"HomographyNet failed: {e}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY: Rotation Generalization")
    print("="*60)

    print(f"\n{'Model':<20} {'Mean Error':<15} {'Gap':<15} {'Gap Ratio':<15}")
    print("-"*65)

    fmt_gap = results['models']['FMT']['gap']
    for name, data in results['models'].items():
        ratio = data['gap'] / fmt_gap if fmt_gap > 0 else float('inf')
        print(f"{name:<20} {data['mean']:.2f}°{'':<10} {data['gap']:.2f}°{'':<10} {ratio:.1f}×")

    # Save results
    results['timestamp'] = datetime.now().isoformat()

    with open(output_dir / 'baseline_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'baseline_comparison.json'}")


if __name__ == '__main__':
    main()
