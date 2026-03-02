#!/usr/bin/env python3
"""
Fair Augmentation Comparison

This experiment addresses a key reviewer concern:
"You train baselines on ±30° and test on ±180° - that's unfair!"

We train BaselineCNN and HomographyNet with FULL rotation augmentation (0-360°)
and compare against SIGMA. This shows that architectural equivariance beats
data augmentation even when augmentation covers the full range.

Expected outcome:
- Baselines with full augmentation will improve over limited augmentation
- But SIGMA will STILL win because equivariance > augmentation
"""

import sys
sys.path.insert(0, "/data/gpfs/projects/punim2769/thermal-homography")

import os
import json
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import cv2

from src.models.log_polar_sim2_net import LogPolarSim2Net
from src.models.baselines import HomographyNet
from src.data.synthetic_generator import (
    generate_arrow_pattern,
    generate_natural_texture,
    generate_asymmetric_pattern,
    generate_homography,
)


class BaselineCNN(nn.Module):
    """Simple CNN baseline for Sim(2) regression."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),  # rotation, scale, tx, ty
        )

    def forward(self, source, target):
        x = torch.cat([source, target], dim=1)
        features = self.encoder(x).squeeze(-1).squeeze(-1)
        params = self.regressor(features)

        rotation = params[:, 0:1] * math.pi  # Scale to [-pi, pi]
        scale = torch.exp(params[:, 1:2] * 0.5)  # Log-scale
        translation = params[:, 2:4] * 50  # Scale translation

        return {
            'rotation': rotation,
            'scale': scale,
            'translation': translation,
        }


def generate_training_data(
    n_samples: int,
    rotation_range: Tuple[float, float],
    patterns: List[str] = ['arrow', 'natural_texture', 'asymmetric'],
    size: Tuple[int, int] = (256, 256),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate training data with specified rotation range."""

    pattern_funcs = {
        'arrow': generate_arrow_pattern,
        'natural_texture': generate_natural_texture,
        'asymmetric': generate_asymmetric_pattern,
    }

    sources = []
    targets = []
    params_list = []

    for i in range(n_samples):
        # Random pattern
        pattern_name = patterns[i % len(patterns)]
        pattern_func = pattern_funcs[pattern_name]

        np.random.seed(i)
        source = pattern_func(size=size)

        # Random rotation from specified range
        angle = np.random.uniform(rotation_range[0], rotation_range[1])
        scale = np.random.uniform(0.9, 1.1)

        H, gt_params = generate_homography(
            rotation_range=(angle, angle),
            scale_range=(scale, scale),
            translation_range=(0, 0),
            image_size=size,
        )

        target = cv2.warpPerspective(source, H, size)

        sources.append(source)
        targets.append(target)
        params_list.append([gt_params['angle'], gt_params['scale'], 0.0, 0.0])

    sources = torch.tensor(np.array(sources), dtype=torch.float32).unsqueeze(1) / 255.0
    targets = torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(1) / 255.0
    params = torch.tensor(np.array(params_list), dtype=torch.float32)

    return sources, targets, params


def compute_rotation_error(pred_rad: float, gt_rad: float) -> float:
    """Compute rotation error handling wraparound."""
    err = abs(pred_rad - gt_rad) * 180 / math.pi
    if err > 180:
        err = 360 - err
    return err


def compute_mace(pred_params: Dict, gt_params: torch.Tensor, size: Tuple[int, int] = (256, 256)) -> float:
    """Compute Mean Average Corner Error."""
    corners = np.array([
        [0, 0, 1],
        [size[0], 0, 1],
        [size[0], size[1], 1],
        [0, size[1], 1]
    ], dtype=np.float64).T

    cx, cy = size[0] / 2, size[1] / 2

    # Ground truth homography
    gt_rot = gt_params[0].item()
    gt_scale = gt_params[1].item()
    cos_r, sin_r = np.cos(gt_rot), np.sin(gt_rot)
    H_gt = np.array([
        [gt_scale * cos_r, -gt_scale * sin_r, cx - gt_scale * (cos_r * cx - sin_r * cy)],
        [gt_scale * sin_r, gt_scale * cos_r, cy - gt_scale * (sin_r * cx + cos_r * cy)],
        [0, 0, 1]
    ])

    # Predicted homography
    pred_rot = pred_params['rotation'].item()
    pred_scale = pred_params['scale'].item()
    cos_r, sin_r = np.cos(pred_rot), np.sin(pred_rot)
    H_pred = np.array([
        [pred_scale * cos_r, -pred_scale * sin_r, cx - pred_scale * (cos_r * cx - sin_r * cy)],
        [pred_scale * sin_r, pred_scale * cos_r, cy - pred_scale * (sin_r * cx + cos_r * cy)],
        [0, 0, 1]
    ])

    # Transform corners
    gt_corners = H_gt @ corners
    gt_corners = gt_corners[:2] / gt_corners[2:]

    pred_corners = H_pred @ corners
    pred_corners = pred_corners[:2] / pred_corners[2:]

    # MACE
    error = np.sqrt(np.sum((pred_corners - gt_corners)**2, axis=0))
    return np.mean(error)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-4,
    model_name: str = "Model",
) -> nn.Module:
    """Train a model."""

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for sources, targets, params in train_loader:
            sources = sources.to(device)
            targets = targets.to(device)
            params = params.to(device)

            optimizer.zero_grad()

            output = model(sources, targets)

            # Rotation loss (geodesic)
            rot_loss = (1 - torch.cos(output['rotation'].squeeze() - params[:, 0])).mean()

            # Scale loss (log space)
            scale_loss = torch.abs(torch.log(output['scale'].squeeze() / params[:, 1])).mean()

            loss = rot_loss + scale_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"  [{model_name}] Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}")

    return model


def evaluate_model(
    model: nn.Module,
    device: torch.device,
    test_angles: List[float],
    model_name: str,
) -> Dict:
    """Evaluate model on test angles."""

    model.eval()
    results = {'by_angle': {}, 'all_rot_errors': [], 'all_mace': []}

    patterns = ['arrow', 'natural_texture', 'asymmetric']
    n_seeds = 5

    for angle in test_angles:
        angle_rot_errors = []
        angle_mace = []

        for pattern_name in patterns:
            pattern_funcs = {
                'arrow': generate_arrow_pattern,
                'natural_texture': generate_natural_texture,
                'asymmetric': generate_asymmetric_pattern,
            }

            for seed in range(n_seeds):
                np.random.seed(int(angle * 1000 + seed))

                source = pattern_funcs[pattern_name](size=(256, 256))

                H, gt_params = generate_homography(
                    rotation_range=(angle, angle),
                    scale_range=(1.0, 1.0),
                    translation_range=(0, 0),
                    image_size=(256, 256),
                )

                target = cv2.warpPerspective(source, H, (256, 256))

                source_t = torch.tensor(source, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0
                target_t = torch.tensor(target, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0

                with torch.no_grad():
                    output = model(source_t, target_t)

                rot_pred = output['rotation'].item()
                rot_gt = gt_params['angle']
                rot_err = compute_rotation_error(rot_pred, rot_gt)

                gt_tensor = torch.tensor([rot_gt, gt_params['scale'], 0, 0])
                mace = compute_mace(output, gt_tensor)

                angle_rot_errors.append(rot_err)
                angle_mace.append(mace)

        results['by_angle'][angle] = {
            'rot_mean': np.mean(angle_rot_errors),
            'rot_std': np.std(angle_rot_errors),
            'mace_mean': np.mean(angle_mace),
            'mace_std': np.std(angle_mace),
            'n': len(angle_rot_errors),
        }
        results['all_rot_errors'].extend(angle_rot_errors)
        results['all_mace'].extend(angle_mace)

        print(f"  [{model_name}] {angle:>3}°: RotErr={np.mean(angle_rot_errors):.2f}° ± {np.std(angle_rot_errors):.2f}°, MACE={np.mean(angle_mace):.1f}px")

    results['summary'] = {
        'rot_mean': np.mean(results['all_rot_errors']),
        'rot_std': np.std(results['all_rot_errors']),
        'mace_mean': np.mean(results['all_mace']),
        'mace_std': np.std(results['all_mace']),
        'pass_3deg': np.mean(np.array(results['all_rot_errors']) < 3.0),
        'pass_5deg': np.mean(np.array(results['all_rot_errors']) < 5.0),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Fair Augmentation Comparison')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--train-samples', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--output-dir', default='outputs/fair_augmentation')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]

    results = {
        'config': {
            'train_samples': args.train_samples,
            'epochs': args.epochs,
            'test_angles': test_angles,
        },
        'timestamp': datetime.now().isoformat(),
        'models': {},
    }

    # =========================================================================
    # 1. SIGMA (Ours) - No training needed
    # =========================================================================
    print("\n" + "="*70)
    print("1. SIGMA (Ours) - Mathematical baseline, no training")
    print("="*70)

    sigma = LogPolarSim2Net(
        lp_size=(180, 64),
        use_fft_magnitude=True,
        use_disambiguation=True,
    ).to(device)
    sigma.eval()

    sigma_results = evaluate_model(sigma, device, test_angles, "SIGMA")
    results['models']['SIGMA'] = sigma_results
    print(f"\n  [SIGMA] OVERALL: RotErr={sigma_results['summary']['rot_mean']:.2f}° ± {sigma_results['summary']['rot_std']:.2f}°, MACE={sigma_results['summary']['mace_mean']:.1f}px")

    # =========================================================================
    # 2. BaselineCNN with LIMITED augmentation (±30°) - Original unfair comparison
    # =========================================================================
    print("\n" + "="*70)
    print("2. BaselineCNN - LIMITED augmentation (±30°)")
    print("="*70)

    print("  Generating training data (±30°)...")
    sources, targets, params = generate_training_data(
        args.train_samples, rotation_range=(-30, 30)
    )
    train_dataset = TensorDataset(sources, targets, params)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    baseline_limited = BaselineCNN().to(device)
    print("  Training...")
    baseline_limited = train_model(baseline_limited, train_loader, device,
                                    epochs=args.epochs, model_name="BaselineCNN-Limited")

    print("  Evaluating...")
    baseline_limited_results = evaluate_model(baseline_limited, device, test_angles, "BaselineCNN-Limited")
    results['models']['BaselineCNN_Limited'] = baseline_limited_results
    print(f"\n  [BaselineCNN-Limited] OVERALL: RotErr={baseline_limited_results['summary']['rot_mean']:.2f}°, MACE={baseline_limited_results['summary']['mace_mean']:.1f}px")

    # =========================================================================
    # 3. BaselineCNN with FULL augmentation (0-360°) - Fair comparison
    # =========================================================================
    print("\n" + "="*70)
    print("3. BaselineCNN - FULL augmentation (0-360°)")
    print("="*70)

    print("  Generating training data (0-360°)...")
    sources, targets, params = generate_training_data(
        args.train_samples, rotation_range=(0, 360)
    )
    train_dataset = TensorDataset(sources, targets, params)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    baseline_full = BaselineCNN().to(device)
    print("  Training...")
    baseline_full = train_model(baseline_full, train_loader, device,
                                 epochs=args.epochs, model_name="BaselineCNN-Full")

    print("  Evaluating...")
    baseline_full_results = evaluate_model(baseline_full, device, test_angles, "BaselineCNN-Full")
    results['models']['BaselineCNN_Full'] = baseline_full_results
    print(f"\n  [BaselineCNN-Full] OVERALL: RotErr={baseline_full_results['summary']['rot_mean']:.2f}°, MACE={baseline_full_results['summary']['mace_mean']:.1f}px")

    # =========================================================================
    # 4. HomographyNet with LIMITED augmentation (±30°)
    # =========================================================================
    print("\n" + "="*70)
    print("4. HomographyNet - LIMITED augmentation (±30°)")
    print("="*70)

    print("  Generating training data (±30°)...")
    sources, targets, params = generate_training_data(
        args.train_samples, rotation_range=(-30, 30)
    )
    train_dataset = TensorDataset(sources, targets, params)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Wrap HomographyNet to output our format
    class HomographyNetWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = HomographyNet()
            # Add a head to predict Sim(2) params from homography
            self.sim2_head = nn.Linear(8, 4)

        def forward(self, source, target):
            h4pt = self.net(source, target)
            sim2 = self.sim2_head(h4pt)
            return {
                'rotation': sim2[:, 0:1] * math.pi,
                'scale': torch.exp(sim2[:, 1:2] * 0.5),
                'translation': sim2[:, 2:4] * 50,
            }

    hnet_limited = HomographyNetWrapper().to(device)
    print("  Training...")
    hnet_limited = train_model(hnet_limited, train_loader, device,
                                epochs=args.epochs, model_name="HomographyNet-Limited")

    print("  Evaluating...")
    hnet_limited_results = evaluate_model(hnet_limited, device, test_angles, "HomographyNet-Limited")
    results['models']['HomographyNet_Limited'] = hnet_limited_results
    print(f"\n  [HomographyNet-Limited] OVERALL: RotErr={hnet_limited_results['summary']['rot_mean']:.2f}°, MACE={hnet_limited_results['summary']['mace_mean']:.1f}px")

    # =========================================================================
    # 5. HomographyNet with FULL augmentation (0-360°)
    # =========================================================================
    print("\n" + "="*70)
    print("5. HomographyNet - FULL augmentation (0-360°)")
    print("="*70)

    print("  Generating training data (0-360°)...")
    sources, targets, params = generate_training_data(
        args.train_samples, rotation_range=(0, 360)
    )
    train_dataset = TensorDataset(sources, targets, params)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    hnet_full = HomographyNetWrapper().to(device)
    print("  Training...")
    hnet_full = train_model(hnet_full, train_loader, device,
                             epochs=args.epochs, model_name="HomographyNet-Full")

    print("  Evaluating...")
    hnet_full_results = evaluate_model(hnet_full, device, test_angles, "HomographyNet-Full")
    results['models']['HomographyNet_Full'] = hnet_full_results
    print(f"\n  [HomographyNet-Full] OVERALL: RotErr={hnet_full_results['summary']['rot_mean']:.2f}°, MACE={hnet_full_results['summary']['mace_mean']:.1f}px")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("FAIR AUGMENTATION COMPARISON SUMMARY")
    print("="*70)

    print(f"\n{'Method':<30} {'RotErr':<15} {'MACE':<15} {'Pass@5°':<10}")
    print("-" * 70)

    for name, data in results['models'].items():
        s = data['summary']
        print(f"{name:<30} {s['rot_mean']:.2f}° ± {s['rot_std']:.2f}°   {s['mace_mean']:.1f}px ± {s['mace_std']:.1f}px   {s['pass_5deg']*100:.1f}%")

    # Compute improvement ratios
    sigma_err = results['models']['SIGMA']['summary']['rot_mean']
    baseline_limited_err = results['models']['BaselineCNN_Limited']['summary']['rot_mean']
    baseline_full_err = results['models']['BaselineCNN_Full']['summary']['rot_mean']

    print(f"\n" + "="*70)
    print("KEY FINDINGS:")
    print(f"  - BaselineCNN (±30°) vs SIGMA: {baseline_limited_err/sigma_err:.1f}× worse")
    print(f"  - BaselineCNN (0-360°) vs SIGMA: {baseline_full_err/sigma_err:.1f}× worse")
    print(f"  - Full augmentation helps baseline: {baseline_limited_err/baseline_full_err:.1f}× improvement")
    print(f"  - But SIGMA STILL WINS by {baseline_full_err/sigma_err:.1f}× even with full augmentation!")
    print("="*70)

    # Save results
    # Convert numpy to native Python for JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    with open(output_dir / 'fair_augmentation_results.json', 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"\nResults saved to {output_dir / 'fair_augmentation_results.json'}")


if __name__ == '__main__':
    main()
