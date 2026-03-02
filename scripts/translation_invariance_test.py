#!/usr/bin/env python3
"""
Translation Invariance Test for ECCV Paper

Proves that FMT achieves translation-invariant rotation estimation.

Protocol:
- Fix rotation at 45°
- Vary translation from 0 to 100 pixels
- Show FMT rotation error stays constant while baseline degrades

Author: ECCV 2026 Submission
"""

import sys
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
from torch.utils.data import DataLoader, Dataset
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, "/data/gpfs/projects/punim2769/thermal-homography")

from src.models.log_polar_sim2_net import LogPolarSim2Net
from src.models.baseline_cnn import BaselineCNN, BaselineLoss
from src.data.synthetic_generator import (
    generate_arrow_pattern,
    generate_checkerboard,
    generate_natural_texture,
    generate_homography,  # Use verified function
)


class TranslationTestDataset(Dataset):
    """Dataset with fixed rotation and varying translation.

    Uses verified generate_homography for consistency.
    """

    def __init__(
        self,
        translation_offsets: List[float],
        samples_per_offset: int = 100,
        fixed_rotation: float = 45.0,
        fixed_scale: float = 1.0,
        image_size: int = 256,
    ):
        self.translation_offsets = translation_offsets
        self.samples_per_offset = samples_per_offset
        self.fixed_rotation = fixed_rotation
        self.fixed_scale = fixed_scale
        self.image_size = image_size
        self.total_samples = len(translation_offsets) * samples_per_offset

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        offset_idx = idx // self.samples_per_offset
        sample_idx = idx % self.samples_per_offset

        trans_range = self.translation_offsets[offset_idx]

        # Generate pattern
        np.random.seed(sample_idx * 1000 + offset_idx)
        pattern_type = sample_idx % 3
        size = (self.image_size, self.image_size)

        if pattern_type == 0:
            source = generate_arrow_pattern(size=size)
        elif pattern_type == 1:
            source = generate_checkerboard(size=size, squares=8)
        else:
            source = generate_natural_texture(size=size)

        # Use generate_homography with fixed rotation/scale but varying translation
        H, params = generate_homography(
            rotation_range=(self.fixed_rotation, self.fixed_rotation),
            scale_range=(self.fixed_scale, self.fixed_scale),
            translation_range=(-trans_range, trans_range) if trans_range > 0 else (0, 0),
            image_size=size,
        )

        target = cv2.warpPerspective(source, H, size)

        source_t = torch.from_numpy(source).unsqueeze(0).float() / 255.0
        target_t = torch.from_numpy(target).unsqueeze(0).float() / 255.0

        return {
            'source': source_t,
            'target': target_t,
            'rotation_gt': params['angle'],  # Already radians
            'scale_gt': params['scale'],
            'tx': params['tx'],
            'ty': params['ty'],
            'trans_range': trans_range,
        }


class TrainDataset(Dataset):
    """Training dataset WITHOUT translation.

    Uses verified generate_homography for consistency.
    """

    def __init__(
        self,
        n_samples: int,
        rotation_range: Tuple[float, float] = (-45, 45),
        scale_range: Tuple[float, float] = (0.95, 1.05),
        image_size: int = 256,
    ):
        self.n_samples = n_samples
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.image_size = image_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        np.random.seed(idx)

        pattern_type = idx % 3
        size = (self.image_size, self.image_size)

        if pattern_type == 0:
            source = generate_arrow_pattern(size=size)
        elif pattern_type == 1:
            source = generate_checkerboard(size=size, squares=8)
        else:
            source = generate_natural_texture(size=size)

        # NO translation during training
        H, params = generate_homography(
            rotation_range=self.rotation_range,
            scale_range=self.scale_range,
            translation_range=(0, 0),  # No translation
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
            'translation': torch.zeros(2),  # No translation
        }


def train_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 25,
    lr: float = 1e-4,
) -> None:
    """Train baseline WITHOUT translation."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = BaselineLoss()

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
            rot_err = torch.abs(output['rotation'] - rot_gt) * 180 / math.pi
            rot_err = torch.where(rot_err > 180, 360 - rot_err, rot_err)
            epoch_rot_err += rot_err.mean().item()
            n_batches += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [BaselineCNN] Epoch {epoch+1}/{epochs}: "
                  f"Loss={epoch_loss/n_batches:.4f}, RotErr={epoch_rot_err/n_batches:.1f}°")


def evaluate_translation_invariance(
    model: nn.Module,
    translation_offsets: List[float],
    device: torch.device,
    samples_per_offset: int = 100,
    fixed_rotation: float = 45.0,
    name: str = "Model",
) -> Dict:
    """Evaluate rotation error at different translation offsets."""
    model.eval()

    dataset = TranslationTestDataset(
        translation_offsets=translation_offsets,
        samples_per_offset=samples_per_offset,
        fixed_rotation=fixed_rotation,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    errors_by_offset = {t: [] for t in translation_offsets}

    with torch.no_grad():
        for batch in loader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            rot_gt = batch['rotation_gt']
            trans_ranges = batch['trans_range'].numpy()

            output = model(source, target)
            rot_pred = output['rotation'].cpu()

            rot_err = torch.abs(rot_pred - rot_gt) * 180 / math.pi
            rot_err = torch.where(rot_err > 180, 360 - rot_err, rot_err)

            for i, tr in enumerate(trans_ranges):
                errors_by_offset[float(tr)].append(rot_err[i].item())

    # Summarize
    results = {'per_offset': {}, 'fixed_rotation': fixed_rotation}

    for offset in translation_offsets:
        errs = errors_by_offset[offset]
        mean_err = np.mean(errs)
        std_err = np.std(errs)
        results['per_offset'][offset] = {'mean': mean_err, 'std': std_err}
        print(f"  [{name}] Trans ±{offset:>3.0f}px: RotErr = {mean_err:.2f}° ± {std_err:.2f}°")

    # Compute degradation ratio
    err_0 = results['per_offset'][translation_offsets[0]]['mean']
    err_max = results['per_offset'][translation_offsets[-1]]['mean']
    results['degradation_ratio'] = err_max / max(err_0, 0.1)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--train-epochs', type=int, default=25)
    parser.add_argument('--train-samples', type=int, default=5000)
    parser.add_argument('--test-samples', type=int, default=100)
    parser.add_argument('--output-dir', default='outputs/translation_invariance')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Translation offsets to test
    translation_offsets = [0, 10, 20, 30, 40, 50, 75, 100]
    fixed_rotation = 45.0  # degrees

    # Training data (NO translation)
    train_dataset = TrainDataset(
        n_samples=args.train_samples,
        rotation_range=(-45, 45),  # Covers 45°
        scale_range=(0.95, 1.05),
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    results = {
        'config': {
            'translation_offsets': translation_offsets,
            'fixed_rotation': fixed_rotation,
            'train_samples': args.train_samples,
            'test_samples_per_offset': args.test_samples,
        },
        'models': {},
    }

    # =========================================================================
    # 1. FMT (Ours) - No training needed
    # =========================================================================
    print("\n" + "="*60)
    print("Evaluating FMT (Ours) - Translation Invariant")
    print("="*60)

    fmt = LogPolarSim2Net(
        lp_size=(180, 64),
        use_fft_magnitude=True,
        use_disambiguation=True,
    ).to(device)

    fmt_results = evaluate_translation_invariance(
        fmt, translation_offsets, device, args.test_samples,
        fixed_rotation, "FMT"
    )
    results['models']['FMT'] = fmt_results

    # =========================================================================
    # 2. BaselineCNN - trained WITHOUT translation
    # =========================================================================
    print("\n" + "="*60)
    print("Training BaselineCNN (NO translation in training)")
    print("="*60)

    baseline = BaselineCNN().to(device)
    train_baseline(baseline, train_loader, device, args.train_epochs)

    print("\nEvaluating BaselineCNN with varying translations...")
    baseline_results = evaluate_translation_invariance(
        baseline, translation_offsets, device, args.test_samples,
        fixed_rotation, "BaselineCNN"
    )
    results['models']['BaselineCNN'] = baseline_results

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY: Translation Invariance Test")
    print("="*60)

    print(f"\nFixed rotation: {fixed_rotation}°")
    print(f"\n{'Model':<15} {'Error@0px':<12} {'Error@100px':<12} {'Degradation':<12}")
    print("-"*55)

    for name, data in results['models'].items():
        err_0 = data['per_offset'][0]['mean']
        err_100 = data['per_offset'][100]['mean']
        deg = data['degradation_ratio']
        print(f"{name:<15} {err_0:.2f}°{'':<7} {err_100:.2f}°{'':<7} {deg:.1f}×")

    print(f"\nFMT is {'TRANSLATION INVARIANT' if results['models']['FMT']['degradation_ratio'] < 1.5 else 'NOT invariant'}")
    print(f"Baseline {'DEGRADES' if results['models']['BaselineCNN']['degradation_ratio'] > 2 else 'is stable'} with translation")

    # =========================================================================
    # Plot
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'FMT': 'blue', 'BaselineCNN': 'red'}

    for name, data in results['models'].items():
        offsets = sorted(data['per_offset'].keys())
        means = [data['per_offset'][t]['mean'] for t in offsets]
        stds = [data['per_offset'][t]['std'] for t in offsets]

        ax.errorbar(offsets, means, yerr=stds, marker='o', label=name,
                    color=colors.get(name, 'gray'), linewidth=2, markersize=6, capsize=4)

    ax.axhline(y=fixed_rotation, color='gray', linestyle='--', alpha=0.5,
               label=f'Target rotation ({fixed_rotation}°)')
    ax.set_xlabel('Translation Offset (pixels)', fontsize=12)
    ax.set_ylabel('Rotation Estimation Error (degrees)', fontsize=12)
    ax.set_title('Translation Invariance Test\n(Rotation fixed at 45°, translation varies)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'translation_invariance.png', dpi=150)
    print(f"\nPlot saved to {output_dir / 'translation_invariance.png'}")

    # Save results
    results['timestamp'] = datetime.now().isoformat()

    with open(output_dir / 'translation_invariance.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir / 'translation_invariance.json'}")


if __name__ == '__main__':
    main()
