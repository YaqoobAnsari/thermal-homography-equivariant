#!/usr/bin/env python3
"""
GENERALIZATION EXPERIMENT - The Killer Result for ECCV

This experiment proves the core claim:
    "Architectural equivariance generalizes to unseen rotations;
     data augmentation does not."

Setup:
    - Train baseline on LIMITED rotation range (±30°)
    - Train FMT model on same range (or use random init)
    - Test BOTH on FULL rotation range (±180°)

Expected Result:
    - Baseline: good at ±30°, FAILS at ±90°, ±180°
    - FMT: good EVERYWHERE (flat error curve)

This is THE experiment that justifies the paper.
"""

import sys
import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import cv2
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, "/data/gpfs/projects/punim2769/thermal-homography")

from src.models.log_polar_sim2_net import LogPolarSim2Net
from src.models.baseline_cnn import BaselineCNN, BaselineLoss
from src.data.synthetic_generator import (
    generate_arrow_pattern,
    generate_natural_texture,
    generate_homography,
)


class LimitedRotationDataset(Dataset):
    """Dataset with LIMITED rotation range for training."""

    def __init__(self, n_samples=2000, rotation_range=(-30, 30),
                 scale_range=(0.95, 1.05), translation_range=(-20, 20),
                 image_size=256):
        self.n_samples = n_samples
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.image_size = image_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        np.random.seed(idx)

        # Generate pattern (mix of types for diversity)
        if idx % 2 == 0:
            source = generate_arrow_pattern(size=(self.image_size, self.image_size))
        else:
            source = generate_natural_texture(size=(self.image_size, self.image_size))

        # Generate transformation within LIMITED range
        H, params = generate_homography(
            rotation_range=self.rotation_range,
            translation_range=self.translation_range,
            scale_range=self.scale_range,
            image_size=(self.image_size, self.image_size),
        )

        target = cv2.warpPerspective(source, H, (self.image_size, self.image_size))

        source_t = torch.from_numpy(source).unsqueeze(0).float() / 255.0
        target_t = torch.from_numpy(target).unsqueeze(0).float() / 255.0

        return {
            'source': source_t,
            'target': target_t,
            'rotation': torch.tensor(math.radians(params['angle'])).float(),
            'scale': torch.tensor(params['scale']).float(),
            'translation': torch.tensor([params['tx'] / 128, params['ty'] / 128]).float(),
        }


def evaluate_at_angles(model, device, test_angles, pattern_type='arrow'):
    """Evaluate model at specific rotation angles."""
    model.eval()

    if pattern_type == 'arrow':
        generator = generate_arrow_pattern
    else:
        generator = generate_natural_texture

    results = {}

    for angle_deg in test_angles:
        np.random.seed(42)
        source = generator(size=(256, 256))

        # Create target with exact rotation
        H, params = generate_homography(
            rotation_range=(angle_deg, angle_deg),
            translation_range=(0, 0),
            scale_range=(1.0, 1.0),
            image_size=(256, 256),
        )
        target = cv2.warpPerspective(source, H, (256, 256))

        src_t = torch.from_numpy(source).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
        tgt_t = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0

        with torch.no_grad():
            output = model(src_t, tgt_t)

        pred_deg = output['rotation'].item() * 180 / math.pi

        # Compute error (handle wrap-around)
        error = abs(pred_deg - angle_deg)
        if error > 180:
            error = 360 - error

        results[angle_deg] = error

    return results


def train_model(model, train_loader, loss_fn, optimizer, device, epochs=10):
    """Train a model and return training history."""
    history = {'loss': [], 'rot_error': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_rot_error = 0
        n_batches = 0

        for batch in train_loader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            rotation_gt = batch['rotation'].to(device)
            scale_gt = batch['scale'].to(device)
            translation_gt = batch['translation'].to(device)

            optimizer.zero_grad()

            output = model(source, target)

            # Prepare target dict
            target_dict = {
                'rotation': rotation_gt,
                'scale': scale_gt,
                'translation': translation_gt,
            }

            loss_dict = loss_fn(output, target_dict)
            loss = loss_dict['total']
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Rotation error
            rot_error = torch.abs(output['rotation'] - rotation_gt) * 180 / math.pi
            rot_error = torch.where(rot_error > 180, 360 - rot_error, rot_error)
            total_rot_error += rot_error.mean().item()

            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_rot_error = total_rot_error / n_batches
        history['loss'].append(avg_loss)
        history['rot_error'].append(avg_rot_error)

        print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, RotErr={avg_rot_error:.1f}°")

    return history


def run_experiment(train_rotation_range=(-30, 30), epochs=20, output_dir='outputs/generalization'):
    """Run the full generalization experiment."""
    print("=" * 80)
    print("GENERALIZATION EXPERIMENT")
    print("=" * 80)
    print(f"\nTrain rotation range: {train_rotation_range}")
    print(f"Test rotation range: full ±180°")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test angles (full range)
    test_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]

    results = {
        'train_range': train_rotation_range,
        'test_angles': test_angles,
        'baseline': {},
        'fmt': {},
    }

    # =========================================================================
    # BASELINE: Train CNN on limited rotation range
    # =========================================================================
    print("=" * 60)
    print("BASELINE: Training CNN on limited rotation range")
    print("=" * 60)

    baseline = BaselineCNN().to(device)
    baseline_loss = BaselineLoss(w_rotation=1.0, w_scale=0.5, w_translation=0.3)
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=1e-4)

    train_dataset = LimitedRotationDataset(
        n_samples=2000,
        rotation_range=train_rotation_range,
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    print(f"\nTraining baseline for {epochs} epochs...")
    baseline_history = train_model(
        baseline, train_loader, baseline_loss, baseline_optimizer, device, epochs=epochs
    )

    print("\nEvaluating baseline at all test angles...")
    baseline_results = evaluate_at_angles(baseline, device, test_angles)
    results['baseline']['errors'] = baseline_results
    results['baseline']['history'] = baseline_history

    print("\nBaseline results:")
    for angle, error in sorted(baseline_results.items()):
        in_train = train_rotation_range[0] <= angle <= train_rotation_range[1]
        marker = "✓ (in train)" if in_train else "✗ (out of train)"
        print(f"  {angle:>4}°: {error:>6.1f}° {marker}")

    # =========================================================================
    # FMT: Our equivariant model (random init - no training needed!)
    # =========================================================================
    print("\n" + "=" * 60)
    print("FMT: Our equivariant model (random init)")
    print("=" * 60)

    fmt_model = LogPolarSim2Net(
        lp_size=(180, 64),
        use_fft_magnitude=True,
        use_disambiguation=True,
    ).to(device)

    print("\nEvaluating FMT at all test angles (NO TRAINING)...")
    fmt_results = evaluate_at_angles(fmt_model, device, test_angles)
    results['fmt']['errors'] = fmt_results
    results['fmt']['history'] = None  # No training

    print("\nFMT results:")
    for angle, error in sorted(fmt_results.items()):
        print(f"  {angle:>4}°: {error:>6.1f}°")

    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON: Baseline vs FMT")
    print("=" * 80)

    print(f"\n{'Angle':<8} | {'Baseline':>10} | {'FMT':>10} | {'Winner':>10}")
    print("-" * 50)

    baseline_wins = 0
    fmt_wins = 0

    for angle in test_angles:
        b_err = baseline_results[angle]
        f_err = fmt_results[angle]

        if b_err < f_err:
            winner = "Baseline"
            baseline_wins += 1
        else:
            winner = "FMT"
            fmt_wins += 1

        print(f"{angle:>4}°    | {b_err:>9.1f}° | {f_err:>9.1f}° | {winner:>10}")

    # Summary statistics
    baseline_errors = list(baseline_results.values())
    fmt_errors = list(fmt_results.values())

    # In-distribution (within training range)
    in_dist_angles = [a for a in test_angles if train_rotation_range[0] <= a <= train_rotation_range[1]]
    out_dist_angles = [a for a in test_angles if a not in in_dist_angles]

    baseline_in_dist = np.mean([baseline_results[a] for a in in_dist_angles]) if in_dist_angles else 0
    baseline_out_dist = np.mean([baseline_results[a] for a in out_dist_angles]) if out_dist_angles else 0
    fmt_in_dist = np.mean([fmt_results[a] for a in in_dist_angles]) if in_dist_angles else 0
    fmt_out_dist = np.mean([fmt_results[a] for a in out_dist_angles]) if out_dist_angles else 0

    print("\n" + "-" * 50)
    print(f"\nSUMMARY STATISTICS:")
    print(f"\n  BASELINE:")
    print(f"    In-distribution ({train_rotation_range}): {baseline_in_dist:.1f}°")
    print(f"    Out-of-distribution:                  {baseline_out_dist:.1f}°")
    print(f"    Generalization gap:                   {baseline_out_dist - baseline_in_dist:.1f}°")
    print(f"    Overall mean:                         {np.mean(baseline_errors):.1f}°")
    print(f"    Std (flatness):                       {np.std(baseline_errors):.1f}°")

    print(f"\n  FMT (OURS):")
    print(f"    In-distribution ({train_rotation_range}): {fmt_in_dist:.1f}°")
    print(f"    Out-of-distribution:                  {fmt_out_dist:.1f}°")
    print(f"    Generalization gap:                   {fmt_out_dist - fmt_in_dist:.1f}°")
    print(f"    Overall mean:                         {np.mean(fmt_errors):.1f}°")
    print(f"    Std (flatness):                       {np.std(fmt_errors):.1f}°")

    results['summary'] = {
        'baseline_in_dist': baseline_in_dist,
        'baseline_out_dist': baseline_out_dist,
        'baseline_gap': baseline_out_dist - baseline_in_dist,
        'baseline_mean': float(np.mean(baseline_errors)),
        'baseline_std': float(np.std(baseline_errors)),
        'fmt_in_dist': fmt_in_dist,
        'fmt_out_dist': fmt_out_dist,
        'fmt_gap': fmt_out_dist - fmt_in_dist,
        'fmt_mean': float(np.mean(fmt_errors)),
        'fmt_std': float(np.std(fmt_errors)),
    }

    # =========================================================================
    # PLOT
    # =========================================================================
    print("\nGenerating plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Error curves
    ax1 = axes[0]
    ax1.plot(test_angles, [baseline_results[a] for a in test_angles], 'r-o', label='Baseline CNN', linewidth=2, markersize=8)
    ax1.plot(test_angles, [fmt_results[a] for a in test_angles], 'b-s', label='FMT (Ours)', linewidth=2, markersize=8)

    # Shade training region
    ax1.axvspan(train_rotation_range[0], train_rotation_range[1], alpha=0.2, color='green', label='Training range')

    ax1.set_xlabel('Test Rotation Angle (degrees)', fontsize=12)
    ax1.set_ylabel('Rotation Error (degrees)', fontsize=12)
    ax1.set_title('Generalization: Baseline vs FMT', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 180)
    ax1.set_ylim(0, max(max(baseline_errors), 90) * 1.1)

    # Plot 2: Bar chart summary
    ax2 = axes[1]
    x = np.arange(2)
    width = 0.35

    in_dist = [baseline_in_dist, fmt_in_dist]
    out_dist = [baseline_out_dist, fmt_out_dist]

    bars1 = ax2.bar(x - width/2, in_dist, width, label='In-distribution', color='green', alpha=0.7)
    bars2 = ax2.bar(x + width/2, out_dist, width, label='Out-of-distribution', color='red', alpha=0.7)

    ax2.set_ylabel('Mean Rotation Error (degrees)', fontsize=12)
    ax2.set_title('Generalization Gap', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Baseline CNN', 'FMT (Ours)'])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars1, in_dist):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}°', ha='center', fontsize=10)
    for bar, val in zip(bars2, out_dist):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}°', ha='center', fontsize=10)

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / 'generalization_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    # Save results
    results_path = output_dir / 'generalization_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    # =========================================================================
    # VERDICT
    # =========================================================================
    print("\n" + "=" * 80)

    if results['summary']['baseline_gap'] > 20 and results['summary']['fmt_gap'] < 5:
        print("✓ SUCCESS: FMT generalizes, baseline does not!")
        print("  → This is the key result for the paper")
        print("  → Baseline has large generalization gap, FMT has near-zero gap")
    elif results['summary']['fmt_mean'] < results['summary']['baseline_mean']:
        print("⚠ PARTIAL: FMT is better overall, but gap difference not dramatic")
        print("  → May need more epochs or narrower training range")
    else:
        print("✗ UNEXPECTED: Baseline competitive with FMT")
        print("  → Check implementation or try narrower training range")

    print("=" * 80)

    return results


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    # Run with limited training range
    results = run_experiment(
        train_rotation_range=(-30, 30),  # Only train on ±30°
        epochs=20,
        output_dir='outputs/generalization'
    )
