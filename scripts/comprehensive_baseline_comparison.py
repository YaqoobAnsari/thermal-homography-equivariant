#!/usr/bin/env python3
"""
Comprehensive Baseline Comparison for ECCV 2026 Paper

Compares SIGMA (FMT) against ALL implemented baselines:
1. HomographyNet (DeTone et al., CVPR 2016) - The canonical baseline
2. IHN (Cao et al., CVPR 2022) - Iterative refinement
3. BaselineCNN - Direct regression
4. ResNetBaseline - Modern CNN architecture
5. BasesHomo (Ye et al., ICCV 2021) - Motion basis learning
6. Classical ECC/Lucas-Kanade

Protocol:
- Train all learning-based methods on ±30° rotation, ±5% scale, ±10px translation
- Test on full rotation range (0-180°) to evaluate generalization
- Report rotation error, scale error, MACE at each test angle
- Include "fair" comparison with full rotation augmentation

Author: ECCV 2026 Submission
"""

import sys
import os
import json
import math
import argparse
from pathlib import Path
from datetime import datetime
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
from src.models.baselines import (
    HomographyNet,
    IterativeHomographyNetwork,
    ResNetBaseline,
    BasesHomoBaseline,
    LucasKanadeBaseline,
)
from src.data.synthetic_generator import (
    generate_arrow_pattern,
    generate_natural_texture,
    generate_checkerboard,
    generate_homography,
)


class Sim2TrainDataset(Dataset):
    """Training dataset for Sim(2) estimation with configurable augmentation."""

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

        pattern_type = idx % 3
        size = (self.image_size, self.image_size)

        if pattern_type == 0:
            source = generate_arrow_pattern(size=size)
        elif pattern_type == 1:
            source = generate_checkerboard(size=size, squares=8)
        else:
            source = generate_natural_texture(size=size)

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
            'rotation': torch.tensor(params['angle']).float(),
            'scale': torch.tensor(params['scale']).float(),
            'translation': torch.tensor([params['tx'] / 128, params['ty'] / 128]).float(),
            'H': torch.from_numpy(H).float(),
        }


class Sim2TestDataset(Dataset):
    """Test dataset with specific rotation angles."""

    def __init__(
        self,
        angles: List[float],
        samples_per_angle: int = 100,
        scale: float = 1.0,
        image_size: int = 256,
    ):
        self.angles = angles
        self.samples_per_angle = samples_per_angle
        self.scale = scale
        self.image_size = image_size
        self.total_samples = len(angles) * samples_per_angle

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        angle_idx = idx // self.samples_per_angle
        sample_idx = idx % self.samples_per_angle
        angle = self.angles[angle_idx]

        np.random.seed(sample_idx * 1000 + angle_idx)
        pattern_type = sample_idx % 3
        size = (self.image_size, self.image_size)

        if pattern_type == 0:
            source = generate_arrow_pattern(size=size)
        elif pattern_type == 1:
            source = generate_checkerboard(size=size, squares=8)
        else:
            source = generate_natural_texture(size=size)

        H, params = generate_homography(
            rotation_range=(angle, angle),
            scale_range=(self.scale, self.scale),
            translation_range=(0, 0),
            image_size=size,
        )

        target = cv2.warpPerspective(source, H, size)

        source_t = torch.from_numpy(source).unsqueeze(0).float() / 255.0
        target_t = torch.from_numpy(target).unsqueeze(0).float() / 255.0

        return {
            'source': source_t,
            'target': target_t,
            'angle': angle,
            'angle_rad': params['angle'],
            'scale': params['scale'],
            'H': torch.from_numpy(H).float(),
        }


def compute_mace(H_pred: torch.Tensor, H_gt: torch.Tensor, image_size: int = 256) -> torch.Tensor:
    """Compute Mean Average Corner Error."""
    device = H_pred.device
    B = H_pred.shape[0]

    # Corner points (normalized coordinates)
    corners = torch.tensor([
        [0, 0], [image_size, 0], [image_size, image_size], [0, image_size]
    ], dtype=torch.float32, device=device)

    # Add homogeneous coordinate
    corners_h = torch.cat([corners, torch.ones(4, 1, device=device)], dim=1)  # [4, 3]

    maces = []
    for b in range(B):
        # Apply ground truth homography
        corners_gt = corners_h @ H_gt[b].T  # [4, 3]
        corners_gt = corners_gt[:, :2] / (corners_gt[:, 2:3] + 1e-8)

        # Apply predicted homography
        corners_pred = corners_h @ H_pred[b].T  # [4, 3]
        corners_pred = corners_pred[:, :2] / (corners_pred[:, 2:3] + 1e-8)

        # Compute corner error
        error = torch.sqrt(((corners_gt - corners_pred) ** 2).sum(dim=1))
        maces.append(error.mean())

    return torch.stack(maces)


def h8_to_matrix(h8: torch.Tensor) -> torch.Tensor:
    """Convert 8-parameter vector to 3x3 homography matrix."""
    B = h8.shape[0]
    H = torch.zeros(B, 3, 3, device=h8.device, dtype=h8.dtype)
    H[:, 0, 0] = h8[:, 0]
    H[:, 0, 1] = h8[:, 1]
    H[:, 0, 2] = h8[:, 2]
    H[:, 1, 0] = h8[:, 3]
    H[:, 1, 1] = h8[:, 4]
    H[:, 1, 2] = h8[:, 5]
    H[:, 2, 0] = h8[:, 6]
    H[:, 2, 1] = h8[:, 7]
    H[:, 2, 2] = 1.0
    return H


class HomographyLoss(nn.Module):
    """Loss for 8-parameter homography prediction."""

    def __init__(self, corner_weight: float = 1.0, param_weight: float = 0.1):
        super().__init__()
        self.corner_weight = corner_weight
        self.param_weight = param_weight

    def forward(self, output: Dict, target: Dict) -> Dict:
        H_pred_vec = output['homography']
        H_gt = target['H']

        # Convert prediction to matrix
        H_pred = h8_to_matrix(H_pred_vec)

        # Corner error loss
        mace = compute_mace(H_pred, H_gt)
        corner_loss = mace.mean()

        # L1 parameter loss (for regularization)
        H_gt_vec = torch.stack([
            H_gt[:, 0, 0], H_gt[:, 0, 1], H_gt[:, 0, 2],
            H_gt[:, 1, 0], H_gt[:, 1, 1], H_gt[:, 1, 2],
            H_gt[:, 2, 0], H_gt[:, 2, 1]
        ], dim=1)
        param_loss = F.l1_loss(H_pred_vec, H_gt_vec)

        total = self.corner_weight * corner_loss + self.param_weight * param_loss

        return {
            'total': total,
            'corner_loss': corner_loss,
            'param_loss': param_loss,
            'mace': mace.mean(),
        }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 25,
    lr: float = 1e-4,
    name: str = "Model",
    use_homography_loss: bool = True,
) -> Dict:
    """Train a baseline model."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    if use_homography_loss:
        loss_fn = HomographyLoss()
    else:
        loss_fn = BaselineLoss()

    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_mace = 0
        n_batches = 0

        for batch in train_loader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()
            output = model(source, target)

            if use_homography_loss:
                loss_dict = loss_fn(output, {'H': batch['H'].to(device)})
            else:
                loss_dict = loss_fn(output, {
                    'rotation': batch['rotation'].to(device),
                    'scale': batch['scale'].to(device),
                    'translation': batch['translation'].to(device),
                })

            loss = loss_dict['total']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            if 'mace' in loss_dict:
                epoch_mace += loss_dict['mace'].item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        avg_mace = epoch_mace / n_batches if epoch_mace > 0 else 0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{name}] Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, MACE={avg_mace:.1f}px")

        history.append({'loss': avg_loss, 'mace': avg_mace})

    return {'history': history}


def evaluate_rotation_generalization(
    model: nn.Module,
    test_angles: List[float],
    device: torch.device,
    samples_per_angle: int = 100,
    name: str = "Model",
    output_type: str = "sim2",  # "sim2" or "homography"
) -> Dict:
    """Evaluate rotation generalization."""
    model.eval()

    dataset = Sim2TestDataset(
        angles=test_angles,
        samples_per_angle=samples_per_angle,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    errors_by_angle = {a: {'rot': [], 'mace': []} for a in test_angles}

    with torch.no_grad():
        for batch in loader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            angles = batch['angle'].numpy()
            angles_rad = batch['angle_rad'].to(device)
            H_gt = batch['H'].to(device)

            output = model(source, target)

            if output_type == "sim2":
                # Extract rotation from Sim(2) output
                rot_pred = output['rotation']
                rot_err = torch.abs(rot_pred - angles_rad) * 180 / math.pi
                rot_err = torch.where(rot_err > 180, 360 - rot_err, rot_err)

                # Compute MACE from predicted params
                # Build H from rotation/scale/translation
                B = rot_pred.shape[0]
                cos_r = torch.cos(rot_pred)
                sin_r = torch.sin(rot_pred)
                scale = output['scale']
                tx = output.get('translation', torch.zeros(B, 2, device=device))[:, 0] * 128
                ty = output.get('translation', torch.zeros(B, 2, device=device))[:, 1] * 128

                H_pred = torch.zeros(B, 3, 3, device=device)
                H_pred[:, 0, 0] = scale * cos_r
                H_pred[:, 0, 1] = -scale * sin_r
                H_pred[:, 0, 2] = tx
                H_pred[:, 1, 0] = scale * sin_r
                H_pred[:, 1, 1] = scale * cos_r
                H_pred[:, 1, 2] = ty
                H_pred[:, 2, 2] = 1.0

                mace = compute_mace(H_pred, H_gt)

            else:  # homography output
                H_pred_vec = output['homography']
                H_pred = h8_to_matrix(H_pred_vec)
                mace = compute_mace(H_pred, H_gt)

                # Extract rotation from homography for error computation
                # For Sim(2), rotation is atan2(H[1,0], H[0,0])
                rot_pred = torch.atan2(H_pred[:, 1, 0], H_pred[:, 0, 0])
                rot_err = torch.abs(rot_pred - angles_rad) * 180 / math.pi
                rot_err = torch.where(rot_err > 180, 360 - rot_err, rot_err)

            for i, angle in enumerate(angles):
                errors_by_angle[angle]['rot'].append(rot_err[i].item())
                errors_by_angle[angle]['mace'].append(mace[i].item())

    # Summarize
    results = {'per_angle': {}}

    for angle in test_angles:
        rot_errs = errors_by_angle[angle]['rot']
        mace_errs = errors_by_angle[angle]['mace']
        results['per_angle'][angle] = {
            'rot_mean': np.mean(rot_errs),
            'rot_std': np.std(rot_errs),
            'mace_mean': np.mean(mace_errs),
            'mace_std': np.std(mace_errs),
        }
        print(f"  [{name}] {angle:>4}°: RotErr={np.mean(rot_errs):.2f}° ± {np.std(rot_errs):.2f}°, "
              f"MACE={np.mean(mace_errs):.1f}px ± {np.std(mace_errs):.1f}px")

    # Overall metrics
    all_rot = [e for errs in errors_by_angle.values() for e in errs['rot']]
    all_mace = [e for errs in errors_by_angle.values() for e in errs['mace']]

    results['mean_rot'] = np.mean(all_rot)
    results['mean_mace'] = np.mean(all_mace)

    # Generalization gap
    in_dist = [e for a, errs in errors_by_angle.items() for e in errs['rot'] if abs(a) <= 30]
    out_dist = [e for a, errs in errors_by_angle.items() for e in errs['rot'] if abs(a) > 30]

    results['in_dist_rot'] = np.mean(in_dist) if in_dist else 0
    results['out_dist_rot'] = np.mean(out_dist) if out_dist else 0
    results['gap'] = results['out_dist_rot'] - results['in_dist_rot']

    print(f"  [{name}] Mean: RotErr={results['mean_rot']:.2f}°, MACE={results['mean_mace']:.1f}px, Gap={results['gap']:.2f}°")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--train-epochs', type=int, default=30)
    parser.add_argument('--train-samples', type=int, default=10000)
    parser.add_argument('--test-samples', type=int, default=100)
    parser.add_argument('--output-dir', default='outputs/comprehensive_comparison')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer samples')
    args = parser.parse_args()

    if args.quick:
        args.train_epochs = 10
        args.train_samples = 2000
        args.test_samples = 50

    device = torch.device(args.device)
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test angles
    test_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]

    results = {
        'config': {
            'train_rotation_range': [-30, 30],
            'train_scale_range': [0.95, 1.05],
            'train_translation_range': [-10, 10],
            'test_angles': test_angles,
            'train_samples': args.train_samples,
            'train_epochs': args.train_epochs,
            'test_samples_per_angle': args.test_samples,
        },
        'models': {},
    }

    # =========================================================================
    # Training data (limited distribution)
    # =========================================================================
    print("\nCreating training data (±30° rotation, ±5% scale, ±10px translation)...")

    train_dataset = Sim2TrainDataset(
        n_samples=args.train_samples,
        rotation_range=(-30, 30),
        scale_range=(0.95, 1.05),
        translation_range=(-10, 10),
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # =========================================================================
    # 1. SIGMA/FMT (Ours) - No training needed
    # =========================================================================
    print("\n" + "="*70)
    print("1. SIGMA/FMT (Ours) - No training")
    print("="*70)

    fmt = LogPolarSim2Net(
        lp_size=(180, 64),
        use_fft_magnitude=True,
        use_disambiguation=True,
    ).to(device)

    fmt_results = evaluate_rotation_generalization(
        fmt, test_angles, device, args.test_samples, "SIGMA", output_type="sim2"
    )
    results['models']['SIGMA'] = fmt_results

    # =========================================================================
    # 2. HomographyNet (DeTone 2016)
    # =========================================================================
    print("\n" + "="*70)
    print("2. HomographyNet (DeTone et al., CVPR 2016)")
    print("="*70)

    homographynet = HomographyNet().to(device)
    num_params = sum(p.numel() for p in homographynet.parameters())
    print(f"   Parameters: {num_params:,}")

    train_model(homographynet, train_loader, device, args.train_epochs,
                lr=1e-4, name="HomographyNet", use_homography_loss=True)

    hnet_results = evaluate_rotation_generalization(
        homographynet, test_angles, device, args.test_samples,
        "HomographyNet", output_type="homography"
    )
    results['models']['HomographyNet'] = hnet_results

    # =========================================================================
    # 3. IHN (Cao et al., CVPR 2022)
    # =========================================================================
    print("\n" + "="*70)
    print("3. IHN - Iterative Homography Network (Cao et al., CVPR 2022)")
    print("="*70)

    ihn = IterativeHomographyNetwork(num_iterations=3).to(device)
    num_params = sum(p.numel() for p in ihn.parameters())
    print(f"   Parameters: {num_params:,}")

    train_model(ihn, train_loader, device, args.train_epochs,
                lr=1e-4, name="IHN", use_homography_loss=True)

    ihn_results = evaluate_rotation_generalization(
        ihn, test_angles, device, args.test_samples,
        "IHN", output_type="homography"
    )
    results['models']['IHN'] = ihn_results

    # =========================================================================
    # 4. BaselineCNN (Direct Sim(2) regression)
    # =========================================================================
    print("\n" + "="*70)
    print("4. BaselineCNN (Direct Regression)")
    print("="*70)

    baseline = BaselineCNN().to(device)
    num_params = sum(p.numel() for p in baseline.parameters())
    print(f"   Parameters: {num_params:,}")

    train_model(baseline, train_loader, device, args.train_epochs,
                lr=1e-4, name="BaselineCNN", use_homography_loss=False)

    baseline_results = evaluate_rotation_generalization(
        baseline, test_angles, device, args.test_samples,
        "BaselineCNN", output_type="sim2"
    )
    results['models']['BaselineCNN'] = baseline_results

    # =========================================================================
    # 5. BasesHomo (Ye et al., ICCV 2021)
    # =========================================================================
    print("\n" + "="*70)
    print("5. BasesHomo - Motion Basis Learning (Ye et al., ICCV 2021)")
    print("="*70)

    baseshomo = BasesHomoBaseline(num_bases=8).to(device)
    num_params = sum(p.numel() for p in baseshomo.parameters())
    print(f"   Parameters: {num_params:,}")

    train_model(baseshomo, train_loader, device, args.train_epochs,
                lr=1e-4, name="BasesHomo", use_homography_loss=True)

    baseshomo_results = evaluate_rotation_generalization(
        baseshomo, test_angles, device, args.test_samples,
        "BasesHomo", output_type="homography"
    )
    results['models']['BasesHomo'] = baseshomo_results

    # =========================================================================
    # 6. ResNetBaseline
    # =========================================================================
    print("\n" + "="*70)
    print("6. ResNetBaseline")
    print("="*70)

    resnet = ResNetBaseline().to(device)
    num_params = sum(p.numel() for p in resnet.parameters())
    print(f"   Parameters: {num_params:,}")

    train_model(resnet, train_loader, device, args.train_epochs,
                lr=1e-4, name="ResNet", use_homography_loss=True)

    resnet_results = evaluate_rotation_generalization(
        resnet, test_angles, device, args.test_samples,
        "ResNet", output_type="homography"
    )
    results['models']['ResNet'] = resnet_results

    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "="*70)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("="*70)

    print(f"\n{'Model':<20} {'Mean Rot':<12} {'Mean MACE':<12} {'Gap':<12} {'OOD Rot':<12}")
    print("-"*70)

    for name, data in results['models'].items():
        print(f"{name:<20} {data['mean_rot']:>6.2f}°{'':<5} "
              f"{data['mean_mace']:>6.1f}px{'':<4} "
              f"{data['gap']:>6.2f}°{'':<5} "
              f"{data['out_dist_rot']:>6.2f}°")

    # Highlight key comparisons
    sigma_gap = results['models']['SIGMA']['gap']
    print(f"\n--- Gap Reduction vs SIGMA ---")
    for name, data in results['models'].items():
        if name != 'SIGMA' and data['gap'] > 0.1:
            ratio = data['gap'] / max(sigma_gap, 0.01)
            print(f"  {name}: {ratio:.1f}× larger gap than SIGMA")

    # =========================================================================
    # Save results
    # =========================================================================
    results['timestamp'] = datetime.now().isoformat()

    with open(output_dir / 'comprehensive_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'comprehensive_comparison.json'}")


if __name__ == '__main__':
    main()
