#!/usr/bin/env python3
"""
SIGMA vs Classical FMT Robustness Comparison

This script compares SIGMA (learned Fourier-Mellin) against Classical FMT
across various image degradation conditions to show when learned features help.

Degradation conditions tested:
1. Gaussian noise at various levels (σ = 0, 5, 10, 20, 30, 50)
2. Gaussian blur at various levels (kernel = 0, 3, 5, 7, 11, 15)
3. Gamma correction (γ = 0.5, 0.75, 1.0, 1.5, 2.0)
4. Contrast scaling (0.25, 0.5, 0.75, 1.0)
5. Brightness shift (-50, -25, 0, +25, +50)

Expected outcome:
- Clean images: SIGMA ≈ Classical FMT (validates mathematical foundation)
- Heavy noise: SIGMA wins (learned features provide robustness)
- Low contrast: SIGMA wins (learned normalization helps)
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.log_polar_sim2_net import LogPolarSim2Net
from src.data.synthetic_generator import SyntheticThermalGenerator


def classical_fmt(source: torch.Tensor, target: torch.Tensor,
                  lp_size=(180, 64), r_min=0.05, r_max=0.9) -> dict:
    """
    Classical Fourier-Mellin Transform for rotation/scale estimation.
    No learned components - pure signal processing.

    FIXED 2026-02-02: Proper 180° disambiguation using spatial NCC.
    Previous version had a bug where disambiguation used the same correlation
    value for both candidates, making it effectively random.
    """
    device = source.device
    B, C, H, W = source.shape

    # 1. Compute FFT magnitude (translation-invariant)
    def fft_log_magnitude(img):
        # FFT
        f = torch.fft.fft2(img)
        f_shifted = torch.fft.fftshift(f, dim=(-2, -1))
        # Log magnitude
        mag = torch.abs(f_shifted)
        log_mag = torch.log(mag + 1e-8)
        return log_mag

    src_mag = fft_log_magnitude(source)
    tgt_mag = fft_log_magnitude(target)

    # 2. Log-polar transform
    n_angles, n_radii = lp_size
    center = torch.tensor([H/2, W/2], device=device)

    # Create log-polar grid
    angles = torch.linspace(0, 2*np.pi, n_angles, device=device)
    log_radii = torch.linspace(np.log(r_min * min(H, W) / 2),
                                np.log(r_max * min(H, W) / 2),
                                n_radii, device=device)
    radii = torch.exp(log_radii)

    # Sample points
    theta_grid, r_grid = torch.meshgrid(angles, radii, indexing='ij')
    x = center[1] + r_grid * torch.cos(theta_grid)
    y = center[0] + r_grid * torch.sin(theta_grid)

    # Normalize to [-1, 1] for grid_sample
    x_norm = 2 * x / W - 1
    y_norm = 2 * y / H - 1
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    # Sample log-polar images
    src_lp = F.grid_sample(src_mag, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    tgt_lp = F.grid_sample(tgt_mag, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    # 3. Phase correlation for rotation/scale
    src_fft = torch.fft.fft2(src_lp)
    tgt_fft = torch.fft.fft2(tgt_lp)

    cross_power = src_fft * torch.conj(tgt_fft)
    cross_power_norm = cross_power / (torch.abs(cross_power) + 1e-8)
    correlation = torch.fft.ifft2(cross_power_norm).real

    # Sum over channels
    correlation = correlation.sum(dim=1)

    # Find peak
    correlation_flat = correlation.view(B, -1)
    peak_idx = correlation_flat.argmax(dim=1)
    peak_y = peak_idx // n_radii
    peak_x = peak_idx % n_radii

    # Handle wrap-around
    peak_y = torch.where(peak_y > n_angles // 2, peak_y - n_angles, peak_y)
    peak_x = torch.where(peak_x > n_radii // 2, peak_x - n_radii, peak_x)

    # Convert to rotation and scale
    angle_res = 2 * np.pi / n_angles
    log_scale_res = (np.log(r_max / r_min)) / n_radii

    rotation = -peak_y.float() * angle_res * 180 / np.pi  # Convert to degrees
    log_scale = peak_x.float() * log_scale_res
    scale = torch.exp(log_scale)

    # 4. FIXED: Proper 180° disambiguation using spatial NCC
    # We de-rotate the target by both θ and θ+180°, then compute spatial
    # normalized cross-correlation with the source to pick the better one.

    def compute_spatial_ncc(img1, img2):
        """Compute normalized cross-correlation between two images."""
        img1_flat = img1.view(img1.size(0), -1)
        img2_flat = img2.view(img2.size(0), -1)

        # Normalize
        img1_norm = img1_flat - img1_flat.mean(dim=1, keepdim=True)
        img2_norm = img2_flat - img2_flat.mean(dim=1, keepdim=True)

        # Compute NCC
        ncc = (img1_norm * img2_norm).sum(dim=1) / (
            torch.sqrt((img1_norm**2).sum(dim=1) * (img2_norm**2).sum(dim=1)) + 1e-8
        )
        return ncc

    def apply_rotation(img, angle_deg, scale_val):
        """Apply rotation and scale using affine transform."""
        angle_rad = angle_deg * np.pi / 180
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Inverse transform matrix (to sample from rotated coordinates)
        # We want to DE-rotate the target, so we apply -angle
        theta = torch.tensor([
            [cos_a / scale_val, sin_a / scale_val, 0],
            [-sin_a / scale_val, cos_a / scale_val, 0]
        ], device=img.device, dtype=img.dtype).unsqueeze(0)

        grid = F.affine_grid(theta, img.shape, align_corners=True)
        rotated = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return rotated

    results = []
    for b in range(B):
        rot1 = rotation[b].item()
        rot2 = rot1 + 180 if rot1 <= 0 else rot1 - 180
        scale_val = scale[b].item()

        src_b = source[b:b+1]
        tgt_b = target[b:b+1]

        # De-rotate target by both candidates
        tgt_derot1 = apply_rotation(tgt_b, -rot1, scale_val)
        tgt_derot2 = apply_rotation(tgt_b, -rot2, scale_val)

        # Compute NCC with source for both
        ncc1 = compute_spatial_ncc(src_b, tgt_derot1).item()
        ncc2 = compute_spatial_ncc(src_b, tgt_derot2).item()

        # Pick the one with higher NCC
        best_rot = rot1 if ncc1 >= ncc2 else rot2

        results.append({
            'rotation': best_rot,
            'scale': scale_val
        })

    return {
        'rotation': torch.tensor([r['rotation'] for r in results], device=device),
        'scale': torch.tensor([r['scale'] for r in results], device=device)
    }


def apply_degradation(image: torch.Tensor, degradation_type: str, level: float) -> torch.Tensor:
    """Apply various degradations to test robustness."""

    if degradation_type == 'none':
        return image

    elif degradation_type == 'gaussian_noise':
        # level is noise standard deviation (0-255 scale, image is 0-1)
        noise = torch.randn_like(image) * (level / 255.0)
        return torch.clamp(image + noise, 0, 1)

    elif degradation_type == 'gaussian_blur':
        # level is kernel size (must be odd)
        if level == 0:
            return image
        kernel_size = int(level)
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = kernel_size / 3.0

        # Create Gaussian kernel
        x = torch.arange(kernel_size, device=image.device) - kernel_size // 2
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

        # Apply convolution
        B, C, H, W = image.shape
        padding = kernel_size // 2
        blurred = F.conv2d(image.view(B*C, 1, H, W), kernel_2d, padding=padding)
        return blurred.view(B, C, H, W)

    elif degradation_type == 'gamma':
        # level is gamma value
        return torch.pow(torch.clamp(image, 1e-8, 1), level)

    elif degradation_type == 'contrast':
        # level is contrast factor (0-1)
        mean = image.mean()
        return torch.clamp((image - mean) * level + mean, 0, 1)

    elif degradation_type == 'brightness':
        # level is brightness shift (-50 to +50, image is 0-1)
        return torch.clamp(image + level / 255.0, 0, 1)

    else:
        raise ValueError(f"Unknown degradation type: {degradation_type}")


def compute_mace(pred_rotation_deg: float, pred_scale: float,
                 gt_rotation_deg: float, gt_scale: float,
                 image_size: tuple = (256, 256)) -> float:
    """
    Compute Mean Average Corner Error (MACE) in pixels.

    MACE measures the average distance between predicted and ground truth
    corner positions after applying the respective homographies.
    """
    H, W = image_size
    cx, cy = W / 2, H / 2

    # Define corners (top-left, top-right, bottom-right, bottom-left)
    corners = np.array([
        [0, 0, 1],
        [W, 0, 1],
        [W, H, 1],
        [0, H, 1]
    ], dtype=np.float32).T  # Shape: (3, 4)

    def build_sim2_homography(rotation_deg, scale, tx=0, ty=0):
        """Build Sim(2) homography matrix."""
        angle_rad = rotation_deg * np.pi / 180
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Rotation about center: T2 @ R @ S @ T1
        # T1: translate center to origin
        # R @ S: rotate and scale
        # T2: translate back + any translation offset
        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
        RS = np.array([
            [scale * cos_a, -scale * sin_a, 0],
            [scale * sin_a, scale * cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        T2 = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], dtype=np.float32)

        H = T2 @ RS @ T1
        return H

    # Build homographies
    H_pred = build_sim2_homography(pred_rotation_deg, pred_scale)
    H_gt = build_sim2_homography(gt_rotation_deg, gt_scale)

    # Transform corners
    corners_pred = H_pred @ corners  # (3, 4)
    corners_gt = H_gt @ corners  # (3, 4)

    # Normalize homogeneous coordinates
    corners_pred = corners_pred[:2] / corners_pred[2:3]  # (2, 4)
    corners_gt = corners_gt[:2] / corners_gt[2:3]  # (2, 4)

    # Compute distances
    distances = np.sqrt(np.sum((corners_pred - corners_gt) ** 2, axis=0))  # (4,)

    # Mean Average Corner Error
    mace = float(np.mean(distances))

    return mace


def evaluate_method(method_name: str, model, source: torch.Tensor, target: torch.Tensor,
                   gt_rotation: float, gt_scale: float, image_size: tuple = (256, 256)) -> dict:
    """Evaluate a method on a single sample. Returns rotation error, scale error, and MACE."""

    with torch.no_grad():
        if method_name == 'SIGMA':
            output = model(source, target)
            pred_rotation = output['rotation_deg'].item()  # Use degrees output
            pred_scale = output['scale'].item()
        elif method_name == 'Classical_FMT':
            output = classical_fmt(source, target)
            pred_rotation = output['rotation'].item()
            pred_scale = output['scale'].item()
        else:
            raise ValueError(f"Unknown method: {method_name}")

    # Compute rotation error
    rot_error = abs(pred_rotation - gt_rotation)
    # Handle wrap-around
    if rot_error > 180:
        rot_error = 360 - rot_error

    # Compute scale error (percentage)
    scale_error = abs(pred_scale / gt_scale - 1) * 100

    # Compute MACE (Mean Average Corner Error in pixels)
    mace = compute_mace(pred_rotation, pred_scale, gt_rotation, gt_scale, image_size)

    return {
        'rotation_error': rot_error,
        'scale_error': scale_error,
        'mace': mace,
        'pred_rotation': pred_rotation,
        'pred_scale': pred_scale
    }


def run_comparison(args):
    """Run the full comparison experiment."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Initialize SIGMA model
    print("Loading SIGMA model...")
    sigma_model = LogPolarSim2Net(
        lp_size=(180, 64),
        r_min=0.05,
        r_max=0.9,
        use_fft_magnitude=True,
        use_disambiguation=True
    ).to(device)
    sigma_model.eval()

    # Pattern types for testing (asymmetric only for unambiguous rotation)
    pattern_types = ['arrow', 'asymmetric', 'L_shape', 'T_shape']

    # Define degradation conditions
    degradations = {
        'gaussian_noise': [0, 5, 10, 20, 30, 50],
        'gaussian_blur': [0, 3, 5, 7, 11, 15],
        'gamma': [0.5, 0.75, 1.0, 1.5, 2.0],
        'contrast': [0.25, 0.5, 0.75, 1.0],
        'brightness': [-50, -25, 0, 25, 50],
    }

    # Test angles
    test_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]

    results = {
        'config': {
            'n_samples_per_condition': args.n_samples,
            'test_angles': test_angles,
            'degradations': degradations,
            'device': str(device)
        },
        'by_degradation': {},
        'summary': {}
    }

    methods = ['SIGMA', 'Classical_FMT']

    for deg_type, levels in degradations.items():
        print(f"\n{'='*60}")
        print(f"Testing degradation: {deg_type}")
        print(f"{'='*60}")

        results['by_degradation'][deg_type] = {}

        for level in levels:
            print(f"\n  Level: {level}")

            method_results = {m: {'rotation_errors': [], 'scale_errors': [], 'mace_errors': []} for m in methods}

            for angle in tqdm(test_angles, desc=f"    Angles"):
                for sample_idx in range(args.n_samples):
                    # Generate clean sample
                    seed = hash((deg_type, level, angle, sample_idx)) % (2**32)

                    # Create generator for this specific angle and pattern
                    pattern = pattern_types[sample_idx % len(pattern_types)]
                    gen = SyntheticThermalGenerator(
                        n_samples=1,
                        image_size=(256, 256),
                        pattern_type=pattern,
                        rotation_range=(angle, angle),
                        scale_range=(1.0, 1.0),
                        translation_range=(0, 0),
                        seed=seed
                    )

                    # Get the sample
                    sample = gen[0]

                    source = sample['image_src'].unsqueeze(0).to(device)
                    target = sample['image_tgt'].unsqueeze(0).to(device)
                    gt_rotation = sample['rotation'].item() * 180 / np.pi  # Convert radians to degrees
                    gt_scale = sample['scale'].item()

                    # Apply degradation to BOTH images
                    source_deg = apply_degradation(source, deg_type, level)
                    target_deg = apply_degradation(target, deg_type, level)

                    # Evaluate both methods
                    for method in methods:
                        try:
                            result = evaluate_method(
                                method,
                                sigma_model if method == 'SIGMA' else None,
                                source_deg, target_deg,
                                gt_rotation, gt_scale
                            )
                            method_results[method]['rotation_errors'].append(result['rotation_error'])
                            method_results[method]['scale_errors'].append(result['scale_error'])
                            method_results[method]['mace_errors'].append(result['mace'])
                        except Exception as e:
                            print(f"      Error with {method}: {e}")
                            method_results[method]['rotation_errors'].append(180.0)
                            method_results[method]['scale_errors'].append(100.0)
                            method_results[method]['mace_errors'].append(999.0)

            # Compute statistics for this level
            level_results = {}
            for method in methods:
                rot_errors = np.array(method_results[method]['rotation_errors'])
                scale_errors = np.array(method_results[method]['scale_errors'])
                mace_errors = np.array(method_results[method]['mace_errors'])

                level_results[method] = {
                    'rotation_mean': float(np.mean(rot_errors)),
                    'rotation_std': float(np.std(rot_errors)),
                    'rotation_median': float(np.median(rot_errors)),
                    'scale_mean': float(np.mean(scale_errors)),
                    'scale_std': float(np.std(scale_errors)),
                    'mace_mean': float(np.mean(mace_errors)),
                    'mace_std': float(np.std(mace_errors)),
                    'mace_median': float(np.median(mace_errors)),
                    'pass_rate_3deg': float(np.mean(rot_errors < 3) * 100),
                    'pass_rate_5deg': float(np.mean(rot_errors < 5) * 100),
                    'pass_rate_3px': float(np.mean(mace_errors < 3) * 100),
                    'pass_rate_5px': float(np.mean(mace_errors < 5) * 100),
                    'n_samples': len(rot_errors)
                }

                print(f"    {method}: rot={level_results[method]['rotation_mean']:.2f}° ± {level_results[method]['rotation_std']:.2f}°, "
                      f"MACE={level_results[method]['mace_mean']:.1f}px, pass@5°={level_results[method]['pass_rate_5deg']:.1f}%")

            results['by_degradation'][deg_type][str(level)] = level_results

    # Compute summary
    print(f"\n{'='*80}")
    print("SUMMARY - ROTATION ERROR")
    print(f"{'='*80}")

    for deg_type in degradations:
        print(f"\n{deg_type}:")
        print(f"  {'Level':<10} {'SIGMA (°)':>12} {'FMT (°)':>12} {'Winner':>10} {'Diff':>10}")
        print(f"  {'-'*55}")

        for level in degradations[deg_type]:
            sigma_err = results['by_degradation'][deg_type][str(level)]['SIGMA']['rotation_mean']
            fmt_err = results['by_degradation'][deg_type][str(level)]['Classical_FMT']['rotation_mean']
            winner = 'SIGMA' if sigma_err < fmt_err else 'FMT' if fmt_err < sigma_err else 'TIE'
            diff = fmt_err - sigma_err

            print(f"  {level:<10} {sigma_err:>12.2f} {fmt_err:>12.2f} {winner:>10} {diff:>+10.2f}")

    print(f"\n{'='*80}")
    print("SUMMARY - MACE (Mean Average Corner Error in pixels)")
    print(f"{'='*80}")

    for deg_type in degradations:
        print(f"\n{deg_type}:")
        print(f"  {'Level':<10} {'SIGMA (px)':>12} {'FMT (px)':>12} {'Winner':>10} {'Diff':>10}")
        print(f"  {'-'*55}")

        for level in degradations[deg_type]:
            sigma_mace = results['by_degradation'][deg_type][str(level)]['SIGMA']['mace_mean']
            fmt_mace = results['by_degradation'][deg_type][str(level)]['Classical_FMT']['mace_mean']
            winner = 'SIGMA' if sigma_mace < fmt_mace else 'FMT' if fmt_mace < sigma_mace else 'TIE'
            diff = fmt_mace - sigma_mace

            print(f"  {level:<10} {sigma_mace:>12.1f} {fmt_mace:>12.1f} {winner:>10} {diff:>+10.1f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results['timestamp'] = datetime.now().isoformat()

    output_file = output_dir / 'sigma_vs_fmt_robustness_fixed.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='SIGMA vs Classical FMT Robustness Comparison')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples per angle per condition')
    parser.add_argument('--output_dir', type=str, default='outputs/sigma_vs_fmt',
                        help='Output directory')

    args = parser.parse_args()

    print("="*60)
    print("SIGMA vs Classical FMT Robustness Comparison")
    print("="*60)
    print(f"Samples per condition: {args.n_samples}")
    print(f"Output directory: {args.output_dir}")

    run_comparison(args)


if __name__ == '__main__':
    main()
