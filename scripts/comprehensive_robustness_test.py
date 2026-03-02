#!/usr/bin/env python3
"""
Comprehensive Robustness Test for SIGMA

Tests SIGMA vs baselines across:
1. Multiple seeds (8 seeds for statistical rigor)
2. Pattern types (asymmetric vs symmetric)
3. Image conditions (clean, noisy, blurry)
4. All rotation angles (0-180°)

This is the definitive test for the paper.

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
import cv2
from scipy import ndimage

sys.path.insert(0, "/data/gpfs/projects/punim2769/thermal-homography")

from src.models.log_polar_sim2_net import LogPolarSim2Net
from src.models.baseline_cnn import BaselineCNN, BaselineLoss
from src.data.synthetic_generator import (
    generate_arrow_pattern,
    generate_asymmetric_pattern,
    generate_L_shape,
    generate_T_shape,
    generate_checkerboard,
    generate_natural_texture,
    generate_homography,
)


# =============================================================================
# Pattern Categories
# =============================================================================

ASYMMETRIC_PATTERNS = {
    'arrow': generate_arrow_pattern,
    'asymmetric': generate_asymmetric_pattern,
    'L_shape': generate_L_shape,
    'T_shape': generate_T_shape,
}

SYMMETRIC_PATTERNS = {
    'checkerboard': lambda size: generate_checkerboard(size=size, squares=8),
    'natural_texture': generate_natural_texture,
}


# =============================================================================
# Image Degradation Functions
# =============================================================================

def add_gaussian_noise(img: np.ndarray, std: float = 25.0) -> np.ndarray:
    """Add Gaussian noise to image."""
    noise = np.random.randn(*img.shape) * std
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_gaussian_blur(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Add Gaussian blur to image."""
    return ndimage.gaussian_filter(img, sigma=sigma).astype(np.uint8)


def add_salt_pepper_noise(img: np.ndarray, amount: float = 0.05) -> np.ndarray:
    """Add salt and pepper noise."""
    noisy = img.copy()
    # Salt
    n_salt = int(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, n_salt) for i in img.shape]
    noisy[coords[0], coords[1]] = 255
    # Pepper
    coords = [np.random.randint(0, i - 1, n_salt) for i in img.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy


def adjust_contrast(img: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """Reduce contrast."""
    mean = img.mean()
    adjusted = mean + factor * (img.astype(np.float32) - mean)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


IMAGE_CONDITIONS = {
    'clean': lambda img: img,
    'noise_light': lambda img: add_gaussian_noise(img, std=15),
    'noise_heavy': lambda img: add_gaussian_noise(img, std=40),
    'blur_light': lambda img: add_gaussian_blur(img, sigma=1.5),
    'blur_heavy': lambda img: add_gaussian_blur(img, sigma=3.0),
    'salt_pepper': lambda img: add_salt_pepper_noise(img, amount=0.03),
    'low_contrast': lambda img: adjust_contrast(img, factor=0.4),
}


# =============================================================================
# Evaluation Functions
# =============================================================================

def generate_test_pair(
    angle: float,
    pattern_func,
    condition_func,
    seed: int,
    size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate a test pair with known rotation."""
    np.random.seed(seed)

    # Generate source
    source = pattern_func(size=size)

    # Apply homography
    H, params = generate_homography(
        rotation_range=(angle, angle),
        scale_range=(1.0, 1.0),
        translation_range=(0, 0),
        image_size=size,
    )
    target = cv2.warpPerspective(source, H, size)

    # Apply degradation to BOTH images (realistic scenario)
    source = condition_func(source)
    target = condition_func(target)

    return source, target, params['angle']  # angle in radians


def compute_rotation_error(pred_rad: float, gt_rad: float) -> float:
    """Compute rotation error in degrees with wraparound handling."""
    err_deg = abs(pred_rad - gt_rad) * 180 / math.pi
    if err_deg > 180:
        err_deg = 360 - err_deg
    return err_deg


def evaluate_model(
    model: nn.Module,
    device: torch.device,
    test_angles: List[float],
    pattern_funcs: Dict,
    condition_funcs: Dict,
    seeds: List[int],
    model_name: str = "Model",
) -> Dict:
    """Evaluate a model across all conditions."""
    model.eval()

    results = {
        'by_angle': {},
        'by_pattern': {},
        'by_condition': {},
        'detailed': [],
    }

    total_tests = len(test_angles) * len(pattern_funcs) * len(condition_funcs) * len(seeds)
    test_count = 0

    for angle in test_angles:
        angle_errors = []

        for pattern_name, pattern_func in pattern_funcs.items():
            for condition_name, condition_func in condition_funcs.items():
                for seed in seeds:
                    source, target, gt_rad = generate_test_pair(
                        angle, pattern_func, condition_func, seed
                    )

                    # To tensor
                    src_t = torch.from_numpy(source).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
                    tgt_t = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0

                    with torch.no_grad():
                        output = model(src_t, tgt_t)
                        pred_rad = output['rotation'].item()

                    error = compute_rotation_error(pred_rad, gt_rad)

                    # Store detailed result
                    results['detailed'].append({
                        'angle': angle,
                        'pattern': pattern_name,
                        'condition': condition_name,
                        'seed': seed,
                        'error': error,
                        'pred_deg': pred_rad * 180 / math.pi,
                        'gt_deg': gt_rad * 180 / math.pi,
                    })

                    angle_errors.append(error)

                    # Aggregate by pattern
                    key = f"{pattern_name}"
                    if key not in results['by_pattern']:
                        results['by_pattern'][key] = []
                    results['by_pattern'][key].append(error)

                    # Aggregate by condition
                    key = f"{condition_name}"
                    if key not in results['by_condition']:
                        results['by_condition'][key] = []
                    results['by_condition'][key].append(error)

                    test_count += 1

        results['by_angle'][angle] = {
            'mean': np.mean(angle_errors),
            'std': np.std(angle_errors),
            'median': np.median(angle_errors),
            'n': len(angle_errors),
        }

    # Summarize by pattern and condition
    for key in results['by_pattern']:
        errs = results['by_pattern'][key]
        results['by_pattern'][key] = {
            'mean': np.mean(errs),
            'std': np.std(errs),
            'n': len(errs),
        }

    for key in results['by_condition']:
        errs = results['by_condition'][key]
        results['by_condition'][key] = {
            'mean': np.mean(errs),
            'std': np.std(errs),
            'n': len(errs),
        }

    # Overall statistics
    all_errors = [d['error'] for d in results['detailed']]
    results['overall'] = {
        'mean': np.mean(all_errors),
        'std': np.std(all_errors),
        'median': np.median(all_errors),
        'pass_3deg': np.mean(np.array(all_errors) < 3.0),
        'pass_5deg': np.mean(np.array(all_errors) < 5.0),
        'pass_10deg': np.mean(np.array(all_errors) < 10.0),
        'n': len(all_errors),
    }

    return results


def print_summary(results: Dict, model_name: str):
    """Print a summary of results."""
    print(f"\n{'='*60}")
    print(f"{model_name} Summary")
    print(f"{'='*60}")

    overall = results['overall']
    print(f"\nOverall ({overall['n']} tests):")
    print(f"  Mean error: {overall['mean']:.2f}° ± {overall['std']:.2f}°")
    print(f"  Median: {overall['median']:.2f}°")
    print(f"  Pass@3°: {overall['pass_3deg']*100:.1f}%")
    print(f"  Pass@5°: {overall['pass_5deg']*100:.1f}%")
    print(f"  Pass@10°: {overall['pass_10deg']*100:.1f}%")

    print(f"\nBy Pattern:")
    for pattern, stats in sorted(results['by_pattern'].items()):
        print(f"  {pattern:<20}: {stats['mean']:.2f}° ± {stats['std']:.2f}°")

    print(f"\nBy Condition:")
    for condition, stats in sorted(results['by_condition'].items()):
        print(f"  {condition:<20}: {stats['mean']:.2f}° ± {stats['std']:.2f}°")

    print(f"\nBy Angle (selected):")
    for angle in [0, 45, 90, 135, 180]:
        if angle in results['by_angle']:
            stats = results['by_angle'][angle]
            print(f"  {angle:>3}°: {stats['mean']:.2f}° ± {stats['std']:.2f}°")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Robustness Test')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seeds', type=int, default=8, help='Number of random seeds')
    parser.add_argument('--output-dir', default='outputs/comprehensive_robustness')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer conditions')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test configuration
    test_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    seeds = list(range(args.seeds))  # [0, 1, 2, ..., 7]

    if args.quick:
        # Quick test - fewer conditions
        pattern_funcs = {'arrow': ASYMMETRIC_PATTERNS['arrow']}
        condition_funcs = {'clean': IMAGE_CONDITIONS['clean'], 'noise_light': IMAGE_CONDITIONS['noise_light']}
    else:
        # Full test - all patterns and conditions
        pattern_funcs = {**ASYMMETRIC_PATTERNS, **SYMMETRIC_PATTERNS}
        condition_funcs = IMAGE_CONDITIONS

    total_tests = len(test_angles) * len(pattern_funcs) * len(condition_funcs) * len(seeds)
    print(f"\nTest Configuration:")
    print(f"  Angles: {len(test_angles)}")
    print(f"  Patterns: {len(pattern_funcs)} ({list(pattern_funcs.keys())})")
    print(f"  Conditions: {len(condition_funcs)} ({list(condition_funcs.keys())})")
    print(f"  Seeds: {len(seeds)}")
    print(f"  Total tests per model: {total_tests}")

    results = {
        'config': {
            'test_angles': test_angles,
            'patterns': list(pattern_funcs.keys()),
            'conditions': list(condition_funcs.keys()),
            'seeds': seeds,
            'total_tests_per_model': total_tests,
        },
        'models': {},
    }

    # =========================================================================
    # 1. SIGMA (Ours)
    # =========================================================================
    print("\n" + "="*60)
    print("1. SIGMA (Ours)")
    print("="*60)

    sigma = LogPolarSim2Net(
        lp_size=(180, 64),
        use_fft_magnitude=True,
        use_disambiguation=True,
    ).to(device)
    sigma.eval()

    sigma_results = evaluate_model(
        sigma, device, test_angles, pattern_funcs, condition_funcs, seeds, "SIGMA"
    )
    results['models']['SIGMA'] = sigma_results
    print_summary(sigma_results, "SIGMA")

    # =========================================================================
    # 2. SIGMA - Asymmetric patterns only
    # =========================================================================
    print("\n" + "="*60)
    print("2. SIGMA (Asymmetric patterns only)")
    print("="*60)

    sigma_asym_results = evaluate_model(
        sigma, device, test_angles, ASYMMETRIC_PATTERNS, condition_funcs, seeds, "SIGMA-Asymmetric"
    )
    results['models']['SIGMA_asymmetric'] = sigma_asym_results
    print_summary(sigma_asym_results, "SIGMA (Asymmetric)")

    # =========================================================================
    # 3. Classical SIFT + RANSAC
    # =========================================================================
    print("\n" + "="*60)
    print("3. SIFT + RANSAC")
    print("="*60)

    sift_results = evaluate_classical_sift(
        device, test_angles, pattern_funcs, condition_funcs, seeds
    )
    results['models']['SIFT_RANSAC'] = sift_results
    print_summary(sift_results, "SIFT+RANSAC")

    # =========================================================================
    # 4. Classical ORB + RANSAC
    # =========================================================================
    print("\n" + "="*60)
    print("4. ORB + RANSAC")
    print("="*60)

    orb_results = evaluate_classical_orb(
        device, test_angles, pattern_funcs, condition_funcs, seeds
    )
    results['models']['ORB_RANSAC'] = orb_results
    print_summary(orb_results, "ORB+RANSAC")

    # =========================================================================
    # Summary Comparison
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)

    print(f"\n{'Model':<25} {'Mean':<10} {'Pass@5°':<10} {'Clean':<12} {'Noisy':<12}")
    print("-"*70)

    for name, data in results['models'].items():
        clean_err = data['by_condition'].get('clean', {}).get('mean', 0)
        noisy_err = data['by_condition'].get('noise_heavy', data['by_condition'].get('noise_light', {})).get('mean', 0)
        print(f"{name:<25} {data['overall']['mean']:.2f}°{'':<5} {data['overall']['pass_5deg']*100:.1f}%{'':<5} {clean_err:.2f}°{'':<7} {noisy_err:.2f}°")

    # Save results
    results['timestamp'] = datetime.now().isoformat()

    # Remove detailed results for JSON (too large)
    results_summary = {k: v for k, v in results.items()}
    for model in results_summary['models']:
        if 'detailed' in results_summary['models'][model]:
            del results_summary['models'][model]['detailed']

    with open(output_dir / 'comprehensive_robustness.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to {output_dir / 'comprehensive_robustness.json'}")


def evaluate_classical_sift(
    device: torch.device,
    test_angles: List[float],
    pattern_funcs: Dict,
    condition_funcs: Dict,
    seeds: List[int],
) -> Dict:
    """Evaluate SIFT + RANSAC."""
    sift = cv2.SIFT_create(nfeatures=2000)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    results = {
        'by_angle': {},
        'by_pattern': {},
        'by_condition': {},
        'detailed': [],
    }

    for angle in test_angles:
        angle_errors = []

        for pattern_name, pattern_func in pattern_funcs.items():
            for condition_name, condition_func in condition_funcs.items():
                for seed in seeds:
                    source, target, gt_rad = generate_test_pair(
                        angle, pattern_func, condition_func, seed
                    )

                    # SIFT matching
                    kp1, desc1 = sift.detectAndCompute(source, None)
                    kp2, desc2 = sift.detectAndCompute(target, None)

                    error = 180.0  # Default failure

                    if desc1 is not None and desc2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
                        try:
                            matches = matcher.knnMatch(desc1, desc2, k=2)
                            good = [m for m, n in matches if len([m, n]) == 2 and m.distance < 0.75 * n.distance]

                            if len(good) >= 4:
                                pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
                                pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

                                H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

                                if H is not None and mask is not None and np.sum(mask) >= 4:
                                    # Extract rotation from homography
                                    pred_rad = math.atan2(H[1, 0], H[0, 0])
                                    error = compute_rotation_error(pred_rad, gt_rad)
                        except:
                            pass

                    results['detailed'].append({
                        'angle': angle,
                        'pattern': pattern_name,
                        'condition': condition_name,
                        'seed': seed,
                        'error': error,
                    })

                    angle_errors.append(error)

                    # Aggregate
                    if pattern_name not in results['by_pattern']:
                        results['by_pattern'][pattern_name] = []
                    results['by_pattern'][pattern_name].append(error)

                    if condition_name not in results['by_condition']:
                        results['by_condition'][condition_name] = []
                    results['by_condition'][condition_name].append(error)

        results['by_angle'][angle] = {
            'mean': np.mean(angle_errors),
            'std': np.std(angle_errors),
            'median': np.median(angle_errors),
            'n': len(angle_errors),
        }

    # Summarize
    for key in results['by_pattern']:
        errs = results['by_pattern'][key]
        results['by_pattern'][key] = {'mean': np.mean(errs), 'std': np.std(errs), 'n': len(errs)}

    for key in results['by_condition']:
        errs = results['by_condition'][key]
        results['by_condition'][key] = {'mean': np.mean(errs), 'std': np.std(errs), 'n': len(errs)}

    all_errors = [d['error'] for d in results['detailed']]
    results['overall'] = {
        'mean': np.mean(all_errors),
        'std': np.std(all_errors),
        'median': np.median(all_errors),
        'pass_3deg': np.mean(np.array(all_errors) < 3.0),
        'pass_5deg': np.mean(np.array(all_errors) < 5.0),
        'pass_10deg': np.mean(np.array(all_errors) < 10.0),
        'n': len(all_errors),
    }

    return results


def evaluate_classical_orb(
    device: torch.device,
    test_angles: List[float],
    pattern_funcs: Dict,
    condition_funcs: Dict,
    seeds: List[int],
) -> Dict:
    """Evaluate ORB + RANSAC."""
    orb = cv2.ORB_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    results = {
        'by_angle': {},
        'by_pattern': {},
        'by_condition': {},
        'detailed': [],
    }

    for angle in test_angles:
        angle_errors = []

        for pattern_name, pattern_func in pattern_funcs.items():
            for condition_name, condition_func in condition_funcs.items():
                for seed in seeds:
                    source, target, gt_rad = generate_test_pair(
                        angle, pattern_func, condition_func, seed
                    )

                    # ORB matching
                    kp1, desc1 = orb.detectAndCompute(source, None)
                    kp2, desc2 = orb.detectAndCompute(target, None)

                    error = 180.0  # Default failure

                    if desc1 is not None and desc2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
                        try:
                            matches = matcher.knnMatch(desc1, desc2, k=2)
                            good = [m for m, n in matches if len([m, n]) == 2 and m.distance < 0.75 * n.distance]

                            if len(good) >= 4:
                                pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
                                pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

                                H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

                                if H is not None and mask is not None and np.sum(mask) >= 4:
                                    pred_rad = math.atan2(H[1, 0], H[0, 0])
                                    error = compute_rotation_error(pred_rad, gt_rad)
                        except:
                            pass

                    results['detailed'].append({
                        'angle': angle,
                        'pattern': pattern_name,
                        'condition': condition_name,
                        'seed': seed,
                        'error': error,
                    })

                    angle_errors.append(error)

                    if pattern_name not in results['by_pattern']:
                        results['by_pattern'][pattern_name] = []
                    results['by_pattern'][pattern_name].append(error)

                    if condition_name not in results['by_condition']:
                        results['by_condition'][condition_name] = []
                    results['by_condition'][condition_name].append(error)

        results['by_angle'][angle] = {
            'mean': np.mean(angle_errors),
            'std': np.std(angle_errors),
            'median': np.median(angle_errors),
            'n': len(angle_errors),
        }

    # Summarize
    for key in results['by_pattern']:
        errs = results['by_pattern'][key]
        results['by_pattern'][key] = {'mean': np.mean(errs), 'std': np.std(errs), 'n': len(errs)}

    for key in results['by_condition']:
        errs = results['by_condition'][key]
        results['by_condition'][key] = {'mean': np.mean(errs), 'std': np.std(errs), 'n': len(errs)}

    all_errors = [d['error'] for d in results['detailed']]
    results['overall'] = {
        'mean': np.mean(all_errors),
        'std': np.std(all_errors),
        'median': np.median(all_errors),
        'pass_3deg': np.mean(np.array(all_errors) < 3.0),
        'pass_5deg': np.mean(np.array(all_errors) < 5.0),
        'pass_10deg': np.mean(np.array(all_errors) < 10.0),
        'n': len(all_errors),
    }

    return results


if __name__ == '__main__':
    main()
