#!/usr/bin/env python3
"""
Runtime Measurement Script

Measures inference time for different models on 256x256 images.
Reports average time per image and FPS.

Usage:
    python scripts/measure_runtime.py --device cuda
"""

import sys
import time
import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.log_polar_sim2_net import LogPolarSim2Net
from src.models.baseline_cnn import BaselineCNN


def measure_inference_time(
    model: nn.Module,
    input_size: tuple = (1, 1, 256, 256),
    n_warmup: int = 50,
    n_measure: int = 200,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Measure model inference time.

    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
        n_warmup: Number of warmup iterations
        n_measure: Number of measurement iterations
        device: Device for inference

    Returns:
        Dict with timing statistics
    """
    model = model.to(device)
    model.eval()

    # Create dummy inputs
    img1 = torch.randn(*input_size, device=device)
    img2 = torch.randn(*input_size, device=device)

    # Warmup
    print(f"  Warming up ({n_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(img1, img2)

    # Synchronize before measurement
    if device == "cuda":
        torch.cuda.synchronize()

    # Measure
    print(f"  Measuring ({n_measure} iterations)...")
    times = []
    with torch.no_grad():
        for _ in range(n_measure):
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(img1, img2)

            if device == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    times = np.array(times)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
        "fps": float(1000 / np.mean(times)),
        "n_measurements": n_measure,
    }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="Measure model runtime")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-warmup", type=int, default=50)
    parser.add_argument("--n-measure", type=int, default=200)
    parser.add_argument("--output", type=str, default="outputs/runtime_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print("RUNTIME MEASUREMENT")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print()

    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print()

    results = {}
    input_size = (args.batch_size, 1, args.image_size, args.image_size)

    # Measure SIGMA (LogPolarSim2Net)
    print("Measuring SIGMA (LogPolarSim2Net)...")
    try:
        model = LogPolarSim2Net()
        results["SIGMA"] = measure_inference_time(
            model, input_size, args.n_warmup, args.n_measure, args.device
        )
        results["SIGMA"]["parameters"] = count_parameters(model)
        print(f"  -> {results['SIGMA']['mean_ms']:.2f} ms ({results['SIGMA']['fps']:.1f} FPS)")
        print(f"  -> Parameters: {results['SIGMA']['parameters']:,}")
        del model
    except Exception as e:
        print(f"  -> Error: {e}")
    print()

    # Measure BaselineCNN
    print("Measuring BaselineCNN...")
    try:
        model = BaselineCNN()
        results["BaselineCNN"] = measure_inference_time(
            model, input_size, args.n_warmup, args.n_measure, args.device
        )
        results["BaselineCNN"]["parameters"] = count_parameters(model)
        print(f"  -> {results['BaselineCNN']['mean_ms']:.2f} ms ({results['BaselineCNN']['fps']:.1f} FPS)")
        print(f"  -> Parameters: {results['BaselineCNN']['parameters']:,}")
        del model
    except Exception as e:
        print(f"  -> Error: {e}")
    print()

    # Measure Classical FMT (no learning, fast)
    print("Measuring Classical FMT (FFT-based)...")
    try:
        from src.models.log_polar_transform import LogPolarTransform
        import torch.nn.functional as F

        class ClassicalFMT(nn.Module):
            """Classical Fourier-Mellin Transform without learned features."""
            def __init__(self):
                super().__init__()
                self.log_polar = LogPolarTransform(
                    output_size=(180, 64),
                    radius_range=(0.05, 0.9),
                )

            def forward(self, img1, img2):
                # FFT magnitude (translation invariant)
                f1 = torch.fft.fft2(img1)
                f2 = torch.fft.fft2(img2)
                mag1 = torch.log(torch.abs(torch.fft.fftshift(f1)) + 1e-8)
                mag2 = torch.log(torch.abs(torch.fft.fftshift(f2)) + 1e-8)

                # Log-polar transform
                lp1 = self.log_polar(mag1)
                lp2 = self.log_polar(mag2)

                # Phase correlation
                f_lp1 = torch.fft.fft2(lp1)
                f_lp2 = torch.fft.fft2(lp2)
                cross_power = f_lp1 * f_lp2.conj()
                cross_power = cross_power / (torch.abs(cross_power) + 1e-8)
                corr = torch.fft.ifft2(cross_power).real

                return {"correlation": corr}

        model = ClassicalFMT()
        results["ClassicalFMT"] = measure_inference_time(
            model, input_size, args.n_warmup, args.n_measure, args.device
        )
        results["ClassicalFMT"]["parameters"] = 0  # No learnable parameters
        print(f"  -> {results['ClassicalFMT']['mean_ms']:.2f} ms ({results['ClassicalFMT']['fps']:.1f} FPS)")
        del model
    except Exception as e:
        print(f"  -> Error: {e}")
        print("  -> Using simple FFT estimate...")
        img1 = torch.randn(*input_size, device=args.device)
        img2 = torch.randn(*input_size, device=args.device)

        times = []
        for _ in range(args.n_measure):
            if args.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            # Simulate FMT: FFT + log-polar + FFT + correlation
            f1 = torch.fft.fft2(img1)
            f2 = torch.fft.fft2(img2)
            mag1 = torch.abs(f1)
            mag2 = torch.abs(f2)
            corr = torch.fft.ifft2(f1 * f2.conj())

            if args.device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        results["ClassicalFMT"] = {
            "mean_ms": float(np.mean(times)),
            "fps": float(1000 / np.mean(times)),
            "parameters": 0,
        }
        print(f"  -> {results['ClassicalFMT']['mean_ms']:.2f} ms ({results['ClassicalFMT']['fps']:.1f} FPS)")
    print()

    # Summary table
    print("=" * 60)
    print("RUNTIME SUMMARY")
    print("=" * 60)
    print(f"{'Method':<20} {'Time (ms)':<15} {'FPS':<10} {'Parameters':<15}")
    print("-" * 60)
    for name, r in results.items():
        params = f"{r.get('parameters', 0):,}" if r.get('parameters', 0) > 0 else "0"
        print(f"{name:<20} {r['mean_ms']:<15.2f} {r['fps']:<10.1f} {params:<15}")
    print("=" * 60)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "device": args.device,
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "n_warmup": args.n_warmup,
                "n_measure": args.n_measure,
            },
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
