"""
Log-Polar Transform for Sim(2) Equivariance.

The log-polar transform converts scale and rotation to translations:
    - Cartesian (x, y) → Log-polar (log(r), θ)
    - Scale by s: log(r') = log(s) + log(r) → translation in log(r)
    - Rotate by φ: θ' = θ + φ → translation in θ

This means a standard CNN in log-polar space becomes scale-rotation equivariant!

Key Properties:
    - Scale change → horizontal shift in log-polar image
    - Rotation → vertical shift in log-polar image
    - Standard cross-correlation can detect both

This is a well-established technique used in Fourier-Mellin transform for
image registration, and is simpler than SESN-style steerable filters.

Author: Yaqoob Ansari
Date: 2026-01-30
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class LogPolarTransform(nn.Module):
    """
    Differentiable log-polar transform.

    Converts Cartesian image to log-polar coordinates:
        (x, y) → (log(r), θ)

    where r = sqrt((x-cx)² + (y-cy)²) and θ = atan2(y-cy, x-cx).

    The output is sampled on a grid where:
        - Horizontal axis: log(r) from log(r_min) to log(r_max)
        - Vertical axis: θ from 0 to 2π

    Args:
        output_size: (height, width) of output log-polar image
                     height = number of angle samples
                     width = number of radius samples
        r_min: Minimum radius (avoid singularity at center)
        r_max: Maximum radius (usually image_size/2)
        center: If None, use image center; else (cx, cy) in [-1, 1] coords
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (180, 64),
        r_min: float = 0.1,
        r_max: Optional[float] = None,
        center: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()

        self.output_height = output_size[0]  # Number of angles
        self.output_width = output_size[1]   # Number of radii
        self.r_min = r_min
        self.r_max = r_max
        self.center = center

        # Pre-compute angle grid (constant)
        # Angles from 0 to 2π (exclusive of 2π for periodicity)
        angles = torch.linspace(0, 2 * math.pi * (1 - 1/self.output_height), self.output_height)
        self.register_buffer("angles", angles)

    def forward(
        self,
        image: Tensor,
        center: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Transform image to log-polar coordinates.

        Args:
            image: [B, C, H, W] input image
            center: [B, 2] optional per-batch center coordinates in [-1, 1]

        Returns:
            log_polar: [B, C, output_height, output_width] log-polar image
        """
        B, C, H, W = image.shape
        device = image.device

        # Determine center
        if center is not None:
            cx, cy = center[:, 0], center[:, 1]  # [B]
        elif self.center is not None:
            cx = torch.full((B,), self.center[0], device=device)
            cy = torch.full((B,), self.center[1], device=device)
        else:
            cx = torch.zeros(B, device=device)
            cy = torch.zeros(B, device=device)

        # Determine max radius
        r_max = self.r_max if self.r_max is not None else 1.0

        # Create log-radius grid
        # log(r) from log(r_min) to log(r_max)
        log_r = torch.linspace(
            math.log(self.r_min),
            math.log(r_max),
            self.output_width,
            device=device,
        )
        r = torch.exp(log_r)  # [output_width]

        # Create sampling grid
        # For each (angle, radius), compute (x, y)
        angles = self.angles.to(device)  # [output_height]

        # Expand for batch and create grid
        # r: [W'], angles: [H']
        # x = cx + r * cos(θ), y = cy + r * sin(θ)

        # Create meshgrid: [H', W']
        theta_grid, r_grid = torch.meshgrid(angles, r, indexing="ij")

        # Compute x, y coordinates: [H', W']
        x_base = r_grid * torch.cos(theta_grid)
        y_base = r_grid * torch.sin(theta_grid)

        # Add center offset and expand for batch: [B, H', W']
        x = x_base.unsqueeze(0) + cx.view(B, 1, 1)
        y = y_base.unsqueeze(0) + cy.view(B, 1, 1)

        # Stack to grid format: [B, H', W', 2]
        grid = torch.stack([x, y], dim=-1)

        # Sample using grid_sample
        # grid_sample expects grid in [-1, 1] range
        log_polar = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        return log_polar

    def extra_repr(self) -> str:
        return (
            f"output_size=({self.output_height}, {self.output_width}), "
            f"r_min={self.r_min}, r_max={self.r_max}"
        )


class InverseLogPolarTransform(nn.Module):
    """
    Inverse log-polar transform (log-polar → Cartesian).

    Useful for visualization or when you need to transform back.
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (64, 64),
        r_min: float = 0.1,
        r_max: float = 1.0,
    ):
        super().__init__()
        self.output_size = output_size
        self.r_min = r_min
        self.r_max = r_max

    def forward(self, log_polar: Tensor) -> Tensor:
        """
        Transform log-polar image back to Cartesian.

        Args:
            log_polar: [B, C, H_lp, W_lp] log-polar image

        Returns:
            cartesian: [B, C, H, W] Cartesian image
        """
        B, C, H_lp, W_lp = log_polar.shape
        H, W = self.output_size
        device = log_polar.device

        # Create Cartesian coordinate grid
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # Convert to polar
        r = torch.sqrt(xx**2 + yy**2)
        theta = torch.atan2(yy, xx)
        theta = (theta + 2 * math.pi) % (2 * math.pi)  # [0, 2π]

        # Convert to log-polar grid coordinates
        # log(r) → x coordinate in log-polar
        log_r = torch.log(r.clamp(min=self.r_min))
        log_r_min = math.log(self.r_min)
        log_r_max = math.log(self.r_max)

        # Normalize to [-1, 1]
        x_lp = 2 * (log_r - log_r_min) / (log_r_max - log_r_min) - 1
        y_lp = 2 * theta / (2 * math.pi) - 1

        # Mask points outside valid range
        mask = (r >= self.r_min) & (r <= self.r_max)

        # Create grid: [H, W, 2]
        grid = torch.stack([x_lp, y_lp], dim=-1)

        # Expand for batch: [B, H, W, 2]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)

        # Sample
        cartesian = F.grid_sample(
            log_polar,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        # Apply mask
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
        cartesian = cartesian * mask.float()

        return cartesian


class LogPolarPhaseCorrelation(nn.Module):
    """
    Phase correlation in log-polar space for scale-rotation estimation.

    In log-polar space:
        - Scale by s → shift by log(s) in the log(r) direction
        - Rotate by φ → shift by φ in the θ direction

    Phase correlation finds these shifts efficiently via FFT.

    This is the core of the Fourier-Mellin transform for image registration.
    """

    def __init__(
        self,
        lp_size: Tuple[int, int] = (180, 64),
        r_min: float = 0.05,
        r_max: float = 0.9,
        high_pass_filter: bool = True,
        window: str = "hann",
    ):
        super().__init__()

        self.lp_size = lp_size
        self.r_min = r_min
        self.r_max = r_max
        self.high_pass_filter = high_pass_filter

        # Log-polar transform
        self.log_polar = LogPolarTransform(
            output_size=lp_size,
            r_min=r_min,
            r_max=r_max,
        )

        # Pre-compute window function
        if window == "hann":
            h_win = torch.hann_window(lp_size[0])
            w_win = torch.hann_window(lp_size[1])
            window_2d = h_win.unsqueeze(1) * w_win.unsqueeze(0)
        else:
            window_2d = torch.ones(lp_size)
        self.register_buffer("window", window_2d)

        # Log scale resolution
        self.log_scale_res = (math.log(r_max) - math.log(r_min)) / lp_size[1]
        self.angle_res = 2 * math.pi / lp_size[0]

        logger.info(f"LogPolarPhaseCorrelation: size={lp_size}, r=[{r_min}, {r_max}]")
        logger.info(f"  log_scale_res={self.log_scale_res:.4f}, angle_res={math.degrees(self.angle_res):.2f}°")

    def _high_pass_filter(self, img: Tensor) -> Tensor:
        """Apply high-pass filter to enhance edges."""
        # Simple Laplacian-like filter
        kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1],
        ], device=img.device, dtype=img.dtype).view(1, 1, 3, 3)

        # Apply per channel
        B, C, H, W = img.shape
        img_flat = img.view(B * C, 1, H, W)
        filtered = F.conv2d(img_flat, kernel, padding=1)
        return filtered.view(B, C, H, W)

    def _phase_correlation(self, img1: Tensor, img2: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute phase correlation between two images.

        Returns:
            correlation: [B, H, W] phase correlation surface
            peak: [B, 2] peak location (y, x)
        """
        B = img1.shape[0]

        # Apply window
        img1 = img1 * self.window
        img2 = img2 * self.window

        # FFT
        f1 = torch.fft.fft2(img1)
        f2 = torch.fft.fft2(img2)

        # Cross-power spectrum
        cross = f1 * f2.conj()
        cross_norm = cross / (cross.abs() + 1e-8)

        # Inverse FFT
        correlation = torch.fft.ifft2(cross_norm).real

        # Sum over channels
        if correlation.dim() == 4:
            correlation = correlation.sum(dim=1)  # [B, H, W]

        # Find peak with soft-argmax for differentiability
        H, W = correlation.shape[-2:]
        correlation_flat = correlation.view(B, -1)

        # Soft-argmax
        temperature = 100.0
        weights = F.softmax(correlation_flat * temperature, dim=-1)
        indices = torch.arange(H * W, device=correlation.device, dtype=torch.float32)

        peak_flat = (weights * indices).sum(dim=-1)
        peak_y = peak_flat // W
        peak_x = peak_flat % W

        # Handle wrap-around (shifts can be negative)
        peak_y = torch.where(peak_y > H / 2, peak_y - H, peak_y)
        peak_x = torch.where(peak_x > W / 2, peak_x - W, peak_x)

        peak = torch.stack([peak_y, peak_x], dim=-1)

        return correlation, peak

    def forward(
        self,
        img_src: Tensor,
        img_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Estimate scale and rotation between images using log-polar phase correlation.

        Args:
            img_src: [B, C, H, W] source image
            img_tgt: [B, C, H, W] target image

        Returns:
            dict with:
                rotation: [B] rotation in radians
                rotation_deg: [B] rotation in degrees
                scale: [B] scale factor
                correlation: [B, H_lp, W_lp] phase correlation surface
                confidence: [B] peak sharpness
        """
        B = img_src.shape[0]

        # Optional high-pass filter
        if self.high_pass_filter:
            img_src = self._high_pass_filter(img_src)
            img_tgt = self._high_pass_filter(img_tgt)

        # Convert to log-polar
        lp_src = self.log_polar(img_src)  # [B, C, H_lp, W_lp]
        lp_tgt = self.log_polar(img_tgt)

        # Phase correlation in log-polar space
        correlation, peak = self._phase_correlation(lp_src, lp_tgt)

        # Convert peak to scale and rotation
        # peak[0] = shift in θ direction → rotation
        # peak[1] = shift in log(r) direction → log(scale)
        #
        # SIGN CONVENTION (corrected 2026-02-02 to match FMT):
        # - rotation = -peak * angle_res (NEGATION for FFT convention)
        # - log_scale = +peak * log_scale_res (NO negation - FMT convention)
        rotation = -peak[:, 0] * self.angle_res  # Negation required for FFT convention
        log_scale = peak[:, 1] * self.log_scale_res  # NO negation - FMT convention
        scale = torch.exp(log_scale)

        # Confidence from peak sharpness
        peak_val = correlation.view(B, -1).max(dim=-1).values
        mean_val = correlation.view(B, -1).mean(dim=-1)
        confidence = (peak_val - mean_val) / (mean_val.abs() + 1e-8)

        return {
            "rotation": rotation,
            "rotation_deg": rotation * 180 / math.pi,
            "scale": scale,
            "log_scale": log_scale,
            "correlation": correlation,
            "peak": peak,
            "confidence": confidence,
            "lp_src": lp_src,
            "lp_tgt": lp_tgt,
        }


class LogPolarScaleRotationEstimator(nn.Module):
    """
    Scale and rotation estimator using log-polar transform with learned features.

    Instead of raw phase correlation, this version:
    1. Converts images to log-polar
    2. Extracts learned features
    3. Uses cross-correlation on features

    This combines the geometric insight of log-polar with learned representations.
    """

    def __init__(
        self,
        lp_size: Tuple[int, int] = (180, 64),
        r_min: float = 0.05,
        r_max: float = 0.9,
        feature_channels: int = 32,
        temperature: float = 10.0,
    ):
        super().__init__()

        self.lp_size = lp_size
        self.r_min = r_min
        self.r_max = r_max

        # Log-polar transform
        self.log_polar = LogPolarTransform(
            output_size=lp_size,
            r_min=r_min,
            r_max=r_max,
        )

        # Feature extractor (operates in log-polar space)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
        )

        # Scale/rotation resolution
        self.log_scale_res = (math.log(r_max) - math.log(r_min)) / lp_size[1]
        self.angle_res = 2 * math.pi / lp_size[0]

        self.temperature = temperature

        logger.info(f"LogPolarScaleRotationEstimator: lp_size={lp_size}, features={feature_channels}")

    def forward(
        self,
        img_src: Tensor,
        img_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Estimate scale and rotation.

        Args:
            img_src: [B, C, H, W] source image
            img_tgt: [B, C, H, W] target image

        Returns:
            dict with rotation, scale, confidence, features
        """
        B = img_src.shape[0]

        # Convert to grayscale if needed
        if img_src.shape[1] > 1:
            img_src = img_src.mean(dim=1, keepdim=True)
            img_tgt = img_tgt.mean(dim=1, keepdim=True)

        # Convert to log-polar
        lp_src = self.log_polar(img_src)  # [B, 1, H_lp, W_lp]
        lp_tgt = self.log_polar(img_tgt)

        # Extract features
        feat_src = self.feature_extractor(lp_src)  # [B, C, H_lp, W_lp]
        feat_tgt = self.feature_extractor(lp_tgt)

        # Cross-correlation via FFT
        f_src = torch.fft.fft2(feat_src)
        f_tgt = torch.fft.fft2(feat_tgt)
        cross = f_src * f_tgt.conj()
        correlation = torch.fft.ifft2(cross).real  # [B, C, H, W]

        # Sum over channels
        correlation = correlation.sum(dim=1)  # [B, H, W]

        # Find peak with soft-argmax
        H, W = correlation.shape[-2:]
        corr_flat = correlation.view(B, -1)

        weights = F.softmax(corr_flat * self.temperature, dim=-1)
        indices = torch.arange(H * W, device=correlation.device, dtype=torch.float32)

        peak_flat = (weights * indices).sum(dim=-1)
        peak_y = peak_flat / W  # Fractional
        peak_x = peak_flat % W

        # Handle wrap-around
        peak_y = torch.where(peak_y > H / 2, peak_y - H, peak_y)
        peak_x = torch.where(peak_x > W / 2, peak_x - W, peak_x)

        # Convert to scale and rotation
        # SIGN CONVENTION (corrected 2026-02-02 to match FMT):
        # - rotation = -peak_y * angle_res (NEGATION for FFT convention)
        # - log_scale = +peak_x * log_scale_res (NO negation - FMT convention)
        rotation = -peak_y * self.angle_res  # Negation required for FFT convention
        log_scale = peak_x * self.log_scale_res  # NO negation - FMT convention
        scale = torch.exp(log_scale)

        # Confidence
        peak_val = corr_flat.max(dim=-1).values
        mean_val = corr_flat.mean(dim=-1)
        confidence = (peak_val - mean_val) / (mean_val.abs() + 1e-8)

        return {
            "rotation": rotation,
            "rotation_deg": rotation * 180 / math.pi,
            "scale": scale,
            "log_scale": log_scale,
            "correlation": correlation,
            "confidence": confidence,
            "feat_src": feat_src,
            "feat_tgt": feat_tgt,
            "lp_src": lp_src,
            "lp_tgt": lp_tgt,
        }
