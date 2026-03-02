"""
Log-Polar Sim(2) Equivariant Network for Similarity Estimation.

This model achieves true Sim(2) equivariance using the log-polar transform:
- Scale → translation in log(r) direction
- Rotation → translation in θ direction

The log-polar transform converts scale-rotation estimation into a 2D
translation problem, which can be solved via phase correlation.

Pipeline:
1. Log-polar phase correlation for scale and rotation
2. De-rotate and de-scale the target image
3. Spatial matching for translation
4. Build similarity S = T(t) @ Scale(s) @ R(θ)

Key Property:
    The log-polar transform makes a standard CNN scale-rotation equivariant
    because scale and rotation become translations in log-polar space.

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
from .log_polar_transform import (
    LogPolarTransform,
    LogPolarPhaseCorrelation,
    LogPolarScaleRotationEstimator,
)
from .differentiable_transforms import (
    rotate_image,
    scale_image,
    build_sim2_homography,
)

logger = get_logger(__name__)


# =============================================================================
# FFT MAGNITUDE FOR TRANSLATION INVARIANCE (FMT)
# =============================================================================

def fft_log_magnitude(image: Tensor) -> Tensor:
    """
    Compute centered FFT log-magnitude spectrum.

    This is TRANSLATION-INVARIANT because translation in spatial domain
    only affects the phase of the FFT, not the magnitude.

    Args:
        image: [B, C, H, W] input image

    Returns:
        log_magnitude: [B, C, H, W] centered log-magnitude spectrum
    """
    # 2D FFT
    f = torch.fft.fft2(image)
    # Shift zero frequency to center
    f_shifted = torch.fft.fftshift(f)
    # Magnitude (translation-invariant!)
    magnitude = torch.abs(f_shifted) + 1e-8
    # Log scale for better dynamic range
    log_magnitude = torch.log(magnitude)
    return log_magnitude


class LearnedLogPolarEncoder(nn.Module):
    """
    Learned feature extractor that operates in log-polar space.

    With use_fft_magnitude=True (FMT mode):
        image → FFT magnitude (translation-invariant) → log-polar → CNN features

    With use_fft_magnitude=False (legacy mode):
        image → log-polar → CNN features

    FMT mode provides translation invariance for rotation/scale estimation.
    """

    def __init__(
        self,
        lp_size: Tuple[int, int] = (180, 64),
        r_min: float = 0.05,
        r_max: float = 0.9,
        feature_channels: int = 64,
        use_fft_magnitude: bool = True,  # Enable FMT by default
    ):
        super().__init__()

        self.lp_size = lp_size
        self.r_min = r_min
        self.r_max = r_max
        self.use_fft_magnitude = use_fft_magnitude

        # Log-polar transform
        self.log_polar = LogPolarTransform(
            output_size=lp_size,
            r_min=r_min,
            r_max=r_max,
        )

        # Feature extractor in log-polar space
        # Standard CNN here is translation-equivariant, which means
        # scale-rotation equivariant in Cartesian space
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feature_channels, 5, padding=2),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )

        mode = "FMT (FFT magnitude)" if use_fft_magnitude else "spatial"
        logger.info(f"LearnedLogPolarEncoder: lp_size={lp_size}, features={feature_channels}, mode={mode}")

    def forward(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Extract features in log-polar space.

        Args:
            image: [B, C, H, W] input image

        Returns:
            lp_image: [B, C, H_lp, W_lp] log-polar representation
            features: [B, C_feat, H_lp, W_lp] features in log-polar space
        """
        # Convert to grayscale if needed
        if image.shape[1] > 1:
            image = image.mean(dim=1, keepdim=True)

        # FMT MODE: Apply FFT magnitude for translation invariance
        if self.use_fft_magnitude:
            # FFT magnitude is TRANSLATION-INVARIANT
            # Translation only affects phase, not magnitude
            image = fft_log_magnitude(image)

        # Transform to log-polar
        lp_image = self.log_polar(image)

        # Extract features
        features = self.encoder(lp_image)

        return lp_image, features


class LogPolarCorrelationEstimator(nn.Module):
    """
    Estimate scale and rotation via correlation in log-polar feature space.

    With use_fft_magnitude=True (FMT mode):
        Provides translation-invariant rotation/scale estimation.
        Returns rotation with 180° ambiguity (needs disambiguation).

    Uses learned features in log-polar space and FFT cross-correlation
    to find the 2D shift that corresponds to scale and rotation.
    """

    def __init__(
        self,
        lp_size: Tuple[int, int] = (180, 64),
        r_min: float = 0.05,
        r_max: float = 0.9,
        feature_channels: int = 64,
        temperature: float = 50.0,
        use_fft_magnitude: bool = True,  # Enable FMT by default
    ):
        super().__init__()

        self.lp_size = lp_size
        self.r_min = r_min
        self.r_max = r_max
        self.temperature = temperature
        self.use_fft_magnitude = use_fft_magnitude

        # Encoder with FMT support
        self.encoder = LearnedLogPolarEncoder(
            lp_size=lp_size,
            r_min=r_min,
            r_max=r_max,
            feature_channels=feature_channels,
            use_fft_magnitude=use_fft_magnitude,
        )

        # Resolution
        self.log_scale_res = (math.log(r_max) - math.log(r_min)) / lp_size[1]
        self.angle_res = 2 * math.pi / lp_size[0]

        # Window for reducing boundary effects
        h_win = torch.hann_window(lp_size[0])
        w_win = torch.hann_window(lp_size[1])
        window = h_win.unsqueeze(1) * w_win.unsqueeze(0)
        self.register_buffer("window", window)

    def forward(
        self,
        img_src: Tensor,
        img_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Estimate scale and rotation between images.

        Args:
            img_src: [B, C, H, W] source image
            img_tgt: [B, C, H, W] target image

        Returns:
            dict with rotation, scale, confidence, etc.
        """
        B = img_src.shape[0]
        device = img_src.device

        # Extract features in log-polar space
        lp_src, feat_src = self.encoder(img_src)
        lp_tgt, feat_tgt = self.encoder(img_tgt)

        # Apply window
        feat_src = feat_src * self.window.unsqueeze(0).unsqueeze(0)
        feat_tgt = feat_tgt * self.window.unsqueeze(0).unsqueeze(0)

        # FFT cross-correlation
        f_src = torch.fft.fft2(feat_src)
        f_tgt = torch.fft.fft2(feat_tgt)

        cross = f_src * f_tgt.conj()
        # Normalize for phase correlation
        cross_norm = cross / (cross.abs() + 1e-8)

        correlation = torch.fft.ifft2(cross_norm).real  # [B, C, H, W]

        # Sum over channels for more robust peak
        correlation = correlation.sum(dim=1)  # [B, H, W]

        # Find peak via soft-argmax (differentiable)
        H, W = correlation.shape[-2:]
        corr_flat = correlation.view(B, -1)

        # Use high temperature for sharper peak selection
        weights = F.softmax(corr_flat * self.temperature, dim=-1)

        # Create coordinate grids
        y_coords = torch.arange(H, device=device, dtype=torch.float32)
        x_coords = torch.arange(W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        coords_flat = torch.stack([yy.flatten(), xx.flatten()], dim=-1)  # [H*W, 2]

        # Weighted average for peak location
        peak = torch.einsum("bn,nd->bd", weights, coords_flat)  # [B, 2]
        peak_y, peak_x = peak[:, 0], peak[:, 1]

        # Handle wrap-around (negative shifts wrap to high indices)
        peak_y = torch.where(peak_y > H / 2, peak_y - H, peak_y)
        peak_x = torch.where(peak_x > W / 2, peak_x - W, peak_x)

        # Convert to scale and rotation
        # peak_y: shift in angle direction → rotation
        # peak_x: shift in log(r) direction → log(scale)
        #
        # SIGN CONVENTION (corrected 2026-02-02 - VERIFIED MATHEMATICALLY):
        #
        # FFT cross-correlation: corr = IFFT(FFT(src) * conj(FFT(tgt)))
        # If target is source rotated by +θ, the FFT peak is at -θ/angle_res (mod H)
        # This is because FFT correlation finds where tgt needs to shift to align with src.
        #
        # Example: rotation_gt = +30° = 15 bins
        #   - Target features are shifted +15 bins from source in log-polar
        #   - FFT peak appears at -15 bins (mod 180) = 165 bins
        #   - After wrap-around: peak_y = -15
        #   - rotation = -peak_y * angle_res = -(-15) * 2° = +30° ✓
        #
        # THEREFORE: rotation = -peak_y * angle_res (NEGATION IS CORRECT)
        #
        # SCALE SIGN (fixed 2026-02-02 based on fmt_full_pipeline.py):
        # log_scale = +peak_x * log_scale_res (NO negation)
        # This matches the working FMT implementation.
        rotation = -peak_y * self.angle_res  # Negation required for FFT convention
        log_scale = peak_x * self.log_scale_res  # NO negation - matches FMT
        scale = torch.exp(log_scale)

        # Confidence from peak sharpness
        peak_val = corr_flat.max(dim=-1).values
        mean_val = corr_flat.mean(dim=-1)
        std_val = corr_flat.std(dim=-1)
        confidence = (peak_val - mean_val) / (std_val + 1e-8)

        return {
            "rotation": rotation,
            "rotation_deg": rotation * 180 / math.pi,
            "scale": scale,
            "log_scale": log_scale,
            "correlation": correlation,
            "confidence": confidence,
            "peak": peak,
            "feat_src": feat_src,
            "feat_tgt": feat_tgt,
            "lp_src": lp_src,
            "lp_tgt": lp_tgt,
        }


class SpatialTranslationEstimator(nn.Module):
    """
    Estimate translation after scale and rotation have been removed.

    Uses spatial cross-correlation or learned matching.
    """

    def __init__(
        self,
        feature_channels: int = 32,
        temperature: float = 20.0,
    ):
        super().__init__()

        self.temperature = temperature

        # Simple feature extractor for translation estimation
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, feature_channels, 5, padding=2),
            nn.BatchNorm2d(feature_channels),
        )

    def forward(
        self,
        img_src: Tensor,
        img_aligned: Tensor,
    ) -> dict[str, Tensor]:
        """
        Estimate translation between source and aligned target.

        Args:
            img_src: [B, C, H, W] source image
            img_aligned: [B, C, H, W] target after de-rotation and de-scaling

        Returns:
            dict with translation_x, translation_y, confidence
        """
        B, _, H, W = img_src.shape
        device = img_src.device

        # Convert to grayscale
        if img_src.shape[1] > 1:
            img_src = img_src.mean(dim=1, keepdim=True)
        if img_aligned.shape[1] > 1:
            img_aligned = img_aligned.mean(dim=1, keepdim=True)

        # Extract features
        feat_src = self.encoder(img_src)
        feat_aligned = self.encoder(img_aligned)

        # FFT cross-correlation
        f_src = torch.fft.fft2(feat_src)
        f_aligned = torch.fft.fft2(feat_aligned)

        cross = f_src * f_aligned.conj()
        correlation = torch.fft.ifft2(cross).real
        correlation = correlation.sum(dim=1)  # [B, H, W]

        # Soft-argmax for peak
        corr_flat = correlation.view(B, -1)
        weights = F.softmax(corr_flat * self.temperature, dim=-1)

        y_coords = torch.arange(H, device=device, dtype=torch.float32)
        x_coords = torch.arange(W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        coords_flat = torch.stack([yy.flatten(), xx.flatten()], dim=-1)

        peak = torch.einsum("bn,nd->bd", weights, coords_flat)
        peak_y, peak_x = peak[:, 0], peak[:, 1]

        # Handle wrap-around
        peak_y = torch.where(peak_y > H / 2, peak_y - H, peak_y)
        peak_x = torch.where(peak_x > W / 2, peak_x - W, peak_x)

        # Normalize to [-1, 1] range
        # NOTE: Translation convention is complex due to feature space downsampling
        # and the alignment step. Keep original convention and let training learn.
        tx = peak_x / (W / 2)
        ty = peak_y / (H / 2)

        # Confidence
        peak_val = corr_flat.max(dim=-1).values
        mean_val = corr_flat.mean(dim=-1)
        confidence = (peak_val - mean_val) / (mean_val.abs() + 1e-8)

        return {
            "translation_x": tx,
            "translation_y": ty,
            "peak_x": peak_x,
            "peak_y": peak_y,
            "correlation": correlation,
            "confidence": confidence,
        }


class LogPolarSim2Net(nn.Module):
    """
    Sim(2)-Equivariant Network using Fourier-Mellin Transform (FMT).

    This achieves true Sim(2) equivariance with TRANSLATION INVARIANCE by:
    1. FFT magnitude extraction (translation-invariant)
    2. Log-polar transform on magnitude spectra
    3. Phase correlation for rotation/scale (with 180° ambiguity)
    4. 180° disambiguation using spatial correlation
    5. Translation estimation after alignment

    Architecture (FMT mode):
        src, tgt → FFT magnitude → LogPolar → correlation → (rotation_raw, scale)
        rotation = disambiguate_180(src, tgt, rotation_raw, scale)
        tgt_aligned = de_rotate(de_scale(tgt, scale), rotation)
        src, tgt_aligned → TranslationEstimator → translation
        H = T(t) @ S(s) @ R(θ)

    Key Properties:
        - FFT magnitude is TRANSLATION-INVARIANT (translation only affects phase)
        - Log-polar converts rotation/scale to shifts
        - 180° disambiguation resolves the FMT ambiguity
    """

    def __init__(
        self,
        lp_size: Tuple[int, int] = (180, 64),
        r_min: float = 0.05,
        r_max: float = 0.9,
        feature_channels: int = 64,
        translation_channels: int = 32,
        sr_temperature: float = 50.0,
        t_temperature: float = 20.0,
        use_learned_features: bool = True,
        use_fft_magnitude: bool = True,  # FMT mode (translation-invariant)
        use_disambiguation: bool = True,  # 180° disambiguation
    ):
        """
        Initialize LogPolarSim2Net.

        Args:
            lp_size: (height, width) of log-polar space = (n_angles, n_radii)
            r_min: Minimum radius (avoid singularity at center)
            r_max: Maximum radius
            feature_channels: Channels for log-polar features
            translation_channels: Channels for translation estimation
            sr_temperature: Temperature for scale-rotation soft-argmax
            t_temperature: Temperature for translation soft-argmax
            use_learned_features: If True, use learned features; else raw phase corr
            use_fft_magnitude: If True, use FFT magnitude for translation invariance
            use_disambiguation: If True, resolve 180° ambiguity via spatial correlation
        """
        super().__init__()

        self.lp_size = lp_size
        self.r_min = r_min
        self.r_max = r_max
        self.use_learned_features = use_learned_features
        self.use_fft_magnitude = use_fft_magnitude
        self.use_disambiguation = use_disambiguation

        # Scale-rotation estimator with FMT support
        if use_learned_features:
            self.sr_estimator = LogPolarCorrelationEstimator(
                lp_size=lp_size,
                r_min=r_min,
                r_max=r_max,
                feature_channels=feature_channels,
                temperature=sr_temperature,
                use_fft_magnitude=use_fft_magnitude,
            )
        else:
            self.sr_estimator = LogPolarPhaseCorrelation(
                lp_size=lp_size,
                r_min=r_min,
                r_max=r_max,
            )

        # Translation estimator
        self.translation_estimator = SpatialTranslationEstimator(
            feature_channels=translation_channels,
            temperature=t_temperature,
        )

        # Log resolution for similarity matrix computation
        self.log_scale_res = (math.log(r_max) - math.log(r_min)) / lp_size[1]
        self.angle_res = 2 * math.pi / lp_size[0]

        mode = "FMT" if use_fft_magnitude else "spatial"
        disamb = "with 180° disambiguation" if use_disambiguation else "no disambiguation"
        logger.info(f"LogPolarSim2Net: lp_size={lp_size}, r=[{r_min}, {r_max}], mode={mode}, {disamb}")
        logger.info(f"  angle_res={math.degrees(self.angle_res):.2f}°, "
                   f"log_scale_res={self.log_scale_res:.4f}")

    def _apply_inverse_transform(
        self,
        image: Tensor,
        rotation: Tensor,
        scale: Tensor,
    ) -> Tensor:
        """
        Apply inverse transform to align target back to source frame.

        Given detected rotation θ and scale s (meaning target ≈ S(s)·R(θ)·source),
        we use grid_sample with the FORWARD transform S(s)·R(θ) because grid_sample
        maps output coordinates to input coordinates.

        Mathematical derivation:
        - Feature at position p in source appears at S(s)·R(θ)·p in target
        - To align: aligned[p] = target[S(s)·R(θ)·p] = source[p]
        - For grid_sample: aligned[out] = target[theta @ out]
        - Therefore: theta = S(s)·R(θ) = [[s·cos(θ), -s·sin(θ)], [s·sin(θ), s·cos(θ)]]

        CRITICAL: Use FORWARD scale (s), not inverse (1/s).
        Previous bug used 1/s which was masked by tests using scale=1.0.
        """
        B = image.shape[0]
        device = image.device

        # Build affine matrix: theta = S(s) @ R(θ)
        # grid_sample maps output coords → input coords, so use forward transform.
        scale_clamped = scale.clamp(min=0.1, max=10.0)
        cos_t = torch.cos(rotation)
        sin_t = torch.sin(rotation)

        # S(s) @ R(θ) = [[s*cos, -s*sin], [s*sin, s*cos]]
        theta = torch.zeros(B, 2, 3, device=device, dtype=image.dtype)
        theta[:, 0, 0] = scale_clamped * cos_t
        theta[:, 0, 1] = -scale_clamped * sin_t
        theta[:, 1, 0] = scale_clamped * sin_t
        theta[:, 1, 1] = scale_clamped * cos_t

        grid = F.affine_grid(theta, image.size(), align_corners=True)
        aligned = F.grid_sample(
            image, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        return aligned

    def _disambiguate_rotation(
        self,
        img_src: Tensor,
        img_tgt: Tensor,
        rotation_candidate: Tensor,
        scale: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Resolve 180° ambiguity using spatial correlation.

        The FMT (Fourier-Mellin Transform) returns rotation with 180° ambiguity
        because |FFT(rotate(img, θ))| = |FFT(rotate(img, θ+180°))| for symmetric
        magnitude spectra.

        We test both θ and θ+180°, de-rotate the target, and pick the one
        that gives higher spatial correlation with the source.

        Args:
            img_src: [B, C, H, W] source image
            img_tgt: [B, C, H, W] target image
            rotation_candidate: [B] rotation in radians (with 180° ambiguity)
            scale: [B] scale factor

        Returns:
            rotation: [B] disambiguated rotation
            confidence: [B] disambiguation confidence (correlation difference)
        """
        B = img_src.shape[0]
        device = img_src.device

        # Convert to grayscale if needed
        if img_src.shape[1] > 1:
            src_gray = img_src.mean(dim=1, keepdim=True)
        else:
            src_gray = img_src
        if img_tgt.shape[1] > 1:
            tgt_gray = img_tgt.mean(dim=1, keepdim=True)
        else:
            tgt_gray = img_tgt

        # Two candidates: θ and θ + 180°
        candidates = [rotation_candidate, rotation_candidate + math.pi]
        best_rotation = rotation_candidate.clone()
        best_corr = torch.full((B,), -float('inf'), device=device)

        for theta in candidates:
            # De-rotate and de-scale target to align with source
            aligned = self._apply_inverse_transform(tgt_gray, theta, scale)

            # Compute spatial phase correlation
            f_src = torch.fft.fft2(src_gray)
            f_aligned = torch.fft.fft2(aligned)
            cross = f_src * f_aligned.conj()
            cross_norm = cross / (cross.abs() + 1e-8)
            correlation = torch.fft.ifft2(cross_norm).real

            # Sum over channels and get peak value
            corr_sum = correlation.sum(dim=1)  # [B, H, W]
            corr_flat = corr_sum.view(B, -1)
            peak_value = corr_flat.max(dim=-1).values  # [B]

            # Update best rotation where this candidate is better
            better = peak_value > best_corr
            best_rotation = torch.where(better, theta, best_rotation)
            best_corr = torch.where(better, peak_value, best_corr)

        # Confidence is how much better the chosen candidate is
        # (This is approximate since we overwrite best_corr)
        confidence = best_corr

        return best_rotation, confidence

    def forward(
        self,
        img_src: Tensor,
        img_tgt: Tensor,
        return_aligned: bool = False,
    ) -> dict[str, Tensor]:
        """
        Estimate Sim(2) transformation from source to target.

        Full FMT Pipeline:
            1. FFT magnitude extraction (translation-invariant) [if use_fft_magnitude]
            2. Log-polar phase correlation → (scale, rotation_raw) with 180° ambiguity
            3. Disambiguate 180° using spatial correlation [if use_disambiguation]
            4. Apply inverse to target → aligned target
            5. Spatial correlation → translation
            6. Build similarity S = T @ Scale @ R

        Args:
            img_src: [B, C, H, W] source image
            img_tgt: [B, C, H, W] target image (transformed version of source)
            return_aligned: If True, return intermediate aligned images

        Returns:
            dict with:
                homography: [B, 3, 3] estimated similarity matrix (kept for backward compat)
                similarity_matrix: [B, 3, 3] estimated Sim(2) matrix (alias for homography)
                rotation: [B] disambiguated rotation in radians
                rotation_raw: [B] raw rotation before disambiguation
                rotation_deg: [B] disambiguated rotation in degrees
                scale: [B] scale factor
                translation: [B, 2] translation (tx, ty) in [-1, 1]
                confidence_sr: [B] scale-rotation confidence
                confidence_t: [B] translation confidence
                confidence_disamb: [B] disambiguation confidence
                (optional) img_aligned: [B, C, H, W] target after de-rotation/scale
        """
        B, C, H, W = img_src.shape
        device = img_src.device

        # Step 1: Estimate scale and rotation via log-polar correlation (FMT)
        # This uses FFT magnitude for translation invariance (if enabled)
        # Returns rotation with 180° ambiguity
        sr_result = self.sr_estimator(img_src, img_tgt)

        rotation_raw = sr_result["rotation"]  # radians, with 180° ambiguity
        scale = sr_result["scale"]

        # Step 2: Disambiguate 180° rotation ambiguity (if enabled)
        # Tests both θ and θ+180°, picks the one with better spatial correlation
        if self.use_disambiguation:
            rotation, disamb_confidence = self._disambiguate_rotation(
                img_src, img_tgt, rotation_raw, scale
            )
        else:
            rotation = rotation_raw
            disamb_confidence = sr_result["confidence"]

        # Step 3: Apply inverse transform to align target to source
        img_aligned = self._apply_inverse_transform(img_tgt, rotation, scale)

        # Step 4: Estimate translation on aligned images
        t_result = self.translation_estimator(img_src, img_aligned)

        tx = t_result["translation_x"]
        ty = t_result["translation_y"]

        # Step 5: Build similarity matrix S = T(center) @ Scale(s) @ R(θ) @ T(-center) @ T(t)
        # Note: The transformation goes from source to target, so we use
        # the forward transformation (not inverse)
        # Convert translation from normalized [-1,1] to pixels
        translation_pixels = torch.stack([
            tx * (W / 2),
            ty * (H / 2),
        ], dim=-1)

        # CRITICAL: Build similarity matrix with rotation/scale about IMAGE CENTER
        # This matches the convention used in synthetic data generation and
        # standard image transformations. Without centering, the similarity
        # matrix would rotate about the origin (0,0), causing large corner
        # errors that increase with rotation angle.
        center = torch.tensor([[W / 2.0, H / 2.0]], device=device, dtype=img_src.dtype)
        center = center.expand(B, -1)  # [B, 2]

        homography = build_sim2_homography(
            rotation=rotation,
            scale=scale,
            translation=translation_pixels,
            center=center,
        )

        result = {
            "homography": homography,
            "rotation": rotation,
            "rotation_raw": rotation_raw,  # Before disambiguation
            "rotation_deg": rotation * 180 / math.pi,  # Disambiguated rotation in degrees
            "scale": scale,
            "log_scale": sr_result.get("log_scale", torch.log(scale)),
            "translation": torch.stack([tx, ty], dim=-1),
            "translation_x": tx,
            "translation_y": ty,
            "confidence_sr": sr_result["confidence"],
            "confidence_t": t_result["confidence"],
            "confidence_disamb": disamb_confidence,  # Disambiguation confidence
            # Always include correlation maps - needed for correlation supervision loss
            "correlation_sr": sr_result["correlation"],
            "correlation_t": t_result["correlation"],
        }

        # Alias: similarity_matrix for new API consumers
        result["similarity_matrix"] = homography  # Alias for 'homography' key

        if return_aligned:
            result["img_aligned"] = img_aligned

        return result


def create_log_polar_sim2_net(
    lp_angles: int = 180,
    lp_radii: int = 64,
    r_min: float = 0.05,
    r_max: float = 0.9,
    feature_channels: int = 64,
    use_learned_features: bool = True,
    **kwargs,
) -> LogPolarSim2Net:
    """Factory function for LogPolarSim2Net."""
    return LogPolarSim2Net(
        lp_size=(lp_angles, lp_radii),
        r_min=r_min,
        r_max=r_max,
        feature_channels=feature_channels,
        use_learned_features=use_learned_features,
        **kwargs,
    )
