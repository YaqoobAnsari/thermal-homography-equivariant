"""
Sim(2) Similarity Loss Function for LogPolarSim2Net.

This module provides a comprehensive loss function that properly supervises
all components of the Sim(2) similarity estimation (4DOF: rotation, scale, tx, ty):
- Rotation (geodesic distance on SO(2))
- Scale (log-space L1)
- Translation (L2)
- Corner reprojection (geometric validation)
- Peak sharpness (correlation quality)
- Correlation supervision (direct peak location supervision)

All losses are normalized to [0, 1] range for balanced gradient contributions.

Key Insight:
    The core paradox: model starts perfect at random init but training destroys
    correlation peaks. This happens because the loss supervises outputs, not the
    intermediate correlation maps. The CNN learns shortcuts that minimize output
    loss but produce diffuse correlation peaks.

    Solution: Directly supervise the correlation map to have a sharp peak at the
    ground truth (θ, log(s)) location using KL divergence to a Gaussian target.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def correlation_supervision_loss(
    correlation: Tensor,
    rotation_gt: Tensor,
    scale_gt: Tensor,
    angle_res: float,
    log_scale_res: float,
    sigma: float = 2.0,
    loss_type: str = "kl",
) -> Tensor:
    """
    Supervise the correlation map to have a sharp peak at the ground truth location.

    This is the key loss that prevents training from destroying correlation peaks.
    Instead of hoping the output losses encourage sharp peaks, we explicitly teach
    the network what the correlation should look like.

    Mathematical basis (CORRECTED 2026-02-02 - VERIFIED WITH FFT MATH):

    FFT cross-correlation: corr = IFFT(FFT(src) * conj(FFT(tgt)))

    If target is source rotated by +θ:
    - Target features in log-polar are shifted +θ from source
    - FFT correlation peak appears at -θ/angle_res (mod H)

    Example: rotation_gt = +30° = +15 bins
    - FFT peak at: -15 mod 180 = 165 bins

    Therefore:
    - peak_y_gt = -rotation_gt / angle_res  (NEGATION REQUIRED!)
    - Model extracts: rotation = -peak_y * angle_res

    This ensures: model output = ground truth ✓

    Args:
        correlation: Raw correlation map [B, H, W] from phase correlation
        rotation_gt: Ground truth rotation in radians [B]
        scale_gt: Ground truth scale factor [B]
        angle_res: Angle resolution (radians per pixel in correlation map)
        log_scale_res: Log-scale resolution (log units per pixel)
        sigma: Standard deviation of Gaussian target (pixels). Smaller = sharper peaks.
        loss_type: "kl" for KL divergence, "mse" for MSE, "focal" for focal loss

    Returns:
        Scalar loss value
    """
    B, H, W = correlation.shape
    device = correlation.device
    dtype = correlation.dtype

    # Compute expected peak position from ground truth
    #
    # FFT CONVENTION (corrected 2026-02-02 to match FMT):
    # - For rotation +θ: peak_y = -θ / angle_res (FFT gives negative shift)
    # - For scale s: peak_x = +log(s) / log_scale_res (FMT convention - NO negation!)
    #
    # Model extraction (LogPolarCorrelationEstimator):
    # - rotation = -peak_y * angle_res  (negation for rotation)
    # - log_scale = +peak_x * log_scale_res  (NO negation for scale - FMT convention)
    #
    # This ensures model output = ground truth when peak is at GT location.
    #
    peak_y_gt = -rotation_gt / angle_res  # [B] - NEGATION for FFT convention
    peak_x_gt = torch.log(scale_gt) / log_scale_res  # [B] - NO negation (FMT convention)

    # Handle wrap-around for angles (correlation map is periodic in angle dimension)
    # Bring peak_y into [0, H) range
    peak_y_gt = peak_y_gt % H

    # Handle wrap-around for scale (may need to wrap if scale < 1)
    peak_x_gt = peak_x_gt % W

    # Create coordinate grids
    y_coords = torch.arange(H, device=device, dtype=dtype)
    x_coords = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [H, W]

    # Expand for batch: [B, H, W]
    yy = yy.unsqueeze(0).expand(B, -1, -1)
    xx = xx.unsqueeze(0).expand(B, -1, -1)

    # Expand peak positions: [B, 1, 1]
    peak_y_gt = peak_y_gt.view(B, 1, 1)
    peak_x_gt = peak_x_gt.view(B, 1, 1)

    # Compute distance from GT peak with circular wrap-around for angle dimension
    # For angle (y): use minimum of direct and wrapped distance
    dy_direct = torch.abs(yy - peak_y_gt)
    dy_wrap = H - dy_direct
    dy = torch.minimum(dy_direct, dy_wrap)

    # For scale (x): use minimum of direct and wrapped distance
    dx_direct = torch.abs(xx - peak_x_gt)
    dx_wrap = W - dx_direct
    dx = torch.minimum(dx_direct, dx_wrap)

    # Squared distance
    dist_sq = dy ** 2 + dx ** 2

    # Create Gaussian target centered at GT peak
    target = torch.exp(-dist_sq / (2 * sigma ** 2))  # [B, H, W]

    # Normalize target to be a probability distribution
    target = target / (target.sum(dim=(-2, -1), keepdim=True) + 1e-8)

    # Convert correlation to probability distribution via softmax
    pred_logits = correlation.view(B, -1)  # [B, H*W]
    pred_prob = F.softmax(pred_logits, dim=-1).view(B, H, W)

    if loss_type == "ce" or loss_type == "cross_entropy":
        # Cross-entropy: -sum(target * log(pred))
        # This is the CORRECT loss: encourages pred to have high prob where target is high
        # Lower loss when peak is at GT location, higher when peak is elsewhere
        log_pred = F.log_softmax(pred_logits, dim=-1).view(B, H, W)
        ce = -(target * log_pred).sum(dim=(-2, -1))  # [B]
        loss = ce.mean()

    elif loss_type == "nll":
        # Negative log-likelihood at GT peak location
        # Directly supervises the probability mass at the expected peak
        # Uses bilinear interpolation for sub-pixel GT locations
        peak_y_norm = (peak_y_gt.squeeze() / (H - 1)) * 2 - 1  # [B] in [-1, 1]
        peak_x_norm = (peak_x_gt.squeeze() / (W - 1)) * 2 - 1  # [B] in [-1, 1]

        # Sample pred_prob at GT location using grid_sample
        grid = torch.stack([peak_x_norm, peak_y_norm], dim=-1).view(B, 1, 1, 2)  # [B, 1, 1, 2]
        pred_prob_at_gt = F.grid_sample(
            pred_prob.unsqueeze(1), grid, mode='bilinear', padding_mode='border', align_corners=True
        ).squeeze()  # [B]

        # Negative log probability
        nll = -torch.log(pred_prob_at_gt + 1e-8)
        loss = nll.mean()

    elif loss_type == "soft_argmax":
        # Soft-argmax localization loss
        # Computes expected peak location and penalizes distance from GT
        y_coords = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
        x_coords = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)

        # Expected peak location (weighted average)
        expected_y = (pred_prob * y_coords).sum(dim=(-2, -1))  # [B]
        expected_x = (pred_prob * x_coords).sum(dim=(-2, -1))  # [B]

        # Distance to GT peak (with wrap-around for angle dimension)
        dy = expected_y - peak_y_gt.squeeze()
        dy = torch.where(torch.abs(dy) > H / 2, dy - torch.sign(dy) * H, dy)
        dx = expected_x - peak_x_gt.squeeze()
        dx = torch.where(torch.abs(dx) > W / 2, dx - torch.sign(dx) * W, dx)

        # L2 distance normalized by map dimensions
        dist = torch.sqrt(dy ** 2 + dx ** 2 + 1e-8)
        loss = dist.mean() / (H / 4)  # Normalize to roughly [0, 1]

    elif loss_type == "kl":
        # KL divergence: D_KL(target || pred)
        # NOTE: This penalizes sharp peaks! Use "ce" instead for better results.
        log_pred = torch.log(pred_prob + 1e-8)
        log_target = torch.log(target + 1e-8)
        kl = (target * (log_target - log_pred)).sum(dim=(-2, -1))  # [B]
        loss = kl.mean()

    elif loss_type == "mse":
        # MSE loss: treat correlation as regression target
        loss = F.mse_loss(pred_prob, target)

    elif loss_type == "focal":
        # Focal loss: focus on hard examples (misplaced peaks)
        # Focal weight: (1 - p_t)^gamma where p_t is prob at target location
        gamma = 2.0
        pt = (pred_prob * target).sum(dim=(-2, -1))  # [B] - prob mass at target
        focal_weight = (1 - pt) ** gamma

        # Cross-entropy with focal weight
        log_pred = torch.log(pred_prob + 1e-8)
        ce = -(target * log_pred).sum(dim=(-2, -1))  # [B]
        loss = (focal_weight * ce).mean()

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. "
                        f"Valid options: ce, nll, soft_argmax, kl, mse, focal")

    return loss


def corner_loss(
    H_pred: Tensor,
    H_gt: Tensor,
    image_size: Tuple[int, int] = (256, 256),
) -> Tensor:
    """
    Compute mean corner reprojection error.

    Args:
        H_pred: Predicted homography [B, 3, 3]
        H_gt: Ground truth homography [B, 3, 3]
        image_size: (H, W) of image

    Returns:
        Corner error per sample [B]
    """
    B = H_pred.shape[0]
    device = H_pred.device
    H, W = image_size

    # Define 4 image corners in homogeneous coordinates
    corners = torch.tensor([
        [0, 0, 1],
        [W, 0, 1],
        [W, H, 1],
        [0, H, 1],
    ], dtype=torch.float32, device=device).T  # [3, 4]

    # Expand for batch
    corners = corners.unsqueeze(0).expand(B, -1, -1)  # [B, 3, 4]

    # Apply homographies
    pred_corners = torch.bmm(H_pred, corners)  # [B, 3, 4]
    gt_corners = torch.bmm(H_gt, corners)  # [B, 3, 4]

    # Convert from homogeneous to Cartesian
    pred_corners = pred_corners[:, :2] / (pred_corners[:, 2:3] + 1e-8)  # [B, 2, 4]
    gt_corners = gt_corners[:, :2] / (gt_corners[:, 2:3] + 1e-8)  # [B, 2, 4]

    # Compute L2 distance per corner
    error = torch.norm(pred_corners - gt_corners, dim=1)  # [B, 4]

    # Mean across corners
    return error.mean(dim=1)  # [B]


class Sim2HomographyLoss(nn.Module):
    """
    Comprehensive loss for Sim(2) similarity estimation training.

    Supervises all Sim(2) components with proper normalization:
    1. Rotation: Geodesic distance on SO(2), normalized by pi
    2. Scale: Log-space L1 loss, clamped to [0, 1]
    3. Translation: L2 distance, normalized by 2
    4. Corner: Log-scaled reprojection error
    5. Peak SR: Scale-rotation peak sharpness
    6. Peak T: Translation peak sharpness

    All losses are normalized to approximately [0, 1] range to ensure
    balanced gradient contributions regardless of weight values.
    """

    def __init__(
        self,
        w_rotation: float = 1.0,
        w_scale: float = 1.0,
        w_translation: float = 0.5,
        w_corner: float = 0.1,
        w_peak_sr: float = 0.2,
        w_peak_t: float = 0.1,
        w_correlation: float = 0.0,
        image_size: Tuple[int, int] = (256, 256),
        lp_size: Tuple[int, int] = (180, 64),
        r_min: float = 0.05,
        r_max: float = 0.9,
        corner_log_scale: bool = True,
        use_dynamic_weighting: bool = False,
        correlation_sigma: float = 2.0,
        correlation_loss_type: str = "soft_argmax",
    ):
        """
        Initialize Sim2HomographyLoss.

        Args:
            w_rotation: Weight for rotation loss (primary)
            w_scale: Weight for scale loss (primary)
            w_translation: Weight for translation loss (secondary)
            w_corner: Weight for corner loss (validation)
            w_peak_sr: Weight for scale-rotation peak sharpness
            w_peak_t: Weight for translation peak sharpness
            w_correlation: Weight for correlation supervision loss (KEY - set > 0 to enable)
            image_size: (H, W) for corner error computation
            lp_size: (H, W) of log-polar correlation map
            r_min: Minimum radius for log-polar transform
            r_max: Maximum radius for log-polar transform
            corner_log_scale: Use log scaling on corner loss to reduce dynamic range
            use_dynamic_weighting: Use learned uncertainty-based weights
            correlation_sigma: Sigma for Gaussian target in correlation loss
            correlation_loss_type: Type of correlation loss ("kl", "mse", "focal")

        Note:
            Setting w_correlation > 0 enables direct supervision of correlation
            peaks. This is CRITICAL for preventing training from destroying the
            correlation structure. Recommended value: 2.0 - 5.0.
        """
        super().__init__()

        self.w_rotation = w_rotation
        self.w_scale = w_scale
        self.w_translation = w_translation
        self.w_corner = w_corner
        self.w_peak_sr = w_peak_sr
        self.w_peak_t = w_peak_t
        self.w_correlation = w_correlation
        self.image_size = image_size
        self.corner_log_scale = corner_log_scale
        self.use_dynamic_weighting = use_dynamic_weighting

        # Correlation supervision parameters
        self.lp_size = lp_size
        self.log_scale_res = (math.log(r_max) - math.log(r_min)) / lp_size[1]
        self.angle_res = 2 * math.pi / lp_size[0]
        self.correlation_sigma = correlation_sigma
        self.correlation_loss_type = correlation_loss_type

        # For dynamic weighting (Kendall et al. uncertainty weighting)
        if use_dynamic_weighting:
            # Learn log-variance for each task (now 7 tasks including correlation)
            self.log_vars = nn.Parameter(torch.zeros(7))

    def rotation_loss(self, rot_pred: Tensor, rot_gt: Tensor) -> Tensor:
        """
        Geodesic distance on SO(2).

        Uses atan2(sin(diff), cos(diff)) for proper wrap-around handling.
        Normalized by pi to get range [0, 1].

        Args:
            rot_pred: Predicted rotation in radians [B]
            rot_gt: Ground truth rotation in radians [B]

        Returns:
            Normalized rotation loss [B]
        """
        diff = rot_pred - rot_gt
        # Proper wrap-around: atan2(sin, cos) gives angle in [-pi, pi]
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        return diff.abs() / math.pi  # Normalize to [0, 1]

    def scale_loss(self, scale_pred: Tensor, scale_gt: Tensor) -> Tensor:
        """
        Log-space L1 loss for scale.

        Log-space makes 2x and 0.5x errors symmetric.
        Clamped to [0, 1] for normalization.

        Args:
            scale_pred: Predicted scale factor [B]
            scale_gt: Ground truth scale factor [B]

        Returns:
            Normalized scale loss [B]
        """
        # Clamp to valid range to avoid log(0)
        log_pred = torch.log(scale_pred.clamp(min=0.1, max=10.0))
        log_gt = torch.log(scale_gt.clamp(min=0.1, max=10.0))
        return (log_pred - log_gt).abs().clamp(max=1.0)  # Normalize to [0, 1]

    def translation_loss(self, trans_pred: Tensor, trans_gt: Tensor) -> Tensor:
        """
        L2 loss for translation.

        Expects translation in normalized [-1, 1] coordinates.
        Normalized by 2 (max distance) to get range [0, 1].

        Args:
            trans_pred: Predicted translation [B, 2]
            trans_gt: Ground truth translation [B, 2]

        Returns:
            Normalized translation loss [B]
        """
        diff = trans_pred - trans_gt
        l2 = torch.norm(diff, dim=-1)  # [B]
        return l2.clamp(max=2.0) / 2.0  # Normalize to [0, 1]

    def corner_reprojection_loss(self, H_pred: Tensor, H_gt: Tensor) -> Tensor:
        """
        Corner reprojection error with optional log scaling.

        Log scaling reduces dynamic range from [0, 200] to [0, 1],
        preventing corner loss from dominating.

        Args:
            H_pred: Predicted homography [B, 3, 3]
            H_gt: Ground truth homography [B, 3, 3]

        Returns:
            Normalized corner loss [B]
        """
        raw_error = corner_loss(H_pred, H_gt, self.image_size)

        if self.corner_log_scale:
            # log1p(x/10) / log(21) maps [0, 200] to approximately [0, 1]
            return torch.log1p(raw_error / 10.0) / math.log(21.0)
        else:
            # Linear scaling
            return raw_error / 50.0

    def peak_sharpness_loss(self, confidence: Tensor, threshold: float = 2.0) -> Tensor:
        """
        Convert peak confidence to loss.

        High confidence (sharp peak) should give low loss.
        Uses sigmoid for smooth gradient.

        Args:
            confidence: Peak confidence values [B]
            threshold: Sigmoid center point

        Returns:
            Peak sharpness loss [B]
        """
        # sigmoid(conf - threshold) gives high values for high confidence
        # 1 - sigmoid gives low loss for high confidence
        return 1.0 - torch.sigmoid(confidence - threshold)

    def forward(
        self,
        pred: Dict[str, Tensor],
        target: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Compute all loss components.

        Args:
            pred: Model output dictionary containing:
                - homography: [B, 3, 3]
                - rotation: [B] in radians
                - scale: [B]
                - translation: [B, 2]
                - confidence_sr: [B] (optional)
                - confidence_t: [B] (optional)
            target: Ground truth dictionary containing:
                - homography: [B, 3, 3]
                - rotation: [B] in radians
                - scale: [B]
                - translation: [B, 2]

        Returns:
            Dictionary with all loss components and total loss
        """
        device = pred['homography'].device
        losses = {}

        # === 1. Rotation Loss ===
        L_rot = self.rotation_loss(pred['rotation'], target['rotation'])
        losses['rotation_raw'] = L_rot.mean()
        L_rot_norm = L_rot.mean()
        losses['rotation'] = L_rot_norm

        # === 2. Scale Loss ===
        L_scale = self.scale_loss(pred['scale'], target['scale'])
        losses['scale_raw'] = L_scale.mean()
        L_scale_norm = L_scale.mean()
        losses['scale'] = L_scale_norm

        # === 3. Translation Loss ===
        L_trans = self.translation_loss(pred['translation'], target['translation'])
        losses['translation_raw'] = L_trans.mean()
        L_trans_norm = L_trans.mean()
        losses['translation'] = L_trans_norm

        # === 4. Corner Reprojection Loss ===
        L_corner = self.corner_reprojection_loss(
            pred['homography'], target['homography']
        )
        losses['corner_raw'] = corner_loss(
            pred['homography'], target['homography'], self.image_size
        ).mean()
        L_corner_norm = L_corner.mean()
        losses['corner'] = L_corner_norm

        # === 5. Peak Sharpness Loss (Scale-Rotation) ===
        if 'confidence_sr' in pred and pred['confidence_sr'] is not None:
            L_peak_sr = self.peak_sharpness_loss(pred['confidence_sr'])
            losses['peak_sr'] = L_peak_sr.mean()
        else:
            L_peak_sr = torch.tensor(0.0, device=device)
            losses['peak_sr'] = L_peak_sr

        # === 6. Peak Sharpness Loss (Translation) ===
        if 'confidence_t' in pred and pred['confidence_t'] is not None:
            L_peak_t = self.peak_sharpness_loss(pred['confidence_t'])
            losses['peak_t'] = L_peak_t.mean()
        else:
            L_peak_t = torch.tensor(0.0, device=device)
            losses['peak_t'] = L_peak_t

        # === 7. Correlation Supervision Loss ===
        # This is the KEY loss that prevents training from destroying correlation peaks.
        # Directly supervises the correlation map to have a sharp peak at GT location.
        if self.w_correlation > 0 and 'correlation_sr' in pred:
            L_corr = correlation_supervision_loss(
                correlation=pred['correlation_sr'],
                rotation_gt=target['rotation'],
                scale_gt=target['scale'],
                angle_res=self.angle_res,
                log_scale_res=self.log_scale_res,
                sigma=self.correlation_sigma,
                loss_type=self.correlation_loss_type,
            )
            losses['correlation'] = L_corr
        else:
            L_corr = torch.tensor(0.0, device=device)
            losses['correlation'] = L_corr

        # === Total Loss ===
        if self.use_dynamic_weighting:
            # Uncertainty-based weighting (Kendall et al.)
            loss_list = [
                L_rot_norm, L_scale_norm, L_trans_norm,
                L_corner_norm, losses['peak_sr'], losses['peak_t'],
                L_corr,
            ]
            total = torch.tensor(0.0, device=device)
            for i, loss in enumerate(loss_list):
                precision = torch.exp(-self.log_vars[i])
                total = total + precision * loss + self.log_vars[i]
        else:
            total = (
                self.w_rotation * L_rot_norm +
                self.w_scale * L_scale_norm +
                self.w_translation * L_trans_norm +
                self.w_corner * L_corner_norm +
                self.w_peak_sr * losses['peak_sr'] +
                self.w_peak_t * losses['peak_t'] +
                self.w_correlation * L_corr
            )

        losses['total'] = total

        return losses

    def get_weight_summary(self) -> Dict[str, float]:
        """Get current weights for logging."""
        if self.use_dynamic_weighting:
            precisions = torch.exp(-self.log_vars).detach().cpu().numpy()
            return {
                'w_rotation': float(precisions[0]),
                'w_scale': float(precisions[1]),
                'w_translation': float(precisions[2]),
                'w_corner': float(precisions[3]),
                'w_peak_sr': float(precisions[4]),
                'w_peak_t': float(precisions[5]),
                'w_correlation': float(precisions[6]),
            }
        else:
            return {
                'w_rotation': self.w_rotation,
                'w_scale': self.w_scale,
                'w_translation': self.w_translation,
                'w_corner': self.w_corner,
                'w_peak_sr': self.w_peak_sr,
                'w_peak_t': self.w_peak_t,
                'w_correlation': self.w_correlation,
            }


def create_loss_fn(
    loss_type: str = "sim2",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create loss function.

    Args:
        loss_type: Type of loss ("sim2" or "corner_only")
        **kwargs: Arguments passed to loss class

    Returns:
        Loss module
    """
    if loss_type == "sim2":
        return Sim2HomographyLoss(**kwargs)
    elif loss_type == "corner_only":
        # Fallback to corner-only loss for comparison
        return Sim2HomographyLoss(
            w_rotation=0.0,
            w_scale=0.0,
            w_translation=0.0,
            w_corner=1.0,
            w_peak_sr=0.0,
            w_peak_t=0.0,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
