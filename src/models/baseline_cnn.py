"""
Baseline CNN for Homography Estimation.

This is a simple regression network that predicts rotation/scale/translation
directly from concatenated image pairs. It relies on DATA AUGMENTATION to
handle different rotations - it has NO architectural equivariance.

This serves as the baseline to show that our FMT approach generalizes
better to unseen rotations.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BaselineCNN(nn.Module):
    """
    Simple CNN baseline for Sim(2) estimation.

    Architecture:
        [src, tgt] concatenated → CNN → FC → (rotation, scale, tx, ty)

    This has NO equivariance - it must learn rotation handling from data.
    """

    def __init__(self, image_size: int = 256):
        super().__init__()

        self.image_size = image_size

        # Encoder: takes concatenated src+tgt (2 channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )

        # Regression head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),  # rotation, scale, tx, ty
        )

        # Initialize final layer small
        nn.init.zeros_(self.head[-1].bias)
        nn.init.normal_(self.head[-1].weight, std=0.01)

    def forward(self, img_src: Tensor, img_tgt: Tensor) -> dict:
        """
        Predict Sim(2) parameters from image pair.

        Args:
            img_src: [B, 1, H, W] source image
            img_tgt: [B, 1, H, W] target image

        Returns:
            dict with rotation, scale, translation
        """
        # Convert to grayscale if needed
        if img_src.shape[1] > 1:
            img_src = img_src.mean(dim=1, keepdim=True)
        if img_tgt.shape[1] > 1:
            img_tgt = img_tgt.mean(dim=1, keepdim=True)

        # Concatenate
        x = torch.cat([img_src, img_tgt], dim=1)  # [B, 2, H, W]

        # Encode and regress
        features = self.encoder(x)
        params = self.head(features)  # [B, 4]

        # Parse outputs
        rotation = params[:, 0]  # radians
        log_scale = params[:, 1]  # log(scale)
        tx = params[:, 2]  # normalized translation
        ty = params[:, 3]

        scale = torch.exp(log_scale.clamp(-1, 1))  # Clamp for stability

        return {
            'rotation': rotation,
            'rotation_deg': rotation * 180 / math.pi,
            'scale': scale,
            'log_scale': log_scale,
            'translation': torch.stack([tx, ty], dim=-1),
            'translation_x': tx,
            'translation_y': ty,
        }


class BaselineLoss(nn.Module):
    """Simple MSE loss for baseline training."""

    def __init__(self, w_rotation=1.0, w_scale=1.0, w_translation=0.5):
        super().__init__()
        self.w_rotation = w_rotation
        self.w_scale = w_scale
        self.w_translation = w_translation

    def forward(self, pred: dict, target: dict) -> dict:
        """
        Compute training loss.

        Args:
            pred: dict with rotation, scale, translation
            target: dict with rotation, scale, translation ground truth
        """
        # Rotation loss (handle wrap-around)
        rot_diff = pred['rotation'] - target['rotation']
        # Wrap to [-pi, pi]
        rot_diff = torch.atan2(torch.sin(rot_diff), torch.cos(rot_diff))
        rot_loss = (rot_diff ** 2).mean()

        # Scale loss (in log space)
        scale_loss = ((pred['log_scale'] - torch.log(target['scale'])) ** 2).mean()

        # Translation loss
        trans_loss = ((pred['translation'] - target['translation']) ** 2).sum(dim=-1).mean()

        total = (self.w_rotation * rot_loss +
                 self.w_scale * scale_loss +
                 self.w_translation * trans_loss)

        return {
            'total': total,
            'rotation': rot_loss,
            'scale': scale_loss,
            'translation': trans_loss,
        }
