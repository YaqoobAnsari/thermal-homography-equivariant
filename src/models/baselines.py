"""
Baseline Models for Comparison

Implements non-equivariant baselines to demonstrate the benefit of E(2) equivariance:
1. ResNetBaseline: Standard CNN regression
2. UNetBaseline: Dense prediction followed by regression
3. CorrelationBaseline: RAFT-style correlation-based estimation
4. HomographyNet: DeTone et al. 2016 - canonical deep homography baseline
5. LucasKanadeBaseline: Classical ECC wrapper for direct method comparison
6. BasesHomoBaseline: ICCV 2021 - motion basis learning
7. IterativeHomographyNetwork: CVPR 2022 - iterative coarse-to-fine refinement
8. NonEquivariantGNN: Standard GNN for ablation study
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ResNetBlock(nn.Module):
    """Basic ResNet block with 2 conv layers."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetBaseline(nn.Module):
    """
    ResNet-based homography regression baseline.

    Similar to HomographyNet (DeTone et al., 2016) but with modern architecture.
    This baseline is NOT rotation-equivariant and must learn rotational
    invariance from data augmentation.

    Input: Concatenated image pair [B, 2, H, W]
    Output: 8-DoF homography [B, 8]
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        num_blocks: list | None = None,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]

        self.in_channels = in_channels

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, 2, 3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )

        # ResNet layers
        self.layer1 = self._make_layer(base_channels, base_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(
            base_channels * 2, base_channels * 4, num_blocks[2], stride=2
        )
        self.layer4 = self._make_layer(
            base_channels * 4, base_channels * 8, num_blocks[3], stride=2
        )

        # Global average pooling + regression head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(base_channels * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 8),
        )

        self._init_weights()

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [ResNetBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize final layer with small weights
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(
        self,
        image_src: Tensor,
        image_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Forward pass.

        Args:
            image_src: [B, C, H, W] source image
            image_tgt: [B, C, H, W] target image

        Returns:
            Dictionary with 'homography' key
        """
        # Ensure single channel
        if image_src.dim() == 3:
            image_src = image_src.unsqueeze(1)
            image_tgt = image_tgt.unsqueeze(1)
        if image_src.shape[1] == 3:
            image_src = image_src.mean(dim=1, keepdim=True)
            image_tgt = image_tgt.mean(dim=1, keepdim=True)

        # Concatenate image pair
        x = torch.cat([image_src, image_tgt], dim=1)  # [B, 2, H, W]

        # Feature extraction
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling + regression
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        homography = self.fc(x)

        return {"homography": homography}


class UNetEncoder(nn.Module):
    """U-Net encoder block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.conv(x)
        pooled = self.pool(features)
        return pooled, features


class UNetDecoder(nn.Module):
    """U-Net decoder block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetBaseline(nn.Module):
    """
    U-Net based homography regression baseline.

    Uses encoder-decoder architecture to extract dense features,
    then regresses homography from global representation.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
    ):
        super().__init__()

        # Encoder
        self.enc1 = UNetEncoder(in_channels, base_channels)
        self.enc2 = UNetEncoder(base_channels, base_channels * 2)
        self.enc3 = UNetEncoder(base_channels * 2, base_channels * 4)
        self.enc4 = UNetEncoder(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, 3, padding=1),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True),
        )

        # Decoder (for dense features)
        self.dec4 = UNetDecoder(base_channels * 16, base_channels * 8)
        self.dec3 = UNetDecoder(base_channels * 8, base_channels * 4)
        self.dec2 = UNetDecoder(base_channels * 4, base_channels * 2)
        self.dec1 = UNetDecoder(base_channels * 2, base_channels)

        # Regression head from bottleneck features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Linear(base_channels * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 8),
        )

        # Initialize final layer
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def forward(
        self,
        image_src: Tensor,
        image_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Forward pass.

        Args:
            image_src: [B, C, H, W] source image
            image_tgt: [B, C, H, W] target image

        Returns:
            Dictionary with 'homography' key
        """
        # Ensure single channel
        if image_src.dim() == 3:
            image_src = image_src.unsqueeze(1)
            image_tgt = image_tgt.unsqueeze(1)
        if image_src.shape[1] == 3:
            image_src = image_src.mean(dim=1, keepdim=True)
            image_tgt = image_tgt.mean(dim=1, keepdim=True)

        # Concatenate
        x = torch.cat([image_src, image_tgt], dim=1)

        # Encoder
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        # Bottleneck
        bottleneck = self.bottleneck(x)

        # Global pooling + regression
        global_feat = self.global_pool(bottleneck)
        global_feat = global_feat.view(global_feat.size(0), -1)
        homography = self.regressor(global_feat)

        return {"homography": homography}


class CorrBlock:
    """
    Multi-scale correlation pyramid (RAFT-style).

    Builds a pyramid of correlation volumes at multiple scales
    for robust matching across different motion magnitudes.
    """

    def __init__(self, num_levels: int = 4):
        self.num_levels = num_levels
        self.corr_pyramid = None

    def build_pyramid(self, fmap1: Tensor, fmap2: Tensor) -> None:
        """Build correlation pyramid from feature maps."""
        B, C, H, W = fmap1.shape

        # Normalize features
        fmap1 = fmap1 / (torch.norm(fmap1, dim=1, keepdim=True) + 1e-8)
        fmap2 = fmap2 / (torch.norm(fmap2, dim=1, keepdim=True) + 1e-8)

        # Compute full correlation volume
        fmap1_flat = fmap1.view(B, C, H * W)  # [B, C, H*W]
        fmap2_flat = fmap2.view(B, C, H * W)  # [B, C, H*W]
        corr = torch.bmm(fmap1_flat.transpose(1, 2), fmap2_flat)  # [B, H*W, H*W]
        corr = corr.view(B, H, W, H, W)  # [B, H, W, H, W]

        # Build pyramid with average pooling
        self.corr_pyramid = [corr]
        for _ in range(1, self.num_levels):
            # Pool the last two dimensions (target H, W)
            corr = corr.view(B, H, W, -1)
            corr = F.avg_pool2d(corr.permute(0, 3, 1, 2), 2, stride=2).permute(0, 2, 3, 1)
            _, h_new, w_new, _ = corr.shape
            corr = corr.view(B, H, W, h_new // W if h_new > W else 1, -1)
            # Simplified: just store pooled versions
            self.corr_pyramid.append(corr.view(B, H, W, -1))

    def lookup(self, coords: Tensor, radius: int = 4) -> Tensor:
        """
        Sample local correlation from pyramid at given coordinates.

        Args:
            coords: [B, H, W, 2] lookup coordinates
            radius: Sampling radius around each coordinate

        Returns:
            Sampled correlations [B, (2*radius+1)^2 * num_levels, H, W]
        """
        if self.corr_pyramid is None:
            raise RuntimeError("Must call build_pyramid first")

        B, H, W, _ = coords.shape
        device = coords.device

        # Create sampling offsets (kept for future RAFT-style local sampling)
        r = radius
        dx = torch.linspace(-r, r, 2 * r + 1, device=device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=device)
        _delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), dim=-1)  # [2r+1, 2r+1, 2]

        # For simplicity, sample from first level and pool
        corr = self.corr_pyramid[0]  # [B, H, W, H, W]
        corr_flat = corr.view(B, H * W, -1).permute(0, 2, 1)  # [B, H*W, H*W]

        # Global average as fallback for simple lookup
        sampled = corr_flat.mean(dim=1).view(B, H, W)

        return sampled.unsqueeze(1)  # [B, 1, H, W]


class ConvGRU(nn.Module):
    """Convolutional GRU for iterative refinement."""

    def __init__(self, hidden_dim: int = 128, input_dim: int = 128):
        super().__init__()

        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        """
        Args:
            h: Hidden state [B, C, H, W]
            x: Input [B, C_in, H, W]

        Returns:
            New hidden state [B, C, H, W]
        """
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


class FlowToHomographyDLT(nn.Module):
    """
    Convert dense flow field to homography using DLT.

    Uses weighted DLT (Direct Linear Transform) to fit a homography
    to the flow field predictions.
    """

    def __init__(self, image_size: tuple[int, int] = (64, 64)):
        super().__init__()
        self.H, self.W = image_size

        # Pre-compute grid coordinates
        y = torch.linspace(0, self.H - 1, self.H)
        x = torch.linspace(0, self.W - 1, self.W)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer("grid_x", grid_x.flatten())
        self.register_buffer("grid_y", grid_y.flatten())

    def forward(self, flow: Tensor, weights: Tensor | None = None) -> Tensor:
        """
        Convert flow to homography.

        Args:
            flow: [B, 2, H, W] dense flow field
            weights: [B, 1, H, W] optional confidence weights

        Returns:
            homography: [B, 8] homography parameters
        """
        B = flow.shape[0]
        device = flow.device

        # Sample points from flow
        flow_flat = flow.view(B, 2, -1)  # [B, 2, H*W]

        # Source points (grid)
        pts_src_x = self.grid_x.unsqueeze(0).expand(B, -1)  # [B, N]
        pts_src_y = self.grid_y.unsqueeze(0).expand(B, -1)  # [B, N]

        # Target points (grid + flow)
        pts_tgt_x = pts_src_x + flow_flat[:, 0]  # [B, N]
        pts_tgt_y = pts_src_y + flow_flat[:, 1]  # [B, N]

        # Subsample for efficiency
        N = pts_src_x.shape[1]
        n_sample = min(256, N)
        indices = torch.randperm(N, device=device)[:n_sample]

        pts_src_x = pts_src_x[:, indices]
        pts_src_y = pts_src_y[:, indices]
        pts_tgt_x = pts_tgt_x[:, indices]
        pts_tgt_y = pts_tgt_y[:, indices]

        # Build DLT matrix (simplified - using least squares)
        # A @ h = b where h is the 8 homography parameters
        ones = torch.ones(B, n_sample, device=device)
        zeros = torch.zeros(B, n_sample, device=device)

        # Construct A matrix for DLT
        A1 = torch.stack(
            [
                pts_src_x,
                pts_src_y,
                ones,
                zeros,
                zeros,
                zeros,
                -pts_src_x * pts_tgt_x,
                -pts_src_y * pts_tgt_x,
            ],
            dim=-1,
        )
        A2 = torch.stack(
            [
                zeros,
                zeros,
                zeros,
                pts_src_x,
                pts_src_y,
                ones,
                -pts_src_x * pts_tgt_y,
                -pts_src_y * pts_tgt_y,
            ],
            dim=-1,
        )

        A = torch.cat([A1, A2], dim=1)  # [B, 2*n_sample, 8]
        b = torch.cat([pts_tgt_x, pts_tgt_y], dim=1)  # [B, 2*n_sample]

        # Solve using least squares
        try:
            h = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)  # [B, 8]
        except RuntimeError:
            # Fallback to identity if solve fails
            h = torch.zeros(B, 8, device=device)
            h[:, 0] = 1.0  # h11 = 1
            h[:, 4] = 1.0  # h22 = 1

        return h


class CorrelationBaseline(nn.Module):
    """
    RAFT-style correlation-based homography estimation.

    Uses multi-scale correlation volumes with iterative GRU refinement
    to estimate dense correspondences, then fits homography via DLT.

    Reference: RAFT (Teed & Deng, ECCV 2020)
    """

    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
        num_iterations: int = 3,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.num_iterations = num_iterations

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

        # Context encoder (for GRU initialization)
        self.context_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_dim * 2, 3, padding=1),
        )

        # Correlation block
        self.corr_block = CorrBlock(num_levels=corr_levels)

        # Flow encoder
        self.flow_encoder = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Correlation encoder (full RAFT uses (2*r+1)^2 * levels channels)
        _corr_input_dim = (2 * corr_radius + 1) ** 2 * corr_levels  # noqa: F841
        self.corr_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 1),  # Simplified from full correlation
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Update block (GRU + flow head)
        self.update_gru = ConvGRU(hidden_dim=hidden_dim, input_dim=64 + 64)
        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, 3, padding=1),
        )

        # Flow to homography converter
        self.flow_to_H = FlowToHomographyDLT(image_size=(64, 64))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Zero-init flow head
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)

    def forward(
        self,
        image_src: Tensor,
        image_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Forward pass with iterative refinement.

        Args:
            image_src: [B, C, H, W] source image
            image_tgt: [B, C, H, W] target image

        Returns:
            Dictionary with 'homography' and 'flow' keys
        """
        # Ensure single channel
        if image_src.dim() == 3:
            image_src = image_src.unsqueeze(1)
            image_tgt = image_tgt.unsqueeze(1)
        if image_src.shape[1] == 3:
            image_src = image_src.mean(dim=1, keepdim=True)
            image_tgt = image_tgt.mean(dim=1, keepdim=True)

        B, _, H_img, W_img = image_src.shape

        # Extract features
        feat_src = self.encoder(image_src)  # [B, C, H/4, W/4]
        feat_tgt = self.encoder(image_tgt)

        _, _, H, W = feat_src.shape

        # Build correlation pyramid
        self.corr_block.build_pyramid(feat_src, feat_tgt)

        # Get context for GRU initialization
        context = self.context_encoder(image_src)
        hidden, context_feat = context.split([self.hidden_dim, self.hidden_dim], dim=1)
        hidden = torch.tanh(hidden)

        # Initialize flow
        flow = torch.zeros(B, 2, H, W, device=image_src.device)

        # Create coordinate grid for correlation lookup
        coords = torch.zeros(B, H, W, 2, device=image_src.device)

        # Iterative refinement
        flow_predictions = []
        for _ in range(self.num_iterations):
            # Lookup correlation
            corr = self.corr_block.lookup(
                coords + flow.permute(0, 2, 3, 1), radius=self.corr_radius
            )

            # Encode flow and correlation
            flow_feat = self.flow_encoder(flow)
            corr_feat = self.corr_encoder(corr)

            # GRU update
            inp = torch.cat([flow_feat, corr_feat], dim=1)
            hidden = self.update_gru(hidden, inp)

            # Predict flow delta
            delta_flow = self.flow_head(hidden)
            flow = flow + delta_flow

            flow_predictions.append(flow)

        # Convert final flow to homography
        # Upsample flow to match DLT expected size
        flow_up = F.interpolate(flow, size=(64, 64), mode="bilinear", align_corners=True)
        homography = self.flow_to_H(flow_up)

        return {
            "homography": homography,
            "flow": flow,
            "flow_predictions": flow_predictions,
        }


class NonEquivariantGNN(nn.Module):
    """
    Standard GNN without E(2) equivariance for ablation study.

    Uses the same graph structure but standard message passing
    (no geometric invariants, no local frames).
    """

    def __init__(
        self,
        in_channels: int = 32,
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_layers: int = 4,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers - 1)])
        self.norms.append(nn.LayerNorm(out_channels))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        return x


class HomographyNet(nn.Module):
    """
    HomographyNet - DeTone et al., CVPR 2016.

    The canonical deep homography baseline. VGG-style architecture
    with 8 conv layers + 2 FC layers, outputs 4-point corner offsets.

    Reference: "Deep Image Homography Estimation" (DeTone et al., 2016)
    """

    def __init__(self, input_channels: int = 2):
        super().__init__()

        self.input_channels = input_channels

        # VGG-style feature extractor: 8 conv layers in 4 blocks
        self.features = nn.Sequential(
            # Block 1: 2 -> 64, pool
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 2: 64 -> 64, pool
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 3: 64 -> 128, pool
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 4: 128 -> 128 (no pool)
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Regression head: 2 FC layers
        # For 256x256 input -> 32x32 after 3 pools -> 128*32*32 = 131072
        # But we use adaptive pool to handle variable sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 16 * 16, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 8),  # 4 corners × 2 coordinates = 8
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize final layer with small weights for stable training
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def forward(
        self,
        image_src: Tensor,
        image_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Forward pass.

        Args:
            image_src: [B, C, H, W] source image
            image_tgt: [B, C, H, W] target image

        Returns:
            Dictionary with 'homography' key (8-parameter vector)
        """
        # Ensure single channel
        if image_src.dim() == 3:
            image_src = image_src.unsqueeze(1)
            image_tgt = image_tgt.unsqueeze(1)
        if image_src.shape[1] == 3:
            image_src = image_src.mean(dim=1, keepdim=True)
            image_tgt = image_tgt.mean(dim=1, keepdim=True)

        # Concatenate image pair (as in original paper)
        x = torch.cat([image_src, image_tgt], dim=1)  # [B, 2, H, W]

        # Feature extraction
        features = self.features(x)

        # Adaptive pooling for variable input sizes
        features = self.adaptive_pool(features)

        # Regression to 4-point offsets
        four_point_offsets = self.regressor(features)

        # Convert 4-point offsets to 8-parameter homography
        homography = self._four_point_to_homography_params(four_point_offsets, image_src.shape)

        return {
            "homography": homography,
            "four_point_offsets": four_point_offsets,
        }

    def _four_point_to_homography_params(
        self,
        offsets: Tensor,
        image_shape: tuple,
    ) -> Tensor:
        """
        Convert 4-point corner offsets to 8-parameter homography.

        The offsets are relative to the four corners of the image.
        We use DLT to convert point correspondences to homography.

        Args:
            offsets: [B, 8] corner offsets (dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4)
            image_shape: Input image shape for corner positions

        Returns:
            [B, 8] homography parameters
        """
        B = offsets.shape[0]
        device = offsets.device
        H, W = image_shape[-2:]

        # Define source corners (image corners)
        corners_src = torch.tensor(
            [
                [0, 0],  # top-left
                [W, 0],  # top-right
                [W, H],  # bottom-right
                [0, H],  # bottom-left
            ],
            dtype=offsets.dtype,
            device=device,
        )
        corners_src = corners_src.unsqueeze(0).expand(B, -1, -1)  # [B, 4, 2]

        # Target corners = source corners + offsets
        offsets_reshaped = offsets.view(B, 4, 2)  # [B, 4, 2]
        corners_tgt = corners_src + offsets_reshaped

        # Compute homography using DLT
        homography = self._dlt_homography(corners_src, corners_tgt)

        return homography

    def _dlt_homography(
        self,
        pts_src: Tensor,
        pts_tgt: Tensor,
    ) -> Tensor:
        """
        Compute homography from point correspondences using DLT.

        Args:
            pts_src: [B, 4, 2] source points
            pts_tgt: [B, 4, 2] target points

        Returns:
            [B, 8] homography parameters
        """
        B = pts_src.shape[0]
        device = pts_src.device

        # Build DLT matrix
        # For each point correspondence (x, y) -> (x', y'):
        # [-x, -y, -1, 0, 0, 0, x*x', y*x', x']
        # [0, 0, 0, -x, -y, -1, x*y', y*y', y']

        x = pts_src[:, :, 0]  # [B, 4]
        y = pts_src[:, :, 1]  # [B, 4]
        xp = pts_tgt[:, :, 0]  # [B, 4]
        yp = pts_tgt[:, :, 1]  # [B, 4]

        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)

        # Construct A matrix rows
        A1 = torch.stack(
            [-x, -y, -ones, zeros, zeros, zeros, x * xp, y * xp, xp], dim=-1
        )  # [B, 4, 9]
        A2 = torch.stack(
            [zeros, zeros, zeros, -x, -y, -ones, x * yp, y * yp, yp], dim=-1
        )  # [B, 4, 9]

        # Interleave rows
        A = torch.zeros(B, 8, 9, device=device, dtype=pts_src.dtype)
        A[:, 0::2, :] = A1
        A[:, 1::2, :] = A2

        # Solve using SVD: find h that minimizes ||Ah||
        try:
            _, _, Vh = torch.linalg.svd(A)
            h = Vh[:, -1, :]  # [B, 9] - last row of V^T
        except RuntimeError:
            # Fallback to identity
            h = torch.zeros(B, 9, device=device, dtype=pts_src.dtype)
            h[:, 0] = 1.0
            h[:, 4] = 1.0
            h[:, 8] = 1.0

        # Normalize so h[8] = 1
        h = h / (h[:, 8:9] + 1e-8)

        # Return first 8 parameters
        return h[:, :8]


class LucasKanadeBaseline(nn.Module):
    """
    Lucas-Kanade / ECC wrapper for classical direct method comparison.

    This is NOT a trainable model - it uses OpenCV's ECC algorithm
    for benchmark comparison only. The forward pass runs the
    iterative optimization for each image pair.

    Reference: Lucas-Kanade 20 Years On (Baker & Matthews, 2004)
    """

    def __init__(
        self,
        num_iterations: int = 5000,
        eps: float = 1e-10,
        motion_type: str = "homography",
    ):
        super().__init__()

        self.num_iterations = num_iterations
        self.eps = eps

        # OpenCV motion type
        self.motion_types = {
            "translation": cv2.MOTION_TRANSLATION,
            "euclidean": cv2.MOTION_EUCLIDEAN,
            "affine": cv2.MOTION_AFFINE,
            "homography": cv2.MOTION_HOMOGRAPHY,
        }
        self.motion_type = self.motion_types.get(motion_type, cv2.MOTION_HOMOGRAPHY)

        # Termination criteria
        self.criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            num_iterations,
            eps,
        )

        # Dummy parameter for device detection
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(
        self,
        image_src: Tensor,
        image_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Run ECC alignment for each image pair in the batch.

        Note: This runs on CPU using OpenCV and is relatively slow.
        Use for benchmark comparison only, not for training.

        Args:
            image_src: [B, C, H, W] source image
            image_tgt: [B, C, H, W] target image

        Returns:
            Dictionary with 'homography' key
        """
        device = self.dummy.device

        # Ensure single channel
        if image_src.dim() == 3:
            image_src = image_src.unsqueeze(1)
            image_tgt = image_tgt.unsqueeze(1)
        if image_src.shape[1] == 3:
            image_src = image_src.mean(dim=1, keepdim=True)
            image_tgt = image_tgt.mean(dim=1, keepdim=True)

        B = image_src.shape[0]
        homographies = []
        converged = []

        for i in range(B):
            # Convert to numpy
            src_np = (image_src[i, 0].detach().cpu().numpy() * 255).astype(np.uint8)
            tgt_np = (image_tgt[i, 0].detach().cpu().numpy() * 255).astype(np.uint8)

            # Initialize with identity
            if self.motion_type == cv2.MOTION_HOMOGRAPHY:
                warp_matrix = np.eye(3, dtype=np.float32)
            else:
                warp_matrix = np.eye(2, 3, dtype=np.float32)

            try:
                # Run ECC
                _, warp_matrix = cv2.findTransformECC(
                    tgt_np,
                    src_np,
                    warp_matrix,
                    self.motion_type,
                    self.criteria,
                )
                success = True
            except cv2.error:
                # Convergence failure - return identity
                if self.motion_type == cv2.MOTION_HOMOGRAPHY:
                    warp_matrix = np.eye(3, dtype=np.float32)
                else:
                    warp_matrix = np.eye(2, 3, dtype=np.float32)
                success = False

            # Convert to homography vector
            if self.motion_type != cv2.MOTION_HOMOGRAPHY:
                # Pad affine to homography
                H = np.eye(3, dtype=np.float32)
                H[:2, :] = warp_matrix
            else:
                H = warp_matrix

            # Normalize
            H = H / (H[2, 2] + 1e-8)

            # Convert to 8-parameter vector
            h_vec = np.array(
                [
                    H[0, 0],
                    H[0, 1],
                    H[0, 2],
                    H[1, 0],
                    H[1, 1],
                    H[1, 2],
                    H[2, 0],
                    H[2, 1],
                ],
                dtype=np.float32,
            )

            homographies.append(torch.from_numpy(h_vec))
            converged.append(success)

        homographies = torch.stack(homographies).to(device)
        converged = torch.tensor(converged, dtype=torch.bool, device=device)

        return {
            "homography": homographies,
            "converged": converged,
        }


class ResNetEncoder(nn.Module):
    """Simple ResNet-style encoder for feature extraction."""

    def __init__(self, in_channels: int = 1, out_channels: int = 256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            # ResBlock 1
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # ResBlock 2
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class BasesHomoBaseline(nn.Module):
    """
    Motion Basis Learning for Unsupervised Deep Homography (ICCV 2021).

    Key insight: Predict homography as weighted sum of learned motion bases.
    This provides better inductive bias for the 8-DoF homography space.

    Reference: "Motion Basis Learning for Unsupervised Deep Homography
               Estimation" (Ye et al., ICCV 2021)
    """

    def __init__(
        self,
        num_bases: int = 8,
        feature_dim: int = 256,
    ):
        super().__init__()

        self.num_bases = num_bases
        self.feature_dim = feature_dim

        # Feature encoder (shared)
        self.encoder = ResNetEncoder(in_channels=1, out_channels=feature_dim)

        # Basis predictor: predicts K motion bases from source features
        self.basis_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_bases * 8),  # K bases × 8 parameters
        )

        # Coefficient predictor: predicts K coefficients from concatenated features
        self.coef_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_bases),  # K coefficients
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize basis predictor to produce near-identity bases
        nn.init.zeros_(self.basis_predictor[-1].weight)
        # Initialize biases to represent identity transforms distributed across bases
        identity_bias = torch.zeros(self.num_bases * 8)
        for i in range(self.num_bases):
            identity_bias[i * 8 + 0] = 1.0  # h11 = 1
            identity_bias[i * 8 + 4] = 1.0  # h22 = 1
        self.basis_predictor[-1].bias.data = identity_bias

    def forward(
        self,
        image_src: Tensor,
        image_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Forward pass.

        Args:
            image_src: [B, C, H, W] source image
            image_tgt: [B, C, H, W] target image

        Returns:
            Dictionary with 'homography', 'bases', and 'coefficients'
        """
        # Ensure single channel
        if image_src.dim() == 3:
            image_src = image_src.unsqueeze(1)
            image_tgt = image_tgt.unsqueeze(1)
        if image_src.shape[1] == 3:
            image_src = image_src.mean(dim=1, keepdim=True)
            image_tgt = image_tgt.mean(dim=1, keepdim=True)

        B = image_src.shape[0]

        # Extract features
        feat_src = self.encoder(image_src)  # [B, C, H', W']
        feat_tgt = self.encoder(image_tgt)

        # Predict motion bases from source features
        bases = self.basis_predictor(feat_src)  # [B, K*8]
        bases = bases.view(B, self.num_bases, 8)  # [B, K, 8]

        # Predict coefficients from concatenated features
        feat_concat = torch.cat([feat_src, feat_tgt], dim=1)  # [B, 2C, H', W']
        coefficients = self.coef_predictor(feat_concat)  # [B, K]
        coefficients = F.softmax(coefficients, dim=-1)  # Normalize to sum to 1

        # Weighted sum of bases
        # homography = sum_k(coef_k * basis_k)
        homography = torch.einsum("bk,bkd->bd", coefficients, bases)  # [B, 8]

        return {
            "homography": homography,
            "bases": bases,
            "coefficients": coefficients,
        }


class FeaturePyramid(nn.Module):
    """Multi-scale feature pyramid for IHN."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        levels: list[int] | None = None,
    ):
        super().__init__()
        if levels is None:
            levels = [32, 64, 128]

        self.levels = len(levels)

        # Build pyramid levels
        self.stems = nn.ModuleList()
        prev_ch = in_channels

        for i, ch in enumerate(levels):
            stride = 2 if i > 0 else 1
            self.stems.append(
                nn.Sequential(
                    nn.Conv2d(prev_ch, ch, 3, stride=stride, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch, ch, 3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                )
            )
            prev_ch = ch

    def forward(self, x: Tensor) -> list[Tensor]:
        """Extract multi-scale features."""
        features = []
        for stem in self.stems:
            x = stem(x)
            features.append(x)
        return features


class HomographyRefinementBlock(nn.Module):
    """Predicts residual homography from feature pair."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(feature_dim * 2, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 8),
        )

        # Initialize to predict near-zero residual
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def forward(self, feat_warped: Tensor, feat_tgt: Tensor) -> Tensor:
        """
        Predict residual homography.

        Args:
            feat_warped: [B, C, H, W] warped source features
            feat_tgt: [B, C, H, W] target features

        Returns:
            [B, 8] residual homography parameters
        """
        combined = torch.cat([feat_warped, feat_tgt], dim=1)
        x = self.conv(combined)
        delta_h = self.regressor(x)
        return delta_h


def warp_features(features: Tensor, H: Tensor) -> Tensor:
    """
    Warp features by homography using grid_sample.

    Args:
        features: [B, C, H, W] feature maps
        H: [B, 3, 3] homography matrices

    Returns:
        Warped features [B, C, H, W]
    """
    B, C, Hf, Wf = features.shape
    device = features.device

    # Create sampling grid
    y = torch.linspace(-1, 1, Hf, device=device)
    x = torch.linspace(-1, 1, Wf, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1)  # [H, W, 3]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 3]

    # Apply homography
    H_inv = torch.linalg.inv(H)
    grid_flat = grid.view(B, -1, 3).transpose(1, 2)  # [B, 3, H*W]
    grid_warped = torch.bmm(H_inv, grid_flat)  # [B, 3, H*W]
    grid_warped = grid_warped.transpose(1, 2).view(B, Hf, Wf, 3)  # [B, H, W, 3]

    # Normalize by w
    grid_warped = grid_warped[..., :2] / (grid_warped[..., 2:3] + 1e-8)

    # Sample features
    warped = F.grid_sample(
        features, grid_warped, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    return warped


def vec_to_matrix(h_vec: Tensor) -> Tensor:
    """Convert 8D homography vector to 3x3 matrix."""
    B = h_vec.shape[0]
    device = h_vec.device

    H = torch.zeros(B, 3, 3, device=device, dtype=h_vec.dtype)
    H[:, 0, 0] = h_vec[:, 0]
    H[:, 0, 1] = h_vec[:, 1]
    H[:, 0, 2] = h_vec[:, 2]
    H[:, 1, 0] = h_vec[:, 3]
    H[:, 1, 1] = h_vec[:, 4]
    H[:, 1, 2] = h_vec[:, 5]
    H[:, 2, 0] = h_vec[:, 6]
    H[:, 2, 1] = h_vec[:, 7]
    H[:, 2, 2] = 1.0

    return H


def matrix_to_vec(H: Tensor) -> Tensor:
    """Convert 3x3 homography matrix to 8D vector."""
    H = H / (H[:, 2:3, 2:3] + 1e-8)
    return torch.stack(
        [
            H[:, 0, 0],
            H[:, 0, 1],
            H[:, 0, 2],
            H[:, 1, 0],
            H[:, 1, 1],
            H[:, 1, 2],
            H[:, 2, 0],
            H[:, 2, 1],
        ],
        dim=-1,
    )


class IterativeHomographyNetwork(nn.Module):
    """
    Iterative Homography Network (IHN) - CVPR 2022.

    Coarse-to-fine iterative refinement with feature warping.
    Each iteration refines the homography estimate by warping
    features and predicting residuals.

    Reference: "Iterative Deep Homography Estimation" (Cao et al., CVPR 2022)
    """

    def __init__(
        self,
        num_iterations: int = 3,
        feature_levels: list[int] | None = None,
    ):
        super().__init__()
        if feature_levels is None:
            feature_levels = [32, 64, 128]

        self.num_iterations = num_iterations
        self.feature_levels = feature_levels

        # Multi-scale feature pyramid
        self.feature_pyramid = FeaturePyramid(
            in_channels=1,
            levels=feature_levels,
        )

        # Refinement blocks (one per iteration, matching feature level)
        # Each block expects features from a specific pyramid level
        self.refinement_blocks = nn.ModuleList()
        for i in range(num_iterations):
            # Map iteration to feature level (coarse to fine)
            level = min(i, len(feature_levels) - 1)
            feat_dim = feature_levels[-(level + 1)]  # Start from coarsest
            self.refinement_blocks.append(HomographyRefinementBlock(feature_dim=feat_dim))

    def forward(
        self,
        image_src: Tensor,
        image_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Forward pass with iterative refinement.

        Args:
            image_src: [B, C, H, W] source image
            image_tgt: [B, C, H, W] target image

        Returns:
            Dictionary with 'homography' and 'intermediate' predictions
        """
        # Ensure single channel
        if image_src.dim() == 3:
            image_src = image_src.unsqueeze(1)
            image_tgt = image_tgt.unsqueeze(1)
        if image_src.shape[1] == 3:
            image_src = image_src.mean(dim=1, keepdim=True)
            image_tgt = image_tgt.mean(dim=1, keepdim=True)

        B = image_src.shape[0]
        device = image_src.device

        # Extract multi-scale features
        feats_src = self.feature_pyramid(image_src)
        feats_tgt = self.feature_pyramid(image_tgt)

        # Initialize with identity homography
        H = torch.eye(3, device=device, dtype=image_src.dtype)
        H = H.unsqueeze(0).expand(B, -1, -1).clone()

        # Iterative refinement (coarse to fine)
        intermediates = []
        for i, refine_block in enumerate(self.refinement_blocks):
            # Use finest level features
            level = min(i, len(feats_src) - 1)
            feat_src = feats_src[-(level + 1)]  # Start from coarsest
            feat_tgt = feats_tgt[-(level + 1)]

            # Warp source features by current homography estimate
            feat_warped = warp_features(feat_src, H)

            # Predict residual homography
            delta_h = refine_block(feat_warped, feat_tgt)

            # Convert residual to matrix
            delta_H = vec_to_matrix(delta_h)

            # Add identity to get the actual delta (since we init with zeros)
            delta_H = delta_H + torch.eye(3, device=device).unsqueeze(0)
            delta_H[:, 2, 2] = 1.0  # Ensure normalized

            # Compose: H_new = delta_H @ H
            H = torch.bmm(delta_H, H)

            # Normalize
            H = H / (H[:, 2:3, 2:3] + 1e-8)

            intermediates.append(matrix_to_vec(H))

        # Final homography
        homography = matrix_to_vec(H)

        return {
            "homography": homography,
            "intermediate": intermediates,
        }
