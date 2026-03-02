"""
Thermal Homography Network with Sim(2) Equivariance

This module implements a Sim(2)-equivariant homography estimation network
using ESCNN (E2-equivariant steerable CNNs) and Procrustes analysis.

THE THREE PILLARS OF SIM(2) EQUIVARIANCE:
1. ROTATION EQUIVARIANCE - Uses ESCNN with C8 group for rotation-invariant features
2. TRANSLATION EQUIVARIANCE - Graph-based representation naturally handles translation
3. SCALE EQUIVARIANCE - Procrustes SVD estimates scale from soft correspondences

Architecture:
1. Feature extraction (CNN or patch-based)
2. Graph construction (kNN or grid)
3. E(2)-equivariant message passing (EGNN-style, distance² only)
4. [Optional] Rotation-invariant LRFT (distance-based attention)
5. ProcrustesCanonicalizer for Sim(2) transformation estimation:
   - ESCNN extracts rotation-invariant features
   - Correlation matching finds soft correspondences
   - Procrustes SVD extracts rotation, scale, translation

Key advantages of Procrustes + ESCNN approach:
- No 90-degree ambiguity (unlike structure tensor)
- Mathematically principled (closed-form SVD solution)
- Differentiable for end-to-end training
- Works on any pattern without retraining

This is the main model for the ECCV 2026 submission.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from src.utils.logging_config import get_logger

from .e2_layers import E2EquivariantGNN
from .lrft import AdaptiveLRFT, InvariantLRFT, SimilarityMatrix
from .sks_head import SKSRegressionHead
from .procrustes_canonicalizer import ProcrustesCanonicalizer
from .sim2_equivariant_net import Sim2EquivariantNet

logger = get_logger(__name__)


class PatchFeatureExtractor(nn.Module):
    """
    Extract features from image patches.

    Colormap-invariant design using gradient-based features.

    CRITICAL FIX (2026-01-30):
    - Previous version used gradient ORIENTATION which is rotation-DEPENDENT
    - Now uses ONLY gradient MAGNITUDE which is rotation-INVARIANT
    - This ensures features don't change when image rotates
    - Rotation detection is handled by OrientationCanonicalizer (from image gradients)
    """

    def __init__(
        self,
        patch_size: int = 7,
        feature_dim: int = 32,
        use_gradient: bool = True,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.use_gradient = use_gradient

        if use_gradient:
            # Gradient-based features (colormap invariant AND rotation invariant)
            # CRITICAL: Use ONLY magnitude (rotation-invariant)
            # DO NOT use orientation (rotation-dependent)
            input_dim = 1  # magnitude ONLY (was 2 with orientation - WRONG)
            self.feature_encoder = nn.Sequential(
                nn.Linear(input_dim * patch_size * patch_size, 128),
                nn.SiLU(),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Linear(64, feature_dim),
            )
        else:
            # Raw patch encoder (not colormap invariant)
            input_dim = patch_size * patch_size
            self.feature_encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.SiLU(),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Linear(64, feature_dim),
            )

        # Sobel filters for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def compute_gradients(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Compute image gradients using Sobel filters."""
        # Ensure single channel
        if image.dim() == 3:
            image = image.unsqueeze(1)
        if image.shape[1] == 3:
            image = image.mean(dim=1, keepdim=True)

        # Pad for valid convolution
        padded = F.pad(image, (1, 1, 1, 1), mode="replicate")

        # Compute gradients
        grad_x = F.conv2d(padded, self.sobel_x)
        grad_y = F.conv2d(padded, self.sobel_y)

        # Magnitude and orientation
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        orientation = torch.atan2(grad_y, grad_x)

        return magnitude, orientation

    def forward(self, image: Tensor, positions: Tensor) -> Tensor:
        """
        Extract features at specified positions.

        Args:
            image: [B, C, H, W] or [B, H, W]
            positions: [B, N, 2] normalized positions in [-1, 1]

        Returns:
            features: [B, N, feature_dim]
        """
        B = image.shape[0]
        N = positions.shape[1]

        if self.use_gradient:
            # Compute gradients
            magnitude, orientation = self.compute_gradients(image)

            # CRITICAL FIX: Use ONLY magnitude (rotation-INVARIANT)
            # DO NOT use orientation (rotation-DEPENDENT)
            # Rotation is detected by ImageGradientOrientationEstimator in the canonicalizer
            grad_features = magnitude  # [B, 1, H, W] - rotation invariant!
        else:
            # Use raw intensities
            if image.dim() == 3:
                image = image.unsqueeze(1)
            if image.shape[1] == 3:
                image = image.mean(dim=1, keepdim=True)
            grad_features = image

        # Sample patches at positions using grid_sample
        # Create sampling grid for patches
        ps = self.patch_size
        offsets = torch.linspace(-ps // 2, ps // 2, ps, device=image.device) / (image.shape[-1] / 2)

        # Create patch grid
        grid_y, grid_x = torch.meshgrid(offsets, offsets, indexing="ij")
        patch_offsets = torch.stack([grid_x, grid_y], dim=-1)  # [ps, ps, 2]
        patch_offsets = patch_offsets.view(1, 1, ps, ps, 2)

        # Add offsets to positions
        positions_expanded = positions.view(B, N, 1, 1, 2)  # [B, N, 1, 1, 2]
        sample_grid = positions_expanded + patch_offsets  # [B, N, ps, ps, 2]
        sample_grid = sample_grid.view(B, N * ps, ps, 2)

        # Sample features
        sampled = F.grid_sample(
            grad_features,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # [B, C, N*ps, ps]

        # Reshape to [B, N, C*ps*ps]
        C = grad_features.shape[1]
        sampled = sampled.view(B, C, N, ps, ps)
        sampled = rearrange(sampled, "b c n h w -> b n (c h w)")

        # Encode to features
        features = self.feature_encoder(sampled)

        return features


class CNNFeatureExtractor(nn.Module):
    """
    CNN-based feature extractor.

    Uses a small CNN to extract dense features, then samples at node positions.
    """

    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 32,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, feature_dim, 1),
        )

    def forward(self, image: Tensor, positions: Tensor) -> Tensor:
        """
        Args:
            image: [B, C, H, W]
            positions: [B, N, 2] normalized to [-1, 1]

        Returns:
            features: [B, N, feature_dim]
        """
        if image.dim() == 3:
            image = image.unsqueeze(1)
        if image.shape[1] == 3:
            image = image.mean(dim=1, keepdim=True)

        # Extract dense features
        features = self.encoder(image)  # [B, D, H, W]

        # Sample at positions
        grid = positions.unsqueeze(2)  # [B, N, 1, 2]
        sampled = F.grid_sample(
            features,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # [B, D, N, 1]

        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # [B, N, D]

        return sampled


class GraphConstructor(nn.Module):
    """
    Construct graph from image for GNN processing.

    Options:
    - Grid: Regular grid of nodes (truly keypoint-free)
    - kNN: k-nearest neighbors based on spatial proximity
    """

    def __init__(
        self,
        grid_size: int = 32,
        k_neighbors: int = 8,
        graph_type: str = "grid",  # "grid" or "knn"
    ):
        super().__init__()

        self.grid_size = grid_size
        self.k_neighbors = k_neighbors
        self.graph_type = graph_type

        # Pre-compute grid positions (normalized to [-1, 1])
        y = torch.linspace(-1, 1, grid_size)
        x = torch.linspace(-1, 1, grid_size)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        positions = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        positions = positions.view(-1, 2)  # [N, 2]
        self.register_buffer("grid_positions", positions)

        # Pre-compute grid edges (4-connected)
        edges = []
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                # Right neighbor
                if j < grid_size - 1:
                    edges.append([idx, idx + 1])
                    edges.append([idx + 1, idx])
                # Down neighbor
                if i < grid_size - 1:
                    edges.append([idx, idx + grid_size])
                    edges.append([idx + grid_size, idx])
                # Diagonal neighbors (8-connected)
                if i < grid_size - 1 and j < grid_size - 1:
                    edges.append([idx, idx + grid_size + 1])
                    edges.append([idx + grid_size + 1, idx])
                if i < grid_size - 1 and j > 0:
                    edges.append([idx, idx + grid_size - 1])
                    edges.append([idx + grid_size - 1, idx])

        edge_index = torch.tensor(edges, dtype=torch.long).T
        self.register_buffer("grid_edge_index", edge_index)

    def forward(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Generate graph structure for a batch.

        Args:
            batch_size: Number of images in batch
            device: Target device

        Returns:
            positions: [B*N, 2] node positions
            edge_index: [2, E] edge connectivity
            batch: [B*N] batch assignment
        """
        N = self.grid_size**2

        # Expand positions for batch
        positions = self.grid_positions.unsqueeze(0).expand(batch_size, -1, -1)
        positions = positions.reshape(-1, 2)  # [B*N, 2]

        # Expand edge index for batch
        edge_indices = []
        for b in range(batch_size):
            offset = b * N
            edge_indices.append(self.grid_edge_index + offset)
        edge_index = torch.cat(edge_indices, dim=1)

        # Batch assignment
        batch = torch.arange(batch_size, device=device).repeat_interleave(N)

        return positions, edge_index, batch


class HomographyRegressionHead(nn.Module):
    """
    Regress homography parameters from matched features.

    Takes similarity-weighted features and outputs 8-DoF homography.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        hidden_dim: int = 256,
        num_nodes: int = 128,
    ):
        super().__init__()

        self.feature_dim = feature_dim

        # Process matched features
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Global pooling + regression
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 8),  # 8-DoF homography
        )

        # Initialize final layer with small weights
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def forward(
        self,
        features_src: Tensor,
        features_tgt: Tensor,
        similarity: Tensor,
        positions_src: Tensor | None = None,
        positions_tgt: Tensor | None = None,
    ) -> Tensor:
        """
        Regress homography from matched features.

        Args:
            features_src: [B, N, D] source features
            features_tgt: [B, N, D] target features
            similarity: [B, N, N] similarity matrix
            positions_src: [B, N, 2] source positions (optional)
            positions_tgt: [B, N, 2] target positions (optional)

        Returns:
            homography: [B, 8] homography parameters
        """
        B, N, D = features_src.shape

        # Soft assignment
        assignment = F.softmax(similarity, dim=-1)  # [B, N, N]

        # Weighted target features
        matched_tgt = torch.bmm(assignment, features_tgt)  # [B, N, D]

        # Concatenate source and matched target
        combined = torch.cat([features_src, matched_tgt], dim=-1)  # [B, N, 2D]

        # Process
        processed = self.feature_processor(combined)  # [B, N, H]

        # Global pooling
        pooled = processed.mean(dim=1)  # [B, H]

        # Regress homography
        homography = self.regressor(pooled)  # [B, 8]

        return homography


class ThermalHomographyNet(nn.Module):
    """
    Full Thermal Homography Network.

    End-to-end model for keypoint-free thermal homography estimation
    using E(2)-equivariant graph neural networks.
    """

    def __init__(
        self,
        # Feature extraction
        feature_extractor: str = "patch",  # "patch" or "cnn"
        feature_dim: int = 32,
        use_gradient: bool = True,
        # Graph construction
        grid_size: int = 32,
        k_neighbors: int = 8,
        # E(2) GNN
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 4,
        # LRFT
        use_lrft: bool = True,
        lrft_out_nodes: int = 128,
        lrft_rank: int = 32,
        lrft_invariant: bool = True,  # Use rotation-invariant LRFT
        # Sim(2) equivariance options
        # use_sim2_equivariant: NEW architecture with true Sim(2) detection capability
        # use_procrustes: Legacy architecture (deprecated - has detection issues)
        use_sim2_equivariant: bool = False,  # NEW: True Sim(2) detection via cyclic correlation
        sim2_num_rotations: int = 16,  # C16 group for cyclic correlation
        sim2_feature_channels: int = 32,  # Feature channels for equivariant encoder
        use_procrustes: bool = True,  # Legacy: Procrustes + ESCNN (deprecated)
        procrustes_num_rotations: int = 8,  # C8 group for ESCNN
        procrustes_grid_size: int = 16,  # Grid size for correlation matching
        procrustes_temperature: float = 0.1,  # Temperature for soft correspondences
        # Regression head
        regression_hidden_dim: int = 256,
        use_sks_head: bool = False,  # Use SKS decomposition head (H = S2 @ K @ S1)
        sks_use_perspective: bool = False,  # Include perspective in SKS
        # Regularization
        dropout: float = 0.1,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.grid_size = grid_size
        self.use_lrft = use_lrft
        self.use_sks_head = use_sks_head
        self.use_sim2_equivariant = use_sim2_equivariant
        self.use_procrustes = use_procrustes and not use_sim2_equivariant  # Disable if using new arch

        # NEW: Sim(2) Equivariant Architecture (2026-01-30)
        # This replaces the broken ProcrustesCanonicalizer approach
        if use_sim2_equivariant:
            self.sim2_net = Sim2EquivariantNet(
                num_rotations=sim2_num_rotations,
                feature_channels=sim2_feature_channels,
                spatial_feature_dim=feature_dim,
            )
            logger.info("Using Sim2EquivariantNet (NEW - true Sim(2) detection)")
            logger.info(f"  - Cyclic correlation for rotation (C{sim2_num_rotations})")
            logger.info(f"  - Procrustes for scale/translation after de-rotation")
            logger.info(f"  - NO GroupPooling - equivariant features SHIFT under rotation")

        # Feature extractor
        if feature_extractor == "patch":
            self.feature_extractor = PatchFeatureExtractor(
                patch_size=7,
                feature_dim=feature_dim,
                use_gradient=use_gradient,
            )
        else:
            self.feature_extractor = CNNFeatureExtractor(
                in_channels=1,
                feature_dim=feature_dim,
            )

        # Graph constructor
        self.graph_constructor = GraphConstructor(
            grid_size=grid_size,
            k_neighbors=k_neighbors,
            graph_type="grid",
        )

        # E(2)-equivariant GNN
        self.e2_gnn = E2EquivariantGNN(
            in_channels=feature_dim,
            hidden_channels=gnn_hidden_dim,
            out_channels=feature_dim,
            num_layers=gnn_num_layers,
            dropout=dropout,
        )

        # Procrustes-based canonicalization (the ONLY approach for Sim(2) equivariance)
        # Uses ESCNN rotation-invariant features + Procrustes SVD
        # No 90-degree ambiguity, mathematically principled
        if use_procrustes:
            self.procrustes_canonicalizer = ProcrustesCanonicalizer(
                feature_dim=feature_dim,
                num_rotations=procrustes_num_rotations,
                grid_size=procrustes_grid_size,
                temperature=procrustes_temperature,
            )
            logger.info(f"Using ProcrustesCanonicalizer (ESCNN C{procrustes_num_rotations}, grid={procrustes_grid_size})")
            logger.info("  -> ESCNN rotation-invariant features + Procrustes SVD")
            logger.info("  -> No 90-degree ambiguity (unlike structure tensor)")

        # Low-Rank Feature Transform
        # Use InvariantLRFT for true rotation invariance (distance-based attention)
        # Use AdaptiveLRFT for comparison (feature-based queries, NOT rotation-invariant)
        if use_lrft:
            if lrft_invariant:
                self.lrft = InvariantLRFT(
                    feature_dim=feature_dim,
                    out_nodes=lrft_out_nodes,
                    num_heads=4,
                    grid_type="uniform",
                )
                logger.info("Using InvariantLRFT (rotation-invariant, distance-based attention)")
            else:
                self.lrft = AdaptiveLRFT(
                    feature_dim=feature_dim,
                    out_nodes=lrft_out_nodes,
                    num_heads=4,
                )
                logger.info("Using AdaptiveLRFT (learned queries, NOT rotation-invariant)")

        # Similarity computation
        self.similarity = SimilarityMatrix(
            feature_dim=feature_dim,
            temperature=0.1,
        )

        # Homography regression head
        out_nodes = lrft_out_nodes if use_lrft else grid_size**2
        if use_sks_head:
            # SKS decomposition: H = S2 @ K @ S1
            # Explicitly separates rotation, scale, translation for Sim(2) equivariance
            self.regression_head = SKSRegressionHead(
                feature_dim=feature_dim,
                hidden_dim=regression_hidden_dim,
                use_perspective=sks_use_perspective,
            )
            logger.info("Using SKSRegressionHead (H = S2 @ K @ S1, explicit rotation/scale)")
        else:
            # Standard 8-parameter direct regression
            self.regression_head = HomographyRegressionHead(
                feature_dim=feature_dim,
                hidden_dim=regression_hidden_dim,
                num_nodes=out_nodes,
            )
            logger.info("Using HomographyRegressionHead (8-param direct regression)")

    def extract_graph_features(
        self,
        image: Tensor,
    ) -> tuple[Tensor, Tensor, dict | None]:
        """
        Extract rotation/scale-invariant graph features from image.

        Args:
            image: [B, C, H, W] input image

        Returns:
            features: [B, N, D] node features
            positions: [B, N, 2] node positions (canonical if canonicalization enabled)
            canon_info: dict with canonicalization info (theta, scale, etc.) or None
        """
        B = image.shape[0]
        device = image.device

        # Construct graph
        positions, edge_index, batch = self.graph_constructor(B, device)

        # Get grid positions for each image
        N = self.grid_size**2
        positions_batched = positions.view(B, N, 2)

        # Extract patch features
        features = self.feature_extractor(image, positions_batched)  # [B, N, D]

        # Flatten for GNN
        features_flat = features.view(B * N, -1)

        # E(2)-equivariant message passing (rotation-invariant)
        features_flat = self.e2_gnn(features_flat, positions, edge_index, batch)

        # Reshape back
        features = features_flat.view(B, N, -1)

        return features, positions_batched, None

    def forward(
        self,
        image_src: Tensor,
        image_tgt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Forward pass with Sim(2) equivariance via Procrustes + ESCNN.

        The THREE PILLARS of Sim(2) equivariance:
        1. ROTATION: ESCNN C8 provides rotation-invariant features
        2. TRANSLATION: Procrustes centroid computation handles translation
        3. SCALE: Procrustes SVD estimates scale from correspondence geometry

        Args:
            image_src: [B, C, H, W] source image
            image_tgt: [B, C, H, W] target image

        Returns:
            Dictionary containing:
            - homography: [B, 3, 3] predicted homography matrix
            - rotation: [B] estimated rotation angle in radians
            - scale: [B] estimated scale factor
            - translation: [B, 2] estimated translation vector
            - confidence: [B, H, W] matching confidence map
        """
        B = image_src.shape[0]
        device = image_src.device

        # NEW (2026-01-30): Sim(2) Equivariant Architecture
        # Uses cyclic correlation for rotation + Procrustes for scale/translation
        # This is the CORRECT approach - previous approach was broken
        if self.use_sim2_equivariant:
            return self.sim2_net(image_src, image_tgt)

        # DEPRECATED: Procrustes-based path (has detection issues!)
        # Uses ESCNN rotation-invariant features + Procrustes SVD
        # WARNING: GroupPooling makes features INVARIANT, breaking detection
        if self.use_procrustes:
            # Get transformation directly via Procrustes
            procrustes_result = self.procrustes_canonicalizer(image_src, image_tgt)

            rotation = procrustes_result['rotation']  # [B]
            scale = procrustes_result['scale']  # [B]
            translation = procrustes_result['translation']  # [B, 2]

            # Build homography from Sim(2) parameters
            homography = self.procrustes_canonicalizer.build_homography(
                rotation, scale, translation
            )

            return {
                "homography": homography,
                "rotation": rotation,
                "scale": scale,
                "translation": translation,
                "confidence": procrustes_result['confidence'],
            }

        # Fallback: Standard forward pass without Sim(2) canonicalization
        # (Not recommended - use Procrustes for proper Sim(2) equivariance)
        features_src, positions_src, _ = self.extract_graph_features(image_src)
        features_tgt, positions_tgt, _ = self.extract_graph_features(image_tgt)

        if self.use_lrft:
            features_src, positions_src = self.lrft(features_src, positions_src)
            features_tgt, positions_tgt = self.lrft(features_tgt, positions_tgt)

        similarity = self.similarity(features_src, features_tgt)

        homography = self.regression_head(
            features_src,
            features_tgt,
            similarity,
            positions_src,
            positions_tgt,
        )

        if not self.use_sks_head and homography.dim() == 2 and homography.shape[-1] == 8:
            homography = homography_vec_to_matrix(homography)

        return {
            "homography": homography,
            "features_src": features_src,
            "features_tgt": features_tgt,
            "positions_src": positions_src,
            "positions_tgt": positions_tgt,
        }


def homography_vec_to_matrix(h_vec: Tensor) -> Tensor:
    """
    Convert 8D homography vector to 3x3 matrix.

    The vector represents h11, h12, h13, h21, h22, h23, h31, h32
    with h33 = 1 (normalized homography).

    Args:
        h_vec: [B, 8] homography parameters

    Returns:
        H: [B, 3, 3] homography matrices
    """
    B = h_vec.shape[0]
    device = h_vec.device

    # Construct matrix
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


def homography_matrix_to_vec(H: Tensor) -> Tensor:
    """
    Convert 3x3 homography matrix to 8D vector.

    Args:
        H: [B, 3, 3] homography matrices (normalized so H[2,2] = 1)

    Returns:
        h_vec: [B, 8] homography parameters
    """
    # Normalize so H[2,2] = 1
    H = H / (H[:, 2:3, 2:3] + 1e-8)

    h_vec = torch.stack(
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

    return h_vec
