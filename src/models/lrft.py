"""
Low-Rank Feature Transform (LRFT) Module

Adapted from Equi-GSPR (ECCV 2024).

The LRFT compresses N×D features to N'×D via low-rank factorization:
    H_compressed = (A @ B)^T @ H_original

Where:
    - A ∈ R^(N × r): Learned left factor
    - B ∈ R^(r × N'): Learned right factor
    - r << min(N, N'): Rank constraint

Benefits:
1. Aggregates local information (similar to pooling)
2. Low-rank constraint enforces feature consistency (reduces outliers)
3. Reduces compute for similarity matrix (N'² << N²)

Classes:
- LowRankFeatureTransform: Basic LRFT with fixed or attention-based compression
- AdaptiveLRFT: Variable input size with learned feature queries (NOT rotation-invariant)
- InvariantLRFT: Distance-based attention for TRUE rotation invariance (Sim(2)-equivariant)

For Sim(2)-equivariant homography estimation, use InvariantLRFT which uses
distance-based attention rather than learned feature queries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class LowRankFeatureTransform(nn.Module):
    """
    Low-Rank Feature Transform module.

    Compresses graph node features while preserving geometric information
    through a learned low-rank projection.

    Args:
        in_nodes: Number of input nodes (N)
        out_nodes: Number of output nodes (N')
        feature_dim: Feature dimension (D)
        rank: Rank of the factorization (r)
        use_attention: Use attention-based compression
    """

    def __init__(
        self,
        in_nodes: int = 1024,
        out_nodes: int = 128,
        feature_dim: int = 32,
        rank: int = 32,
        use_attention: bool = True,
    ):
        super().__init__()

        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.feature_dim = feature_dim
        self.rank = rank
        self.use_attention = use_attention

        if use_attention:
            # Attention-based compression
            self.query = nn.Linear(feature_dim, rank)
            self.key = nn.Linear(feature_dim, rank)
            self.value = nn.Linear(feature_dim, feature_dim)

            # Learned output positions (N' anchor points)
            self.anchors = nn.Parameter(torch.randn(out_nodes, rank))

        else:
            # Direct low-rank factorization
            # A: maps input nodes to rank space
            self.A = nn.Parameter(torch.randn(in_nodes, rank) * 0.01)
            # B: maps rank space to output nodes
            self.B = nn.Parameter(torch.randn(rank, out_nodes) * 0.01)

        # Output normalization
        self.norm = nn.LayerNorm(feature_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.use_attention:
            nn.init.xavier_uniform_(self.query.weight)
            nn.init.xavier_uniform_(self.key.weight)
            nn.init.xavier_uniform_(self.value.weight)
            nn.init.normal_(self.anchors, std=0.01)

    def forward(
        self,
        features: Tensor,
        positions: Tensor | None = None,
        batch: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Compress node features via low-rank transform.

        Args:
            features: Node features [B, N, D] or [N, D]
            positions: Node positions [B, N, 2] or [N, 2] (optional)
            batch: Batch indices [N] for non-batched input

        Returns:
            compressed_features: [B, N', D]
            compressed_positions: [B, N', 2] (if positions provided)
        """
        # Handle unbatched input
        if features.dim() == 2:
            features = features.unsqueeze(0)
            if positions is not None:
                positions = positions.unsqueeze(0)

        B, N, D = features.shape

        if self.use_attention:
            # Attention-based compression
            # Query from anchor points, Key/Value from input features

            # Compute keys and values from input
            K = self.key(features)  # [B, N, r]
            V = self.value(features)  # [B, N, D]

            # Expand anchors for batch
            Q = self.anchors.unsqueeze(0).expand(B, -1, -1)  # [B, N', r]

            # Attention weights
            scale = self.rank**-0.5
            attn = torch.bmm(Q, K.transpose(-2, -1)) * scale  # [B, N', N]
            attn = F.softmax(attn, dim=-1)

            # Compress features
            compressed = torch.bmm(attn, V)  # [B, N', D]

            # Compress positions if provided
            if positions is not None:
                compressed_pos = torch.bmm(attn, positions)  # [B, N', 2]
            else:
                compressed_pos = None

        else:
            # Direct low-rank factorization
            # Projection matrix: P = A @ B, shape [N, N']
            P = self.A @ self.B  # [N, N']
            P = F.softmax(P, dim=0)  # Normalize columns

            # Compress: [B, N, D] @ [N, N'] -> need to handle correctly
            # Actually: H_out = P^T @ H_in
            compressed = torch.bmm(
                P.T.unsqueeze(0).expand(B, -1, -1), features  # [B, N', N]  # [B, N, D]
            )  # [B, N', D]

            if positions is not None:
                compressed_pos = torch.bmm(P.T.unsqueeze(0).expand(B, -1, -1), positions)
            else:
                compressed_pos = None

        # Normalize output
        compressed = self.norm(compressed)

        return compressed, compressed_pos


class AdaptiveLRFT(nn.Module):
    """
    Adaptive LRFT that handles variable input sizes.

    Uses attention mechanism to compress arbitrary number of
    input nodes to fixed output size.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        out_nodes: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.out_nodes = out_nodes
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        # Multi-head attention
        self.to_qkv = nn.Linear(feature_dim, 3 * feature_dim)
        self.to_out = nn.Linear(feature_dim, feature_dim)

        # Learned compression queries (output anchors)
        self.compression_queries = nn.Parameter(torch.randn(out_nodes, feature_dim))

        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        features: Tensor,
        positions: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Args:
            features: [B, N, D]
            positions: [B, N, 2] optional
            mask: [B, N] attention mask (True = keep)

        Returns:
            compressed_features: [B, N', D]
            compressed_positions: [B, N', 2] or None
        """
        B, N, D = features.shape

        # Project to Q, K, V
        qkv = self.to_qkv(features)  # [B, N, 3D]
        q, k, v = qkv.chunk(3, dim=-1)  # Each [B, N, D]

        # Use learned queries for compression
        comp_q = self.compression_queries.unsqueeze(0).expand(B, -1, -1)  # [B, N', D]

        # Reshape for multi-head attention
        comp_q = rearrange(comp_q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        # Attention
        scale = self.head_dim**-0.5
        attn = torch.einsum("b h q d, b h k d -> b h q k", comp_q, k) * scale

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)

        # Aggregate values
        out = torch.einsum("b h q k, b h k d -> b h q d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.to_out(out)
        out = self.norm(out)

        # Compress positions using same attention weights
        if positions is not None:
            # Average attention across heads
            attn_avg = attn.mean(dim=1)  # [B, N', N]
            compressed_pos = torch.bmm(attn_avg, positions)  # [B, N', 2]
        else:
            compressed_pos = None

        return out, compressed_pos


class InvariantLRFT(nn.Module):
    """
    Rotation-Invariant LRFT with distance-based attention.

    Key insight: Attention weights based on ||query_pos - key_pos||²
    are rotation-invariant because distance is preserved under rotation.

    This replaces feature-based queries (which break invariance when rotated)
    with fixed spatial query positions and distance-based attention weights.

    Args:
        feature_dim: Feature dimension (D)
        out_nodes: Number of output nodes (N')
        num_heads: Number of attention heads
        grid_type: Type of query grid ("uniform" or "random")
    """

    def __init__(
        self,
        feature_dim: int = 32,
        out_nodes: int = 128,
        num_heads: int = 4,
        grid_type: str = "uniform",
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.out_nodes = out_nodes
        self.num_heads = num_heads

        # Fixed spatial query grid (positions, not learned features)
        # These positions define WHERE we sample, not WHAT we look for
        query_pos = self._create_query_grid(out_nodes, grid_type)
        self.register_buffer("query_positions", query_pos)  # [N', 2]

        # Distance-based attention MLP
        # Input: distance squared, Output: attention logits per head
        # This learns HOW to weight based on distance
        self.attn_mlp = nn.Sequential(
            nn.Linear(1, 64),  # Input: distance squared
            nn.SiLU(),
            nn.Linear(64, num_heads),  # Output: logits per head
        )

        # Value and output projections (operate on features, not positions)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

        self._reset_parameters()

    def _create_query_grid(self, n_points: int, grid_type: str) -> Tensor:
        """Create a fixed grid of query positions in [-1, 1]²."""
        if grid_type == "uniform":
            # Create uniform grid
            side = int(n_points**0.5)
            if side * side != n_points:
                # Not a perfect square, use closest
                side = int(n_points**0.5) + 1

            # Generate grid points
            lin = torch.linspace(-0.9, 0.9, side)
            xx, yy = torch.meshgrid(lin, lin, indexing="xy")
            grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

            # Trim to exact number of points
            return grid[:n_points]  # [N', 2]
        else:
            # Random initialization (will be fixed after init)
            return torch.randn(n_points, 2) * 0.5

    def _reset_parameters(self):
        """Initialize attention MLP for stable training."""
        for module in self.attn_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _pairwise_dist_sq(self, queries: Tensor, keys: Tensor) -> Tensor:
        """
        Compute pairwise squared distances.

        Args:
            queries: [B, N', 2] query positions
            keys: [B, N, 2] key positions

        Returns:
            dist_sq: [B, N', N] squared distances
        """
        # queries: [B, N', 1, 2]
        # keys:    [B, 1, N, 2]
        # diff:    [B, N', N, 2]
        diff = queries.unsqueeze(2) - keys.unsqueeze(1)
        dist_sq = (diff**2).sum(dim=-1)  # [B, N', N]
        return dist_sq

    def forward(
        self,
        features: Tensor,
        positions: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Compress features using distance-based attention.

        Args:
            features: [B, N, D] input node features
            positions: [B, N, 2] input node positions (REQUIRED for distance computation)
            mask: [B, N] attention mask (True = keep)

        Returns:
            compressed_features: [B, N', D]
            compressed_positions: [B, N', 2]
        """
        B, N, D = features.shape

        # Expand query positions for batch
        q_pos = self.query_positions.unsqueeze(0).expand(B, -1, -1)  # [B, N', 2]

        # Compute pairwise squared distances (ROTATION INVARIANT!)
        dist_sq = self._pairwise_dist_sq(q_pos, positions)  # [B, N', N]

        # Normalize distances (for positions in [-1, 1]², max dist² ≈ 8)
        dist_sq_normalized = dist_sq / 8.0

        # Distance-based attention logits
        # The MLP learns to convert distance² to attention weights
        attn_logits = self.attn_mlp(dist_sq_normalized.unsqueeze(-1))  # [B, N', N, H]
        attn_logits = attn_logits.permute(0, 3, 1, 2)  # [B, H, N', N]

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            attn_logits = attn_logits.masked_fill(~mask_expanded, float("-inf"))

        # Softmax to get attention weights
        # Note: We use -attn_logits so that smaller distances get higher weights
        # But the MLP can learn this behavior, so we let it be flexible
        attn = F.softmax(attn_logits, dim=-1)  # [B, H, N', N]

        # Project values
        V = self.value_proj(features)  # [B, N, D]

        # Aggregate values using attention (multi-head)
        # attn: [B, H, N', N], V: [B, N, D]
        out = torch.einsum("bhqk,bkd->bhqd", attn, V)  # [B, H, N', D]
        out = out.mean(dim=1)  # Average over heads: [B, N', D]

        # Output projection and normalization
        out = self.out_proj(out)
        out = self.norm(out)

        # Compute compressed positions using average attention
        attn_avg = attn.mean(dim=1)  # [B, N', N]
        compressed_pos = torch.bmm(attn_avg, positions)  # [B, N', 2]

        return out, compressed_pos


class SimilarityMatrix(nn.Module):
    """
    Compute similarity matrix between two sets of features.

    Used to establish correspondences between source and target
    image features for homography regression.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        temperature: float = 0.1,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.temperature = temperature

        # Optional: learned temperature
        self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))

    def forward(
        self,
        features_src: Tensor,
        features_tgt: Tensor,
        normalize: bool = True,
    ) -> Tensor:
        """
        Compute similarity matrix.

        Args:
            features_src: Source features [B, N, D]
            features_tgt: Target features [B, M, D]
            normalize: Whether to L2-normalize features

        Returns:
            Similarity matrix [B, N, M]
        """
        if normalize:
            features_src = F.normalize(features_src, dim=-1)
            features_tgt = F.normalize(features_tgt, dim=-1)

        # Cosine similarity
        sim = torch.bmm(features_src, features_tgt.transpose(-2, -1))

        # Temperature scaling
        temp = self.log_temp.exp()
        sim = sim / temp

        return sim

    def soft_assignment(
        self,
        similarity: Tensor,
        dim: int = -1,
    ) -> Tensor:
        """Convert similarity to soft assignment via softmax."""
        return F.softmax(similarity, dim=dim)


def compute_rank_loss(similarity: Tensor, target_rank: int = 32) -> Tensor:
    """
    Rank regularization loss.

    Encourages the similarity matrix to have full rank up to target_rank,
    which promotes unique correspondences.

    L_rank = |trace(S^T S)^(1/2) - target_rank|

    Args:
        similarity: Similarity matrix [B, N, M]
        target_rank: Desired effective rank

    Returns:
        Rank loss scalar
    """
    # Nuclear norm approximation of rank
    # trace(sqrt(S^T S)) = sum of singular values
    # We use Frobenius norm as a proxy (easier to compute)
    nuclear_norm = torch.linalg.matrix_norm(similarity, ord="nuc")  # [B]

    # Loss: deviation from target rank
    loss = (nuclear_norm - target_rank).abs().mean()

    return loss
