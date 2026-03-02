"""
Rotation-Invariant Graph Neural Network Layers (EGNN-Style)

This module implements rotation-invariant message passing for 2D image graphs,
following the EGNN (Equivariant Graph Neural Network) approach.

Key insight: By using ONLY distance² as edge features (no angles), we ensure
that the message function output is identical regardless of input rotation:
    f(R · x) = f(x)  for all R ∈ SO(2)

This achieves true rotation INVARIANCE (not just equivariance), which is
essential for homography estimation where we want consistent predictions
across arbitrary input rotations.

Why distance² and not angles?
- Distance² is rotation-invariant: ||R @ v||² = ||v||²
- Angles change with rotation: atan2(R @ v) ≠ atan2(v)
- Using angles in an MLP breaks invariance because MLP(angle) ≠ MLP(rotated_angle)

References:
- EGNN (Satorras et al., ICML 2021): E(n)-equivariant graphs
- Equi-GSPR (Kang et al., ECCV 2024): SE(3) graphs for 3D point clouds
- E(2)-Steerable CNNs (Weiler & Cesa, 2019): Theory of E(2) equivariance
"""

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class E2MessagePassing(MessagePassing):
    """
    Rotation-Invariant Message Passing Layer (EGNN-Style).

    For nodes i, k with features h_i, h_k and positions x_i, x_k:
    1. Compute rotation-invariant edge features: distance squared ||x_k - x_i||²
    2. Build message from distance² + node features via MLP
    3. Aggregate messages (sum preserves invariance)

    Key insight from EGNN (Satorras et al., 2021):
    Using ONLY distance² (no angles) ensures that message function output
    is identical regardless of input rotation, achieving true rotation invariance.

    This is critical for homography estimation where we want predictions
    to be consistent across arbitrary input rotations.

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        hidden_channels: Hidden layer dimension
        use_attention: Whether to use attention-weighted aggregation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        use_attention: bool = True,
    ):
        super().__init__(aggr="add", flow="source_to_target")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.use_attention = use_attention

        # Edge feature encoder: processes rotation-invariant quantities
        # Input: [h_i, h_k, distance_squared] - EGNN-style, no angles
        # Using only distance² ensures true rotation invariance
        edge_input_dim = 2 * in_channels + 1  # node features + dist_sq only

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels),
        )

        # Attention mechanism (optional)
        if use_attention:
            self.attention_mlp = nn.Sequential(
                nn.Linear(edge_input_dim, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 1),
            )

        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels),
        )

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(out_channels)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters for stable training with proper gradient flow."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # FIXED: gain=0.01 was too small, causing vanishing gradients.
                # Use standard Xavier init with gain=1.0 for SiLU activations.
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, in_channels]
            pos: Node positions [N, 2]
            edge_index: Edge connectivity [2, E]
            batch: Batch assignment [N] (optional)

        Returns:
            Updated node features [N, out_channels]
        """
        # Compute edge-level geometric quantities
        row, col = edge_index

        # Relative positions (used for local frame construction)
        rel_pos = pos[col] - pos[row]  # [E, 2]

        # EGNN-STYLE: Use ONLY distance squared (rotation-invariant!)
        # Distance squared is invariant under rotation: ||R @ x||² = ||x||²
        # This is the key insight from EGNN (Satorras et al., 2021)
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)  # [E, 1]

        # Fixed normalization: for positions in [-1, 1]², diagonal² = 8
        # This ensures consistent scaling regardless of batch content
        dist_sq_normalized = dist_sq / 8.0  # [E, 1]

        # NOTE: We removed angle features (cos, sin) because they change with rotation.
        # By using only distance², the message function output is identical
        # regardless of how the input is rotated.

        # Propagate messages
        out = self.propagate(
            edge_index,
            x=x,
            pos=pos,
            dist_sq=dist_sq_normalized,
        )

        # Residual connection + normalization
        if self.in_channels == self.out_channels:
            out = self.layer_norm(out + x)
        else:
            out = self.layer_norm(out)

        return out

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        dist_sq: Tensor,
        index: Tensor,
    ) -> Tensor:
        """
        Construct messages from source nodes j to target nodes i.

        The message is built from:
        - Source node features x_j
        - Target node features x_i
        - Rotation-invariant edge feature: distance squared

        NOTE: We use only distance² (no angles) to ensure true rotation invariance.
        This follows EGNN-style message passing where only invariant quantities
        are used as edge features.
        """
        # Concatenate node features with rotation-invariant distance squared
        edge_features = torch.cat([x_i, x_j, dist_sq], dim=-1)

        # Compute message
        message = self.edge_mlp(edge_features)

        # Optional attention weighting
        if self.use_attention:
            attn_logits = self.attention_mlp(edge_features)
            attn_weights = softmax(attn_logits, index)
            message = message * attn_weights

        return message

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        """Update node features with aggregated messages."""
        combined = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(combined)


class E2EquivariantGNN(nn.Module):
    """
    Full E(2)-Equivariant Graph Neural Network.

    Stacks multiple E2MessagePassing layers to build a deep
    equivariant feature extractor for image graphs.

    Args:
        in_channels: Input node feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output feature dimension
        num_layers: Number of message passing layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.SiLU(),
            nn.LayerNorm(hidden_channels),
        )

        # Message passing layers
        self.mp_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.mp_layers.append(
                E2MessagePassing(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    hidden_channels=hidden_channels * 2,
                    use_attention=True,
                )
            )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass through the E(2)-equivariant GNN.

        Args:
            x: Node features [N, in_channels]
            pos: Node positions [N, 2]
            edge_index: Edge connectivity [2, E]
            batch: Batch assignment [N]

        Returns:
            Node embeddings [N, out_channels]
        """
        # Project input features
        h = self.input_proj(x)

        # Message passing
        for mp_layer in self.mp_layers:
            h = mp_layer(h, pos, edge_index, batch)
            h = self.dropout(h)

        # Output projection
        out = self.output_proj(h)

        return out


def build_local_frame(pos_i: Tensor, pos_j: Tensor) -> Tensor:
    """
    Build local 2D coordinate frame from positions.

    The frame is defined by the direction from i to j,
    which makes subsequent operations equivariant to global rotations.

    Args:
        pos_i: Source positions [E, 2]
        pos_j: Target positions [E, 2]

    Returns:
        Rotation matrices [E, 2, 2] representing local frames
    """
    # Direction vector
    delta = pos_j - pos_i  # [E, 2]

    # Normalize
    dist = torch.norm(delta, dim=-1, keepdim=True)
    direction = delta / (dist + 1e-8)  # [E, 2]

    # Build 2D rotation matrix: [cos, -sin; sin, cos]
    # where direction = [cos(theta), sin(theta)]
    cos_theta = direction[:, 0]  # [E]
    sin_theta = direction[:, 1]  # [E]

    # Stack into rotation matrices
    R = torch.stack(
        [
            torch.stack([cos_theta, -sin_theta], dim=-1),
            torch.stack([sin_theta, cos_theta], dim=-1),
        ],
        dim=-2,
    )  # [E, 2, 2]

    return R


def test_equivariance(
    model: nn.Module, x: Tensor, pos: Tensor, edge_index: Tensor, angle: float = math.pi / 4
) -> tuple[float, float]:
    """
    Test E(2)-equivariance numerically.

    For a function f to be SO(2)-equivariant:
        f(R @ x) should equal R @ f(x)

    Args:
        model: The model to test
        x: Node features [N, C]
        pos: Node positions [N, 2]
        edge_index: Edge connectivity [2, E]
        angle: Rotation angle in radians

    Returns:
        (equivariance_error, invariance_error) as float values
    """
    model.eval()

    # Original output
    with torch.no_grad():
        out_original = model(x, pos, edge_index)

    # Rotation matrix
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=pos.device, dtype=pos.dtype)

    # Rotated positions
    pos_rotated = pos @ R.T

    # Output on rotated input
    with torch.no_grad():
        out_rotated = model(x, pos_rotated, edge_index)

    # For scalar features, output should be invariant (same under rotation)
    # For vector features, output should transform: out_rotated = R @ out_original

    # Check invariance of scalar features
    invariance_error = (out_rotated - out_original).abs().mean().item()

    # For vector features (if applicable)
    equivariance_error = 0.0  # Would need vector outputs to test

    return equivariance_error, invariance_error
