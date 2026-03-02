"""
SKS Decomposition Regression Head for Homography Estimation

SKS decomposition: H = S2 @ K @ S1

Where:
- S1: Similarity transform in source frame (θ, s, tx, ty)
- S2: Similarity transform in target frame (θ, s, tx, ty)
- K: Kernel/projective transform (optional, for perspective)

Benefits:
1. Makes rotation and scale EXPLICIT (not tangled in 8D vector)
2. Identity initialization is natural (S1 = S2 = I, K = I)
3. Learned components are interpretable
4. Better optimization landscape for Sim(2) transformations

The key insight is that most homographies in practice are dominated by
similarity transforms (rotation, scale, translation), with only small
perspective distortion. By decomposing H = S2 @ K @ S1, we make the
similarity components explicit and easier to learn.

References:
- LoFTR (Sun et al., 2021): Coarse-to-fine matching
- Deep Homography Estimation (DeTone et al., 2016): 4-pt parameterization
- Supervised Learning of Semantics-Preserving Hash Functions (Li et al.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class SKSRegressionHead(nn.Module):
    """
    Regress homography using SKS decomposition: H = S2 @ K @ S1

    Predicts:
    - S1: Similarity transform (θ1, s1, tx1, ty1) - 4 params
    - S2: Similarity transform (θ2, s2, tx2, ty2) - 4 params
    - K: Kernel (optional perspective) - 2 params

    Total: 8 or 10 parameters, same DoF as standard homography.

    Args:
        feature_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        use_perspective: Whether to predict perspective (K) component
        scale_range: Expected range of scale factors (for initialization)
    """

    def __init__(
        self,
        feature_dim: int = 32,
        hidden_dim: int = 256,
        use_perspective: bool = False,
        scale_range: tuple[float, float] = (0.5, 2.0),
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.use_perspective = use_perspective

        # Shared feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Similarity heads: predict (θ, s, tx, ty)
        # θ: rotation angle
        # s: scale factor
        # tx, ty: translation
        self.sim1_head = nn.Linear(hidden_dim, 4)
        self.sim2_head = nn.Linear(hidden_dim, 4)

        # Optional perspective head: predict (p1, p2)
        # These control perspective distortion
        if use_perspective:
            self.persp_head = nn.Linear(hidden_dim, 2)

        self._init_identity()

    def _init_identity(self):
        """Initialize to predict identity transform."""
        # Zero weights
        nn.init.zeros_(self.sim1_head.weight)
        nn.init.zeros_(self.sim2_head.weight)

        # Bias: θ=0, s=1, tx=0, ty=0
        # NOTE: Scale goes through softplus(s - 1.0) + 0.1
        # To get output scale = 1.0, we need softplus(s_bias - 1.0) + 0.1 = 1.0
        # => softplus(s_bias - 1.0) = 0.9
        # => s_bias - 1.0 = inverse_softplus(0.9) = ln(exp(0.9) - 1) ≈ 0.378
        # => s_bias ≈ 1.378
        with torch.no_grad():
            self.sim1_head.bias.data = torch.tensor([0.0, 1.378, 0.0, 0.0])
            self.sim2_head.bias.data = torch.tensor([0.0, 1.378, 0.0, 0.0])

        if self.use_perspective:
            nn.init.zeros_(self.persp_head.weight)
            nn.init.zeros_(self.persp_head.bias)

    def _params_to_similarity(self, params: Tensor) -> Tensor:
        """
        Convert (θ, s, tx, ty) to 3x3 similarity matrix.

        Similarity matrix form:
        | s*cos(θ)  -s*sin(θ)  tx |
        | s*sin(θ)   s*cos(θ)  ty |
        |    0          0       1 |

        Args:
            params: [B, 4] containing (θ, s, tx, ty)

        Returns:
            S: [B, 3, 3] similarity matrices
        """
        theta, s, tx, ty = params.unbind(dim=-1)

        # Ensure scale is positive
        # softplus(x) = ln(1 + exp(x)), always positive
        # With s_bias=1.378: softplus(1.378 - 1.0) + 0.1 = softplus(0.378) + 0.1 ≈ 0.9 + 0.1 = 1.0
        s = F.softplus(s - 1.0) + 0.1

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        B = params.shape[0]
        device = params.device
        dtype = params.dtype

        S = torch.zeros(B, 3, 3, device=device, dtype=dtype)

        S[:, 0, 0] = s * cos_theta
        S[:, 0, 1] = -s * sin_theta
        S[:, 0, 2] = tx
        S[:, 1, 0] = s * sin_theta
        S[:, 1, 1] = s * cos_theta
        S[:, 1, 2] = ty
        S[:, 2, 2] = 1.0

        return S

    def _params_to_kernel(self, params: Tensor) -> Tensor:
        """
        Convert perspective parameters to kernel matrix.

        Kernel matrix form (perspective only):
        | 1  0  0 |
        | 0  1  0 |
        | p1 p2 1 |

        Args:
            params: [B, 2] containing (p1, p2)

        Returns:
            K: [B, 3, 3] kernel matrices
        """
        p1, p2 = params.unbind(dim=-1)

        B = params.shape[0]
        device = params.device
        dtype = params.dtype

        K = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
        K[:, 2, 0] = p1
        K[:, 2, 1] = p2

        return K

    def forward(
        self,
        features_src: Tensor,
        features_tgt: Tensor,
        similarity_matrix: Tensor,
        positions_src: Tensor | None = None,
        positions_tgt: Tensor | None = None,
    ) -> Tensor:
        """
        Regress homography using SKS decomposition.

        Args:
            features_src: [B, N, D] source features
            features_tgt: [B, N, D] target features
            similarity_matrix: [B, N, N] soft correspondence matrix
            positions_src: [B, N, 2] source positions (optional)
            positions_tgt: [B, N, 2] target positions (optional)

        Returns:
            H: [B, 3, 3] homography matrices
        """
        B = features_src.shape[0]

        # Soft matching: get matched target features
        assignment = F.softmax(similarity_matrix, dim=-1)  # [B, N, N]
        matched_tgt = torch.bmm(assignment, features_tgt)  # [B, N, D]

        # Concatenate source and matched target
        combined = torch.cat([features_src, matched_tgt], dim=-1)  # [B, N, 2D]

        # Global pooling
        pooled = combined.mean(dim=1)  # [B, 2D]

        # Encode
        encoded = self.encoder(pooled)  # [B, H]

        # Predict similarity parameters
        sim1_params = self.sim1_head(encoded)  # [B, 4]
        sim2_params = self.sim2_head(encoded)  # [B, 4]

        # Build similarity matrices
        S1 = self._params_to_similarity(sim1_params)  # [B, 3, 3]
        S2 = self._params_to_similarity(sim2_params)  # [B, 3, 3]

        # Build kernel (identity or with perspective)
        if self.use_perspective:
            persp_params = self.persp_head(encoded)  # [B, 2]
            K = self._params_to_kernel(persp_params)  # [B, 3, 3]
        else:
            K = torch.eye(3, device=encoded.device, dtype=encoded.dtype)
            K = K.unsqueeze(0).expand(B, -1, -1)

        # Compose: H = S2 @ K @ S1
        H = torch.bmm(torch.bmm(S2, K), S1)

        # Normalize so H[2,2] = 1
        H = H / (H[:, 2:3, 2:3] + 1e-8)

        return H

    def get_decomposition(
        self,
        features_src: Tensor,
        features_tgt: Tensor,
        similarity_matrix: Tensor,
    ) -> dict[str, Tensor]:
        """
        Get the full decomposition for analysis.

        Returns dictionary with:
        - H: [B, 3, 3] composed homography
        - S1: [B, 3, 3] source similarity
        - S2: [B, 3, 3] target similarity
        - K: [B, 3, 3] kernel
        - sim1_params: [B, 4] (θ, s, tx, ty) for source
        - sim2_params: [B, 4] (θ, s, tx, ty) for target
        """
        B = features_src.shape[0]

        assignment = F.softmax(similarity_matrix, dim=-1)
        matched_tgt = torch.bmm(assignment, features_tgt)
        combined = torch.cat([features_src, matched_tgt], dim=-1)
        pooled = combined.mean(dim=1)
        encoded = self.encoder(pooled)

        sim1_params = self.sim1_head(encoded)
        sim2_params = self.sim2_head(encoded)

        S1 = self._params_to_similarity(sim1_params)
        S2 = self._params_to_similarity(sim2_params)

        if self.use_perspective:
            persp_params = self.persp_head(encoded)
            K = self._params_to_kernel(persp_params)
        else:
            K = torch.eye(3, device=encoded.device).unsqueeze(0).expand(B, -1, -1)
            persp_params = torch.zeros(B, 2, device=encoded.device)

        H = torch.bmm(torch.bmm(S2, K), S1)
        H = H / (H[:, 2:3, 2:3] + 1e-8)

        return {
            "H": H,
            "S1": S1,
            "S2": S2,
            "K": K,
            "sim1_params": sim1_params,
            "sim2_params": sim2_params,
            "persp_params": persp_params if self.use_perspective else None,
        }


class DirectRegressionHead(nn.Module):
    """
    Standard 8-parameter homography regression (for comparison).

    This is the baseline approach used in most deep homography methods.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 8),
        )

        # Initialize to identity
        nn.init.zeros_(self.regressor[-1].weight)
        with torch.no_grad():
            # Identity homography as 8D vector: [1,0,0,0,1,0,0,0]
            self.regressor[-1].bias.data = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32)

    def forward(
        self,
        features_src: Tensor,
        features_tgt: Tensor,
        similarity_matrix: Tensor,
        positions_src: Tensor | None = None,
        positions_tgt: Tensor | None = None,
    ) -> Tensor:
        """
        Regress 8D homography vector.

        Returns:
            h: [B, 8] homography parameters
        """
        assignment = F.softmax(similarity_matrix, dim=-1)
        matched_tgt = torch.bmm(assignment, features_tgt)
        combined = torch.cat([features_src, matched_tgt], dim=-1)
        pooled = combined.mean(dim=1)

        encoded = self.encoder(pooled)
        h = self.regressor(encoded)

        return h


def decompose_homography(H: Tensor) -> dict[str, Tensor]:
    """
    Decompose a homography matrix into similarity components.

    Uses SVD-based decomposition to extract rotation, scale, translation.
    Useful for analysis and visualization.

    Args:
        H: [B, 3, 3] homography matrices

    Returns:
        Dictionary with extracted components
    """
    B = H.shape[0]

    # Extract 2x2 linear part
    A = H[:, :2, :2]  # [B, 2, 2]

    # SVD to get rotation and scale
    U, S, Vh = torch.linalg.svd(A)

    # Rotation: R = U @ Vh
    R = torch.bmm(U, Vh)  # [B, 2, 2]

    # Handle reflection (det(R) = -1)
    det_R = R[:, 0, 0] * R[:, 1, 1] - R[:, 0, 1] * R[:, 1, 0]
    flip_mask = det_R < 0
    if flip_mask.any():
        R[flip_mask, :, 1] *= -1

    # Scale: geometric mean of singular values
    scale = S.prod(dim=-1).sqrt()  # [B]

    # Rotation angle
    theta = torch.atan2(R[:, 1, 0], R[:, 0, 0])  # [B]

    # Translation
    tx = H[:, 0, 2]  # [B]
    ty = H[:, 1, 2]  # [B]

    # Perspective
    p1 = H[:, 2, 0]  # [B]
    p2 = H[:, 2, 1]  # [B]

    return {
        "theta": theta,  # rotation angle in radians
        "scale": scale,  # scale factor
        "tx": tx,  # x translation
        "ty": ty,  # y translation
        "p1": p1,  # perspective param 1
        "p2": p2,  # perspective param 2
    }
