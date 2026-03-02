# Implementation Plan: Sim(2)-Equivariant Thermal Homography Network

**Goal:** Build a network with full **translation, rotation, and scale invariance** (Sim(2) = SO(2) × R+ × R²)

**Novel Contribution:** First Sim(2)-Equivariant Deep Homography Estimator

---

## Overview

| Phase | Focus | Invariance Added | Verification |
|-------|-------|------------------|--------------|
| 2a | Fix E2MessagePassing | Rotation | Flat line on rotation test |
| 2b | Fix LRFT | Rotation (preserved) | Rotation test still passes |
| 3a | Add scale canonicalization | Scale | Flat line on scale test |
| 3b | Full Sim(2) verification | Rotation + Scale | Joint rotation-scale test |
| 4 | SKS regression head | Better convergence | Lower corner error |
| 5 | Integration + ablations | All combined | SOTA comparison |

---

## Phase 2a: Fix Rotation Equivariance (CRITICAL)

### Problem
Current `E2MessagePassing` uses `cos(angle), sin(angle)` of edge directions, which change with rotation.

### Solution: EGNN-Style Message Passing

Use ONLY rotation-invariant quantities:
- Distance squared: `||x_i - x_j||²` ✓ (invariant)
- Distance ratios: `||x_i - x_j|| / ||x_i - x_k||` ✓ (invariant)
- NO absolute angles

### Code Changes

**File:** `src/models/e2_layers.py`

```python
# BEFORE (broken - lines 140-143):
angle = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])
cos_angle = torch.cos(angle).unsqueeze(-1)
sin_angle = torch.sin(angle).unsqueeze(-1)

# AFTER (EGNN-style):
# Remove angle computation entirely
# Use only distance squared
dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)  # [E, 1]

# Fixed normalization (NOT batch-dependent)
# For positions in [-1, 1]², diagonal = 2*sqrt(2) ≈ 2.83
# Square of diagonal = 8
dist_sq_normalized = dist_sq / 8.0  # [E, 1]
```

**Updated message function:**
```python
# BEFORE (line 181):
edge_features = torch.cat([x_i, x_j, dist, cos_angle, sin_angle], dim=-1)

# AFTER:
edge_features = torch.cat([x_i, x_j, dist_sq_normalized], dim=-1)
# Reduced from 2*in_channels + 3 to 2*in_channels + 1
```

### Verification Checkpoint

Run the rotation equivariance test:
```bash
python scripts/phase1_synthetic_validation.py --config checkerboard_low_noise --epochs 30
```

**Success criteria:**
- Error std across angles < 5 px (was 82 px)
- Error should be approximately FLAT from 0° to 180°

---

## Phase 2b: Fix LRFT Rotation Equivariance

### Problem
`AdaptiveLRFT` uses fixed learned queries that don't rotate with input.

### Solution: Distance-Based Attention

Replace feature-based queries with position-based attention using distances.

**File:** `src/models/lrft.py`

```python
class InvariantLRFT(nn.Module):
    """
    LRFT with rotation-invariant attention.

    Key insight: Attention weights based on ||query_pos - key_pos||²
    are rotation-invariant because distance is preserved under rotation.
    """

    def __init__(self, feature_dim=32, out_nodes=128, num_heads=4):
        super().__init__()
        self.out_nodes = out_nodes
        self.num_heads = num_heads

        # Fixed spatial query grid (positions, not features)
        query_pos = self._create_uniform_grid(out_nodes)
        self.register_buffer('query_positions', query_pos)

        # Distance-based attention
        self.attn_mlp = nn.Sequential(
            nn.Linear(1, 64),  # Input: distance squared
            nn.SiLU(),
            nn.Linear(64, num_heads),
        )

        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features, positions):
        B, N, D = features.shape

        # Compute pairwise distances: [B, N_out, N_in]
        q = self.query_positions.unsqueeze(0).expand(B, -1, -1)  # [B, N', 2]
        dist_sq = self._pairwise_dist_sq(q, positions)  # [B, N', N]

        # Distance-based attention (ROTATION INVARIANT!)
        attn_logits = self.attn_mlp(dist_sq.unsqueeze(-1))  # [B, N', N, heads]
        attn_logits = attn_logits.permute(0, 3, 1, 2)  # [B, heads, N', N]
        attn = F.softmax(-attn_logits, dim=-1)  # Closer = higher weight

        # Aggregate values
        V = self.value_proj(features)  # [B, N, D]
        out = torch.einsum('bhqk,bkd->bhqd', attn, V)
        out = out.mean(dim=1)  # Average over heads

        out = self.out_proj(out)
        out = self.norm(out)

        # Compressed positions
        attn_avg = attn.mean(dim=1)
        compressed_pos = torch.bmm(attn_avg, positions)

        return out, compressed_pos
```

### Verification Checkpoint

Re-run rotation test after LRFT fix:
- Should still show flat line
- May have slightly different absolute error (fine-tuning needed)

---

## Phase 3a: Add Scale Canonicalization

### Concept (from EquiBot)

Scale equivariance = canonicalization + rotation equivariance + rescaling

```
Input → Estimate Scale → Normalize by Scale → Process (rotation-equivariant) → Rescale
```

### Implementation

**New file:** `src/models/scale_canonicalizer.py`

```python
class ScaleCanonicalizer(nn.Module):
    """
    Estimate and canonicalize scale for Sim(2) equivariance.

    If input is scaled by s, the canonicalized positions are the same,
    making downstream processing scale-invariant.
    """

    def __init__(self, feature_dim):
        super().__init__()

        # Scale predictor from global features
        self.scale_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.SiLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Softplus(),  # Ensure positive scale
        )

        # Minimum scale to avoid division by zero
        self.min_scale = 0.1

    def forward(self, features, positions):
        """
        Args:
            features: [B, N, D] node features
            positions: [B, N, 2] node positions

        Returns:
            canonical_positions: [B, N, 2] scale-normalized positions
            estimated_scale: [B, 1] estimated scale factor
        """
        B = features.shape[0]

        # Global feature aggregation
        global_feat = features.mean(dim=1)  # [B, D]

        # Estimate scale
        estimated_scale = self.scale_net(global_feat)  # [B, 1]
        estimated_scale = estimated_scale.clamp(min=self.min_scale)

        # Canonicalize positions
        canonical_positions = positions / estimated_scale.unsqueeze(-1)

        return canonical_positions, estimated_scale

    def rescale_output(self, output, estimated_scale):
        """Rescale output back to original scale."""
        # For homography, scale affects translation components
        # H' = S @ H @ S^{-1} where S = diag(s, s, 1)
        return output  # Implement based on output format
```

### Integration

**File:** `src/models/graph_network.py`

```python
class ThermalHomographyNet(nn.Module):
    def __init__(self, ...):
        # ... existing code ...

        # NEW: Scale canonicalization
        self.scale_canonicalizer = ScaleCanonicalizer(feature_dim)

    def extract_graph_features(self, image):
        # ... existing feature extraction ...

        # NEW: Canonicalize scale before GNN
        positions_canonical, scale = self.scale_canonicalizer(
            features, positions_batched
        )

        # Use canonical positions for GNN
        features_flat = self.e2_gnn(
            features_flat, positions_canonical.view(-1, 2), edge_index, batch
        )

        return features, positions_canonical, scale
```

### Verification Checkpoint

Create scale equivariance test:
```python
def test_scale_equivariance(model, image_src, image_tgt, scales=[0.5, 1.0, 2.0]):
    """Test that predictions are consistent across input scales."""
    predictions = []
    for s in scales:
        # Scale images
        scaled_src = F.interpolate(image_src, scale_factor=s)
        scaled_tgt = F.interpolate(image_tgt, scale_factor=s)

        # Predict
        H_pred = model(scaled_src, scaled_tgt)

        # Transform H back to original scale
        # H_orig = S^{-1} @ H_scaled @ S
        H_orig = scale_homography_inverse(H_pred, s)
        predictions.append(H_orig)

    # Check consistency
    std = torch.stack(predictions).std(dim=0).mean()
    return std  # Should be close to 0
```

**Success criteria:**
- Prediction std across scales < 1 px corner error
- Should work for scales 0.5x to 2.0x

---

## Phase 3b: Full Sim(2) Verification

### Joint Rotation-Scale Test

```python
def test_sim2_equivariance(model, image_src, image_tgt):
    """Test joint rotation and scale equivariance."""
    angles = [0, 45, 90, 135, 180]
    scales = [0.7, 1.0, 1.5]

    errors = []
    for angle in angles:
        for scale in scales:
            # Apply rotation and scale
            transformed_src = rotate_and_scale(image_src, angle, scale)
            transformed_tgt = rotate_and_scale(image_tgt, angle, scale)

            # Predict
            H_pred = model(transformed_src, transformed_tgt)

            # Transform H back
            H_orig = inverse_sim2_transform(H_pred, angle, scale)

            # Compare to baseline (0°, 1.0x)
            errors.append(corner_error(H_orig, H_baseline))

    return np.std(errors)  # Should be small
```

**Success criteria:**
- Error std across all (angle, scale) combinations < 5 px
- No systematic bias in any direction

---

## Phase 4: SKS Decomposition Regression Head

### Motivation
- Current 8-parameter vector couples rotation, scale, translation, perspective
- SKS decomposition separates similarity (Sim(2)) from projective components
- Makes rotation and scale explicitly learnable

### Implementation

**File:** `src/models/sks_head.py`

```python
class SKSRegressionHead(nn.Module):
    """
    Regress homography using SKS decomposition:
    H = S2 @ K @ S1

    Where:
    - S1: Similarity transform in source frame (4 params: θ, s, tx, ty)
    - S2: Similarity transform in target frame (4 params: θ, s, tx, ty)
    - K: Kernel/projective transform (optional, for perspective)

    This makes Sim(2) components explicit and directly learnable.
    """

    def __init__(self, feature_dim=32, hidden_dim=256, use_perspective=False):
        super().__init__()

        self.use_perspective = use_perspective

        # Shared feature processor
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Similarity heads (rotation, scale, translation)
        self.sim1_head = nn.Linear(hidden_dim, 4)  # θ1, s1, tx1, ty1
        self.sim2_head = nn.Linear(hidden_dim, 4)  # θ2, s2, tx2, ty2

        # Optional perspective head
        if use_perspective:
            self.persp_head = nn.Linear(hidden_dim, 2)  # p1, p2

        self._init_weights()

    def _init_weights(self):
        # Initialize to identity transform
        nn.init.zeros_(self.sim1_head.weight)
        nn.init.zeros_(self.sim2_head.weight)

        # Bias: θ=0, s=1, tx=0, ty=0
        with torch.no_grad():
            self.sim1_head.bias.data = torch.tensor([0, 1, 0, 0], dtype=torch.float32)
            self.sim2_head.bias.data = torch.tensor([0, 1, 0, 0], dtype=torch.float32)

    def forward(self, features_src, features_tgt, similarity_matrix):
        B = features_src.shape[0]

        # Soft matching
        assignment = F.softmax(similarity_matrix, dim=-1)
        matched_tgt = torch.bmm(assignment, features_tgt)

        # Concatenate and encode
        combined = torch.cat([features_src, matched_tgt], dim=-1)
        encoded = self.encoder(combined.mean(dim=1))  # Global pooling

        # Predict Sim(2) components
        sim1_params = self.sim1_head(encoded)  # [B, 4]
        sim2_params = self.sim2_head(encoded)  # [B, 4]

        # Construct similarity matrices
        S1 = self._params_to_similarity(sim1_params)  # [B, 3, 3]
        S2 = self._params_to_similarity(sim2_params)  # [B, 3, 3]

        # Kernel (identity if not using perspective)
        if self.use_perspective:
            persp = self.persp_head(encoded)
            K = self._params_to_kernel(persp)
        else:
            K = torch.eye(3, device=encoded.device).unsqueeze(0).expand(B, -1, -1)

        # Compose: H = S2 @ K @ S1
        H = S2 @ K @ S1

        # Normalize
        H = H / (H[:, 2:3, 2:3] + 1e-8)

        return H

    def _params_to_similarity(self, params):
        """Convert (θ, s, tx, ty) to 3x3 similarity matrix."""
        θ, s, tx, ty = params.unbind(dim=-1)

        cos_θ = torch.cos(θ)
        sin_θ = torch.sin(θ)

        B = params.shape[0]
        S = torch.zeros(B, 3, 3, device=params.device)

        S[:, 0, 0] = s * cos_θ
        S[:, 0, 1] = -s * sin_θ
        S[:, 0, 2] = tx
        S[:, 1, 0] = s * sin_θ
        S[:, 1, 1] = s * cos_θ
        S[:, 1, 2] = ty
        S[:, 2, 2] = 1.0

        return S
```

### Verification

- Corner error should decrease (better parameterization)
- Rotation/scale components should match ground truth more closely

---

## Phase 5: Full Integration and Ablations

### Ablation Studies

| Ablation | What We Remove | Expected Impact |
|----------|----------------|-----------------|
| No rotation fix | Keep angle features | +100% error on rotations |
| No scale canonicalization | Skip scale normalization | +50% error on scale variations |
| No SKS decomposition | Use 8-param vector | Slightly worse convergence |
| No LRFT | Direct similarity | Similar accuracy, slower |

### Final Architecture

```
Input: (I_src, I_tgt)
    ↓
Feature Extraction (Steerable CNN or Patch-based)
    ↓
Graph Construction (16x16 or 32x32 grid)
    ↓
Scale Canonicalization ← NEW
    ↓
EGNN-Style Message Passing (distance only, no angles) ← FIXED
    ↓
Invariant LRFT (distance-based attention) ← FIXED
    ↓
Similarity Matrix Computation
    ↓
SKS Regression Head ← NEW
    ↓
Output: H (3x3 homography)
```

### ECCV Comparison

| Method | Rotation Robust | Scale Robust | Thermal | Corner Error |
|--------|-----------------|--------------|---------|--------------|
| HomographyNet | No | No | No | ~X px |
| Content-Aware | Augmentation | Augmentation | No | ~X px |
| SE2-LoFTR | Yes (C_N) | No | No | ~X px |
| **Ours (Sim(2))** | **Yes (SO(2))** | **Yes (R+)** | **Yes** | **~X px** |

---

## Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | 2a | Fixed E2MessagePassing (no angles) |
| 1 | 2b | Rotation test passes (flat line) |
| 2 | 3a | Scale canonicalization implemented |
| 2 | 3b | Sim(2) test passes |
| 3 | 4 | SKS regression head |
| 4 | 5 | Ablations complete |
| 5-6 | - | Real data experiments, paper writing |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Rotation fix doesn't work | Fall back to distance + relative angles between edges |
| Scale canonicalization unstable | Use multi-scale feature pyramid instead |
| SKS decomposition slow to converge | Pre-train components separately |
| ECCV deadline pressure | Prioritize rotation fix (most impact) |

---

*Created: 2026-01-30*
*Target: ECCV 2026*
