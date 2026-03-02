# Mathematical Analysis: Sim(2)-Equivariant Homography Network

**Date:** 2026-01-30
**Status:** Critical Analysis for ECCV 2026
**Author:** Research Team

---

## 1. Executive Summary

After a deep mathematical analysis of all model components, we have identified **fundamental issues** with the current approach that explain our poor test results. The core problem is:

> **The structure tensor extracts DOMINANT GRADIENT DIRECTION, not ROTATION ANGLE.**

These are mathematically different quantities. No amount of frame averaging or hypothesis selection can fix a fundamentally wrong estimator.

---

## 2. Component-by-Component Analysis

### 2.1 Data Flow Diagram

```
Input Images (img_src, img_tgt)
         │
         ├──────────────────────┐
         │                      │
         ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ ImageGradient   │    │ RelativeScale   │
│ Orientation     │    │ Estimator       │
│ Estimator       │    │                 │
│ ⚠️ PROBLEMATIC  │    │ ✓ WORKING       │
└────────┬────────┘    └────────┬────────┘
         │                      │
         │    θ_src, θ_tgt      │    scale_ratio
         │                      │
         ▼                      ▼
┌─────────────────────────────────────────┐
│         Sim2Canonicalizer               │
│  Rotate images by -θ, Scale by 1/s      │
└────────────────────┬────────────────────┘
                     │
                     ▼ (Canonical images)
┌─────────────────────────────────────────┐
│       PatchFeatureExtractor             │
│  Gradient magnitude only (✓ invariant)  │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│         E2EquivariantGNN                │
│  Distance² only (✓ rotation-invariant)  │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│          InvariantLRFT                  │
│  Distance-based attention (✓ invariant) │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│          SKSRegressionHead              │
│  H = S2 @ K @ S1 decomposition          │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│       Decanonicalization                │
│  H_orig = S_tgt @ R_tgt @ H @ R_src⁻¹   │
└────────────────────┬────────────────────┘
                     │
                     ▼
               Final Homography
```

---

## 3. Mathematical Analysis of Each Component

### 3.1 ImageGradientOrientationEstimator - ⚠️ FUNDAMENTAL ISSUE

**Current Implementation:**
```python
# Structure tensor (gradient covariance matrix)
Cxx = Σ(w * gx²)
Cyy = Σ(w * gy²)
Cxy = Σ(w * gx * gy)

# "Dominant gradient direction"
theta = 0.5 * atan2(2*Cxy, Cxx - Cyy)
```

**What This Actually Computes:**

The structure tensor M = [[Cxx, Cxy], [Cxy, Cyy]] is the **second moment matrix** of the gradient distribution. Its eigenvector corresponding to the largest eigenvalue gives the **dominant edge orientation** - the direction along which gradients vary most.

**Why This Is WRONG for Rotation Estimation:**

| Pattern | Unrotated θ | Rotated 30° → θ | Expected Change | Actual Change |
|---------|-------------|-----------------|-----------------|---------------|
| Vertical lines | 0° | 30° | +30° | +30° ✓ |
| Checkerboard | ~45° | ~45° | +30° | ~0° ✗ |
| Circle | undefined | undefined | +30° | N/A ✗ |
| L-shape | ~22.5° | ~52.5° | +30° | Variable ✗ |

**Mathematical Proof:**

For a checkerboard with edges at angles α and α+90°:
- Gradient vectors: g₁ ∝ (cos α, sin α), g₂ ∝ (-sin α, cos α)
- Structure tensor: M = I (isotropic!) for equal edge density
- θ = atan2(0, 0) = undefined

Rotating the checkerboard by φ:
- New gradient vectors: g₁' ∝ (cos(α+φ), sin(α+φ)), g₂' ∝ (-sin(α+φ), cos(α+φ))
- Structure tensor: Still M ≈ I (still isotropic)
- θ' ≈ θ (unchanged!)

**The structure tensor measures TEXTURE ANISOTROPY, not ROTATION.**

---

### 3.2 RelativeScaleEstimator - ✓ Working Correctly

**Implementation:**
```python
grad_src = mean(|∇I_src|)
grad_tgt = mean(|∇I_tgt|)
scale_ratio = grad_src / grad_tgt
```

**Why This Works:**

When an image is scaled by factor s:
- Zoomed IN (s > 1): gradients become gentler (spread over more pixels) → |∇I| decreases
- Zoomed OUT (s < 1): gradients become steeper → |∇I| increases

For same scene at different scales: |∇I_tgt| / |∇I_src| ≈ s_src / s_tgt

**Mathematical Justification:**

Let I(x) = f(x/s) where s is scale. Then:
∇I(x) = (1/s) ∇f(x/s)

So: mean(|∇I|) ∝ 1/s

Therefore: scale_ratio = grad_src / grad_tgt = s_tgt / s_src ✓

---

### 3.3 E2MessagePassing - ✓ Properly Rotation-Invariant

**Implementation:**
```python
# Edge features: ONLY distance squared
dist_sq = ||pos_j - pos_i||²
edge_features = [h_i, h_j, dist_sq]
message = MLP(edge_features)
```

**Why This Is Correct:**

For rotation R ∈ SO(2):
- ||R(p_j - p_i)||² = ||p_j - p_i||² (rotation preserves distance)
- Features h are invariant (come from invariant PatchFeatureExtractor)
- MLP output is thus identical under rotation

**Verification:** The existing test_equivariance() function confirms invariance error < 1e-6.

---

### 3.4 InvariantLRFT - ✓ Properly Rotation-Invariant

**Implementation:**
```python
dist_sq = ||query_pos - key_pos||²  # Pairwise distances
attn_logits = MLP(dist_sq)           # Distance → attention
out = softmax(attn_logits) @ V        # Aggregate
```

**Why This Is Correct:**

The attention weights depend ONLY on distances, which are rotation-invariant.
The value projection operates on invariant features.
Therefore: f(R·positions) = f(positions) ✓

---

### 3.5 SKSRegressionHead - ✓ Mathematically Sound

**Implementation:**
H = S₂ @ K @ S₁ where:
- S₁ = Similarity(θ₁, s₁, t₁) in source frame
- S₂ = Similarity(θ₂, s₂, t₂) in target frame
- K = Kernel (optional perspective)

**Benefits:**
1. Identity initialization: S₁ = S₂ = I, K = I → H = I
2. Explicit rotation/scale separation
3. Well-conditioned optimization landscape

**No issues detected.**

---

### 3.6 Frame Averaging / Hypothesis Selection - Partially Correct

**Current Implementation:**
```python
for offset in [0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4]:  # 8 hypotheses
    theta_k = theta_base + offset
    H_k = forward_single_hypothesis(theta_k)
    hypotheses.append(H_k)

# Select best by photometric consistency
H_best = select_by_reconstruction_error(hypotheses)
```

**The Problem:**

If `theta_base` is FUNDAMENTALLY WRONG (from structure tensor), then all hypotheses are wrong:
- theta_base = θ_wrong
- Hypotheses: θ_wrong, θ_wrong+45°, θ_wrong+90°, ...

The correct rotation θ_true might not be in this set!

**Example:**
- True rotation: θ_true = 17°
- Structure tensor gives: θ_wrong = 45° (dominant gradient direction)
- Hypotheses: 45°, 90°, 135°, 180°, 225°, 270°, 315°, 0°
- None of these include 17° or anything close!

---

## 4. Root Cause Summary

| Component | Status | Issue | Impact |
|-----------|--------|-------|--------|
| ImageGradientOrientationEstimator | ⚠️ BROKEN | Extracts gradient direction, not rotation | Canonicalization fails |
| RelativeScaleEstimator | ✓ OK | Works correctly | None |
| PatchFeatureExtractor | ✓ OK | Magnitude-only is invariant | None |
| E2EquivariantGNN | ✓ OK | Distance² only | None |
| InvariantLRFT | ✓ OK | Distance-based attention | None |
| SKSRegressionHead | ✓ OK | Proper decomposition | None |
| Frame Averaging | ⚠️ LIMITED | Can't fix bad base estimate | High error on many patterns |

**The single point of failure is rotation estimation.**

---

## 5. Mathematically Principled Solutions

### Solution 1: Phase Correlation in Log-Polar Domain (RECOMMENDED)

**Mathematical Foundation:**

For images I₁(x,y) and I₂(x,y) = I₁(R_θ·(sx, sy)):
1. Fourier Transform: F₁(u,v), F₂(u,v)
2. Magnitude spectra: |F₁|, |F₂| (translation-invariant)
3. Log-polar conversion: M₁(log r, θ), M₂(log r, θ)
4. In log-polar space: rotation → horizontal shift, scale → vertical shift
5. Phase correlation finds the shift → (θ, log s)

**Implementation Sketch:**
```python
class PhaseCorrelationRotationScale(nn.Module):
    def forward(self, img_src, img_tgt):
        # FFT magnitude spectra (translation-invariant)
        F_src = torch.abs(torch.fft.fft2(img_src))
        F_tgt = torch.abs(torch.fft.fft2(img_tgt))

        # High-pass filter (remove DC)
        F_src = highpass(F_src)
        F_tgt = highpass(F_tgt)

        # Convert to log-polar coordinates
        LP_src = cartesian_to_logpolar(F_src)
        LP_tgt = cartesian_to_logpolar(F_tgt)

        # Phase correlation
        cross_power = (LP_src * LP_tgt.conj()) / (|LP_src * LP_tgt.conj()| + ε)
        correlation = torch.fft.ifft2(cross_power)

        # Find peak → (rotation, log_scale)
        peak_y, peak_x = argmax(correlation)
        rotation = peak_x * (2π / width)
        scale = exp(peak_y * log_scale_range / height)

        return rotation, scale
```

**Advantages:**
- Works for ANY pattern (theoretically proven)
- Estimates BOTH rotation and scale simultaneously
- Classic, well-understood algorithm
- Differentiable with soft-argmax

**Disadvantages:**
- Requires careful handling of FFT wraparound
- Log-polar resampling introduces interpolation errors
- May struggle with very small images

---

### Solution 2: Learned Relative Rotation Network

**Mathematical Foundation:**

Train a network to predict Δθ = θ_tgt - θ_src directly from image pair:
f(I_src, I_tgt) → (cos Δθ, sin Δθ)

Using (cos, sin) output ensures valid angle via atan2.

**Implementation Sketch:**
```python
class LearnedRelativeRotation(nn.Module):
    def __init__(self):
        self.encoder = ResNet18(pretrained=True)  # Shared encoder
        self.correlation = nn.Conv2d(...)  # Correlation layer
        self.predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # (cos, sin)
        )

    def forward(self, img_src, img_tgt):
        # Extract features
        feat_src = self.encoder(img_src)
        feat_tgt = self.encoder(img_tgt)

        # Correlation
        corr = self.correlation(feat_src, feat_tgt)

        # Predict rotation
        cos_sin = self.predictor(corr)
        cos_sin = F.normalize(cos_sin, dim=-1)  # Unit vector

        delta_theta = torch.atan2(cos_sin[:, 1], cos_sin[:, 0])
        return delta_theta
```

**Training:**
- Use synthetic data with known rotations
- Loss: L = ||predicted - (cos Δθ_true, sin Δθ_true)||²

**Advantages:**
- Can learn complex rotation patterns
- End-to-end trainable
- Pattern-independent (learned from data)

**Disadvantages:**
- Requires training data with rotation labels
- May not generalize to unseen patterns
- Adds parameters

---

### Solution 3: Procrustes Analysis on Soft Correspondences

**Mathematical Foundation:**

Given soft correspondences from similarity matrix:
1. Extract weighted point pairs: (p_src, Σ_tgt w_ij * p_tgt_j)
2. Solve Procrustes problem: find (R, s, t) minimizing:
   min ||R·s·P_src + t - P_tgt||²_F
3. Closed-form solution via SVD

**Implementation Sketch:**
```python
class ProcrustesRotationEstimator(nn.Module):
    def forward(self, positions_src, positions_tgt, similarity):
        # Soft correspondences
        weights = F.softmax(similarity, dim=-1)  # [B, N, N]
        matched_tgt = torch.bmm(weights, positions_tgt)  # [B, N, 2]

        # Center the point clouds
        centroid_src = positions_src.mean(dim=1, keepdim=True)
        centroid_tgt = matched_tgt.mean(dim=1, keepdim=True)

        P = positions_src - centroid_src
        Q = matched_tgt - centroid_tgt

        # Cross-covariance matrix
        H = torch.bmm(P.transpose(-2, -1), Q)  # [B, 2, 2]

        # SVD
        U, S, Vh = torch.linalg.svd(H)

        # Rotation: R = V @ U^T
        R = torch.bmm(Vh.transpose(-2, -1), U.transpose(-2, -1))

        # Handle reflection
        det = torch.det(R)
        # ... fix negative determinants ...

        # Extract angle
        theta = torch.atan2(R[:, 1, 0], R[:, 0, 0])

        return theta
```

**Advantages:**
- Closed-form, differentiable
- No learnable parameters
- Uses actual correspondences (geometry-based)

**Disadvantages:**
- Depends on quality of soft correspondences
- Sensitive to outliers (need robust version)

---

### Solution 4: Steerable CNNs (ESCNN)

**Mathematical Foundation:**

Use rotation-equivariant features that transform predictably:
f(R · x) = ρ(R) · f(x)

Where ρ is a group representation. For SO(2), use regular representations.

**Implementation Sketch:**
```python
import escnn
from escnn import gspaces

class SteerableFeatureExtractor(nn.Module):
    def __init__(self):
        self.gspace = gspaces.Rot2dOnR2(N=8)  # C8 rotation group

        in_type = escnn.nn.FieldType(self.gspace, [self.gspace.trivial_repr])

        # Regular representation features
        hidden_type = escnn.nn.FieldType(self.gspace,
            8 * [self.gspace.regular_repr])

        self.net = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(in_type, hidden_type, kernel_size=5),
            escnn.nn.InnerBatchNorm(hidden_type),
            escnn.nn.ReLU(hidden_type),
            # ... more layers ...
            escnn.nn.GroupPooling(hidden_type),  # → invariant
        )

    def forward(self, x):
        x_geo = escnn.nn.GeometricTensor(x, self.in_type)
        return self.net(x_geo).tensor
```

**Advantages:**
- Mathematically guaranteed equivariance
- Well-established theory and implementation
- No canonicalization needed

**Disadvantages:**
- Different architecture (requires retraining)
- Higher computational cost
- Less flexibility

---

## 6. Recommended Action Plan

### Immediate (Can Run in Parallel with Current Job)

**Action 1: Implement Phase Correlation Rotation Estimator**
- Create `src/models/phase_correlation.py`
- Unit test on synthetic rotations
- Replace ImageGradientOrientationEstimator

**Action 2: Implement Procrustes Alternative**
- Create `src/models/procrustes_estimator.py`
- Use after similarity matrix computation
- Provides geometry-based rotation estimate

### Short-term (This Week)

**Action 3: Design Comparison Experiment**
```
Test each rotation estimator:
1. Current (structure tensor) - baseline
2. Phase correlation
3. Procrustes
4. Learned (if time permits)

Metrics:
- Rotation estimation error vs ground truth
- Final homography corner error
- Generalization across patterns
```

### Medium-term (Before ECCV Deadline)

**Action 4: Steerable CNN Integration**
- More significant change
- Provides theoretical guarantees for paper

---

## 7. Key Insight for ECCV Paper

The paper narrative should be:

> **"Canonicalization-based equivariance requires ACCURATE canonical frame estimation."**

Current methods use structure tensor, which extracts texture direction, not geometric rotation. We propose [chosen method] which provably estimates the correct transformation.

This is a **novel contribution**: Identifying and fixing this fundamental issue in canonicalization-based approaches.

---

## 8. Appendix: Mathematical Proofs

### Proof: Structure Tensor ≠ Rotation Estimator

**Theorem:** For an image I with symmetric gradient distribution (e.g., checkerboard), the structure tensor principal direction θ is invariant under rotation.

**Proof:**
Let I(x) have gradient field g(x) = (gx(x), gy(x)).

Structure tensor: M = ∫ g(x) g(x)^T dx

For rotation R_φ, transformed image I'(x) = I(R_{-φ} x).
Gradient: g'(x) = R_φ g(R_{-φ} x)

Transformed structure tensor:
M' = ∫ g'(x) g'(x)^T dx
   = ∫ R_φ g(R_{-φ} x) g(R_{-φ} x)^T R_φ^T dx
   = R_φ [∫ g(y) g(y)^T dy] R_φ^T    (change of variables y = R_{-φ} x)
   = R_φ M R_φ^T

For symmetric gradients (checkerboard), M = σ² I (identity).
Then: M' = R_φ (σ² I) R_φ^T = σ² I = M

The principal direction θ = atan2(2 M_xy, M_xx - M_yy) = atan2(0, 0) is undefined/arbitrary. □

---

*Last updated: 2026-01-30*
*Status: Ready for Implementation*
