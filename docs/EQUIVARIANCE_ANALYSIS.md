# Equivariance Analysis and Solutions

**Date:** 2026-01-30
**Status:** Comprehensive Analysis Complete

---

## Executive Summary

The Sim(2) equivariance framework has been thoroughly tested across 7 patterns, 13 rotation angles, and 11 scale factors. Key findings:

1. **Rotation canonicalization is pattern-dependent**: Arrow patterns work perfectly (14.5px error), but spiral/L-shapes fail catastrophically (>200px error)
2. **RelativeScaleEstimator is working correctly**: Outputs ~1.0 when both images have same scale (correct behavior)
3. **Structure tensor fundamentally limited**: Only detects "dominant gradient direction", not true rotation
4. **90° angle is a blind spot**: 187.72px error vs 75.76px at 30°

---

## Test Results Summary (2026-01-30)

### Rotation-Only Test (13 angles × 7 patterns)

| Pattern | Error (px) | Rot Err (°) | Coherence | Status |
|---------|-----------|-------------|-----------|--------|
| **arrow** | **14.49 ± 0.31** | **0.56°** | **0.270** | ✅ EXCELLENT |
| diagonal_gradient | 51.46 ± 62.54 | 15.88° | 0.410 | ⚠️ OK |
| blob | 97.15 ± 113.12 | 33.26° | 0.007 | ❌ |
| checkerboard | 128.01 ± 111.96 | 41.68° | 0.007 | ❌ |
| T_shape | 159.60 ± 111.17 | 57.05° | **0.000** | ❌ FAIL |
| L_shape | 179.56 ± 88.76 | 64.94° | 0.004 | ❌ FAIL |
| **spiral** | **216.59 ± 83.63** | **75.98°** | 0.129 | ❌ WORST |

### Performance by Rotation Angle

| Angle | Error (px) | Rot Error (°) | Notes |
|-------|-----------|---------------|-------|
| 30° | 75.76 ± 87.06 | 22.99° | **Best** |
| 15° | 83.42 ± 85.13 | 26.26° | Good |
| 45° | 96.22 ± 83.36 | 30.29° | Moderate |
| **90°** | **187.72 ± 84.30** | **65.86°** | **WORST** |
| 135° | 159.98 ± 140.01 | 59.68° | Bad |

### Scale-Only Test (11 scales × 7 patterns)

| Scale | Error (px) | Scale Err | Detected Ratio | Notes |
|-------|-----------|-----------|----------------|-------|
| 2.0x | 55.55 ± 38.90 | 0.0958 | 0.924 | **Best** |
| 1.1x | 89.91 ± 83.85 | 0.0893 | 0.975 | Good |
| 1.0x | 119.51 ± 98.92 | 0.0657 | 0.937 | Baseline |
| **0.5x** | **207.74 ± 131.15** | 0.0263 | 1.027 | **WORST** |

**Important**: Detected ratios are all ~1.0 because the test transforms BOTH images by the same factor. This is **correct behavior** for RelativeScaleEstimator!

---

## Root Cause Analysis

### Problem 1: Structure Tensor Limitations

The `ImageGradientOrientationEstimator` uses structure tensor eigenanalysis:

```python
# What it actually computes:
Cxx = Σ(w * gx²)  # Second moments of gradients
Cyy = Σ(w * gy²)
Cxy = Σ(w * gx * gy)
theta = 0.5 * atan2(2*Cxy, Cxx - Cyy)  # Dominant gradient direction
```

**Fundamental limitation**: This extracts the **dominant gradient direction**, NOT the rotation angle. For symmetric patterns:
- 90° rotation may give same θ (4-fold symmetry)
- Circular patterns have no dominant direction (low coherence)
- Pattern-specific gradients don't correlate with rotation

### Problem 2: 90° Blind Spot

The 90° angle shows dramatically worse performance (187.72px vs 75.76px at 30°) because:

1. Structure tensor has 180° ambiguity (θ and θ+180° are equivalent)
2. 90° rotation flips gradient components: (gx, gy) → (-gy, gx)
3. For symmetric patterns, Cxx ≈ Cyy, making atan2 unstable

### Problem 3: Pattern-Dependent Coherence

| Pattern | Coherence | Works? | Why |
|---------|-----------|--------|-----|
| diagonal_gradient | 0.410 | ✅ | Strong single-direction gradient |
| arrow | 0.270 | ✅ | Clear asymmetric gradient structure |
| spiral | 0.129 | ❌ | Mixed gradient directions |
| checkerboard | 0.007 | ❌ | Equal X and Y gradients (4-fold symmetric) |
| T_shape | **0.000** | ❌ | No dominant direction |

---

## Innovative Solutions

### Solution 1: Learned Rotation Estimator (HIGH PRIORITY)

**Approach**: Train a small CNN to predict relative rotation between image pairs.

```python
class LearnedRelativeRotation(nn.Module):
    """Learn rotation difference directly from image pairs."""

    def __init__(self, feature_dim=64):
        super().__init__()
        # Shared encoder (rotation-invariant features)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Global pooling
        )
        # Cross-attention for relative features
        self.cross_attn = nn.MultiheadAttention(32, 4)
        # Output: sin(Δθ), cos(Δθ)
        self.predictor = nn.Linear(32, 2)

    def forward(self, img_src, img_tgt):
        feat_src = self.encoder(img_src)
        feat_tgt = self.encoder(img_tgt)
        # Compute relative features
        rel_feat, _ = self.cross_attn(feat_src, feat_tgt, feat_tgt)
        # Predict rotation as (sin, cos) pair
        out = self.predictor(rel_feat.squeeze())
        return torch.atan2(out[:, 0], out[:, 1])
```

**Advantages**:
- Pattern-independent (learned from data)
- Directly estimates relative rotation (like RelativeScaleEstimator)
- No arbitrary reference needed

### Solution 2: Steerable CNNs (e2cnn/escnn)

**Approach**: Replace the feature extractor with rotation-equivariant steerable CNNs.

```python
import escnn
from escnn import gspaces

class SteerableFeatureExtractor(nn.Module):
    """Rotation-equivariant feature extraction using steerable CNNs."""

    def __init__(self, out_channels=32):
        super().__init__()
        self.gspace = gspaces.Rot2dOnR2(N=8)  # C8 rotation group
        self.in_type = escnn.nn.FieldType(self.gspace, [self.gspace.trivial_repr])

        # Build equivariant network
        self.net = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(self.in_type, ...),
            escnn.nn.NormNonlinearity(...),
            ...
        )

    def forward(self, x):
        x_geo = escnn.nn.GeometricTensor(x, self.in_type)
        return self.net(x_geo).tensor
```

**Advantages**:
- Guaranteed equivariance by construction
- No canonicalization needed
- Proven in multiple computer vision tasks

### Solution 3: Multi-Hypothesis Frame Averaging with Adaptive Selection

**Approach**: Increase hypotheses from 4 to 8 and use coherence-weighted selection.

```python
class AdaptiveHypothesisSelection(nn.Module):
    """Select best hypothesis based on coherence and consistency."""

    def __init__(self, num_hypotheses=8):
        self.offsets = torch.linspace(0, 2*np.pi, num_hypotheses+1)[:-1]

    def forward(self, img_src, img_tgt, base_theta_src, base_theta_tgt):
        homographies = []
        confidences = []

        for offset in self.offsets:
            theta_src = base_theta_src + offset
            theta_tgt = base_theta_tgt + offset

            # Canonicalize
            img_src_canon = rotate(img_src, -theta_src)
            img_tgt_canon = rotate(img_tgt, -theta_tgt)

            # Predict homography
            H_k = self.predict(img_src_canon, img_tgt_canon)

            # Compute consistency score
            conf = self.compute_consistency(img_src, img_tgt, H_k)

            homographies.append(H_k)
            confidences.append(conf)

        # Soft selection via softmax
        weights = F.softmax(torch.stack(confidences), dim=0)
        return sum(w * H for w, H in zip(weights, homographies))
```

### Solution 4: Phase Correlation for Rotation Estimation

**Approach**: Use Fourier-domain phase correlation to estimate rotation.

```python
class PhaseCorrelationRotation(nn.Module):
    """Estimate rotation using log-polar FFT phase correlation."""

    def forward(self, img_src, img_tgt):
        # Convert to frequency domain
        F_src = torch.fft.fft2(img_src)
        F_tgt = torch.fft.fft2(img_tgt)

        # Convert to log-polar coordinates
        # (rotation becomes translation in log-polar space)
        lp_src = self.cartesian_to_logpolar(torch.abs(F_src))
        lp_tgt = self.cartesian_to_logpolar(torch.abs(F_tgt))

        # Phase correlation to find translation (= rotation in original)
        cross_power = (lp_src * lp_tgt.conj()) / (torch.abs(lp_src * lp_tgt.conj()) + 1e-8)
        correlation = torch.fft.ifft2(cross_power)

        # Find peak
        peak = torch.argmax(correlation.abs())
        rotation = peak_to_angle(peak)

        return rotation
```

**Advantages**:
- Mathematically principled
- Works for any pattern
- Rotation-invariant to pattern content

### Solution 5: Keypoint-Based Relative Rotation

**Approach**: Detect keypoints and use RANSAC to estimate relative rotation.

```python
class KeypointRelativeRotation(nn.Module):
    """Estimate rotation from matched keypoints."""

    def __init__(self):
        self.detector = nn.Sequential(...)  # Learnable detector
        self.descriptor = nn.Sequential(...)  # Learnable descriptor

    def forward(self, img_src, img_tgt):
        # Detect and describe keypoints
        kp_src, desc_src = self.detect_and_describe(img_src)
        kp_tgt, desc_tgt = self.detect_and_describe(img_tgt)

        # Match keypoints
        matches = self.match(desc_src, desc_tgt)

        # RANSAC for rotation (in similarity group)
        rotation = self.ransac_rotation(kp_src[matches[:, 0]],
                                         kp_tgt[matches[:, 1]])

        return rotation
```

---

## Recommended Action Plan

### Phase 1: Quick Wins (1-2 days)

1. **Increase hypotheses to 8**: Simple change, may help with 45° angles
2. **Add consistency-based selection**: Use photometric consistency to select best hypothesis
3. **Filter by coherence threshold**: Skip canonicalization for low-coherence patterns

### Phase 2: Novel Research (1-2 weeks)

1. **Implement LearnedRelativeRotation**: Train on synthetic data with known rotations
2. **Evaluate steerable CNNs**: Test ESCNN integration for feature extraction

### Phase 3: Paper-Ready (2-3 weeks)

1. **Comprehensive ablation study**: Compare all solutions
2. **Real-world dataset evaluation**: Test on thermal-visible datasets
3. **Theoretical analysis**: Prove equivariance guarantees for chosen method

---

## Success Metrics

| Metric | Current | Phase 1 Target | Phase 2 Target |
|--------|---------|---------------|----------------|
| Rotation std (scale=1.0) | 111.69 px | < 50 px | < 20 px |
| 90° angle error | 187.72 px | < 100 px | < 30 px |
| Mean coherence | 0.118 | > 0.3 | > 0.5 |
| Worst pattern error | 216.59 px | < 150 px | < 50 px |

---

*Last updated: 2026-01-30 14:00*
