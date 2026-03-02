# Sim(2) Equivariant Homography Estimation

**Date:** 2026-01-30, 7:49 PM AEDT
**Status:** Implementation Plan Approved
**Last Updated:** 2026-01-30

---

## ⚠️ CRITICAL CLARIFICATION: Equivariance Status (2026-01-30)

### What We Actually Have

| Component | Equivariant? | Method | Notes |
|-----------|--------------|--------|-------|
| **Rotation** | ✅ Yes | ESCNN C₁₆ regular repr → cyclic shift → FFT correlation | True SE(2) equivariance |
| **Translation** | ✅ Yes | Inherent to CNNs (spatial features shift with input) | Standard property |
| **Scale** | ❌ **No** | Procrustes on matched correspondences | Classical, NOT equivariant |

### What This Means

**The current architecture is SE(2)-equivariant (rotation + translation), but NOT Sim(2)-equivariant.**

Scale is estimated via Procrustes after de-rotation - this is a **classical geometric method**, not a scale-equivariant neural network approach. For true Sim(2) equivariance, we would need **scale-equivariant features**.

### Is This a Problem?

**For practical Sim(2) estimation**: No - the hybrid approach (equivariant rotation + classical scale) works well.

**For claiming "Sim(2)-equivariant architecture"**: Yes - this is technically incorrect.

### Options for True Scale Equivariance

1. **Scale-Space Networks**: Process images at multiple scales, equivariant features across scale pyramid
2. **Log-Polar Transform**: Convert scale changes to translations, which CNNs handle equivariantly
3. **Steerable Scale Networks**: Scale-equivariant convolutions (e.g., SESN - Scale-Equivariant Steerable Networks)
4. **Dilated Convolution Pyramids**: Different dilation rates capture different scales

**See PHASE 9 below for detailed planning.**

---

## Problem Analysis (Critical Discovery)

### What Went Wrong

The previous architecture used `GroupPooling` which produces rotation-**INVARIANT** features:
```
f(R·img) = f(img)  ← Same feature regardless of rotation!
```

**Consequence:** When matching `(img_src, rotate(img_src, 45°))`, both produce identical features → can't detect the 45° rotation.

### Evidence (2026-01-30)

```
Scale Detection Test:
GT Scale     Pred Scale   Error
0.70         0.1000       ✗ (constant!)
1.00         0.1000       ✗
1.30         0.1000       ✗

Rotation Detection Test:
GT Angle     Pred Angle   Error
45°          118°         73°  ✗
90°          69°          21°  ✗
```

The model predicted **constant scale = 0.1** and **random rotations** because invariant features contain no geometric information.

### Root Cause

```
ESCNN + GroupPooling produces:
    f(img) = f(R·img) = f(S·img)  (approximately)

When matching (img, T·img):
    features_src ≈ features_tgt  ← IDENTICAL!

Result:
    correlation_matrix ≈ uniform
    matched_positions ≈ source_positions (identity)
    estimated_transform ≈ identity (scale hits clamp at 0.1)
```

---

## Solution: Cyclic Correlation + Procrustes Architecture

### Key Insight

**Use EQUIVARIANT (not invariant) features:**
```
f(R·img) = ρ(R) · f(img)  ← Features TRANSFORM predictably
```

For ESCNN with regular representation:
- Rotation by `k × (360°/N)` cyclically shifts feature channels by `k` positions
- Find the shift via cross-correlation → recover rotation angle

### Architecture

```
Input: (img_src, img_tgt)
         │
         ▼
┌────────────────────────────┐
│  E2EquivariantEncoder      │  ← Regular repr, NO GroupPooling
│  Output: [B, C, N_rot, H, W]│  ← Features SHIFT under rotation
└────────────────────────────┘
         │
         ▼
┌────────────────────────────┐
│  CyclicRotationEstimator   │  ← FFT-based cross-correlation
│  Output: rotation θ        │  ← Peak = rotation angle
└────────────────────────────┘
         │
         ▼
┌────────────────────────────┐
│  De-rotate target image    │  ← Align images using θ
│  AlignedDenseMatcher       │  ← Find spatial correspondences
└────────────────────────────┘
         │
         ▼
┌────────────────────────────┐
│  Procrustes SVD            │  ← Extract scale s, translation t
└────────────────────────────┘
         │
         ▼
Output: H = build_homography(θ, s, t)
```

---

## Mathematical Foundation

### Theorem: Sim(2) Decomposition
Any Sim(2) transformation H can be uniquely decomposed as:
```
H = T(t) @ S(s) @ R(θ)
```

### Rotation Detection via Cyclic Cross-Correlation

For C_N equivariant features with regular representation:
```python
# Cross-correlation over rotation dimension
corr[k] = Σ_{c,r} F_src[c, r] · F_tgt[c, (r+k) mod N]

# Peak at index k → rotation = k × (360°/N)
```

**Efficient FFT implementation (O(N log N)):**
```python
F_src_fft = torch.fft.fft(feat_src, dim=-1)  # [B, C, N_rot]
F_tgt_fft = torch.fft.fft(feat_tgt, dim=-1)
corr = torch.fft.ifft(F_src_fft * F_tgt_fft.conj(), dim=-1).real
rotation_idx = soft_argmax(corr.sum(dim=1))
```

### Scale & Translation via Procrustes

After de-rotation, standard weighted Procrustes:
```python
scale = ||Q||_F / ||P||_F  # Frobenius norm ratio
translation = centroid_tgt - scale × centroid_src
```

---

## Implementation Plan

### Files to Create

| File | Purpose |
|------|---------|
| `src/models/sim2_equivariant_net.py` | Main model integrating all components |
| `src/models/cyclic_rotation_estimator.py` | FFT-based cyclic cross-correlation |
| `src/models/dense_spatial_matcher.py` | Dense correspondence matching |
| `src/models/procrustes_st.py` | Scale + Translation Procrustes |
| `src/models/differentiable_transforms.py` | Differentiable rotate/scale/translate |
| `scripts/test_true_sim2_detection.py` | Proper detection tests |

### Files to Modify

| File | Changes |
|------|---------|
| `src/models/e2_feature_extractor.py` | Add `E2EquivariantEncoder` (no GroupPooling) |
| `src/models/__init__.py` | Export new classes |
| `src/models/graph_network.py` | Add `use_sim2_equivariant` option |

---

## Component Specifications

### 1. E2EquivariantEncoder

```python
class E2EquivariantEncoder(nn.Module):
    """
    Equivariant encoder - features SHIFT under rotation.
    NO GroupPooling - we need features that transform predictably.
    """
    def __init__(self, num_rotations=16, feature_channels=32):
        self.gspace = gspaces.rot2dOnR2(N=num_rotations)
        in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])
        hidden_type = enn.FieldType(self.gspace, feature_channels * [self.gspace.regular_repr])

        self.encoder = enn.SequentialModule(
            enn.R2Conv(in_type, hidden_type, kernel_size=7, padding=3),
            enn.InnerBatchNorm(hidden_type),
            enn.ReLU(hidden_type),
            enn.R2Conv(hidden_type, hidden_type, kernel_size=5, padding=2),
            enn.InnerBatchNorm(hidden_type),
            enn.ReLU(hidden_type),
            enn.R2Conv(hidden_type, hidden_type, kernel_size=3, padding=1),
            enn.InnerBatchNorm(hidden_type),
            enn.ReLU(hidden_type),
            # NO GroupPooling!
        )

    def forward(self, image):
        # Output: [B, C, N_rot, H, W]
        x = self.encoder(enn.GeometricTensor(image, self.in_type))
        B, _, H, W = x.tensor.shape
        return x.tensor.view(B, self.feature_channels, self.num_rotations, H, W)
```

### 2. CyclicRotationEstimator

```python
class CyclicRotationEstimator(nn.Module):
    """Estimate rotation via FFT-based cyclic cross-correlation."""

    def forward(self, feat_src, feat_tgt):
        # Global spatial pooling: [B, C, N_rot, H, W] → [B, C, N_rot]
        feat_src_global = feat_src.mean(dim=[3, 4])
        feat_tgt_global = feat_tgt.mean(dim=[3, 4])

        # FFT-based cyclic cross-correlation
        F_src = torch.fft.fft(feat_src_global, dim=-1)
        F_tgt = torch.fft.fft(feat_tgt_global, dim=-1)
        correlation = torch.fft.ifft(F_src * F_tgt.conj(), dim=-1).real
        correlation = correlation.sum(dim=1)  # [B, N_rot]

        # Differentiable soft-argmax
        weights = F.softmax(correlation * 10.0, dim=-1)
        indices = torch.arange(self.num_rotations, device=correlation.device).float()
        rotation_idx = (weights * indices).sum(dim=-1)

        return rotation_idx * self.angle_resolution  # radians
```

### 3. Sim2EquivariantNet (Main Model)

```python
class Sim2EquivariantNet(nn.Module):
    def forward(self, img_src, img_tgt):
        # 1. Extract equivariant features
        feat_equiv_src = self.equivariant_encoder(img_src)
        feat_equiv_tgt = self.equivariant_encoder(img_tgt)

        # 2. Detect rotation via cyclic correlation
        rotation = self.rotation_estimator(feat_equiv_src, feat_equiv_tgt)

        # 3. De-rotate target, then match spatially
        img_tgt_aligned = rotate_image(img_tgt, -rotation)
        feat_inv_src = self.invariant_encoder(img_src)
        feat_inv_tgt = self.invariant_encoder(img_tgt_aligned)
        src_pos, matched_tgt_pos, confidence = self.spatial_matcher(feat_inv_src, feat_inv_tgt)

        # 4. Procrustes for scale and translation
        scale, translation = self.procrustes(src_pos, matched_tgt_pos, confidence)

        # 5. Build homography
        H = self.build_homography(rotation, scale, translation)

        return {'homography': H, 'rotation': rotation, 'scale': scale, 'translation': translation}
```

---

## Verification Tests

### Test 1: Rotation Detection
```python
for theta in [0, 15, 30, 45, 60, 90, 120, 180]:
    img_tgt = rotate(img_src, theta)
    pred = model(img_src, img_tgt)
    error = angular_error(pred['rotation'], theta)
    assert error < 15°
```

### Test 2: Scale Detection
```python
for s in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
    img_tgt = scale(img_src, s)
    pred = model(img_src, img_tgt)
    assert abs(pred['scale'] - s) / s < 0.2
```

### Test 3: Combined Sim(2)
```python
for R, s, t in test_transforms:
    img_tgt = apply_sim2(img_src, R, s, t)
    pred = model(img_src, img_tgt)
    assert angular_error(pred['rotation'], R) < 15°
    assert abs(pred['scale'] - s) / s < 0.2
    assert (pred['translation'] - t).norm() < 0.1
```

---

## Key Differences from Broken Version

| Aspect | Broken (Previous) | Fixed (New) |
|--------|-------------------|-------------|
| Features | Rotation-INVARIANT | Rotation-EQUIVARIANT |
| GroupPooling | Used (destroys info) | NOT used |
| Rotation detection | Procrustes (fails) | Cyclic cross-correlation |
| Feature shape | `[B, D, H, W]` | `[B, C, N_rot, H, W]` |
| Test type | Equivariance (misleading) | DETECTION (correct) |

---

## Running Commands

```bash
# Test after implementation
python scripts/test_true_sim2_detection.py --device cuda

# SLURM submission
sbatch slurm/slurm_test_sim2_equivariant.sh
```

---

## References

- **E2-CNN (Weiler & Cesa, NeurIPS 2019):** Steerable convolutions with group theory
- **Existing implementation:** `src/models/e2_rotation_detector.py` shows correct cyclic correlation
- **Procrustes (Schönemann, 1966):** Orthogonal Procrustes problem

---

## Phase 9: True Sim(2) Equivariance via SESN

**Date:** 2026-01-30, 9:05 PM AEDT
**Status:** APPROVED - Implementation Starting
**Approach:** SESN (Scale-Equivariant Steerable Networks) + ESCNN Composition

### Motivation

The current architecture is **SE(2)-equivariant** (rotation + translation) but NOT Sim(2)-equivariant. Scale is handled via Procrustes on correspondences (classical geometric method, not neural equivariance).

For true Sim(2) equivariance, we need scale-equivariant neural network layers.

### Solution: SESN-Style Scale Equivariance

**SESN (Sosnovik et al., ICLR 2020)** achieves scale equivariance using Hermite polynomial basis:

```
ψ_{n,m,σ}(x,y) = H_n(x/σ) · H_m(y/σ) · exp(-(x²+y²)/(2σ²))
```

- **Hermite polynomials** H_n(x) form an orthogonal basis
- **Gaussian envelope** provides localization
- **Scale parameter σ** determines the scale level
- **Efficiency:** 7×7 kernel with order-4 Hermite needs only 10 params (vs 49)

### Architecture Overview

**Compose scale + rotation equivariance:**

```
Input Image [B, 1, H, W]
        ↓
ScaleEquivariantEncoder (Hermite basis)
        ↓
[B, C, N_scales, H, W]  ← Features shift in scale dim under scaling
        ↓
ESCNN R2Conv (per scale level)
        ↓
[B, C, N_scales, N_rot, H, W]  ← Features shift in both dims
        ↓
CyclicScaleEstimator (FFT correlation over scale dim)
        ↓
Scale estimate
        ↓
CyclicRotationEstimator (FFT correlation over rotation dim)
        ↓
Rotation estimate
        ↓
Translation via spatial matching
        ↓
H = T(t) @ S(s) @ R(θ)
```

### Implementation Plan

#### New Files to Create

| File | Purpose |
|------|---------|
| `src/models/scale_equivariant_basis.py` | Hermite polynomial basis with Gaussian envelope |
| `src/models/scale_equivariant_conv.py` | Scale-equivariant 2D convolution layer |
| `src/models/scale_equivariant_encoder.py` | Scale-equivariant feature encoder |
| `src/models/sim2_encoder.py` | Combined rotation+scale equivariant encoder |
| `src/models/cyclic_scale_estimator.py` | Scale detection via FFT correlation |
| `scripts/test_scale_equivariance.py` | Verification tests |

#### Files to Modify

| File | Changes |
|------|---------|
| `src/models/sim2_equivariant_net.py` | Add `TrueSim2EquivariantNet` class |
| `src/models/__init__.py` | Export new modules |

### Mathematical Foundation

**Scale Equivariance Property:**
For scale-equivariant features with N_scales discrete levels:
```
f(Scale(x, s_k))[:, :, i, :, :] = f(x)[:, :, (i-k) mod N_scales, :, :]
```

This is analogous to ESCNN's rotation equivariance where features cyclically shift.

**Scale Detection via Cyclic Correlation:**
```python
# Pool over spatial dimensions
feat_global = feat.mean(dim=[-2, -1])  # [B, C, N_scales]

# FFT cross-correlation
F_src = torch.fft.fft(feat_src_global, dim=-1)
F_tgt = torch.fft.fft(feat_tgt_global, dim=-1)
correlation = torch.fft.ifft(F_src * F_tgt.conj(), dim=-1).real

# Peak → scale index → scale factor
scale_idx = soft_argmax(correlation)
scale = scale_factors[scale_idx]
```

### Verification Tests

1. **Scale equivariance property:**
   ```python
   for k, s in enumerate(scale_factors):
       feat_scaled = encoder(scale(img, s))
       feat_orig = encoder(img)
       # Check: feat_scaled[:,:,i] ≈ feat_orig[:,:,(i-k)%N]
   ```

2. **Scale detection accuracy:**
   ```python
   for s in [0.5, 0.7, 1.0, 1.3, 2.0]:
       pred = model(img, scale(img, s))
       assert abs(pred['scale'] - s) / s < 0.15
   ```

3. **Full Sim(2) detection:**
   - Test grid: 13 angles × 5 scales × translations
   - Target: rotation < 15°, scale < 15%, translation < 0.1

### References

- **SESN:** Sosnovik et al., "Scale-Equivariant Steerable Networks", ICLR 2020
- **GitHub:** https://github.com/ISosnovik/sesn
- **ESCNN:** Weiler & Cesa, "General E(2)-Equivariant Steerable CNNs", NeurIPS 2019

---

## Phase 9b: Log-Polar Sim(2) Equivariance (RECOMMENDED)

**Date:** 2026-01-30, 11:00 PM AEDT
**Status:** ✅ SUCCESS - 6/6 tests PASSED
**Approach:** Log-Polar Transform + Phase Correlation
**Job IDs:** 21243894 (initial), 21243907 (with scale fix)

### Motivation

The SESN approach (Phase 9) requires careful implementation of Hermite polynomial basis kernels. Initial tests showed that while the cyclic estimator works perfectly on synthetic feature shifts, the actual ScaleEquivariantEncoder may not produce properly shifted features.

**Alternative: Log-Polar Transform** is a well-established technique (Fourier-Mellin transform) that:
- Converts scale and rotation to translations
- Enables standard CNNs to become scale-rotation equivariant
- Is simpler to implement and more robust

### Key Insight

In log-polar coordinates (log(r), θ):
```
- Scale by s: r' = s·r → log(r') = log(s) + log(r) → SHIFT in log(r)
- Rotate by φ: θ' = θ + φ → SHIFT in θ
```

**Therefore:** Scale and rotation become 2D translations in log-polar space!

A standard translation-equivariant CNN in log-polar space becomes scale-rotation equivariant in Cartesian space.

### Architecture

```
Input Images [B, 1, H, W]
        ↓
LogPolarTransform
        ↓
Log-polar images [B, 1, H_θ, W_logr]
        ↓
CNN Feature Extractor (translation-equivariant → scale-rot equivariant!)
        ↓
Features [B, C, H_θ, W_logr]
        ↓
FFT Phase Correlation
        ↓
Peak shift (Δθ, Δlog_r) = (rotation, log(scale))
        ↓
scale = exp(Δlog_r), rotation = Δθ
        ↓
De-rotate and de-scale target
        ↓
Spatial correlation for translation
        ↓
H = T(t) @ S(s) @ R(θ)
```

### Implementation

#### New Files Created

| File | Purpose |
|------|---------|
| `src/models/log_polar_transform.py` | LogPolarTransform, InverseLogPolarTransform, LogPolarPhaseCorrelation |
| `src/models/log_polar_sim2_net.py` | LogPolarSim2Net - full Sim(2) estimation pipeline |
| `scripts/test_log_polar_sim2.py` | Verification tests |
| `slurm/slurm_test_log_polar.sh` | SLURM test script |

#### Key Classes

**LogPolarTransform:**
```python
class LogPolarTransform(nn.Module):
    """
    Differentiable log-polar transform.
    Input: [B, C, H, W] Cartesian image
    Output: [B, C, H_angles, W_radii] log-polar image

    Parameters:
        output_size: (n_angles, n_radii)
        r_min: Minimum radius (avoid singularity)
        r_max: Maximum radius
    """
```

**LogPolarPhaseCorrelation:**
```python
class LogPolarPhaseCorrelation(nn.Module):
    """
    Phase correlation in log-polar space.
    Finds 2D shift = (rotation, log(scale)).

    Uses FFT for efficiency and soft-argmax for differentiability.
    """
```

**LogPolarSim2Net:**
```python
class LogPolarSim2Net(nn.Module):
    """
    Full Sim(2) estimation using log-polar transform.

    Pipeline:
    1. Log-polar correlation → (scale, rotation)
    2. De-rotate and de-scale target
    3. Spatial correlation → translation
    4. Build H = T @ S @ R
    """
```

### Advantages over SESN

| Aspect | SESN (Hermite) | Log-Polar |
|--------|----------------|-----------|
| Complexity | High (custom kernels) | Low (standard ops) |
| Robustness | Depends on kernel design | Well-established |
| Equivariance | Discrete scales | Continuous via log |
| Artifacts | None | Boundary effects |
| Center dependence | None | Requires center estimate |

### Verification Tests

1. **Log-polar property:** Scale → horizontal shift, rotation → vertical shift
2. **Phase correlation accuracy:** Detect known (scale, rotation) pairs
3. **Full pipeline:** Estimate Sim(2) transformations end-to-end
4. **Multiple patterns:** Test robustness with different image types

### SLURM Job

```bash
sbatch slurm/slurm_test_log_polar.sh
```

### Test Results (6/6 PASSED)

| Test Case | GT | Detected | Status |
|-----------|-----|----------|--------|
| Identity | (0°, 1.0) | (0.0°, 1.000) | ✅ PASS |
| Rotation 30° | (30°, 1.0) | (28.0°, 1.000) | ✅ PASS |
| Scale 1.2x | (0°, 1.2) | (0.0°, 1.198) | ✅ PASS |
| Rot+Scale | (45°, 0.8) | (46.0°, 0.894) | ✅ PASS |
| Rot+Scale | (-30°, 1.3) | (-30.0°, 1.283) | ✅ PASS |
| Rotation 90° | (90°, 1.0) | (90.0°, 1.000) | ✅ PASS |

### Bug Fix Applied

Initial scale detection was inverted (1.2 → 0.83). Fixed by negating `log_scale` in all estimators.

### Comparison: SESN vs Log-Polar

| Approach | Scale Detection | Complexity | Status |
|----------|-----------------|------------|--------|
| **SESN** | ❌ Failed | High | Not recommended |
| **Log-Polar** | ✅ 6/6 PASSED | Low | **RECOMMENDED** |

### Verdict

**Log-Polar is the recommended approach for TRUE Sim(2) equivariance.**

The architecture now achieves full Sim(2) detection:
- **Rotation:** ✅ via log-polar phase correlation
- **Scale:** ✅ via log-polar phase correlation
- **Translation:** ✅ via spatial cross-correlation

---

## Phase 10: ECCV 2026 Evaluation Plan (Current)

**Date:** 2026-01-30
**Status:** Active
**Target:** ECCV 2026 Submission

### Executive Summary

Now that LogPolarSim2Net achieves TRUE Sim(2) equivariance, the next step is to **prove it scientifically** through controlled experiments that demonstrate:

1. **Generalization to unseen transformations** - train on restricted range, test on full range
2. **Flat error curves** - consistent accuracy regardless of rotation/scale
3. **Competitive real-world performance** - HPatches benchmark

### The Core Scientific Claim

> A Sim(2)-equivariant architecture generalizes to unseen rotations and scales **without data augmentation**, while non-equivariant baselines fail catastrophically outside their training distribution.

### Experimental Design

#### Experiment 1: Rotation Generalization

| Training | Testing | What We Prove |
|----------|---------|---------------|
| [-30°, +30°] only | [0°, 45°, 90°, 135°, 180°] | Ours handles unseen rotations; baseline fails |

**Expected Results:**
```
              Error (pixels)
           60 ┤                      HomographyNet (fails at unseen)
              │                   ╭──╮
           45 ┤                 ╭─╯  ╰─╮
              │               ╭─╯      ╰──
           30 ┤             ╭─╯
              │
           15 ┤         ╭─╯
              │       ╭─╯
            5 ┤───────────────────────────────── Ours: FLAT LINE
              │
            0 ┼───┬───┬───┬───┬───┬───┬───┬───
                0° 15° 30° 45° 60° 75° 90° 135°180°
                                Rotation
```

**Key Metric:** Coefficient of Variation (CV)
- Ours: CV < 10%
- Baseline: CV > 50%

#### Experiment 2: Scale Generalization

| Training | Testing | What We Prove |
|----------|---------|---------------|
| [0.9x, 1.1x] only | [0.5x, 0.7x, 1.0x, 1.3x, 2.0x] | Ours handles unseen scales |

#### Experiment 3: Combined Sim(2) Grid

| Grid | Purpose |
|------|---------|
| 24 rotation angles × 16 scale factors = 384 test cases | Generate 2D error heatmap |

**Expected:** Ours shows uniform low error; baseline shows increasing error away from training distribution.

#### Experiment 4: HPatches Real-World Validation

| Split | Sequences | What We Prove |
|-------|-----------|---------------|
| Viewpoint (59) | Real perspective changes | Works on genuine images |
| Illumination (57) | Lighting variations | Robust to appearance |

### Datasets

| Dataset | Size | Purpose | Status |
|---------|------|---------|--------|
| MS-COCO val2017 | 5,000 images (1.6GB) | Development & testing | ✅ Downloaded |
| MS-COCO train2017 | 118,287 images (19GB) | Full training | ⏳ Downloading |
| HPatches | 116 sequences (1.3GB) | Real-world benchmark | ✅ Downloaded |

### Training Protocol (Important!)

**All methods trained on IDENTICAL restricted data:**
```
Rotation: Uniform[-30°, +30°]
Scale: Uniform[0.9, 1.1]
Translation: Uniform[-32, +32] pixels
```

This is **deliberately restrictive** to test generalization. Baselines trained with data augmentation get more variety, but we show equivariance beats augmentation.

### Baselines

| Method | Type | Why Include |
|--------|------|-------------|
| **HomographyNet** | Supervised CNN | Standard baseline |
| **HomographyNet+Aug** | CNN + aggressive augmentation | Shows equivariance > augmentation |
| **SIFT+RANSAC** | Classical | Establishes deep learning advantage |

### Model Comparison

| Model | Parameters | Equivariance |
|-------|------------|--------------|
| HomographyNet | 34M | None (learns from data) |
| MCNet | 2.5M | None |
| **LogPolarSim2Net** | **130K** | TRUE Sim(2) (architectural) |

### Publication Figures

| Figure | Content |
|--------|---------|
| Fig 1 | Log-polar transform illustration |
| Fig 2 | Rotation sweep: error vs angle |
| Fig 3 | Scale sweep: error vs scale |
| Fig 4 | 2D error heatmap (rotation × scale) |
| Fig 5 | HPatches qualitative results |

### Publication Tables

| Table | Content |
|-------|---------|
| Tab 1 | MS-COCO benchmark comparison (ACE, AUC@3, AUC@5) |
| Tab 2 | Generalization metrics (CV, gap) |
| Tab 3 | HPatches results (viewpoint vs illumination) |
| Tab 4 | Ablation study |

### Current Job Status

| Job ID | Script | Purpose | Status |
|--------|--------|---------|--------|
| 21246611 | `slurm_test_procrustes.sh` | Sim(2) pillars verification | Submitted |
| 21246612 | `slurm_phase6_sim2_test.sh` | Phase 6 equivariance validation | Submitted |
| 21246613 | `slurm_phase1.sh` | Phase 1 synthetic training | **Running on H100** |
| TBD | `slurm_download_datasets.sh coco_train` | MS-COCO train2017 download | To submit |

### Success Criteria

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| Rotation CV | < 10% | Prove rotation equivariance |
| Scale CV | < 15% | Prove scale equivariance |
| Generalization gap | < 20% | Ours vs baseline gap |
| HPatches AUC@5 | > 80% | Competitive real-world |

---

*This plan serves as the roadmap for ECCV 2026 submission.*
