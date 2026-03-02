# Evaluation: Sim(2)-Equivariant Thermal Homography

---

## Core Hypothesis

**Claim:** A Sim(2)-equivariant architecture (via log-polar + Fourier-Mellin transform) generalizes to unseen rotations and scales without data augmentation, while baselines fail outside their training distribution.

**Success Criteria:**
- Error variance < 10% across rotation angles (flat error curve)
- Generalization gap < 20% on unseen transformations

---

## Architecture: LogPolarSim2Net

```
Source/Target Images (256x256)
       |
   FFT Magnitude (translation-invariant)
       |
   Log-Polar Transform (180 angles x 64 radii)
       |
   CNN Encoder (learned features on LP magnitude)
       |
   Phase Correlation -> correlation_sr [B, 180, 64]
       |
   Soft-Argmax -> rotation (theta), scale (s)  <- has 180 deg ambiguity
       |
   180 deg Disambiguation (spatial correlation)
       |
   Inverse Transform (align target to source)
       |
   Translation Estimator -> (tx, ty)
       |
   Homography H = T @ R(theta) @ S(s)
```

**Key Parameters:**
- `lp_size=(180, 64)` - 2 deg angle resolution, 64 radial samples
- `r_min=0.05, r_max=0.9` - Radial range
- `sr_temperature=50.0, t_temperature=20.0` - Soft-argmax temperatures
- `use_fft_magnitude=True` - FMT mode (translation-invariant)
- `use_disambiguation=True` - Resolve 180 deg ambiguity

---

## Loss Function: Sim2HomographyLoss

All components normalized to [0, 1] for balanced gradients:

| Component | Weight | Formula | Purpose |
|-----------|--------|---------|---------|
| Rotation | 1.0 | Geodesic on SO(2) | Primary supervision |
| Scale | 1.0 | `|log(s_pred/s_gt)|` | Log-space symmetry |
| Translation | 0.5 | L2 normalized | Secondary |
| Corner | 0.1 | `log1p(err/10)` | Validation only |
| Peak SR | 0.2 | Sharpness loss | Encourage sharp peaks |
| Correlation | 2.0 | Soft-argmax to GT | Direct peak supervision |

---

## Test Patterns (by Symmetry Tier)

| Tier | Patterns | Expected Pass Rate |
|------|----------|-------------------|
| **Tier 1** (Asymmetric) | asymmetric, arrow, L_shape, T_shape, corner_marker | >80% |
| **Tier 2** (Semi) | natural, blobs, stripes, ellipse, gradient | 50-80% |
| **Tier 3** (Symmetric) | checkerboard, cross, concentric | <20% (stress test) |
| **Tier 4** (Thermal) | thermal_hotspot, multi_hotspot | >60% |

**GO/NO-GO:** Tier 1 + Tier 4 pass -> Proceed to real-world benchmarks

---

## Datasets

| Dataset | Size | Status | Phase |
|---------|------|--------|-------|
| Synthetic | Generated | Ready | 1 (validation) |
| HPatches | 3 GB | Downloaded | 2 (real-world) |
| MS-COCO (val+train) | 20 GB | Downloaded | 2 (real-world) |
| STHN | 122 GB | Deferred | 3 (cross-modal) |

---

## Validated Results

### Raw FMT Pipeline (No Training)

| Metric | Result |
|--------|--------|
| Rotation error (mean) | **0.10 deg** |
| Rotation error (max) | 1.00 deg |
| Scale error (mean) | 0.0036 |
| Scale error (max) | 0.0262 |
| Pass rate | **21/21 (100%)** |

Test cases: basic rotations, 180 deg ambiguity, with translation, with scale, full Sim(2).

### Phase 1A: Rotation-Only Training

| Metric | Before Training | After Training | Best (Epoch 9) |
|--------|-----------------|----------------|----------------|
| Mean rotation error | 1.1 deg | 0.6 deg | **0.3 deg** |
| Max rotation error | 2.0 deg | 2.0 deg | 2.0 deg |

Conclusion: Rotation/scale architecture validated. Log-polar phase correlation works correctly when translation is absent.

---

## Success Criteria

| Metric | Threshold | Measures |
|--------|-----------|----------|
| Error at 0 deg | < 10 px | Basic correctness |
| Error std across angles | < 3 px | **Equivariance** |
| Mean corner error | < 10 px | Accuracy |
| PSR (Peak-Sidelobe Ratio) | > 15 | Peak sharpness |

---

## Key Learnings

1. **Corner-only loss destroys equivariance** - Model compensates rotation errors via translation
2. **Correlation supervision is essential** - Without it, CNN learns features that diffuse peaks
3. **Sign conventions must be consistent** - Between rotation detection and inverse transform
4. **Symmetric patterns are fundamentally ambiguous** - Checkerboards have 90 deg rotational symmetry
5. **FFT magnitude provides translation invariance** - Critical for real-world applicability

---

## References

1. DeTone et al., "Deep Image Homography Estimation", 2016
2. HPatches Benchmark, CVPR 2017
3. MCNet, CVPR 2024
4. HomoFM, 2025
5. STHN, RA-L 2024
