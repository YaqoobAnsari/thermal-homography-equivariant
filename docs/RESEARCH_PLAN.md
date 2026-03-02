# Research Plan: ECCV 2026 Submission

## Thermal Homography with E(2)-Equivariant Graph Networks

**Target:** ECCV 2026 (European Conference on Computer Vision)
**Submission Deadline:** ~March 2026
**Status:** Phase 0 - Critical Bug Fixes Required

---

## What Are We Doing? (Plain English)

### The Problem
We have two thermal images of the same scene taken from slightly different positions/angles. We want to find the **homography** - the mathematical transformation (8 numbers) that maps points from one image to the other. This is fundamental for:
- Stitching panoramas
- Augmented reality
- Robot navigation
- Medical image alignment

### Why Is This Hard for Thermal Images?
Traditional methods (SIFT, ORB, SuperPoint) find distinctive "keypoints" and match them between images. But thermal images are:
- **Low texture** - smooth temperature gradients, few edges
- **Colormap-dependent** - same scene looks different in "iron", "rainbow", "grayscale"
- **Noisy** - thermal sensors have lower resolution than RGB

Result: Keypoint methods fail catastrophically on thermal images.

### Our Solution: E(2)-Equivariant Graph Neural Networks

Instead of finding keypoints, we:
1. Place a **regular grid** of points on both images
2. Build a **graph** connecting nearby points
3. Extract **gradient features** (edges) that are colormap-invariant
4. Use **E(2)-equivariant layers** so the network "understands" rotations/translations

**The Key Insight:** If image A is rotated by 30°, then image B (the transformed version) should have features that are also rotated by 30°. This is called **equivariance** - the output transforms in a predictable way with the input.

### Why Equivariance Matters
Without equivariance, the network must **memorize** every possible rotation during training. With equivariance, it **generalizes** automatically:
- Train on 0°-30° rotations → works on 0°-360°
- Fewer parameters needed
- More robust to unseen transformations

---

## Why Synthetic Data First? (Methodological Justification)

### The Scientific Method Applied to Deep Learning

**Hypothesis:** E(2)-equivariant features improve homography estimation for rotated images.

To test this hypothesis rigorously, we need to **isolate variables**:

| Factor | Synthetic Data | Real Data |
|--------|---------------|-----------|
| Ground truth homography | **Exact** (we generate it) | Approximate (from RANSAC/manual) |
| Image noise | Controlled | Uncontrolled |
| Scene complexity | Minimal | High |
| Lighting variations | None | Present |
| Calibration errors | None | Present |

### Why Not Skip to Real Data?

**Scenario:** We train on real thermal images and get poor results. Why?

Possible causes:
1. Our equivariance approach is fundamentally flawed
2. The implementation has bugs
3. The ground truth labels are noisy
4. The images are too challenging
5. Hyperparameters are wrong
6. All of the above

With real data, **we can't distinguish these causes**. We'd waste months debugging the wrong thing.

### The Synthetic-First Strategy

```
Phase 0: Fix known bugs
    ↓
Phase 1: Synthetic data (simple patterns)
    ↓ Does equivariance work in principle?
    ├─ YES → Phase 2: Real data
    └─ NO → Debug model/theory (saved months of wasted effort)
```

**Analogy:** Before testing a new drug on humans, you test in controlled lab conditions. If it fails in the lab, it will definitely fail in humans. But if it works in the lab and fails in humans, you know the issue is with biological complexity, not the drug's mechanism.

### What Synthetic Data Tells Us

1. **Equivariance Verification:** Feed in image rotated by θ, check if features rotate by θ
2. **Numerical Stability:** No NaN, gradient explosions
3. **Convergence:** Loss decreases, model learns
4. **Rotation Generalization:** Train on ±30°, test on ±180° - error should be flat

### ECCV Reviewer Perspective

Reviewers will ask:
> "How do you know your model is actually equivariant?"

Answer: "We verified numerically on synthetic data where ground truth is exact. Figure X shows error vs rotation angle - the flat line proves equivariance."

This is **much stronger** than:
> "We trained on real data and it works."

Which invites:
> "Maybe you just overfit to the training distribution?"

---

## Critical Issues Found (Must Fix First)

### 🚨 CRITICAL: Homography Not Scaled on Resize
**File:** `src/data/transforms.py:178`
**Issue:** When images resize from 640×480 to 256×256, the ground-truth homography is NOT adjusted. Every training label is wrong.
**Impact:** Model learns incorrect transformations.
**Fix:** Scale homography: `H_scaled = S @ H @ S_inv` where `S = diag(scale_x, scale_y, 1)`

### 🚨 CRITICAL: Train/Test Distribution Mismatch
**File:** `src/data/synthetic.py:45-48`
**Issue:** Synthetic data generates rotations in [-30°, +30°] but we claim equivariance over [-180°, +180°].
**Impact:** We cannot claim rotation equivariance if we never train/test on large rotations.
**Fix:** Generate full rotation range, or clearly document limitation.

### 🚨 HIGH: Distance Normalization Breaks Equivariance
**File:** `src/models/graph_network.py:89`
**Issue:** `edge_attr = edge_attr / edge_attr.max()` normalizes by batch maximum, which changes with rotation.
**Impact:** Features are not truly equivariant.
**Fix:** Use fixed normalization: `edge_attr = edge_attr / image_diagonal`

### ⚠️ MEDIUM: Terminology Issue
**Issue:** Code uses `gpool` (group pooling) which produces **invariant** features, not equivariant.
**Impact:** Paper claims will be technically incorrect.
**Decision Needed:** Are we claiming equivariance or invariance? Both are valid, but different.

---

## Revised Research Phases

### Phase 0: Foundation Fixes (Week 0-1) 🔧 **CURRENT**

**Goal:** Fix all critical bugs so experiments produce valid results.

| Task | Priority | File |
|------|----------|------|
| Fix homography scaling on resize | CRITICAL | `src/data/transforms.py` |
| Fix distance normalization | CRITICAL | `src/models/graph_network.py` |
| Expand synthetic rotation range | CRITICAL | `src/data/synthetic.py` |
| Fix weight initialization (gain=0.01→1.0) | HIGH | `src/models/e2_layers.py` |
| Add `.detach()` in visualization | MEDIUM | `src/training/callbacks.py` |
| Create unit tests for transforms | HIGH | `tests/test_transforms.py` |
| Create equivariance verification script | HIGH | `scripts/verify_equivariance.py` |

**Verification:**
```bash
# All tests must pass before proceeding
pytest tests/ -v
python scripts/verify_equivariance.py --threshold 1e-5
```

**Exit Criteria:**
- [ ] All unit tests pass
- [ ] Equivariance error < 1e-5 on identity transformation
- [ ] Homography scaling verified correct

---

### Phase 1: Synthetic Validation (Week 2-3) 🧪

**Goal:** Prove the method works on controlled, simple data.

**Synthetic Data Types:**
| Pattern | Purpose | Expected Difficulty |
|---------|---------|---------------------|
| Checkerboard | Clear gradients, easy alignment | Easy |
| Gaussian blobs | Thermal-like appearance | Medium |
| Random textures | Stress test | Hard |

**Key Experiment: Rotation Generalization**
```
Train: rotations in [-180°, +180°] (uniform)
Test:  rotations at [0°, 15°, 30°, ..., 180°] (specific angles)
Plot:  Error vs Test Rotation Angle
```

**Expected Result (if equivariance works):**
```
Error (px)
   |
 5 |  -------- flat line --------
   |
 0 +---------------------------→ Rotation (°)
   0°   45°   90°   135°  180°
```

**Expected Result (if equivariance broken):**
```
Error (px)
   |                    *
20 |              *          *
   |        *                    *
 5 | *                                *
   +----------------------------------→ Rotation (°)
   0°   45°   90°   135°  180°   (trained on)
```

**Success Criteria:**
- [ ] Error < 5px across ALL rotation angles
- [ ] Error standard deviation < 1px across rotations
- [ ] No NaN during training
- [ ] Converges within 50 epochs

---

### Phase 2: Architecture Search (Week 4-6) 🔍

**Goal:** Find optimal hyperparameters systematically.

**Search Space:**
| Parameter | Values | Why |
|-----------|--------|-----|
| `feature_dim` | [16, 32, 64, 128] | Capacity vs compute |
| `grid_size` | [16, 32, 48] | Resolution vs memory |
| `gnn_layers` | [2, 4, 6, 8] | Depth vs gradient flow |
| `learning_rate` | [1e-5, 5e-5, 1e-4, 5e-4] | Convergence |
| `lrft_rank` | [4, 8, 16, 32] | Compression factor |

**Method:** Wandb Bayesian optimization (50-100 trials)

**SLURM:**
```bash
sbatch --array=1-50%8 scripts/slurm_sweep.sh
```

**Success Criteria:**
- [ ] Best config beats default by ≥20%
- [ ] Top-3 configs identified
- [ ] Sensitivity analysis documented

---

### Phase 3: Real Data Training (Week 7-10) 📊

**Goal:** Train on real thermal imagery and establish benchmark results.

**Datasets:**
| Dataset | Pairs | Modality | Use |
|---------|-------|----------|-----|
| KAIST | 10K | Thermal/RGB | Primary |
| FLIR ADAS | 10K | Thermal driving | Secondary |
| Synthetic | 50K | Generated | Pre-training |

**Training Strategy:**
1. Pre-train on synthetic (builds rotation robustness)
2. Fine-tune on real thermal (adapts to real noise/textures)
3. Evaluate on held-out test set

**Success Criteria:**
- [ ] Corner error < 5px on test set
- [ ] Beat HomographyNet baseline by ≥20%
- [ ] Beat RANSAC+SIFT baseline
- [ ] Stable training (no NaN, smooth loss)

---

### Phase 4: Ablations & Analysis (Week 11-14) 📈

**Goal:** Prove each component is necessary (reviewers will ask).

**Ablation Studies:**
| Remove This | Expected Impact | Why |
|-------------|-----------------|-----|
| E(2) equivariance | +50% error on rotations | Core contribution |
| LRFT module | Similar accuracy, 2× slower | Efficiency claim |
| Graph structure | +30% error | Spatial reasoning |
| Gradient features | Colormap-dependent | Robustness claim |
| Pre-training | +20% error | Transfer learning |

**Publication Figures:**
1. **Fig 1:** Teaser (problem + our solution)
2. **Fig 2:** Architecture diagram
3. **Fig 3:** Rotation equivariance plot (the flat line)
4. **Fig 4:** Baseline comparison table
5. **Fig 5:** Qualitative results (success + failure cases)
6. **Fig 6:** Ablation results

---

### Phase 5: Paper Writing (Week 15-18) ✍️

**Structure (8 pages + supplementary):**
1. Abstract (150 words)
2. Introduction (1 page) - Problem + contributions
3. Related Work (1 page)
4. Method (2 pages) - E(2) theory + architecture
5. Experiments (2 pages)
6. Ablations (1 page)
7. Conclusion (0.5 page)

**Key Claims to Support:**
1. "First E(2)-equivariant approach for homography" → Literature review
2. "Achieves rotation equivariance" → Figure 3 (flat error plot)
3. "Outperforms baselines" → Table 1 (quantitative results)
4. "Each component necessary" → Table 2 (ablations)
5. "Works on real thermal" → Figure 5 (qualitative)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Equivariance doesn't help | Low | High | Synthetic tests will catch early |
| Worse than baselines | Medium | High | Extensive hyperparameter search |
| Training instability | Medium | Medium | Gradient clipping, careful init |
| Ground truth noise | Low | Medium | Synthetic pre-training |
| Missing deadline | Low | High | Start writing early |

---

## Immediate Next Steps

```bash
# 1. Fix critical bugs (this week)
# 2. Create unit tests
# 3. Run equivariance verification
# 4. Submit Phase 1 training job
```

---

*Last updated: 2026-01-30*
*Status: Phase 0 - Critical Bug Fixes Required*
