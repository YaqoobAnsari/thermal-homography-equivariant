# IEEE TIP Paper Plan: Sim(2)-Equivariant Thermal Similarity Estimation

## Paper in One Sentence

> *"A learned Fourier-Mellin Transform with provable Sim(2)-equivariance for thermal image similarity estimation — achieving flat error curves across all rotations and scales without augmentation, where classical and learned baselines fail."*

---

## The 3 Critical Problems (ranked by severity)

| # | Problem | Why it matters | Difficulty |
|---|---------|---------------|------------|
| **1** | Training config points to dead `e2_gnn` model, not `LogPolarSim2Net` | We have **never properly trained** our primary model | Easy |
| **2** | Training degrades performance vs untrained baseline | The learned encoder destroys correlation peak structure | Hard (core research) |
| **3** | Everything says "homography" but we do similarity (4DOF) | Framing mismatch — reviewers will reject on this alone | Medium (tedious) |

---

## Step-by-Step Plan

### STEP 1: Terminology Reframing (Codebase-wide)
**Goal:** Every user-facing string says "similarity" not "homography"
**Status:** TODO

What changes:
- `README.md` — reframe title, description, architecture diagram
- `EVALUATION.md` — update all references
- `pyproject.toml` — project description
- `src/training/sim2_losses.py` — class names, docstrings
- `src/training/losses.py` — `HomographyLoss` → keep internal name but update docstrings
- `src/training/metrics.py` — update metric descriptions
- `src/models/log_polar_sim2_net.py` — output dict key `'homography'` → `'similarity_matrix'`
- All evaluation scripts — update print statements and result labels
- All test files — update assertions for renamed output keys

What does NOT change:
- Internal variable names (H matrix is still H)
- `src/utils/homography.py` — math is the same, keep filename
- The actual 3x3 matrix format — a similarity IS a restricted homography

---

### STEP 2: Fix Training Config
**Goal:** A training config that actually trains `LogPolarSim2Net`
**Status:** TODO

Create `configs/sim2_similarity.yaml`:
```yaml
model_type: "log_polar_sim2"
model:
  lp_size: [180, 64]
  r_min: 0.05
  r_max: 0.9
  feature_channels: 32
  use_fft_magnitude: true
  use_disambiguation: true
  sr_temperature: 50.0
  t_temperature: 20.0
data:
  synthetic: true
  patterns: ["asymmetric", "arrow", "L_shape", "thermal_hotspot", "multi_hotspot", "natural", "blobs"]
  n_train: 5000
  n_val: 1000
  image_size: 256
  rotation_range: [-180, 180]  # Full circle - equivariance should handle it
  scale_range: [0.7, 1.4]      # Wider range
  translation_range: [-40, 40]
loss:
  rotation_weight: 1.0
  scale_weight: 1.0
  translation_weight: 0.5
  correlation_weight: 2.0
  peak_sharpness_weight: 0.2
training:
  batch_size: 16
  lr: 3e-4
  max_epochs: 200
  optimizer: adamw
  scheduler: cosine
  precision: 32
```

Key changes from old config:
- Uses `LogPolarSim2Net` not `E2EquivariantGNN`
- Full rotation range [-180, 180] (test equivariance properly)
- Wider scale range [0.7, 1.4]
- Multiple asymmetric patterns (not just checkerboard)
- Correlation supervision enabled (weight 2.0)

---

### STEP 3: Fix Training Degradation (Core Research Problem)
**Goal:** Learned features must improve over raw FFT magnitude
**Status:** TODO (next round of edits)

The problem: the CNN encoder in `LearnedLogPolarEncoder` learns features that blur/destroy the sharp correlation peaks that the classical FMT naturally produces.

**Approach A — Residual encoder with identity initialization:**
```python
class LearnedLogPolarEncoder(nn.Module):
    def __init__(self, ...):
        self.conv_layers = nn.Sequential(...)
        self.residual_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        identity = x
        learned = self.conv_layers(x)
        alpha = torch.sigmoid(self.residual_gate)
        return (1 - alpha) * identity + alpha * learned
```

Guarantees:
- At init, model = classical FMT (which works)
- Training can only improve by learning useful refinements
- If CNN hurts, gradient pushes gate back toward 0

**Approach B — Freeze encoder, train only soft-argmax temperatures:**
- Keep encoder frozen (pass-through)
- Only learn `sr_temperature` and `t_temperature`

**Approach C — Contrastive feature learning:**
- Train encoder with contrastive loss
- Auxiliary loss: correlation peak PSR must increase

**Recommendation:** Start with A, validate with B as ablation.

---

### STEP 4: Fix Translation Gradient Flow
**Goal:** End-to-end differentiable pipeline
**Status:** TODO (next round of edits)

Current issue: translation estimation uses `grid_sample` for de-rotation/de-scaling before computing translation correlation. The `grid_sample` + FFT chain breaks gradient flow.

**Fix:** Use straight-through estimator (STE) pattern or reformulate to avoid `grid_sample`.

---

### STEP 5: Build Proper Evaluation Framework
**Goal:** Metrics that measure what we claim (Sim(2) equivariance)
**Status:** TODO

**Primary metrics (per-component):**
| Metric | Formula | What it measures |
|--------|---------|-----------------|
| Rotation MAE | `\|θ_pred - θ_gt\|` (degrees) | Rotation accuracy |
| Scale MAE | `\|log(s_pred/s_gt)\|` | Scale accuracy (log-space) |
| Translation MAE | `\|\|t_pred - t_gt\|\|₂` (pixels) | Translation accuracy |
| Similarity RMSE | corner error from 4-param similarity | Overall accuracy |

**Equivariance metrics (the key selling point):**
| Metric | Formula | What it measures |
|--------|---------|-----------------|
| Error variance across angles | `std(err(θ))` for θ ∈ [0°, 360°) | **Flat error curve** |
| Error variance across scales | `std(err(s))` for s ∈ [0.5, 2.0] | Scale equivariance |
| Generalization gap | `err_unseen / err_seen` | OOD performance |

**Threshold:** Error variance < 10% of mean → TRUE equivariance

**Baselines:**
1. Classical FMT (untrained — our lower bound)
2. SIFT + RANSAC
3. ORB + RANSAC
4. HomographyNet (DeTone et al. 2016)
5. ResNet regression baseline
6. Same architecture + augmentation (ablation)

---

### STEP 6: Thermal-Specific Experiments
**Goal:** Demonstrate why thermal needs this approach
**Status:** TODO

- **6a:** Keypoint failure on thermal (SIFT/ORB <40% success)
- **6b:** Colormap invariance (iron, rainbow, grayscale, hot)
- **6c:** Low-texture robustness (progressive degradation)
- **6d:** Real thermal data (STHN subset or custom)

---

### STEP 7: Equivariance Demonstration (Core Contribution)
**Goal:** The plot that sells the paper
**Status:** TODO

The key figure: Error vs. rotation angle
```
Error (deg)
  10 |  * * * *           ← Baseline (trained [-30,30])
     |    *   * *
   5 |          * * *
     |              * * * * * * * *  ← Baseline + augmentation
   1 |_________________________     ← Ours (FLAT LINE)
     0   30  60  90  120 150 180  Rotation angle (deg)
```

---

### STEP 8: Ablation Studies
**Goal:** Justify every design choice
**Status:** TODO

| Ablation | What it tests |
|----------|--------------|
| With vs without FFT magnitude | Translation invariance contribution |
| With vs without log-polar | Scale equivariance contribution |
| With vs without learned encoder | Does learning help over classical? |
| With vs without correlation supervision | Does direct supervision prevent degradation? |
| With vs without 180° disambiguation | Ambiguity resolution |
| Residual gate value analysis | How much the CNN contributes |
| Temperature sensitivity | Soft-argmax sharpness |

---

### STEP 9: Paper Writing (IEEE TIP format, 13 pages)
**Goal:** Submit-ready manuscript
**Status:** TODO

1. **Introduction** (1.5p) — Thermal alignment is hard, keypoints fail, equivariance is the answer
2. **Related Work** (1.5p) — Homography estimation, thermal registration, equivariant networks, FMT
3. **Method** (3p) — Sim(2) group theory, log-polar FMT, learned encoder, full pipeline
4. **Theoretical Analysis** (1p) — Prove equivariance properties formally
5. **Experiments** (4p) — Synthetic validation, equivariance plots, thermal experiments, ablations, baselines
6. **Discussion** (1p) — Limitations, failure cases, applications
7. **Conclusion** (0.5p)
8. **References** (0.5p)

---

### STEP 10: Final Cleanup and Submission
**Goal:** Submit via ScholarOne
**Status:** TODO

---

## Execution Dependencies

```
STEP 1 (Terminology)     ─── can start immediately
STEP 2 (Config)          ─── can start immediately
STEP 3 (Training fix)    ─── depends on Step 2
STEP 4 (Gradient flow)   ─── depends on Step 2
STEP 5 (Eval framework)  ─── can start immediately
STEP 6 (Thermal exps)    ─── depends on Steps 3, 5
STEP 7 (Equivariance)    ─── depends on Steps 3, 5
STEP 8 (Ablations)       ─── depends on Steps 3, 5
STEP 9 (Paper writing)   ─── depends on Steps 6, 7, 8
STEP 10 (Submission)     ─── depends on Step 9
```

---

## Target Journal

- **IEEE Transactions on Image Processing (TIP)**
- IF 13.7, CiteScore 22.5
- 13-page limit (double-column IEEE format)
- EDICS: TEC-ISR, ARS-IVA, SMR-REP
- Submit: mc.manuscriptcentral.com/tip-ieee
