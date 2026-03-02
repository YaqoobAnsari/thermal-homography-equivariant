# Literature Review: Geometric Equivariance for Homography Estimation

This document compiles research findings on equivariant neural networks, homography estimation methods, and geometric deep learning relevant to our ECCV 2026 submission.

**Target:** Sim(2)-Equivariant Thermal Homography Estimation
**Invariances:** Translation + Rotation + Scale

---

## Table of Contents

1. [E(2)/SE(2) Equivariant Networks](#1-e2se2-equivariant-networks)
2. [SO(3) and 3D Equivariant Methods](#2-so3-and-3d-equivariant-methods)
3. [Scale-Equivariant Networks](#3-scale-equivariant-networks)
4. [Sim(2)/Sim(3) Equivariant Networks](#4-sim2sim3-equivariant-networks)
5. [Deep Homography Estimation](#5-deep-homography-estimation)
6. [Lie Algebra Approaches](#6-lie-algebra-approaches)
7. [Loss Functions for Geometric Tasks](#7-loss-functions-for-geometric-tasks)
8. [Key Libraries and Implementations](#8-key-libraries-and-implementations)

---

## 1. E(2)/SE(2) Equivariant Networks

### 1.1 e2cnn / escnn: Steerable CNNs (Weiler & Cesa, NeurIPS 2019)

**Paper:** [General E(2)-Equivariant Steerable CNNs](https://arxiv.org/abs/1911.08251)
**Code:** [github.com/QUVA-Lab/escnn](https://github.com/QUVA-Lab/escnn)

**Key Concepts:**
- **Steerable filters**: Kernel space constrained by representation theory
- **Field types**: Features organized with explicit transformation laws
- **Group pooling**: Convert equivariant → invariant features

**How it achieves equivariance:**
```python
# Under rotation g, feature maps transform as:
f'(x) = ρ(g) · f(g^{-1}x)
# where ρ is the representation

# Convolution kernel K must satisfy:
K(g·r) = ρ_out(g) · K(r) · ρ_in(g)^{-1}
```

**Representation types:**
- **Trivial**: Scalars, invariant under rotation
- **Regular**: Group convolution style, |G| channels per field
- **Irreducible (irreps)**: Minimal building blocks (for SO(2): 2D rotation matrices)

**Computational cost:** 1.5-3× overhead vs standard CNNs

**Relevance:** Foundation for rotation equivariance in 2D images.

---

### 1.2 EGNN: E(n)-Equivariant Graph Neural Networks (Satorras et al., ICML 2021)

**Paper:** [E(n) Equivariant Graph Neural Networks](https://arxiv.org/abs/2102.09844)
**Code:** [github.com/vgsatorras/egnn](https://github.com/vgsatorras/egnn)

**Key Innovation:** Separate scalar features `h` from vector coordinates `x`

**Message passing (CRITICAL - this is what we need):**
```python
# EGNN Update:
m_ij = φ_e(h_i, h_j, ||x_i - x_j||², a_ij)  # Invariant message (distance only!)
x_i' = x_i + Σ_j (x_i - x_j) · φ_x(m_ij)     # Equivariant position update
h_i' = φ_h(h_i, Σ_j m_ij)                     # Invariant feature update
```

**Why it works:**
- Messages use ONLY `||x_i - x_j||²` (distance squared) - rotation invariant
- No angles! Angles change with rotation and break equivariance
- Position updates use equivariant vector operations

**Performance:** 32% error reduction over non-equivariant methods in molecular dynamics

**Relevance:** Direct fix for our current E2MessagePassing bug.

---

### 1.3 SE2-LoFTR (CVPRW 2022)

**Paper:** [SE(2)-Equivariant Feature Matching](https://arxiv.org/abs/2204.10144)
**Code:** [github.com/georg-bn/se2-loftr](https://github.com/georg-bn/se2-loftr)

**Key Contribution:** Replaced LoFTR's backbone with escnn steerable convolutions

**Results:** Rotation-robust feature matching without augmentation

**Relevance:** Proof that escnn works for image matching tasks.

---

### 1.4 Vector Neurons (Deng et al., ICCV 2021)

**Paper:** [Vector Neurons: A General Framework for SO(3)-Equivariant Networks](https://arxiv.org/abs/2104.12229)
**Code:** [github.com/FlyingGiraffe/vnn](https://github.com/FlyingGiraffe/vnn)

**Key Idea:** Extend neurons from 1D scalars to 3D vectors (adaptable to 2D)

```python
# Vector Neuron ReLU (2D version):
def vn_relu(V):  # V: [B, C, 2]
    norm = torch.norm(V, dim=-1, keepdim=True)
    k = V / (norm + eps)
    return torch.relu(norm) * k
```

**Properties:**
- Full SO(3)/SO(2) equivariance without group convolutions
- ~2/9× fewer parameters than scalar equivalents
- 89.9% accuracy on rotated ModelNet40

**Relevance:** Alternative to steerable convolutions for vector-valued features.

---

## 2. SO(3) and 3D Equivariant Methods

### 2.1 Equi-GSPR (Kang et al., ECCV 2024)

**Paper:** [SE(3)-Equivariant Graph Networks for Point Cloud Registration](https://arxiv.org/abs/2410.05729)

**Key Contributions:**
- Spherical SE(3) message passing
- Low-Rank Feature Transformation (LRFT) for outlier rejection
- Data-efficient sparse point cloud registration

**Relevance:** Our LRFT module is inspired by this work; however, SE(3) is for 3D, not 2D projective transforms.

---

### 2.2 GATr: Geometric Algebra Transformer (NeurIPS 2023)

**Paper:** [Geometric Algebra Transformer](https://arxiv.org/abs/2305.18415)
**Code:** [github.com/Qualcomm-AI-research/geometric-algebra-transformer](https://github.com/Qualcomm-AI-research/geometric-algebra-transformer)

**Key Innovation:** Uses projective geometric (Clifford) algebra for representations

- 16-dimensional vectors represent points, lines, planes
- E(3) equivariance by construction
- Extended to Lorentz group (NeurIPS 2024)

**Relevance:** Geometric algebra provides principled homography representation (future work).

---

### 2.3 Equiformer / EquiformerV2 (ICLR 2023, 2024)

**Papers:**
- [Equiformer](https://arxiv.org/abs/2206.11990)
- [EquiformerV2](https://arxiv.org/abs/2306.12059)

**Key Features:**
- Combines transformers with spherical harmonics
- Uses irreducible representations up to degree L
- MLP attention replaces dot-product attention

**Why NOT ideal for our 2D problem:**
- Designed for 3D molecular/atomistic graphs
- Spherical harmonics overhead unnecessary for E(2)
- escnn with C_N groups is more efficient for 2D

---

## 3. Scale-Equivariant Networks

### 3.1 SESN: Scale-Equivariant Steerable Networks (ICML 2020)

**Paper:** [Scale-Equivariant Steerable Networks](https://arxiv.org/abs/1910.11093)
**Code:** [github.com/ISosnovik/sesn](https://github.com/ISosnovik/sesn)

**Approach:**
- Scale-steerable filters via basis functions
- Inverse mapping for scale transformations
- Discrete scale groups

**Limitation:** Only discrete scales, not continuous.

---

### 3.2 Polar Transformer Networks (ICLR 2018)

**Paper:** [Polar Transformer Networks](https://arxiv.org/abs/1709.01889)

**Key Idea:** Transform to log-polar coordinates
```
(x, y) → (log(r), θ)
```
- Scale becomes translation in log-radius
- Standard convolution becomes scale-equivariant

**Limitation:** Requires center estimation; boundary effects.

---

### 3.3 SeLFM: Scale-Equivariant Feature Matching (Visual Computer 2024)

**Paper:** [Scale-Equivariant Local Feature Matching](https://link.springer.com/article/10.1007/s00371-024-03389-0)

**Relevance:** Recent work on scale equivariance for feature matching.

---

## 4. Sim(2)/Sim(3) Equivariant Networks

### 4.1 EquiBot: SIM(3)-Equivariant Diffusion Policy (CoRL 2025)

**Paper:** [EquiBot: SIM(3)-Equivariant Diffusion Policy](https://arxiv.org/abs/2407.01479)

**Key Innovation: Scale Canonicalization**
```python
# 1. Extract scale parameter Θs from input
estimated_scale = scale_predictor(features)

# 2. Normalize inputs
canonical_pos = positions / estimated_scale

# 3. Process with SO(3)-equivariant network
features_out = so3_equivariant_net(canonical_pos)

# 4. Rescale outputs
output = features_out * estimated_scale
```

**Critical Insight:**
> Scale equivariance = canonicalization + SO(n) equivariance + rescaling

**Relevance:** Direct template for our Sim(2) approach.

---

### 4.2 SIM(2) Lie Group-CNNs (Neural Networks 2024)

**Paper:** [Lie Group-CNNs with Scale-Rotation Equivariance](https://www.sciencedirect.com/science/article/abs/pii/S0893608024009092)

**Architecture:**
1. **Lifting module**: Transfer input from Euclidean space to SIM(2) Lie group space
2. **Group convolution**: Parameterized through FC networks using Lie algebra coefficients
3. **Global pooling + classification**

**Key Innovation:** Explicit mapping between SIM(2) and its Lie algebra

**Performance:**
- 97.50% on blood cell dataset
- 77.90% on HAM10000
- Outperforms SESN and spatial transformer networks

**Relevance:** Most directly applicable paper for our Sim(2) goal.

---

### 4.3 SIM2E Benchmark (ECCV 2022 Workshop)

**Paper:** [SIM2E: Similarity-Equivariant Benchmark](https://link.springer.com/chapter/10.1007/978-3-031-25056-9_47)

**Contribution:** Benchmark for evaluating Sim(2) equivariance

**Relevance:** Evaluation framework for our method.

---

## 5. Deep Homography Estimation

### 5.1 HomographyNet (DeTone et al., 2016) - Foundational

**Paper:** [Deep Image Homography Estimation](https://arxiv.org/abs/1606.03798)

**Key Contributions:**
- First deep learning approach for homography
- 4-point parameterization (8 corner offsets)
- Supervised training on synthetically warped MS-COCO

**Baseline:** Our method must beat this.

---

### 5.2 Content-Aware Unsupervised Deep Homography (ECCV 2020 Oral)

**Paper:** [Content-Aware Unsupervised Deep Homography](https://link.springer.com/chapter/10.1007/978-3-030-58452-8_38)
**Code:** [github.com/JirongZhang/DeepHomography](https://github.com/JirongZhang/DeepHomography)

**Key Innovations:**
- **Content-aware mask**: Reject outlier regions (like RANSAC)
- **Learned deep features** for loss instead of photometric
- **Triplet loss** for unsupervised learning

**Quote:** "Only regions that are really fit for alignment are taken into account"

**Relevance:** State-of-the-art unsupervised approach; content-aware masking is valuable.

---

### 5.3 LocalTrans (ICCV 2021)

**Paper:** [LocalTrans: Multiscale Local Transformer](https://arxiv.org/abs/2106.04067)

**Contribution:** Cross-resolution homography (10× resolution gaps)

---

### 5.4 BasesHomo (ICCV 2021)

**Paper:** [BasesHomo: Homography Flow Bases](https://github.com/megvii-research/BasesHomo)

**Key Innovation:**
- 8 pre-defined homography flow bases
- Weighted sum formulation instead of direct regression
- Implicit regularization through subspace constraint

**Relevance:** Alternative homography representation.

---

### 5.5 IHN: Iterative Homography Network (CVPR 2022)

**Paper:** [IHN: Trainable Iterative Refinement](https://arxiv.org/abs/2203.15982)

**Performance:**
- 95% error reduction through iteration
- 32.7 fps (8× faster than IC-LK)

---

### 5.6 biHomE: Bidirectional Homography (CVPR 2021)

**Paper:** [biHomE](https://github.com/NeurAI-Lab/biHomE)

**Key Innovation:** Symmetric bidirectional loss in feature space

**Relevance:** Loss function design for our method.

---

### 5.7 CodingHomo (2025) - Latest SOTA

**Paper:** [CodingHomo: Unsupervised Homography](https://arxiv.org/html/2504.12165v1)

**Features:**
- Mask-Guided Fusion for beneficial feature identification
- Coarse-to-fine refinement

---

### 5.8 homoViG: GNN-based Homography (Electronics 2024)

**Paper:** [homoViG](https://www.mdpi.com/2079-9292/13/21/4173)

**Relevance:** Recent GNN approach for homography.

---

### 5.9 HomoFM: Flow Matching for Homography (2025)

**Paper:** [HomoFM](https://arxiv.org/html/2601.18222)

**Relevance:** Latest generative approach.

---

## 6. Lie Algebra Approaches

### 6.1 Warped Convolutional Networks (WCN) - Neurocomputing 2024

**Paper:** [Warped Convolutional Networks](https://arxiv.org/abs/2206.11657)

**Key Insight:** Homography decomposes into six commutative SL(3) subgroups:
1. Rotation
2. Scale
3. Stretch (x-direction)
4. Stretch (y-direction)
5. Shear
6. Perspective (two components)

```python
# sl(3) generators for homography decomposition
G_rot = [[0, -1, 0], [1, 0, 0], [0, 0, 0]]      # Rotation
G_scale = [[1, 0, 0], [0, 1, 0], [0, 0, -2]]    # Scale
G_stretch_x = [[1, 0, 0], [0, -1, 0], [0, 0, 0]] # Stretch X
G_stretch_y = [[0, 1, 0], [1, 0, 0], [0, 0, 0]]  # Stretch Y (shear)
G_persp_1 = [[0, 0, 1], [0, 0, 0], [0, 0, 0]]   # Perspective 1
G_persp_2 = [[0, 0, 0], [0, 0, 1], [0, 0, 0]]   # Perspective 2
```

**Approach:** Decompose homography learning into separate pseudo-translation regressions

**Relevance:** Principled homography parameterization for our regression head.

---

### 6.2 DeepSKS: SKS Decomposition (May 2025)

**Paper:** [DeepSKS: Similarity-Kernel-Similarity Decomposition](https://arxiv.org/abs/2505.16599)

**Key Innovation:**
```python
# H = S2 @ K @ S1 (SKS decomposition)
# S1, S2: Similarity transforms (rotation + scale + translation each)
# K: Kernel transform (projective residual)

# Predict directly:
θ1, s1, tx1, ty1 = similarity_head_1(features)  # 4 params
θ2, s2, tx2, ty2 = similarity_head_2(features)  # 4 params
# Total: 8 params, but geometrically meaningful
```

**Advantages:**
- Decoupled similarity from projective components
- Direct matrix multiplication (no DLT solving)
- Rotation and scale are explicit parameters

**Relevance:** Excellent regression head design for Sim(2)-equivariant output.

---

### 6.3 Lie Neurons (NeurIPS 2023)

**Paper:** [Lie Neurons](https://arxiv.org/abs/2310.04521)

**Contribution:** Neural network layers based on Lie algebra structure

---

### 6.4 Lie Group Decompositions (ICLR 2024)

**Paper:** [Lie Group Decompositions for Equivariant Networks](https://arxiv.org/html/2310.11366)

**Contribution:** Systematic approach to decomposing equivariance requirements

---

## 7. Loss Functions for Geometric Tasks

### 7.1 Corner Reprojection Loss (Standard)

```python
def corner_loss(H_pred, H_gt, corners):
    corners_pred = apply_homography(H_pred, corners)
    corners_gt = apply_homography(H_gt, corners)
    return (corners_pred - corners_gt).norm(dim=-1).mean()
```

### 7.2 Symmetric Bidirectional Loss (biHomE)

```python
def symmetric_loss(H_pred, H_gt):
    loss_fwd = corner_loss(H_pred, H_gt)
    loss_bwd = corner_loss(torch.inverse(H_pred), torch.inverse(H_gt))
    return loss_fwd + loss_bwd
```

### 7.3 Multi-Scale Loss

```python
def multiscale_loss(H_pred, H_gt, scales=[1.0, 0.5, 0.25]):
    total = 0
    for s in scales:
        corners_scaled = corners * s
        total += corner_loss_at_points(H_pred, H_gt, corners_scaled)
    return total / len(scales)
```

### 7.4 Rotation-Explicit Loss

```python
def rotation_aware_loss(H_pred, H_gt):
    # Decompose homographies
    θ_pred = extract_rotation_angle(H_pred)
    θ_gt = extract_rotation_angle(H_gt)

    # Geodesic distance on SO(2)
    angle_diff = θ_pred - θ_gt
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

    return corner_loss(H_pred, H_gt) + 0.3 * angle_diff.abs().mean()
```

### 7.5 Photometric Loss (Unsupervised)

```python
def photometric_loss(img_src, img_tgt, H_pred):
    warped = warp_image(img_src, H_pred)
    return (warped - img_tgt).abs().mean()
```

### 7.6 Perceptual Loss (Content-Aware)

```python
def perceptual_loss(img_src, img_tgt, H_pred, feature_extractor):
    warped = warp_image(img_src, H_pred)
    feat_warped = feature_extractor(warped)
    feat_tgt = feature_extractor(img_tgt)
    return (feat_warped - feat_tgt).pow(2).mean()
```

---

## 8. Key Libraries and Implementations

### 8.1 escnn (E(2)-Equivariant)

**Repository:** [github.com/QUVA-Lab/escnn](https://github.com/QUVA-Lab/escnn)

```bash
pip install escnn>=1.0.0
```

**Usage:**
```python
from escnn import gspaces, nn as enn

# Define group (C_16 = 16 rotations)
gspace = gspaces.rot2dOnR2(N=16)

# Define field types
in_type = enn.FieldType(gspace, [gspace.trivial_repr])
out_type = enn.FieldType(gspace, 8 * [gspace.regular_repr])

# Steerable convolution
conv = enn.R2Conv(in_type, out_type, kernel_size=5, padding=2)

# Group pooling (regular → trivial = equivariant → invariant)
pool = enn.GroupPooling(out_type)
```

### 8.2 EGNN PyTorch

**Repository:** [github.com/lucidrains/egnn-pytorch](https://github.com/lucidrains/egnn-pytorch)

```python
from egnn_pytorch import EGNN

layer = EGNN(dim=32, edge_dim=1, m_dim=16, num_nearest_neighbors=8)
feats_out, coords_out = layer(feats, coords, edges)
```

### 8.3 PyTorch Geometric

**Already in use.** Continue using for graph construction and message passing infrastructure.

### 8.4 e3nn (3D Equivariance)

**Repository:** [github.com/e3nn/e3nn](https://github.com/e3nn/e3nn)

Note: Primarily for 3D, but has some 2D functionality.

### 8.5 SE2-LoFTR Reference Implementation

**Repository:** [github.com/georg-bn/se2-loftr](https://github.com/georg-bn/se2-loftr)

Good reference for integrating escnn with image matching.

---

---

## 9. Canonicalization Approaches (KEY FINDING)

### 9.1 The Fundamental Problem: Invariance vs Prediction

**Critical Discovery (Phase 5 Failure):**

A rotation-INVARIANT network produces `f(R @ x) = f(x)` - identical outputs for rotated inputs. But the ground truth homography CHANGES: `H_new = R @ H @ R^{-1}`.

**Result:** A purely invariant model CANNOT predict rotation because it has NO rotation signal.

This is mathematically impossible to fix with invariant features alone.

### 9.2 The Solution: Canonicalization

Instead of building invariant features (which lose transformation info):

1. **ESTIMATE** the canonical frame (orientation + scale) from each image
2. **TRANSFORM** to canonical space
3. **PROCESS** in canonical space (now transformations are factored out)
4. **DE-CANONICALIZE** the output

**Key insight:** The transformation IS the canonicalization. The relative rotation = θ_tgt - θ_src.

### 9.3 EquiAdapt: Model-Agnostic Equivariance (NeurIPS 2023)

**Paper:** [Equivariant Adaptation of Large Pretrained Models](https://arxiv.org/abs/2310.01647)
**Code:** [github.com/arnab39/equiadapt](https://github.com/arnab39/equiadapt)

**Key Contribution:** Makes ANY pretrained model equivariant via learned canonicalization.

```python
# EquiAdapt pattern:
Input → Canonicalizer(x) → T → T⁻¹(x) → Prediction Network → T(output)
```

**Relevance:** Direct template for our Sim(2) canonicalization.

### 9.4 Frame Averaging (ICML 2023-2024)

**Papers:**
- [FAENet: Frame Averaging Equivariant GNN](https://arxiv.org/pdf/2305.05577) (ICML 2023)
- [Minimal Frame Averaging](https://arxiv.org/abs/2406.07598) (ICML 2024)
- [Impossibility of Continuous Canonicalization](https://arxiv.org/abs/2402.16077) (ICML 2024)

**Key Insight:** Direct canonicalization can be discontinuous. Solution: weighted frame averaging.

```python
# Frame averaging for ambiguity (e.g., 180° rotation):
hypotheses = [θ, θ + π]  # Multiple canonical frames
confidences = model.predict_confidence(x)
output = Σ confidence[k] * process(canonicalize(x, hypotheses[k]))
```

**Relevance:** Handles 180° ambiguity in orientation estimation.

### 9.5 Local Canonicalization (2025)

**Paper:** [Local Canonicalization for Equivariant GNNs](https://arxiv.org/html/2509.26499v1)

**Key Innovation:** Per-node local frames + tensorial messages
- 4-5x faster than Equiformer
- Exact equivariance preserved

**Relevance:** Potential efficiency improvement for our approach.

### 9.6 Canonicalization vs Equivariance Comparison

| Approach | Transformation Info | Pros | Cons |
|----------|---------------------|------|------|
| **Invariant** | Lost (f(Tx)=f(x)) | Simple | Cannot predict T |
| **Equivariant** | Encoded (f(Tx)=T'f(x)) | Principled | Complex architecture |
| **Canonicalization** | Extracted explicitly | Flexible, interpretable | Training stability |

**Our choice:** Canonicalization - best for homography prediction where we NEED the transformation.

---

## Summary: Key Takeaways for Our Approach

### Why Rotation-Invariant Approach Failed (Phase 5)

1. **Invariant features lose transformation information**
2. **Model predicts constant H, but GT varies with rotation**
3. **Error increases from 181px (0°) to 491px (180°)**
4. **This is FUNDAMENTAL, not a bug**

### Sim(2) Equivariance Strategy (Phase 6 - CANONICALIZATION)

1. **Translation**: Learned from correspondences in canonical space
2. **Rotation**: Estimated via OrientationCanonicalizer, then factored out
3. **Scale**: Estimated via GeometricScaleEstimator, then factored out

```
Image_src → Sim2Canonicalizer → (θ_src, s_src, canonical_pos_src)
Image_tgt → Sim2Canonicalizer → (θ_tgt, s_tgt, canonical_pos_tgt)

Relative rotation = θ_tgt - θ_src  (EXTRACTED, not predicted!)
Relative scale = s_tgt / s_src

Process in canonical space → H_canonical (translation only)
De-canonicalize → H_final = R_tgt @ S_tgt @ H_canonical @ S_src⁻¹ @ R_src⁻¹
```

### Novel Contribution

**"First Sim(2)-Equivariant Deep Homography Estimator via Canonicalization"**

Key innovations:
- Orientation canonicalization (PCA-based + learned refinement)
- Multi-hypothesis handling for 180° ambiguity
- Full Sim(2) = SO(2) × R⁺ × R² de-canonicalization
- SKS decomposition for interpretable output

### Comparison with Prior Art

| Method | Translation | Rotation | Scale | Approach |
|--------|-------------|----------|-------|----------|
| HomographyNet | Augmentation | Augmentation | Augmentation | Data-driven |
| SE2-LoFTR | Yes | Yes (C_N) | No | Steerable CNNs |
| SESN | No | No | Yes (discrete) | Scale-steerable |
| EquiBot | Yes | Yes | Yes | Canonicalization (3D) |
| **Ours** | **Yes** | **Yes (continuous)** | **Yes (continuous)** | **Canonicalization (2D)** |

---

*Last updated: 2026-01-30*
*For ECCV 2026 submission*
