# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial HPC environment setup for Spartan cluster (University of Melbourne)
- SLURM job submission scripts for GPU training (`scripts/slurm_train.sh`)
- Quick test script for 1-hour validation (`scripts/slurm_quick_test.sh`)
- Comprehensive SLURM and GPU documentation (`SLURM_GPU.md`)
- Research plan document with ECCV 2026 timeline (`RESEARCH_PLAN.md`)
- User quotas and billing information in SLURM documentation
- Email notifications for job status
- This CHANGELOG.md file for tracking project evolution
- Equivariance verification script (`scripts/verify_equivariance.py`) for testing rotation consistency
- Unit tests for homography scaling (`tests/test_data.py::TestHomographyScaling`)
- Unit tests for distance normalization (`tests/test_data.py::TestDistanceNormalization`)

### Changed
- Updated `requirements.txt` with exact tested versions for reproducibility
- Added numpy version constraint (`<2.0.0`) for lie-learn/escnn compatibility
- Updated installation documentation for HPC environments
- Enhanced SLURM scripts with better logging and error handling
- **BREAKING**: Synthetic data now defaults to full rotation range [-180, 180] instead of [-30, 30]
  - This is necessary for proper E(2)-equivariance validation

### Fixed
- NumPy version incompatibility with lie-learn (constrained to 1.26.4)
- **CRITICAL**: Homography scaling on image resize (`src/data/thermal_dataset.py`)
  - Previously, when images were resized, the homography was not scaled accordingly
  - All training labels were incorrect; this invalidated all previous experiments
  - Now correctly applies `H_scaled = S @ H @ S^{-1}` transformation
- **CRITICAL**: Distance normalization in E(2) GNN (`src/models/e2_layers.py`)
  - Previously normalized by `dist.max()` which varies with rotation, breaking equivariance
  - Now uses fixed normalization by image diagonal (2*sqrt(2) for [-1,1] coordinates)
- **HIGH**: Weight initialization gain (`src/models/e2_layers.py`)
  - Previously `gain=0.01` caused vanishing gradients
  - Now uses `gain=1.0` for proper gradient flow

---

## [0.1.0] - 2026-01-30

### Added

#### Core Architecture
- **E(2)-Equivariant GNN** (`src/models/e2_layers.py`)
  - Custom message passing with rotation/translation invariance
  - Attention-weighted aggregation
  - Local coordinate frame construction

- **ThermalHomographyNet** (`src/models/graph_network.py`)
  - End-to-end architecture: Feature extraction → GNN → LRFT → Regression
  - Colormap-invariant gradient-based features (Sobel)
  - Dense 32x32 grid (1024 nodes) keypoint-free approach

- **Low-Rank Feature Transform** (`src/models/lrft.py`)
  - Adapted from Equi-GSPR (ECCV 2024)
  - Compresses 1024→128 nodes for efficient similarity computation
  - Attention-based and direct factorization modes

- **8 Baseline Models** (`src/models/baselines.py`)
  - ResNetBaseline, UNetBaseline, CorrelationBaseline
  - HomographyNet (DeTone 2016), LucasKanadeBaseline
  - BasesHomoBaseline (ICCV 2021), IterativeHomographyNetwork (CVPR 2022)
  - NonEquivariantGNN (ablation)

#### Data Pipeline
- **ThermalPairDataset** (`src/data/thermal_dataset.py`)
  - Multi-format support: pairs, roadscene, synthetic
  - JSON-based homography loading
  - Robust error handling

- **Augmentation Pipeline** (`src/data/augmentation.py`)
  - Synchronized geometric augmentation for pairs
  - Homography update on transform
  - Albumentations integration

- **Synthetic Data Generator** (`src/data/synthetic_generator.py`)
  - Checkerboard and thermal blob patterns
  - On-the-fly homography generation
  - Rotation equivariance test dataset

#### Training Infrastructure
- **PyTorch Lightning Module** (`src/training/train.py`)
  - Model registry for easy switching
  - NaN detection with automatic halt
  - Gradient monitoring per layer
  - Live training visualization
  - Config-saving checkpoints

- **Multi-Loss Training** (`src/training/losses.py`)
  - Corner reprojection loss (primary)
  - Geodesic rotation loss (SO(2) distance)
  - Translation loss
  - Rank regularization loss
  - Dynamic loss weighting (Kendall et al.)

- **Comprehensive Metrics** (`src/training/metrics.py`)
  - Corner/rotation/translation errors
  - Registration recall at thresholds
  - Precision-recall curves
  - Per-scene breakdown
  - Inference benchmarking (FPS, latency)

#### Evaluation & Visualization
- **Evaluation Protocols** (`src/evaluation/eval_splits.py`)
  - Standard evaluation
  - Rotation equivariance testing
  - Geometric split (train <30°, test >45°)
  - Colormap split

- **Visualization Tools** (`src/evaluation/visualize.py`)
  - Homography visualization (4-panel)
  - t-SNE/UMAP feature projection
  - Attention map visualization
  - Failure case reports

#### Configuration & Utilities
- **Centralized Config** (`src/config.py`)
  - Environment variable support
  - Platform-independent paths
  - Schema validation with helpful errors

- **Constants** (`src/constants.py`)
  - All magic numbers centralized
  - Default hyperparameters

- **Logging Infrastructure** (`src/utils/logging_config.py`)
  - Colored terminal output
  - File rotation (10MB, 5 files)
  - Structured JSON logging option

- **Geometry Utilities** (`src/utils/`)
  - Homography manipulation
  - Geometric operations
  - E(2) equivariance tests

#### Project Infrastructure
- Three-phase training configs (phase1/phase2/phase3)
- Comprehensive test suite (models, data, training, integration)
- Setup scripts and verification tools
- Multi-seed experiment runner
- Experiment comparison tool

### Documentation
- README.md with quick start guide
- INSTALL.md with troubleshooting
- CONTRIBUTING.md with code style guidelines
- TODO.md tracking 87 improvements

---

## Development Notes

### Research Timeline
| Phase | Duration | Status | Goal |
|-------|----------|--------|------|
| Phase 1 | Week 1-2 | In Progress | Validate E(2) equivariance on synthetic data |
| Phase 2 | Week 3-5 | Pending | Architecture search & hyperparameter tuning |
| Phase 3 | Week 6-9 | Pending | Full training & evaluation |
| Phase 4 | Week 10-13 | Pending | Ablations, writing, figures |
| Phase 5 | Week 14-16 | Pending | Revisions & submission |

### Key Dependencies
- PyTorch 2.5.1+cu121
- torch-geometric 2.7.0
- escnn 1.0.11
- numpy 1.26.4 (constraint: <2.0.0)
- pytorch-lightning 2.6.0

### HPC Environment
- Cluster: Spartan (University of Melbourne)
- OS: RHEL 9.6
- GPU Partitions: gpu-a100, gpu-h100, gpu-l40s
- CUDA: 12.1 (via pip wheels)
