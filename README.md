# Sim(2)-Equivariant Thermal Homography

A keypoint-free approach to thermal image alignment achieving **true Sim(2) equivariance** through log-polar transform for joint scale-rotation detection.

## Overview

This project implements a novel method for thermal homography estimation that:

- Works without keypoint detection (handles low-texture thermal images)
- Achieves **TRUE Sim(2) = SO(2) x R+ x R2** equivariance (rotation + scale + translation)
- Uses **log-polar + Fourier-Mellin transform** to convert scale-rotation into translations (enabling CNN equivariance)
- Generalizes to unseen rotations and scales **without** data augmentation
- Is colormap-invariant (works on any thermal colormap)

### Key Innovation: Learned Fourier-Mellin Transform

In log-polar coordinates (log(r), theta):
- **Scale by s** -> shift by log(s) in the log(r) direction
- **Rotate by phi** -> shift by phi in the theta direction

Combined with FFT magnitude (translation-invariant), this means a standard **translation-equivariant CNN** in log-polar-frequency space becomes **Sim(2)-equivariant** in Cartesian space.

## Architecture: LogPolarSim2Net

```
Input: (img_src, img_tgt)
         |
    FFT Magnitude (translation-invariant)
         |
    Log-Polar Transform (Cartesian -> Log-polar)
         |
    LearnedLogPolarEncoder (CNN features)
         |
    FFT Phase Correlation
         |
    Peak detection (soft-argmax)
         |
    (scale, rotation) = (exp(d_log_r), d_theta)
         |
    180 deg Disambiguation (spatial correlation)
         |
    De-rotate and de-scale target
         |
    Spatial cross-correlation for translation
         |
    H = T(t) @ S(s) @ R(theta)
         |
Output: H (3x3 homography matrix)
```

### Results

| Test Case | Ground Truth | Detected | Status |
|-----------|-------------|----------|--------|
| Identity | (0 deg, 1.0x) | (0.0 deg, 1.000x) | PASS |
| Rotation 30 deg | (30 deg, 1.0x) | (30.0 deg, 1.000x) | PASS |
| Scale 1.2x | (0 deg, 1.2x) | (0.0 deg, 1.198x) | PASS |
| Rot+Scale (45 deg, 0.8x) | (45 deg, 0.8x) | (45.0 deg, 0.798x) | PASS |
| Rot+Scale (-30 deg, 1.3x) | (-30 deg, 1.3x) | (-30.0 deg, 1.300x) | PASS |
| Rotation 90 deg | (90 deg, 1.0x) | (90.0 deg, 1.000x) | PASS |
| **With translation** | 30 deg + (20,20)px | 30.0 deg | PASS |
| **Full pipeline** | 21/21 tests | Mean error 0.10 deg | **100% PASS** |

## Quick Start

### 1. Setup Environment

```bash
git clone https://github.com/YaqoobAnsari/thermal-homography-equivariant.git
cd thermal-homography-equivariant

conda env create -f environment.yml
conda activate thermal-homography
pip install -e .
```

### 2. Usage

```python
from src.models import LogPolarSim2Net

# Create the model
model = LogPolarSim2Net(
    lp_size=(180, 64),      # (n_angles, n_radii)
    r_min=0.05,             # Avoid center singularity
    r_max=0.9,              # Maximum radius
    feature_channels=32,    # CNN feature dimension
    use_fft_magnitude=True, # Fourier-Mellin mode
    use_disambiguation=True,# Resolve 180 deg ambiguity
)

# Forward pass
result = model(img_src, img_tgt)
homography = result['homography']   # [B, 3, 3]
rotation = result['rotation_deg']   # Detected rotation in degrees
scale = result['scale']             # Detected scale factor
```

### 3. Training

```bash
# Submit SLURM training job
sbatch slurm/slurm_train.sh

# Or train directly
python -m src.training.train --config-name=phase1_synthetic
```

### 4. Evaluation

```bash
# HPatches benchmark
python scripts/evaluate_hpatches.py

# MS-COCO benchmark
python scripts/evaluate_mscoco.py

# Robustness evaluation
python scripts/comprehensive_robustness_test.py

# Baseline comparison
python scripts/comprehensive_baseline_comparison.py
```

## Project Structure

```
thermal-homography/
├── src/
│   ├── config.py              # Centralized configuration
│   ├── constants.py           # Project constants
│   ├── models/
│   │   ├── log_polar_sim2_net.py   # LogPolarSim2Net (recommended)
│   │   ├── log_polar_transform.py  # Log-polar + phase correlation
│   │   ├── baselines.py            # CNN baselines for comparison
│   │   ├── e2_layers.py            # E(2)-equivariant message passing
│   │   ├── graph_network.py        # Graph-based model
│   │   └── ...
│   ├── data/
│   │   ├── thermal_dataset.py
│   │   ├── synthetic_generator.py
│   │   ├── hpatches_dataset.py
│   │   └── mscoco_dataset.py
│   ├── training/
│   │   ├── train.py           # Training loop (PyTorch Lightning)
│   │   ├── losses.py          # Homography loss functions
│   │   ├── sim2_losses.py     # Sim(2)-specific losses
│   │   └── metrics.py         # Evaluation metrics
│   ├── evaluation/
│   │   ├── eval_splits.py
│   │   └── visualize.py
│   └── utils/
│       ├── logging_config.py
│       ├── homography.py
│       ├── equivariance_tests.py
│       └── geometry.py
├── configs/
│   └── phase1_synthetic.yaml
├── scripts/                   # Evaluation scripts
├── slurm/                     # SLURM job scripts
└── tests/
    ├── test_models.py
    ├── test_data.py
    ├── test_training.py
    └── test_integration.py
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `THERMAL_DATA_DIR` | `./data` | Root directory for datasets |
| `THERMAL_OUTPUT_DIR` | System temp | Output directory for results |
| `THERMAL_CHECKPOINTS_DIR` | `./checkpoints` | Checkpoint directory |
| `THERMAL_LOGS_DIR` | `./logs` | Logs directory |
| `THERMAL_LOG_LEVEL` | `INFO` | Logging level |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Citation

```bibtex
@article{ansari2026thermal,
  title={Keypoint-Free Thermal Homography via Learned Fourier-Mellin Transform
         with Sim(2) Equivariance},
  author={Ansari, Yaqoob},
  year={2026}
}
```

## License

MIT License - See LICENSE file for details.
