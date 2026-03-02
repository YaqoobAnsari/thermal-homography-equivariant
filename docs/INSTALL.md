# Installation Guide

## Quick Start (Server)

```bash
# 1. Extract and enter directory
tar -xzf thermal-homography.tar.gz
cd thermal-homography

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip wheel setuptools

# 4. Install PyTorch (adjust for your CUDA version)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# For CPU only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. Install torch-geometric
pip install torch-geometric

# 6. Install escnn (has tricky build - use older setuptools)
pip install numpy scipy
pip install 'setuptools<69'
pip install lie-learn
pip install --upgrade setuptools
pip install escnn

# 7. Install remaining dependencies
pip install -r requirements.txt

# 8. Install package in dev mode
pip install -e .

# 9. Verify installation
python -c "from src.models import ThermalHomographyNet; print('OK')"
```

## Environment Variables

```bash
# Optional: Set data directories
export THERMAL_DATA_DIR=/path/to/your/data
export THERMAL_OUTPUT_DIR=/path/to/outputs
export THERMAL_CHECKPOINTS_DIR=/path/to/checkpoints
```

## Running Training

```bash
# Quick validation (synthetic data)
python scripts/run_validation.py --epochs 50 --device cuda

# Full training
python -m src.training.train config=configs/phase1_synthetic.yaml

# Or use the entry point
thermal-train config=configs/phase1_synthetic.yaml
```

## Troubleshooting

### lie-learn build fails
```bash
# Use older setuptools
pip install 'setuptools<69'
pip install lie-learn
pip install --upgrade setuptools
```

### CUDA out of memory
```bash
# Reduce batch size in config or use gradient accumulation
python -m src.training.train training.batch_size=4
```

### Import errors
```bash
# Make sure package is installed
pip install -e .
```
