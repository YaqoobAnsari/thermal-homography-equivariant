#!/bin/bash
#SBATCH --job-name=phase1-synthetic
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/phase1_%j.out
#SBATCH --error=logs/phase1_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ansarimohammedyaqoob01@gmail.com

# Phase 1: Synthetic Validation Experiments
# This is the GO/NO-GO test for E(2)-equivariance

echo "=========================================="
echo "PHASE 1: SYNTHETIC VALIDATION"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Create logs directory
mkdir -p logs

# Load environment
module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thermal-homography

# Verify environment
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c "import torch; assert torch.cuda.is_available()"; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# Navigate to project directory
cd /data/gpfs/projects/punim2769/thermal-homography

# Run Phase 1 experiments
# Config can be: all, checkerboard_low_noise, checkerboard_medium_noise,
#               thermal_blobs, rotation_only, high_noise_stress, seed_check_123
CONFIG=${1:-all}

echo "Running config: $CONFIG"
echo ""

python scripts/phase1_synthetic_validation.py \
    --config "$CONFIG" \
    --device cuda \
    --output-dir outputs/phase1_$(date +%Y%m%d_%H%M%S)

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
