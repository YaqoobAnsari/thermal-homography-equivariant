#!/bin/bash
#SBATCH --job-name=thermal-train
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ansarimohammedyaqoob01@gmail.com

# =============================================================================
# SLURM Training Script for Thermal Homography
# =============================================================================
# Usage:
#   sbatch scripts/slurm_train.sh                    # Phase 1: Synthetic
#   sbatch scripts/slurm_train.sh phase2_search      # Phase 2: HPO
#   sbatch scripts/slurm_train.sh phase3_full        # Phase 3: Full
#
# Resource Limits (punim2769):
#   - Max 200 jobs in queue
#   - Max 48 GPUs concurrent
#   - Max 384 CPUs concurrent
#   - Max 5.9TB memory concurrent
# =============================================================================

set -e  # Exit on error

# Default to phase1 if no argument provided
CONFIG=${1:-phase1_synthetic}

# Timestamp for logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================"
echo "Thermal Homography Training"
echo "========================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Job Name:   $SLURM_JOB_NAME"
echo "Node:       $SLURMD_NODENAME"
echo "Partition:  $SLURM_JOB_PARTITION"
echo "GPUs:       $SLURM_GPUS_ON_NODE"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "Memory:     $SLURM_MEM_PER_NODE MB"
echo "Config:     $CONFIG"
echo "Timestamp:  $TIMESTAMP"
echo "========================================"

# Load modules
module load Anaconda3/2024.02-1

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate thermal-homography

# Verify Python and key packages
echo ""
echo "Environment Check:"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Verify GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
echo ""

# Set environment variables
export THERMAL_DATA_DIR=/data/gpfs/projects/punim2769/thermal-homography/data
export THERMAL_OUTPUT_DIR=/data/gpfs/projects/punim2769/thermal-homography/outputs
export THERMAL_CHECKPOINTS_DIR=/data/gpfs/projects/punim2769/thermal-homography/checkpoints
export THERMAL_LOGS_DIR=/data/gpfs/projects/punim2769/thermal-homography/logs

# Performance settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42

# Disable wandb if not configured
if [ -z "$WANDB_API_KEY" ]; then
    export WANDB_MODE=disabled
    echo "Wandb: Disabled (no API key)"
else
    echo "Wandb: Enabled"
fi

# Create output directories
mkdir -p $THERMAL_OUTPUT_DIR $THERMAL_CHECKPOINTS_DIR $THERMAL_LOGS_DIR

# Change to project directory
cd /data/gpfs/projects/punim2769/thermal-homography

# Run training
echo ""
echo "========================================"
echo "Starting training at $(date)"
echo "Config: configs/${CONFIG}.yaml"
echo "========================================"

# Use unbuffered output for real-time logging
python -u -m src.training.train --config-name=${CONFIG}

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Training completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
