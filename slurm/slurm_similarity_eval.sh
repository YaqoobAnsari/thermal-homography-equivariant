#!/bin/bash
#SBATCH --job-name=sim2-eval
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_sim2eval_%j.out
#SBATCH --error=logs/slurm_sim2eval_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ansarimohammedyaqoob01@gmail.com

# =============================================================================
# SLURM Evaluation Script for Sim(2) Similarity Estimation
# =============================================================================
# Runs the comprehensive evaluation comparing LogPolarSim2Net against
# classical baselines (SIFT+RANSAC, ORB+RANSAC, ECC) across:
# - Full rotation range (0-330 degrees)
# - Full scale range (0.5x - 2.0x)
# - Translation range (0-50 pixels)
# - Thermal colormap invariance
#
# Usage:
#   sbatch slurm/slurm_similarity_eval.sh
#   sbatch slurm/slurm_similarity_eval.sh checkpoints/best.pt
# =============================================================================

set -e  # Exit on error

# Optional checkpoint argument
CHECKPOINT=${1:-""}

# Timestamp for logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================"
echo "Sim(2) Similarity Estimation Evaluation"
echo "========================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Job Name:   $SLURM_JOB_NAME"
echo "Node:       $SLURMD_NODENAME"
echo "Partition:  $SLURM_JOB_PARTITION"
echo "GPUs:       $SLURM_GPUS_ON_NODE"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "Memory:     $SLURM_MEM_PER_NODE MB"
echo "Checkpoint: ${CHECKPOINT:-None (untrained FMT baseline)}"
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
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONHASHSEED=42

# Create output directories
OUTPUT_DIR=/data/gpfs/projects/punim2769/thermal-homography/outputs/similarity_evaluation
mkdir -p $OUTPUT_DIR
mkdir -p /data/gpfs/projects/punim2769/thermal-homography/logs

# Change to project directory
cd /data/gpfs/projects/punim2769/thermal-homography

# Build command
CMD="python -u scripts/evaluate_similarity.py --device cuda --n-samples 100 --output-dir $OUTPUT_DIR"

if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

# Run evaluation
echo ""
echo "========================================"
echo "Starting evaluation at $(date)"
echo "Command: $CMD"
echo "========================================"

$CMD

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Evaluation completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results at: $OUTPUT_DIR"
echo "========================================"

exit $EXIT_CODE
