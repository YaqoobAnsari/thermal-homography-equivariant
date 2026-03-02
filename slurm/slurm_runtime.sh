#!/bin/bash
#SBATCH --job-name=runtime
#SBATCH --output=logs/runtime_%j.out
#SBATCH --error=logs/runtime_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# =============================================================================
# RUNTIME MEASUREMENT
# Measures inference time for all models on 256x256 images
# =============================================================================

echo "=========================================="
echo "RUNTIME MEASUREMENT"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="

cd /data/gpfs/projects/punim2769/thermal-homography

# Load conda module and activate environment
module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thermal-homography

echo "Python: $(which python)"

mkdir -p logs outputs

nvidia-smi

echo ""
echo "Measuring runtime..."
python scripts/measure_runtime.py \
    --device cuda \
    --image-size 256 \
    --batch-size 1 \
    --n-warmup 100 \
    --n-measure 500 \
    --output outputs/runtime_results.json

echo ""
echo "=========================================="
echo "RUNTIME COMPLETE: $(date)"
echo "Results in: outputs/runtime_results.json"
echo "=========================================="
