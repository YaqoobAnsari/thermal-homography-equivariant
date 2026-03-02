#!/bin/bash
#SBATCH --job-name=hpatches
#SBATCH --output=logs/hpatches_%j.out
#SBATCH --error=logs/hpatches_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# =============================================================================
# HPATCHES BENCHMARK EVALUATION
# Tests SIGMA on real images with viewpoint/illumination changes
# =============================================================================

echo "=========================================="
echo "HPATCHES BENCHMARK EVALUATION"
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

mkdir -p logs outputs/hpatches

nvidia-smi

echo ""
echo "Running HPatches evaluation..."
python scripts/evaluate_hpatches.py \
    --data-root data/hpatches/hpatches-sequences-release \
    --model-type log_polar_sim2 \
    --device cuda \
    --batch-size 8 \
    --output outputs/hpatches/hpatches_results.json

echo ""
echo "=========================================="
echo "HPATCHES COMPLETE: $(date)"
echo "Results in: outputs/hpatches/"
echo "=========================================="
