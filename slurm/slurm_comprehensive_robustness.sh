#!/bin/bash
#SBATCH --job-name=robust_test
#SBATCH --output=logs/comprehensive_robustness_%j.out
#SBATCH --error=logs/comprehensive_robustness_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "=========================================="
echo "COMPREHENSIVE ROBUSTNESS TEST: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

cd /data/gpfs/projects/punim2769/thermal-homography

# Load conda
module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thermal-homography

echo "Python: $(which python)"
mkdir -p logs outputs/comprehensive_robustness

nvidia-smi

echo ""
echo "Running COMPREHENSIVE robustness test..."
echo "Testing: SIGMA vs SIFT+RANSAC"
echo "Across: 6 pattern types, 7 image conditions, 8 seeds"
echo ""

python scripts/comprehensive_robustness_test.py \
    --device cuda \
    --seeds 8 \
    --output-dir outputs/comprehensive_robustness

echo ""
echo "=========================================="
echo "COMPREHENSIVE ROBUSTNESS TEST COMPLETE: $(date)"
echo "Results in: outputs/comprehensive_robustness/"
echo "=========================================="
