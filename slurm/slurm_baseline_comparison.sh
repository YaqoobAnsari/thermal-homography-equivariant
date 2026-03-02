#!/bin/bash
#SBATCH --job-name=baseline_cmp
#SBATCH --output=logs/baseline_comparison_%j.out
#SBATCH --error=logs/baseline_comparison_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "=========================================="
echo "BASELINE COMPARISON: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

cd /data/gpfs/projects/punim2769/thermal-homography

# Load conda
module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thermal-homography

echo "Python: $(which python)"

mkdir -p logs outputs/baseline_comparison

nvidia-smi

echo ""
echo "Running baseline comparison..."
python scripts/baseline_comparison.py \
    --device cuda \
    --train-epochs 25 \
    --train-samples 5000 \
    --test-samples 100 \
    --output-dir outputs/baseline_comparison

echo ""
echo "=========================================="
echo "BASELINE COMPARISON COMPLETE: $(date)"
echo "Results in: outputs/baseline_comparison/"
echo "=========================================="
