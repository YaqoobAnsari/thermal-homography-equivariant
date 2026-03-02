#!/bin/bash
#SBATCH --job-name=fair_cmp
#SBATCH --output=logs/fair_comparison_%j.out
#SBATCH --error=logs/fair_comparison_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

echo "=========================================="
echo "FAIR AUGMENTATION COMPARISON: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "=========================================="

# Setup
cd /data/gpfs/projects/punim2769/thermal-homography
module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thermal-homography
mkdir -p outputs/fair_augmentation

echo "Python: $(which python)"
nvidia-smi

echo ""
echo "Running fair augmentation comparison..."
echo "This trains baselines with BOTH limited (±30°) and FULL (0-360°) augmentation"
echo "to address reviewer concern about unfair comparison."
echo ""

python scripts/fair_augmentation_comparison.py \
    --train-samples 5000 \
    --epochs 50 \
    --output-dir outputs/fair_augmentation

echo ""
echo "=========================================="
echo "FAIR COMPARISON COMPLETE: $(date)"
echo "Results in: outputs/fair_augmentation/"
echo "=========================================="
