#!/bin/bash
#SBATCH --job-name=sim2_eval
#SBATCH --output=logs/comprehensive_eval_%j.out
#SBATCH --error=logs/comprehensive_eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# =============================================================================
# COMPREHENSIVE Sim(2) EVALUATION
# =============================================================================
# This runs all 4 experiments:
# 1. Rotation generalization
# 2. Scale generalization
# 3. Translation invariance
# 4. Combined Sim(2)
#
# Expected runtime: 2-3 hours on A100
# =============================================================================

echo "=========================================="
echo "COMPREHENSIVE Sim(2) EVALUATION"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="

# Setup
cd /data/gpfs/projects/punim2769/thermal-homography
source ~/.bashrc
conda activate thermal

# Create output directories
mkdir -p logs
mkdir -p outputs/comprehensive_eval

# Check GPU
nvidia-smi

# Run evaluation
echo ""
echo "Running comprehensive evaluation..."
python scripts/comprehensive_sim2_evaluation.py

echo ""
echo "=========================================="
echo "EXPERIMENT COMPLETE: $(date)"
echo "Results in: outputs/comprehensive_eval/"
echo "=========================================="
