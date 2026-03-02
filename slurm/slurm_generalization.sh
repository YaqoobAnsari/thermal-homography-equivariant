#!/bin/bash
#SBATCH --job-name=gen_exp
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/generalization_%j.out
#SBATCH --error=logs/generalization_%j.err

set -e
cd /data/gpfs/projects/punim2769/thermal-homography
mkdir -p logs outputs/generalization

PYTHON=/home/yansari/.conda/envs/thermal-homography/bin/python

echo "=========================================="
echo "GENERALIZATION EXPERIMENT"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=========================================="

$PYTHON scripts/generalization_experiment.py

echo ""
echo "=========================================="
echo "EXPERIMENT COMPLETE: $(date)"
echo "Results in: outputs/generalization/"
echo "=========================================="
