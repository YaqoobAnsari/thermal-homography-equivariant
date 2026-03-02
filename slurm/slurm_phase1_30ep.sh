#!/bin/bash
#SBATCH --job-name=phase1-30ep
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-44
#SBATCH --output=logs/phase1_30ep_%A_%a.out
#SBATCH --error=logs/phase1_30ep_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ansarimohammedyaqoob01@gmail.com

# Phase 1 Comprehensive Testing: 30 Epochs
#
# EXPANDED TEST SUITE: 15 patterns x 3 noise levels = 45 scenarios
#
# Pattern Tiers (by symmetry):
#   Tier 1 (asymmetric):  asymmetric, arrow, L_shape, T_shape, corner_marker
#   Tier 2 (semi):        natural, blobs, stripes, ellipse, gradient
#   Tier 3 (symmetric):   checkerboard, cross, concentric
#   Tier 4 (realistic):   thermal_hotspot, multi_hotspot
#
# Quick test: 1 seed (default), Full validation: 5 seeds (--full-validation)

echo "=========================================="
echo "PHASE 1 COMPREHENSIVE TEST - 30 EPOCHS"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Create logs directory
mkdir -p logs

# Load environment
module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thermal-homography

cd /data/gpfs/projects/punim2769/thermal-homography

# Verify environment
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c "import torch; assert torch.cuda.is_available()"; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# Array index to scenario mapping
# 15 patterns x 3 noise levels = 45 scenarios
SCENARIOS=(
    # Tier 1: Fully Asymmetric (indices 0-14)
    "asymmetric_low"
    "asymmetric_medium"
    "asymmetric_high"
    "arrow_low"
    "arrow_medium"
    "arrow_high"
    "L_shape_low"
    "L_shape_medium"
    "L_shape_high"
    "T_shape_low"
    "T_shape_medium"
    "T_shape_high"
    "corner_marker_low"
    "corner_marker_medium"
    "corner_marker_high"
    # Tier 2: Semi-Asymmetric (indices 15-29)
    "natural_low"
    "natural_medium"
    "natural_high"
    "blobs_low"
    "blobs_medium"
    "blobs_high"
    "stripes_low"
    "stripes_medium"
    "stripes_high"
    "ellipse_low"
    "ellipse_medium"
    "ellipse_high"
    "gradient_low"
    "gradient_medium"
    "gradient_high"
    # Tier 3: High Symmetry - Stress Tests (indices 30-38)
    "checkerboard_low"
    "checkerboard_medium"
    "checkerboard_high"
    "cross_low"
    "cross_medium"
    "cross_high"
    "concentric_low"
    "concentric_medium"
    "concentric_high"
    # Tier 4: Realistic Thermal (indices 39-44)
    "thermal_hotspot_low"
    "thermal_hotspot_medium"
    "thermal_hotspot_high"
    "multi_hotspot_low"
    "multi_hotspot_medium"
    "multi_hotspot_high"
)

SCENARIO=${SCENARIOS[$SLURM_ARRAY_TASK_ID]}

echo "Running scenario: $SCENARIO (30 epochs)"
echo ""

# Create output directory with date
OUTPUT_DIR="outputs/phase1_30ep_$(date +%Y%m%d)"
mkdir -p "$OUTPUT_DIR"

# Run comprehensive test (default: 1 seed for quick test)
# Add --full-validation for 5 seeds
python scripts/phase1_comprehensive_test.py \
    --scenario "$SCENARIO" \
    --epochs 30 \
    --device cuda \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Scenario $SCENARIO completed"
else
    echo "FAILED: Scenario $SCENARIO failed"
fi

exit $EXIT_CODE
