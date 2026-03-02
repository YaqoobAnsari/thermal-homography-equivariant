#!/bin/bash
#SBATCH --job-name=mscoco
#SBATCH --output=logs/mscoco_%j.out
#SBATCH --error=logs/mscoco_%j.err
#SBATCH --time=3:00:00
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# =============================================================================
# MS-COCO BENCHMARK EVALUATION
# Full evaluation: standard + rotation sweep + scale sweep + Sim(2) grid
# =============================================================================

echo "=========================================="
echo "MS-COCO BENCHMARK EVALUATION"
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

mkdir -p logs outputs/mscoco

nvidia-smi

echo ""
echo "Running MS-COCO evaluation (all modes)..."
python scripts/evaluate_mscoco.py \
    --mode all \
    --coco-root data/coco/val2017 \
    --model-type log_polar_sim2 \
    --device cuda \
    --n-samples 1000 \
    --n-samples-per-config 100 \
    --output-dir outputs/mscoco

echo ""
echo "=========================================="
echo "MS-COCO COMPLETE: $(date)"
echo "Results in: outputs/mscoco/"
echo "=========================================="
