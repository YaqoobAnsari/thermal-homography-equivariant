# SLURM & GPU Guide for Spartan HPC

This document provides comprehensive guidance for running thermal-homography experiments on the Spartan HPC cluster at the University of Melbourne.

## Table of Contents

1. [Cluster Overview](#cluster-overview)
2. [Available GPU Partitions](#available-gpu-partitions)
3. [Environment Setup](#environment-setup)
4. [Job Submission](#job-submission)
5. [Job Scripts](#job-scripts)
6. [Monitoring Jobs](#monitoring-jobs)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Cluster Overview

**Cluster Name:** Spartan
**Institution:** University of Melbourne
**Operating System:** Red Hat Enterprise Linux 9.6
**Scheduler:** SLURM (Simple Linux Utility for Resource Management)
**Storage:** GPFS at `/data/gpfs/projects/`

### Key Paths
```bash
# Project directory
/data/gpfs/projects/punim2769/thermal-homography

# Conda environments
/home/$USER/.conda/envs/thermal-homography

# Shared modules
/apps/easybuild-2022/easybuild/modules/all/Core
```

---

## Available GPU Partitions

### Production Partitions

| Partition | GPU Type | GPUs/Node | Nodes | Max Time | Memory/Node | Recommended For |
|-----------|----------|-----------|-------|----------|-------------|-----------------|
| `gpu-a100` | NVIDIA A100 80GB | 4 | 27 | 7 days | 500 GB | Full training, large batches |
| `gpu-a100-short` | NVIDIA A100 80GB | 4 | 2 | **4 hours** | 500 GB | Quick experiments, debugging |
| `gpu-h100` | NVIDIA H100 80GB | 4 | 16 | 7 days | 1 TB | Fastest training, large models |
| `gpu-l40s` | NVIDIA L40S 48GB | 4 | 10 | 7 days | 1 TB | Inference, medium training |

### User Quotas (punim2769)

**Per-User Limits on gpu-a100:**
| Resource | Limit | Notes |
|----------|-------|-------|
| Max Submit Jobs | 200 | Total jobs in queue |
| Max GPUs Concurrent | **48** | 12 nodes × 4 GPUs |
| Max CPUs Concurrent | 384 | ~8 CPUs per GPU |
| Max Memory Concurrent | 5.9 TB | Across all jobs |

**Account Status:**
- **FairShare:** 1.0 (full priority, no prior usage)
- **Storage:** 467 GB total, ~410 GB available
- **Default per GPU:** 1 CPU, 4 GB memory (increase as needed)

### Preemptible Partitions (Lower Priority, May Be Interrupted)

| Partition | GPU Type | Nodes | Notes |
|-----------|----------|-------|-------|
| `gpu-a100-preempt` | A100 | 22 | Can be preempted by higher-priority jobs |
| `gpu-l40s-preempt` | L40S | 12 | Good for exploratory work |

### MIG Partition (Multi-Instance GPU)

| Partition | GPU Slices | Use Case |
|-----------|------------|----------|
| `gpu-a100-mig` | 1g.20gb, 1g.10gb | Smaller jobs, inference |

### GPU Specifications

| GPU | VRAM | FP32 | TF32 | FP16/BF16 | Architecture |
|-----|------|------|------|-----------|--------------|
| A100 | 80 GB HBM2e | 19.5 TFLOPS | 156 TFLOPS | 312 TFLOPS | Ampere |
| H100 | 80 GB HBM3 | 51 TFLOPS | 756 TFLOPS | 1513 TFLOPS | Hopper |
| L40S | 48 GB GDDR6 | 91.6 TFLOPS | 183 TFLOPS | 362 TFLOPS | Ada Lovelace |

### Billing Weights (Important for Planning)

| Resource | Weight | Notes |
|----------|--------|-------|
| GPU | 100 | Primary billing factor |
| CPU | 1 | Minimal impact |
| Memory | 0.125/GB | ~8 GB = 1 CPU equivalent |

**Cost-effective strategy:** Request minimal CPUs (8 per GPU) and memory (32-64GB) unless needed.

---

## Environment Setup

### 1. Load Anaconda Module

```bash
module load Anaconda3/2024.02-1
```

### 2. Activate Environment

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate thermal-homography
```

### 3. One-Line Activation (for scripts)

```bash
module load Anaconda3/2024.02-1 && source $(conda info --base)/etc/profile.d/conda.sh && conda activate thermal-homography
```

### 4. Add to ~/.bashrc (Optional)

```bash
# Thermal Homography environment alias
alias th-env='module load Anaconda3/2024.02-1 && source $(conda info --base)/etc/profile.d/conda.sh && conda activate thermal-homography'
```

---

## Job Submission

### Basic SLURM Commands

| Command | Description |
|---------|-------------|
| `sbatch script.sh` | Submit a batch job |
| `squeue --me` | View your jobs |
| `squeue -j <job_id>` | View specific job |
| `scancel <job_id>` | Cancel a job |
| `sinfo -p gpu-a100` | View partition info |
| `sacct -j <job_id>` | View job accounting |

### Interactive GPU Session

```bash
# Quick 1-hour session on A100
srun --partition=gpu-a100-short --gres=gpu:1 --mem=16G --time=1:00:00 --pty bash

# Longer 4-hour session
srun --partition=gpu-a100-short --gres=gpu:1 --mem=32G --cpus-per-task=8 --time=4:00:00 --pty bash

# H100 session (faster)
srun --partition=gpu-h100 --gres=gpu:1 --mem=32G --time=2:00:00 --pty bash
```

### Submit Training Job

```bash
# Phase 1: Synthetic validation
sbatch scripts/slurm_train.sh

# Phase 2: Hyperparameter search
sbatch scripts/slurm_train.sh phase2_search

# Phase 3: Full training
sbatch scripts/slurm_train.sh phase3_full
```

---

## Job Scripts

### Basic Training Script

```bash
#!/bin/bash
#SBATCH --job-name=thermal-train
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Load environment
module load Anaconda3/2024.02-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate thermal-homography

# Set environment variables
export THERMAL_DATA_DIR=/data/gpfs/projects/punim2769/thermal-homography/data
export THERMAL_CHECKPOINTS_DIR=/data/gpfs/projects/punim2769/thermal-homography/checkpoints
export THERMAL_LOGS_DIR=/data/gpfs/projects/punim2769/thermal-homography/logs

# Run training
cd /data/gpfs/projects/punim2769/thermal-homography
python -m src.training.train --config-name=phase1_synthetic
```

### Multi-GPU Training Script

```bash
#!/bin/bash
#SBATCH --job-name=thermal-multigpu
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

module load Anaconda3/2024.02-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate thermal-homography

cd /data/gpfs/projects/punim2769/thermal-homography

# Multi-GPU with PyTorch Lightning
python -m src.training.train \
    --config-name=phase3_full \
    training.devices=4 \
    training.strategy=ddp
```

### Job Array for Hyperparameter Search

```bash
#!/bin/bash
#SBATCH --job-name=thermal-sweep
#SBATCH --partition=gpu-a100
#SBATCH --array=1-10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err

module load Anaconda3/2024.02-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate thermal-homography

cd /data/gpfs/projects/punim2769/thermal-homography

# Use array task ID for different seeds/configs
python scripts/run_multi_seed.py --seed $SLURM_ARRAY_TASK_ID
```

### Dependency Chain (Sequential Jobs)

```bash
# Submit first job
JOB1=$(sbatch --parsable scripts/slurm_train.sh phase1_synthetic)
echo "Submitted phase 1: $JOB1"

# Submit second job after first completes successfully
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/slurm_train.sh phase2_search)
echo "Submitted phase 2: $JOB2 (depends on $JOB1)"

# Submit third job after second
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 scripts/slurm_train.sh phase3_full)
echo "Submitted phase 3: $JOB3 (depends on $JOB2)"
```

---

## Monitoring Jobs

### Real-Time Monitoring

```bash
# Watch your job queue
watch -n 5 squeue --me

# View job output in real-time
tail -f logs/slurm_<job_id>.out

# Check GPU utilization (on compute node)
nvidia-smi -l 1

# Detailed job info
scontrol show job <job_id>
```

### Job Efficiency Check

```bash
# After job completes
seff <job_id>

# Detailed accounting
sacct -j <job_id> --format=JobID,JobName,Partition,MaxRSS,MaxVMSize,Elapsed,State
```

### Check GPU Availability

```bash
# See which nodes have free GPUs
sinfo -p gpu-a100 -N -l

# Check specific partition
squeue -p gpu-a100 --format="%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
```

---

## Best Practices

### 1. Resource Requests

```bash
# Start conservative, scale up if needed
#SBATCH --mem=32G          # Start with 32GB, increase if OOM
#SBATCH --cpus-per-task=8  # 8 CPUs per GPU is usually enough
#SBATCH --time=24:00:00    # Estimate conservatively

# Request only needed GPUs
#SBATCH --gres=gpu:1       # Single GPU for most experiments
```

### 2. Output Management

```bash
# Separate stdout and stderr
#SBATCH --output=logs/%x_%j.out  # %x=job name, %j=job id
#SBATCH --error=logs/%x_%j.err

# Or combined
#SBATCH --output=logs/%x_%j.log
```

### 3. Email Notifications

```bash
#SBATCH --mail-user=your.email@unimelb.edu.au
#SBATCH --mail-type=BEGIN,END,FAIL
```

### 4. Checkpointing

Always save checkpoints frequently:
```python
# In your training config
trainer:
  callbacks:
    - class_path: pytorch_lightning.callbacks.ModelCheckpoint
      init_args:
        save_top_k: 3
        every_n_epochs: 1
        save_last: true
```

### 5. Environment Variables

Set in job script:
```bash
# Reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42

# Performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# NCCL for multi-GPU (if needed)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # If InfiniBand issues
```

---

## Troubleshooting

### Common Issues

#### 1. Job Pending Forever
```bash
# Check reason
squeue -j <job_id> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Common reasons:
# - Resources: Requested resources not available
# - Priority: Lower priority than other jobs
# - QOSMaxJobsPerUserLimit: Too many jobs running
```

#### 2. Out of Memory (OOM)
```bash
# Check memory usage after job
sacct -j <job_id> --format=MaxRSS,MaxVMSize

# Solutions:
# - Increase --mem
# - Reduce batch_size
# - Enable gradient checkpointing
# - Use mixed precision (fp16/bf16)
```

#### 3. CUDA Out of Memory
```python
# In training config
training:
  batch_size: 4  # Reduce from 8
  precision: 16  # Use mixed precision
  accumulate_grad_batches: 2  # Effective batch = 8
```

#### 4. Module Not Found
```bash
# Ensure environment is activated in job script
module load Anaconda3/2024.02-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate thermal-homography

# Verify
which python
python -c "import torch; print(torch.__version__)"
```

#### 5. GPU Not Detected
```bash
# Check in interactive session
srun --partition=gpu-a100-short --gres=gpu:1 --time=0:10:00 --pty bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Useful Debug Commands

```bash
# Check node features
scontrol show node spartan-gpgpu101

# Check partition limits
scontrol show partition gpu-a100

# View your recent job history
sacct --starttime=2026-01-01 --format=JobID,JobName,Partition,State,ExitCode,Elapsed

# Cancel all your pending jobs
scancel --state=PENDING --user=$USER
```

---

## Quick Reference Card

```bash
# Activate environment
module load Anaconda3/2024.02-1 && conda activate thermal-homography

# Interactive GPU session
srun -p gpu-a100-short --gres=gpu:1 --mem=16G -t 1:00:00 --pty bash

# Submit training job
sbatch scripts/slurm_train.sh phase1_synthetic

# Check your jobs
squeue --me

# Cancel job
scancel <job_id>

# View job output
tail -f logs/slurm_<job_id>.out

# Check job efficiency
seff <job_id>
```

---

## Contact & Resources

- **Spartan Documentation:** https://dashboard.hpc.unimelb.edu.au/
- **Help Desk:** hpc-support@unimelb.edu.au
- **Job Script Generator:** https://dashboard.hpc.unimelb.edu.au/job_submit/

---

*Last updated: 2026-01-30*
