#!/bin/bash
#SBATCH --job-name=dye_lr_sweep
#SBATCH --partition=general
#SBATCH --array=0-24
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/home/kdoherty/logs/dye_patch/tuning/lr/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/dye_patch/tuning/lr/%A_%a.err

# 20 jobs = 4 LRs × 5 seeds (real_only data, shared across all configs)
# Index mapping: divmod(idx, 5) → (lr_idx, seed)

source ~/.bashrc
mamba activate dye_patch

export HF_HOME=/data/hf_cache
export HF_DATASETS_OFFLINE=1
export TMPDIR=/scratch/kdoherty && mkdir -p $TMPDIR

mkdir -p /home/kdoherty/logs/dye_patch/tuning/lr

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"

python -u -m patch.tuning.sweep_lr --idx $SLURM_ARRAY_TASK_ID
