#!/bin/bash
#SBATCH --job-name=dye_loss_cmp
#SBATCH --partition=general
#SBATCH --array=0-5
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=/home/kdoherty/logs/dye_patch/debug/loss_fn/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/dye_patch/debug/loss_fn/%A_%a.err

# 6 jobs: 3 loss fns × 2 LRs (1e-3, 1e-4), seed=0, 15 epochs

source ~/.bashrc
mamba activate dye_patch

export HF_HOME=/data/hf_cache
export HF_DATASETS_OFFLINE=1
export TMPDIR=/scratch/kdoherty && mkdir -p $TMPDIR

mkdir -p /home/kdoherty/logs/dye_patch/debug/loss_fn

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"

python -u -m patch.debug.check_loss_fn --idx $SLURM_ARRAY_TASK_ID
