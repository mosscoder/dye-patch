#!/bin/bash
#SBATCH --job-name=dye_epoch_full
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --array=0-19
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/home/kdoherty/logs/dye_patch/tuning/epochs_full/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/dye_patch/tuning/epochs_full/%A_%a.err

# 20 jobs = 4 configs × 5 folds (full-data epoch tuning for data_source.py)
# Index: divmod(idx, 5) → (config, fold)

source ~/.bashrc
mamba activate dye_patch

export HF_HOME=/data/hf_cache
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TMPDIR=/scratch/kdoherty && mkdir -p $TMPDIR

mkdir -p /home/kdoherty/logs/dye_patch/tuning/epochs_full

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"

python -u -m patch.tuning.sweep_epochs --mode full --idx $SLURM_ARRAY_TASK_ID
