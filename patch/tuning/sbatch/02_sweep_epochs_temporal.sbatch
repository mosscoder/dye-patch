#!/bin/bash
#SBATCH --job-name=dye_epoch_temporal
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --array=0-59
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --exclude=babel-l5-[16,20,24],babel-y5-16,babel-x9-24
#SBATCH --output=/home/kdoherty/logs/dye_patch/tuning/epochs_temporal/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/dye_patch/tuning/epochs_temporal/%A_%a.err

# 60 jobs = 4 configs × 3 months × 5 folds (single-month epoch tuning for temporal_holdout.py)
# Index: divmod(idx, 5) → (config_month, fold)
#        divmod(config_month, 3) → (config, month)

source ~/.bashrc
mamba activate dye_patch

export HF_HOME=/data/hf_cache
export TMPDIR=/scratch/kdoherty && mkdir -p $TMPDIR

mkdir -p /home/kdoherty/logs/dye_patch/tuning/epochs_temporal

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"

python -u -m patch.tuning.sweep_epochs --mode temporal --idx $SLURM_ARRAY_TASK_ID
