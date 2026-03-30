#!/bin/bash
#SBATCH --job-name=dye_temporal
#SBATCH --partition=general
#SBATCH --array=0-11
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-m5-32,babel-m9-16,babel-m9-20,babel-n9-20
#SBATCH --time=4:00:00
#SBATCH --output=/home/kdoherty/logs/dye_patch/eval/temporal/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/dye_patch/eval/temporal/%A_%a.err

# 12 jobs = 4 configs × 3 months, seed=0
# Index mapping: divmod(idx, 3) → (config, month)

source ~/.bashrc
mamba activate dye_patch

export HF_HOME=/data/hf_cache
export HF_DATASETS_OFFLINE=1
export TMPDIR=/scratch/kdoherty && mkdir -p $TMPDIR

mkdir -p /home/kdoherty/logs/dye_patch/eval/temporal

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"

python -u -m patch.eval.temporal_holdout --idx $SLURM_ARRAY_TASK_ID
