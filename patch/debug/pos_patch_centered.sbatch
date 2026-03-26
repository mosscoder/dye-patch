#!/bin/bash
#SBATCH --job-name=dye_centroid
#SBATCH --partition=general
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=babel-m5-32,babel-l5-16
#SBATCH --time=2:00:00
#SBATCH --output=/home/kdoherty/logs/dye_patch/debug/centroid/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/dye_patch/debug/centroid/%A_%a.err

# 8 jobs: 2 seeds × 4 configs (2x/5e-4, 10x/5e-4, all/5e-4, all/1e-3). Ternary, Large, CE, 50 epochs.

source ~/.bashrc
mamba activate dye_patch

export HF_HOME=/data/hf_cache
export HF_DATASETS_OFFLINE=1
export TMPDIR=/scratch/kdoherty && mkdir -p $TMPDIR

mkdir -p /home/kdoherty/logs/dye_patch/debug/centroid

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"

python -u -m patch.debug.pos_patch_centered --idx $SLURM_ARRAY_TASK_ID
