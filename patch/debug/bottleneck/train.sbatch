#!/bin/bash
#SBATCH --job-name=dye_bottleneck
#SBATCH --partition=general
#SBATCH --array=0-5
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-m5-32,babel-m9-16,babel-m9-20,babel-n9-20
#SBATCH --time=4:00:00
#SBATCH --output=/home/kdoherty/logs/dye_patch/debug/bottleneck/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/dye_patch/debug/bottleneck/%A_%a.err

# 6 jobs: 2 configs (baseline, bottleneck_32) × 3 seeds. 50 epochs.

source ~/.bashrc
mamba activate dye_patch

export HF_HOME=/data/hf_cache
export HF_DATASETS_OFFLINE=1
export TMPDIR=/scratch/kdoherty && mkdir -p $TMPDIR

mkdir -p /home/kdoherty/logs/dye_patch/debug/bottleneck

python -u -m patch.debug.bottleneck.train --idx $SLURM_ARRAY_TASK_ID
