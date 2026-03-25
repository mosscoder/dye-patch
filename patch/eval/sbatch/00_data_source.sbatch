#!/bin/bash
#SBATCH --job-name=dye_data_source
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --array=0-3
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --exclude=babel-l5-[16,20,24],babel-y5-16,babel-x9-24
#SBATCH --output=/home/kdoherty/logs/dye_patch/eval/data_source/%A_%a.out
#SBATCH --error=/home/kdoherty/logs/dye_patch/eval/data_source/%A_%a.err

# 4 jobs = 4 configs, seed=0

source ~/.bashrc
mamba activate dye_patch

export HF_HOME=/data/hf_cache
export TMPDIR=/scratch/kdoherty && mkdir -p $TMPDIR

mkdir -p /home/kdoherty/logs/dye_patch/eval/data_source

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"

python -u -m patch.eval.data_source --idx $SLURM_ARRAY_TASK_ID
