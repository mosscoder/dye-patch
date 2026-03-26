#!/bin/bash
#SBATCH --job-name=dye_test_load
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00
#SBATCH --output=/home/kdoherty/logs/dye_patch/debug/%j.out
#SBATCH --error=/home/kdoherty/logs/dye_patch/debug/%j.err

# Minimal test: load cached datasets + model on a compute node.
# Matches wolverines env var pattern (no HF_HUB_OFFLINE).

source ~/.bashrc
mamba activate dye_patch

export HF_HOME=/data/hf_cache
export HF_DATASETS_OFFLINE=1
export TMPDIR=/scratch/kdoherty && mkdir -p $TMPDIR

mkdir -p /home/kdoherty/logs/dye_patch/debug

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

python -u -m patch.debug.test_data_model_loading
