"""
Pre-cache model weights and HuggingFace datasets on the cluster.

Run once before submitting SLURM jobs. Downloads everything to the shared
HF_HOME location so compute nodes don't each download independently.

Usage (on cluster login node):
  HF_HOME=/data/hf_cache python patch/preprocessing/cache_model_and_data.py

The same HF_HOME is set in all sbatch files, so cached files are found
automatically by downstream scripts.
"""

import os

# Ensure HF_HOME is set (sbatch files set this to /data/hf_cache)
hf_home = os.environ.get("HF_HOME")
if hf_home:
    print(f"HF_HOME={hf_home}")
else:
    print("WARNING: HF_HOME not set. Using default ~/.cache/huggingface/")

from patch.utils.config import MODEL_NAME

HF_REPO = "mpg-ranch/dye_patch"
HF_CONFIGS = ["sprayed", "unsprayed_annex", "offsite"]


def cache_model():
    """Download model weights to HF_HOME/hub/."""
    from transformers import AutoModel, AutoImageProcessor

    print(f"\nCaching model: {MODEL_NAME}")
    print("  Downloading image processor...")
    AutoImageProcessor.from_pretrained(MODEL_NAME)
    print("  Downloading model weights (this may take a while)...")
    AutoModel.from_pretrained(MODEL_NAME)
    print("  Model cached.")


def cache_datasets():
    """Download all HF dataset configs to HF_HOME/datasets/."""
    from datasets import load_dataset

    for config in HF_CONFIGS:
        print(f"\nCaching dataset: {HF_REPO} / {config}")
        for split in ["train", "test"]:
            try:
                ds = load_dataset(HF_REPO, config, split=split)
                print(f"  {config}/{split}: {len(ds)} rows")
            except Exception:
                # Some configs only have "train" split
                pass
    print("  Datasets cached.")


if __name__ == "__main__":
    cache_model()
    cache_datasets()
    print("\nAll cached. Ready for SLURM jobs.")
