"""
Pre-cache model weights and HuggingFace datasets on the cluster.

Run once before submitting SLURM jobs. Downloads everything to the shared
HF_HOME location so compute nodes don't each download independently.

Usage (on cluster login node):
  HF_HOME=/data/hf_cache python -m patch.preprocessing.cache_model_and_data
"""

import os

from patch.utils.config import MODEL_NAME

HF_REPO = "mpg-ranch/dye_patch"
HF_CONFIGS_SPLITS = [
    ("sprayed", "train"),
    ("sprayed", "test"),
    ("unsprayed_annex", "train"),
    ("offsite", "train"),
]


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
    """Download all HF dataset configs. Cache location controlled by HF_HOME."""
    from datasets import load_dataset

    for config, split in HF_CONFIGS_SPLITS:
        print(f"\nCaching dataset: {HF_REPO} / {config} / {split}")
        ds = load_dataset(HF_REPO, config, split=split)
        print(f"  {config}/{split}: {len(ds)} rows")

    print("  Datasets cached.")


if __name__ == "__main__":
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        print(f"HF_HOME={hf_home}")
    else:
        print("WARNING: HF_HOME not set. Using default ~/.cache/huggingface/")

    cache_model()
    cache_datasets()
    print("\nAll cached. Ready for SLURM jobs.")
