"""
Pre-cache model weights and HuggingFace datasets on the cluster.

Run once before submitting SLURM jobs. Downloads everything to the shared
HF_HOME location so compute nodes don't each download independently.

Datasets are saved to disk (arrow format) at HF_HOME/dye_patch/{config}/{split}
so they can be loaded offline with load_from_disk, bypassing Hub resolution
which is broken in datasets>=4 offline mode.

Usage (on cluster login node):
  HF_HOME=/data/hf_cache python -m patch.preprocessing.cache_model_and_data
"""

import os

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
    """Download all HF dataset configs and save to disk for offline loading."""
    from datasets import load_dataset

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    disk_root = os.path.join(hf_home, "dye_patch")

    for config in HF_CONFIGS:
        print(f"\nCaching dataset: {HF_REPO} / {config}")
        for split in ["train", "test"]:
            try:
                ds = load_dataset(HF_REPO, config, split=split)
                disk_path = os.path.join(disk_root, config, split)
                ds.save_to_disk(disk_path)
                print(f"  {config}/{split}: {len(ds)} rows → {disk_path}")
            except Exception:
                # Some configs only have "train" split
                pass
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
