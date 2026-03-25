"""Minimal test: can we load cached HF datasets and model on a compute node?"""

import os

from patch.utils.config import MODEL_NAME

HF_REPO = "mpg-ranch/dye-patch"
CONFIGS_SPLITS = [
    ("sprayed", "train"),
    ("sprayed", "test"),
    ("unsprayed_annex", "train"),
    ("offsite", "train"),
]


def print_env():
    import datasets
    import huggingface_hub

    print(f"HF_HOME={os.environ.get('HF_HOME', '(not set)')}")
    print(f"HF_DATASETS_OFFLINE={os.environ.get('HF_DATASETS_OFFLINE', '(not set)')}")
    print(f"datasets=={datasets.__version__}")
    print(f"huggingface_hub=={huggingface_hub.__version__}")


def test_datasets():
    from datasets import load_dataset

    print("\n--- Datasets ---")
    for config, split in CONFIGS_SPLITS:
        try:
            ds = load_dataset(HF_REPO, config, split=split)
            print(f"  OK   {config}/{split}: {len(ds)} rows")
        except Exception as e:
            print(f"  FAIL {config}/{split}: {type(e).__name__}: {e}")


def test_model():
    print("\n--- Model ---")
    from transformers import AutoModel

    try:
        model = AutoModel.from_pretrained(MODEL_NAME)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  OK   {MODEL_NAME} ({n_params:.0f}M params)")
    except Exception as e:
        print(f"  FAIL {e}")


if __name__ == "__main__":
    print_env()
    test_datasets()
    test_model()
