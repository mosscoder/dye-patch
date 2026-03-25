"""Minimal test: can we load cached HF datasets and model on a compute node?"""

import os
import subprocess

from patch.utils.config import MODEL_NAME

HF_REPO = "mpg-ranch/dye_patch"
TEST_CONFIG = "sprayed"
TEST_SPLIT = "train"


def print_env():
    import datasets
    import huggingface_hub

    print(f"HF_HOME={os.environ.get('HF_HOME', '(not set)')}")
    print(f"HF_DATASETS_OFFLINE={os.environ.get('HF_DATASETS_OFFLINE', '(not set)')}")
    print(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '(not set)')}")
    print(f"datasets=={datasets.__version__}")
    print(f"huggingface_hub=={huggingface_hub.__version__}")


def inspect_cache():
    hf_home = os.environ.get("HF_HOME", "")
    if not hf_home:
        print("  HF_HOME not set, skipping cache inspection")
        return

    # Check hub cache (where dataset metadata lives)
    hub_ds = os.path.join(hf_home, "hub", "datasets--mpg-ranch--dye_patch")
    print(f"\n  Hub cache: {hub_ds}")
    print(f"    exists: {os.path.isdir(hub_ds)}")
    refs = os.path.join(hub_ds, "refs")
    if os.path.isdir(refs):
        for f in os.listdir(refs):
            content = open(os.path.join(refs, f)).read().strip()
            print(f"    refs/{f}: {content}")
    snapshots = os.path.join(hub_ds, "snapshots")
    if os.path.isdir(snapshots):
        for d in os.listdir(snapshots):
            files = os.listdir(os.path.join(snapshots, d))
            print(f"    snapshots/{d[:12]}...: {files}")

    # Check datasets cache
    ds_cache = os.path.join(hf_home, "datasets")
    print(f"\n  Datasets cache: {ds_cache}")
    print(f"    exists: {os.path.isdir(ds_cache)}")
    if os.path.isdir(ds_cache):
        entries = [e for e in os.listdir(ds_cache) if "dye" in e.lower() or "mpg" in e.lower()]
        for e in entries[:5]:
            print(f"    {e}")


def test_with_offline_flag():
    """Test 1: load with HF_DATASETS_OFFLINE=1 (current sbatch setup)."""
    print("\n--- Test 1: HF_DATASETS_OFFLINE=1 ---")
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset

    try:
        ds = load_dataset(HF_REPO, TEST_CONFIG, split=TEST_SPLIT)
        print(f"  OK   {len(ds)} rows")
    except Exception as e:
        print(f"  FAIL {type(e).__name__}: {e}")


def test_without_offline_flag():
    """Test 2: load WITHOUT offline flag (let Hub access fail naturally)."""
    print("\n--- Test 2: HF_DATASETS_OFFLINE unset ---")
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    os.environ.pop("HF_HUB_OFFLINE", None)

    # Need fresh import to pick up env change
    import importlib
    import datasets.config
    importlib.reload(datasets.config)

    from datasets import load_dataset

    try:
        ds = load_dataset(HF_REPO, TEST_CONFIG, split=TEST_SPLIT)
        print(f"  OK   {len(ds)} rows")
    except Exception as e:
        print(f"  FAIL {type(e).__name__}: {e}")


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

    print("\n--- Cache Inspection ---")
    inspect_cache()

    test_with_offline_flag()
    test_without_offline_flag()
    test_model()
