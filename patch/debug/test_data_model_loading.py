"""Minimal test: can we load cached HF datasets and model on a compute node?"""

import os

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
        print("  HF_HOME not set, skipping")
        return

    # Inspect dye_patch hub cache (2 levels deep)
    hub_ds = os.path.join(hf_home, "hub", "datasets--mpg-ranch--dye_patch")
    print(f"\n--- dye_patch hub cache ---")
    print(f"  {hub_ds} exists: {os.path.isdir(hub_ds)}")

    # Find the snapshot dir
    snapshots_dir = os.path.join(hub_ds, "snapshots")
    if os.path.isdir(snapshots_dir):
        for snap in os.listdir(snapshots_dir):
            snap_path = os.path.join(snapshots_dir, snap)
            print(f"\n  snapshot/{snap[:12]}...:")
            for entry in sorted(os.listdir(snap_path)):
                entry_path = os.path.join(snap_path, entry)
                if os.path.isdir(entry_path):
                    children = sorted(os.listdir(entry_path))
                    # Show file sizes for first few
                    details = []
                    for c in children[:5]:
                        cp = os.path.join(entry_path, c)
                        sz = os.path.getsize(cp) if os.path.isfile(cp) else "dir"
                        details.append(f"{c} ({sz})")
                    print(f"    {entry}/ [{len(children)} files]: {details}")
                else:
                    sz = os.path.getsize(entry_path)
                    print(f"    {entry} ({sz} bytes)")

    # Compare with wolverines cache
    wol_ds = os.path.join(hf_home, "hub", "datasets--kdoherty--wolverines")
    print(f"\n--- wolverines hub cache (for comparison) ---")
    print(f"  {wol_ds} exists: {os.path.isdir(wol_ds)}")
    wol_snaps = os.path.join(wol_ds, "snapshots")
    if os.path.isdir(wol_snaps):
        for snap in os.listdir(wol_snaps):
            snap_path = os.path.join(wol_snaps, snap)
            entries = sorted(os.listdir(snap_path))
            print(f"  snapshot/{snap[:12]}...: {entries[:10]}")
            # Show first config dir if it exists
            for entry in entries[:3]:
                ep = os.path.join(snap_path, entry)
                if os.path.isdir(ep):
                    children = sorted(os.listdir(ep))[:5]
                    details = []
                    for c in children:
                        cp = os.path.join(ep, c)
                        sz = os.path.getsize(cp) if os.path.isfile(cp) else "dir"
                        details.append(f"{c} ({sz})")
                    print(f"    {entry}/ : {details}")


def test_load():
    """Try loading sprayed/train with various approaches."""
    from datasets import load_dataset

    # Test 1: offline flag set
    print("\n--- Test 1: HF_DATASETS_OFFLINE=1 ---")
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        ds = load_dataset(HF_REPO, TEST_CONFIG, split=TEST_SPLIT)
        print(f"  OK   {len(ds)} rows")
    except Exception as e:
        print(f"  FAIL {type(e).__name__}: {e}")

    # Test 2: offline flag unset
    print("\n--- Test 2: HF_DATASETS_OFFLINE unset ---")
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        ds = load_dataset(HF_REPO, TEST_CONFIG, split=TEST_SPLIT)
        print(f"  OK   {len(ds)} rows")
    except Exception as e:
        print(f"  FAIL {type(e).__name__}: {e}")

    # Test 3: load with explicit data_dir pointing to snapshot
    print("\n--- Test 3: explicit data_dir from snapshot ---")
    hf_home = os.environ.get("HF_HOME", "")
    snap_dir = os.path.join(hf_home, "hub", "datasets--mpg-ranch--dye_patch", "snapshots")
    if os.path.isdir(snap_dir):
        snaps = os.listdir(snap_dir)
        if snaps:
            data_dir = os.path.join(snap_dir, snaps[0])
            print(f"  data_dir={data_dir}")
            try:
                ds = load_dataset(HF_REPO, TEST_CONFIG, split=TEST_SPLIT, data_dir=data_dir)
                print(f"  OK   {len(ds)} rows")
            except Exception as e:
                print(f"  FAIL {type(e).__name__}: {e}")

    # Test 4: load wolverines for comparison
    print("\n--- Test 4: wolverines (known working) ---")
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        ds = load_dataset("kdoherty/wolverines", "reidentification", split="train")
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
    inspect_cache()
    test_load()
    test_model()
