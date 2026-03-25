"""Minimal test: can we load cached HF datasets and model on a compute node?"""

import os
import subprocess
import sys

from patch.utils.config import MODEL_NAME

HF_REPO = "mpg-ranch/dye_patch"
TEST_CONFIG = "sprayed"
TEST_SPLIT = "train"


def print_env():
    import datasets
    import huggingface_hub

    print(f"HF_HOME={os.environ.get('HF_HOME', '(not set)')}")
    print(f"HF_DATASETS_OFFLINE={os.environ.get('HF_DATASETS_OFFLINE', '(not set)')}")
    print(f"datasets=={datasets.__version__}")
    print(f"huggingface_hub=={huggingface_hub.__version__}")


def inspect_cache():
    hf_home = os.environ.get("HF_HOME", "")

    # What's in the datasets processed cache?
    ds_cache = os.path.join(hf_home, "datasets")
    print(f"\n--- Processed cache: {ds_cache} ---")
    if os.path.isdir(ds_cache):
        entries = sorted(os.listdir(ds_cache))
        for e in entries:
            ep = os.path.join(ds_cache, e)
            if os.path.isdir(ep):
                children = sorted(os.listdir(ep))[:5]
                print(f"  {e}/ : {children}")
            else:
                print(f"  {e} ({os.path.getsize(ep)} bytes)")
    else:
        print("  DOES NOT EXIST")

    # What does wolverines processed cache look like?
    print(f"\n--- Wolverines processed cache (for comparison) ---")
    if os.path.isdir(ds_cache):
        wol = [e for e in os.listdir(ds_cache) if "wolverine" in e.lower()]
        for e in wol:
            ep = os.path.join(ds_cache, e)
            if os.path.isdir(ep):
                children = sorted(os.listdir(ep))[:5]
                print(f"  {e}/ : {children}")


def test_in_subprocess(label, env_overrides, load_code):
    """Run a load_dataset test in a fresh subprocess to avoid cached env vars."""
    env = os.environ.copy()
    env.update(env_overrides)
    # Remove keys set to None
    for k, v in list(env.items()):
        if v is None:
            del env[k]

    code = f"""
import sys
try:
    {load_code}
    print("OK " + str(len(ds)) + " rows")
except Exception as e:
    print(f"FAIL {{type(e).__name__}}: {{e}}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env, capture_output=True, text=True, timeout=60,
    )
    output = (result.stdout + result.stderr).strip().split("\n")[-1]
    print(f"  {label}: {output}")


def test_load():
    hf_home = os.environ.get("HF_HOME", "")

    # Test 1: offline flag (current sbatch setup)
    test_in_subprocess(
        "HF_DATASETS_OFFLINE=1",
        {"HF_DATASETS_OFFLINE": "1"},
        f'from datasets import load_dataset; ds = load_dataset("{HF_REPO}", "{TEST_CONFIG}", split="{TEST_SPLIT}")',
    )

    # Test 2: no offline flag (fresh subprocess, clean env)
    env2 = {}
    if "HF_DATASETS_OFFLINE" in os.environ:
        env2["HF_DATASETS_OFFLINE"] = None  # signals removal
    if "HF_HUB_OFFLINE" in os.environ:
        env2["HF_HUB_OFFLINE"] = None
    # Build clean env for subprocess
    clean_env = os.environ.copy()
    clean_env.pop("HF_DATASETS_OFFLINE", None)
    clean_env.pop("HF_HUB_OFFLINE", None)
    code2 = f"""
import sys
try:
    from datasets import load_dataset
    ds = load_dataset("{HF_REPO}", "{TEST_CONFIG}", split="{TEST_SPLIT}")
    print("OK " + str(len(ds)) + " rows")
except Exception as e:
    print(f"FAIL {{type(e).__name__}}: {{e}}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code2],
        env=clean_env, capture_output=True, text=True, timeout=120,
    )
    output = (result.stdout + result.stderr).strip().split("\n")[-1]
    print(f"  No offline flags: {output}")

    # Test 3: wolverines with offline flag
    test_in_subprocess(
        "wolverines/pelage offline",
        {"HF_DATASETS_OFFLINE": "1"},
        'from datasets import load_dataset; ds = load_dataset("kdoherty/wolverines", "pelage", split="train")',
    )


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

    print("\n--- Load tests (subprocesses for clean env) ---")
    test_load()
    test_model()
