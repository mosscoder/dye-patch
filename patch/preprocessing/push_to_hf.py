"""
Build and push the patch-level HuggingFace dataset.

Three configs:
  - sprayed: train/test splits (70/30 stratified by point, color/concentration/size)
  - unsprayed_annex: single split (eastern block tiles)
  - offsite: single split (European grassland tiles)

Usage:
  load_dataset("mpg-ranch/dye_patch", "sprayed", split="train")
  load_dataset("mpg-ranch/dye_patch", "unsprayed_annex")
  load_dataset("mpg-ranch/dye_patch", "offsite")
"""

import argparse
import os

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image, Value

from patch.utils.config import TUNING_TEST_FRAC
from patch.utils.dataset import stratified_split


MPG_TILE_DIR = "data/images/mpg_ranch/tiles"
OFFSITE_TILE_DIR = "data/images/european_grassland/tiles"
HF_REPO = "mpg-ranch/dye_patch"

FEATURES = Features({
    "image": Image(),
    "tile_type": Value("string"),
    "month": Value("string"),
    "color": Value("string"),
    "concentration": Value("string"),
    "spray_size_m": Value("float32"),
    "latitude": Value("float64"),
    "longitude": Value("float64"),
    "point_name": Value("string"),
})


def _load_mpg_manifest(tile_dir: str = MPG_TILE_DIR) -> pd.DataFrame:
    """Load MPG Ranch tile manifest."""
    manifest_path = os.path.join(tile_dir, "manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"MPG Ranch manifest not found at {manifest_path}")
    return pd.read_csv(manifest_path)


def _load_offsite_manifest(tile_dir: str = OFFSITE_TILE_DIR) -> pd.DataFrame:
    """Load European grassland tile manifest."""
    manifest_path = os.path.join(tile_dir, "offsite_manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Offsite manifest not found at {manifest_path}")
    return pd.read_csv(manifest_path)


def _manifest_to_dataset(manifest: pd.DataFrame) -> Dataset:
    """Convert manifest DataFrame to HuggingFace Dataset."""
    records = []
    for _, row in manifest.iterrows():
        tile_path = row["tile_path"]
        if not os.path.exists(tile_path):
            print(f"Warning: missing tile {tile_path}, skipping")
            continue

        records.append({
            "image": tile_path,
            "tile_type": str(row.get("tile_type", "unknown")),
            "month": str(row.get("month", "none")),
            "color": str(row.get("color", "none")),
            "concentration": str(row.get("concentration", "none")),
            "spray_size_m": float(row.get("spray_size_m", 0.0)),
            "latitude": float(row.get("latitude", 0.0)),
            "longitude": float(row.get("longitude", 0.0)),
            "point_name": str(row.get("point_name", "")),
        })

    if not records:
        raise ValueError("No valid tiles found")

    return Dataset.from_dict(
        {k: [r[k] for r in records] for k in records[0]},
        features=FEATURES,
    )


def push(
    mpg_tile_dir: str = MPG_TILE_DIR,
    offsite_tile_dir: str = OFFSITE_TILE_DIR,
    repo: str = HF_REPO,
    seed: int = 0,
):
    """Build three dataset configs and push to HuggingFace."""
    mpg_manifest = _load_mpg_manifest(mpg_tile_dir)

    # Split MPG Ranch into sprayed and unsprayed_annex
    sprayed_df = mpg_manifest[mpg_manifest["tile_type"] == "sprayed"]
    annex_df = mpg_manifest[mpg_manifest["tile_type"] == "eastern_block"]

    print(f"Sprayed tiles: {len(sprayed_df)}")
    print(f"Unsprayed annex tiles: {len(annex_df)}")

    # Sprayed: 70/30 stratified train/test split (point-level)
    sprayed_ds = _manifest_to_dataset(sprayed_df)
    sprayed_train, sprayed_test = stratified_split(sprayed_ds, test_frac=TUNING_TEST_FRAC, seed=seed)
    print(f"Sprayed train: {len(sprayed_train)}, test: {len(sprayed_test)}")

    sprayed_train.push_to_hub(repo, config_name="sprayed", split="train")
    sprayed_test.push_to_hub(repo, config_name="sprayed", split="test")
    print(f"Pushed sprayed config to {repo}")

    # Unsprayed annex: single split
    if len(annex_df) > 0:
        annex_ds = _manifest_to_dataset(annex_df)
        annex_ds.push_to_hub(repo, config_name="unsprayed_annex", split="train")
        print(f"Pushed unsprayed_annex config to {repo} ({len(annex_ds)} tiles)")
    else:
        print("Warning: no eastern block tiles found, skipping unsprayed_annex config")

    # Offsite: single split (European grassland)
    try:
        offsite_manifest = _load_offsite_manifest(offsite_tile_dir)
        offsite_ds = _manifest_to_dataset(offsite_manifest)
        offsite_ds.push_to_hub(repo, config_name="offsite", split="train")
        print(f"Pushed offsite config to {repo} ({len(offsite_ds)} tiles)")
    except FileNotFoundError:
        print("Warning: no offsite manifest found, skipping offsite config")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push patch dataset to HuggingFace")
    parser.add_argument("--mpg-tile-dir", default=MPG_TILE_DIR)
    parser.add_argument("--offsite-tile-dir", default=OFFSITE_TILE_DIR)
    parser.add_argument("--repo", default=HF_REPO)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    push(args.mpg_tile_dir, args.offsite_tile_dir, args.repo, args.seed)
