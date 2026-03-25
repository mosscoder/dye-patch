"""
Figure: 2-panel — concentration and size factors.

Uses per-tile predictions from data_source.py (defaults to real_only config).
Panel 1: F1 by concentration (standard vs high).
Panel 2: F1 by size (0.1m vs 0.5m).
Both pooled across months.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from patch.utils.config import EVAL_CROP_OFFSET
from patch.utils.dataset import generate_patch_labels

RESULTS_DIR = "patch/eval/results/data_source"
OUTPUT_DIR = "patch/visualizations/figures"


def compute_f1_by_factor(results: dict, factor_key: str) -> dict:
    """Compute spray-level F1 grouped by a factor (concentration or size)."""
    center_offset = EVAL_CROP_OFFSET
    groups = {}

    for tile in results.get("per_tile", []):
        if tile.get("tile_type") != "sprayed":
            continue

        color = tile.get("color", "none")
        if color == "none":
            continue

        factor_val = str(tile.get(factor_key, "unknown"))
        if factor_val not in groups:
            groups[factor_val] = {"tp": 0, "fn": 0}

        pred = np.array(tile["pred"])
        mask = generate_patch_labels(
            crop_offset=(center_offset, center_offset),
            spray_size_m=tile.get("spray_size_m", 0.0),
            spray_color=color,
        )
        spray_patches = mask > 0
        if spray_patches.any() and (pred[spray_patches] > 0).any():
            groups[factor_val]["tp"] += 1
        else:
            groups[factor_val]["fn"] += 1

    f1_by_group = {}
    for val, counts in groups.items():
        tp, fn = counts["tp"], counts["fn"]
        recall = tp / max(tp + fn, 1)
        f1_by_group[val] = recall

    return f1_by_group


def plot(config: str = "real_only", output_path: str = None):
    """Generate 2-panel figure: concentration + size."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    path = os.path.join(RESULTS_DIR, f"{config}.json")
    if not os.path.exists(path):
        print(f"No results for {config}")
        return

    with open(path) as f:
        results = json.load(f)

    conc_f1 = compute_f1_by_factor(results, "concentration")
    size_f1 = compute_f1_by_factor(results, "spray_size_m")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Panel 1: Concentration
    conc_labels = sorted(conc_f1.keys())
    conc_vals = [conc_f1[k] for k in conc_labels]
    ax1.bar(conc_labels, conc_vals, color=["#2ca02c", "#98df8a"])
    ax1.set_ylabel("Spray-level F1")
    ax1.set_xlabel("Concentration")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("By concentration")

    # Panel 2: Size
    size_labels = sorted(size_f1.keys())
    size_vals = [size_f1[k] for k in size_labels]
    ax2.bar(size_labels, size_vals, color=["#ff7f0e", "#ffbb78"])
    ax2.set_ylabel("Spray-level F1")
    ax2.set_xlabel("Spray size (m)")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("By spray size")

    fig.suptitle(f"Factor analysis ({config})", fontsize=13)
    plt.tight_layout()
    out = output_path or os.path.join(OUTPUT_DIR, f"factors_{config}.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="real_only")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    plot(args.config, args.output)
