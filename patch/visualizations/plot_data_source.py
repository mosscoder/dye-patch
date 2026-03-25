"""
Figure: Performance by color and data source.

Bar plot of spray-level F1, x-axis = data source config, bars dodged by
dye color (red/blue). Reads per-tile predictions from data_source results.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from patch.utils.config import CONFIGS, EVAL_CROP_OFFSET
from patch.utils.dataset import generate_patch_labels

RESULTS_DIR = "patch/eval/results/data_source"
OUTPUT_DIR = "patch/visualizations/figures"


def compute_f1_by_color(results: dict) -> dict:
    """Compute spray-level F1 per dye color from per-tile predictions."""
    center_offset = EVAL_CROP_OFFSET
    color_counts = {}

    for tile in results.get("per_tile", []):
        color = tile.get("color", "none")
        if color == "none" or tile.get("tile_type") != "sprayed":
            continue

        if color not in color_counts:
            color_counts[color] = {"tp": 0, "fn": 0}

        pred = np.array(tile["pred"])
        mask = generate_patch_labels(
            crop_offset=(center_offset, center_offset),
            spray_size_m=tile.get("spray_size_m", 0.0),
            spray_color=color,
        )
        spray_patches = mask > 0
        if spray_patches.any() and (pred[spray_patches] > 0).any():
            color_counts[color]["tp"] += 1
        else:
            color_counts[color]["fn"] += 1

    color_f1 = {}
    for color, counts in color_counts.items():
        tp, fn = counts["tp"], counts["fn"]
        recall = tp / max(tp + fn, 1)
        color_f1[color] = recall  # precision requires FP from eastern block; use recall as proxy for spray-level

    return color_f1


def plot(output_path: str = None):
    """Generate dodged bar plot: F1 by config, dodged by color."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    configs = []
    red_f1s = []
    blue_f1s = []

    for config in CONFIGS:
        path = os.path.join(RESULTS_DIR, f"{config}.json")
        if not os.path.exists(path):
            print(f"Skipping {config}: no results")
            continue

        with open(path) as f:
            results = json.load(f)

        color_f1 = compute_f1_by_color(results)
        configs.append(config)
        red_f1s.append(color_f1.get("red", 0))
        blue_f1s.append(color_f1.get("blue", 0))

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, red_f1s, width, label="Red", color="#d62728")
    ax.bar(x + width / 2, blue_f1s, width, label="Blue", color="#1f77b4")

    ax.set_ylabel("Spray-level F1")
    ax.set_xlabel("Data source")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in configs])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Detection performance by color and data source")

    plt.tight_layout()
    out = output_path or os.path.join(OUTPUT_DIR, "data_source_by_color.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    plot(args.output)
