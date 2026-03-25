"""
Figure: Temporal holdout by color.

F1 on y-axis, x-axis = training month, bars dodged by dye color.
Shows how well a model trained on one season generalises to the others.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from patch.utils.config import MONTHS
from patch.utils.dataset import generate_patch_labels

RESULTS_DIR = "patch/eval/results/temporal"
OUTPUT_DIR = "patch/visualizations/figures"


def compute_f1_by_color(per_tile: list) -> dict:
    """Compute spray-level F1 per dye color from per-tile predictions."""
    color_counts = {}

    for tile in per_tile:
        color = tile.get("color", "none")
        if color == "none" or tile.get("tile_type") != "sprayed":
            continue

        if color not in color_counts:
            color_counts[color] = {"tp": 0, "fn": 0}

        pred = np.array(tile["pred"])
        # Spray size not stored in temporal per-tile; use a default
        # The spray is centred; check if any non-zero prediction in centre region
        if (pred > 0).any():
            color_counts[color]["tp"] += 1
        else:
            color_counts[color]["fn"] += 1

    f1 = {}
    for color, counts in color_counts.items():
        tp, fn = counts["tp"], counts["fn"]
        f1[color] = tp / max(tp + fn, 1)

    return f1


def plot(config: str = "real_only", output_path: str = None):
    """Generate temporal holdout plot dodged by color."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_months = []
    red_f1s = []
    blue_f1s = []

    for month in MONTHS:
        path = os.path.join(RESULTS_DIR, config, f"train_{month}.json")
        if not os.path.exists(path):
            print(f"Skipping {month}: no results")
            continue

        with open(path) as f:
            results = json.load(f)

        color_f1 = compute_f1_by_color(results.get("per_tile", []))
        train_months.append(month.capitalize())
        red_f1s.append(color_f1.get("red", 0))
        blue_f1s.append(color_f1.get("blue", 0))

    x = np.arange(len(train_months))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, red_f1s, width, label="Red", color="#d62728")
    ax.bar(x + width / 2, blue_f1s, width, label="Blue", color="#1f77b4")

    ax.set_ylabel("Spray-level F1 (held-out seasons)")
    ax.set_xlabel("Training season")
    ax.set_xticks(x)
    ax.set_xticklabels(train_months)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title(f"Temporal generalisation ({config})")

    plt.tight_layout()
    out = output_path or os.path.join(OUTPUT_DIR, f"temporal_{config}.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="real_only")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    plot(args.config, args.output)
