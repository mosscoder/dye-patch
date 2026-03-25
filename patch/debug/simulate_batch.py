"""
Visualize transformed training samples: image (left) + label mask (right).

Shows a grid of samples from one batch, with augmentations and optional
synthetic overlay applied. Useful for verifying the data pipeline visually.

Usage:
  python -m patch.debug.simulate_batch
  python -m patch.debug.simulate_batch --config hybrid --n 8
"""

import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets

from patch.utils.config import GRID_DIM, MODEL_INPUT_SIZE, NORM_MEAN, NORM_STD, VIT_PATCH_SIZE
from patch.utils.dataset import DyePatchDataset, tuning_split
from patch.utils.synthetic import SyntheticDyeOverlay

HF_REPO = "mpg-ranch/dye-patch"


def get_data_for_config(config: str):
    if config == "real_only":
        return load_dataset(HF_REPO, "sprayed", split="train")
    elif config == "hybrid":
        sprayed = load_dataset(HF_REPO, "sprayed", split="train")
        annex = load_dataset(HF_REPO, "unsprayed_annex", split="train")
        return concatenate_datasets([sprayed, annex])
    elif config == "synth_local":
        return load_dataset(HF_REPO, "unsprayed_annex", split="train")
    elif config == "synth_offsite":
        return load_dataset(HF_REPO, "offsite", split="train")


def denormalize(tensor):
    """Undo ImageNet-style normalization for display."""
    mean = torch.tensor(NORM_MEAN).view(3, 1, 1)
    std = torch.tensor(NORM_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def main(config: str, n: int, seed: int):
    hf_data = get_data_for_config(config)
    train_hf, _ = tuning_split(hf_data, seed=seed)

    overlay = SyntheticDyeOverlay() if config != "real_only" else None
    ds = DyePatchDataset(train_hf, overlay=overlay, training=True)

    ncols = 2
    fig, axes = plt.subplots(n, ncols, figsize=(6, 2.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        tensor, mask, meta = ds[i]

        # Left: denormalized image with ViT patch grid overlay
        img = denormalize(tensor).permute(1, 2, 0).numpy()
        ax = axes[i, 0]
        ax.imshow(img)
        for g in range(1, GRID_DIM):
            px = g * VIT_PATCH_SIZE
            ax.axhline(px, color="white", linewidth=0.3, alpha=0.3)
            ax.axvline(px, color="white", linewidth=0.3, alpha=0.3)
        # Highlight dye patches
        mask_np = mask.numpy()
        for r in range(GRID_DIM):
            for c in range(GRID_DIM):
                if mask_np[r, c] > 0:
                    rect = mpatches.Rectangle(
                        (c * VIT_PATCH_SIZE, r * VIT_PATCH_SIZE),
                        VIT_PATCH_SIZE, VIT_PATCH_SIZE,
                        linewidth=1, edgecolor="red", facecolor="none", alpha=0.6,
                    )
                    ax.add_patch(rect)
        ax.set_title(
            f"{meta.get('color', '?')} {meta.get('spray_size_m', 0):.1f}m "
            f"({meta.get('month', '?')})",
            fontsize=9,
        )
        ax.axis("off")

        # Right: label mask
        axes[i, 1].imshow(mask.numpy(), cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
        dye_count = int(mask.sum())
        axes[i, 1].set_title(f"mask ({dye_count} dye patches)", fontsize=9)
        axes[i, 1].axis("off")

    fig.suptitle(f"Config: {config}", fontsize=11, fontweight="bold")
    plt.tight_layout()

    out_path = f"patch/debug/batch_{config}_seed{seed}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="real_only", choices=["real_only", "hybrid", "synth_local", "synth_offsite"])
    parser.add_argument("--n", type=int, default=6, help="Number of samples to show")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.config, args.n, args.seed)
