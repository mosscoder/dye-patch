"""
Example tile visualizations with synthetic overlays and ground truth patch outlines.

Loads tiles from HuggingFace dataset configs (sprayed, unsprayed_annex, offsite).

4 columns (cols 0-1 red, cols 2-3 blue), 3 rows:
  Row 1: Sprayed tiles + synthetic overlays + GT bbox
  Row 2: Annex tiles + synthetic overlays
  Row 3: Offsite (European grassland) tiles + synthetic overlays

Tiles use random 384px crops from 512px pre-crops (matching training behaviour).
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datasets import load_dataset

from patch.utils.config import (
    GRID_DIM,
    PRECROP_SIZE,
    VIT_PATCH_SIZE,
)
from patch.utils.augmentations import patch_aligned_crop
from patch.utils.dataset import generate_patch_labels
from patch.utils.synthetic import SyntheticDyeOverlay

HF_REPO = "mpg-ranch/dye_patch"
OUTPUT_DIR = "patch/visualizations/figures"
N_COLS = 6

GT_OUTLINE_COLORS = {"red": "#ff4444", "blue": "#4488ff"}
SEED = 42
random.seed(SEED)


def make_overlay() -> SyntheticDyeOverlay:
    """Create overlay using empirical HSV deltas (or defaults)."""
    return SyntheticDyeOverlay()


def apply_overlay(tile_384: np.ndarray, color_name: str, spray_mask: np.ndarray | None = None):
    """Apply synthetic overlay and return the result + final label mask."""
    overlay = make_overlay()
    label_mask = spray_mask.copy() if spray_mask is not None else np.zeros((GRID_DIM, GRID_DIM), dtype=np.int8)
    img = tile_384.astype(np.float32) / 255.0
    img, label_mask = overlay(img, label_mask, color_name=color_name)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8), label_mask


def draw_gt_bbox(ax, gt_mask: np.ndarray, color: str):
    """Draw a single bounding box around the ground truth spray region."""
    outline_color = GT_OUTLINE_COLORS.get(color, "white")
    rows, cols = np.where(gt_mask > 0)
    if len(rows) == 0:
        return
    y0 = int(rows.min()) * VIT_PATCH_SIZE
    y1 = (int(rows.max()) + 1) * VIT_PATCH_SIZE
    x0 = int(cols.min()) * VIT_PATCH_SIZE
    x1 = (int(cols.max()) + 1) * VIT_PATCH_SIZE
    rect = mpatches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=2, edgecolor=outline_color, facecolor="none",
    )
    ax.add_patch(rect)


def sample_tiles(dataset, n: int, color: str | None = None) -> list[dict]:
    """Sample n tiles from a HF dataset, optionally filtered by color."""
    if color is not None:
        indices = [i for i in range(len(dataset)) if dataset[i].get("color") == color]
    else:
        indices = list(range(len(dataset)))
    random.shuffle(indices)
    return [dataset[i] for i in indices[:n]]


def hf_image_to_numpy(row) -> np.ndarray:
    """Convert a HF dataset image to a numpy array."""
    img = row["image"]
    if hasattr(img, "convert"):
        img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def build_row(tiles: list[dict], color_name: str, with_gt: bool = False) -> tuple[list, list]:
    """Build a row of images with optional GT bboxes."""
    imgs = []
    masks = []
    for tile in tiles:
        arr = hf_image_to_numpy(tile)

        # Ensure 512x512 — resize if needed (offsite tiles are already 512)
        if arr.shape[0] != PRECROP_SIZE or arr.shape[1] != PRECROP_SIZE:
            from PIL import Image
            arr = np.array(Image.fromarray(arr).resize((PRECROP_SIZE, PRECROP_SIZE)), dtype=np.uint8)

        tile_384, crop_offset = patch_aligned_crop(arr)

        gt_mask = None
        if with_gt:
            gt_mask = generate_patch_labels(
                crop_offset=crop_offset,
                spray_size_m=tile.get("spray_size_m", 0.0),
                spray_color=tile.get("color", "none"),
            )

        synth, _ = apply_overlay(tile_384, color_name, gt_mask)
        imgs.append(synth)
        masks.append(gt_mask)

    return imgs, masks


def plot(output_path: str = None):
    """Generate example grid from HuggingFace datasets.

    4 columns: cols 0-1 = red overlay, cols 2-3 = blue overlay.
    Rows: sprayed, annex, offsite (if available).
    """
    print(f"Seed: {SEED}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    N_PER_COLOR = N_COLS // 2  # 3 red + 3 blue = 6 columns

    # Load datasets
    print("Loading sprayed config...")
    sprayed = load_dataset(HF_REPO, "sprayed", split="train")
    print("Loading unsprayed_annex config...")
    annex = load_dataset(HF_REPO, "unsprayed_annex", split="train")

    has_offsite = True
    try:
        print("Loading offsite config...")
        offsite = load_dataset(HF_REPO, "offsite", split="train")
    except Exception as e:
        print(f"Offsite config not available: {e}")
        has_offsite = False

    # Row 1: Sprayed — 2 red + 2 blue, with GT bbox
    spray_red = sample_tiles(sprayed, N_PER_COLOR, color="red")
    spray_blue = sample_tiles(sprayed, N_PER_COLOR, color="blue")
    r1_red_imgs, r1_red_masks = build_row(spray_red, "red", with_gt=True)
    r1_blue_imgs, r1_blue_masks = build_row(spray_blue, "blue", with_gt=True)
    row1_imgs = r1_red_imgs + r1_blue_imgs
    row1_masks = r1_red_masks + r1_blue_masks
    row1_colors = ["red"] * N_PER_COLOR + ["blue"] * N_PER_COLOR

    # Row 2: Annex — 2 red + 2 blue
    annex_r = sample_tiles(annex, N_PER_COLOR)
    annex_b = sample_tiles(annex, N_PER_COLOR)
    r2_red_imgs, _ = build_row(annex_r, "red")
    r2_blue_imgs, _ = build_row(annex_b, "blue")
    row2_imgs = r2_red_imgs + r2_blue_imgs

    # Row 3: Offsite — 2 red + 2 blue
    if has_offsite:
        off_r = sample_tiles(offsite, N_PER_COLOR)
        off_b = sample_tiles(offsite, N_PER_COLOR)
        r3_red_imgs, _ = build_row(off_r, "red")
        r3_blue_imgs, _ = build_row(off_b, "blue")
        row3_imgs = r3_red_imgs + r3_blue_imgs

    # --- Plot ---
    n_rows = 3 if has_offsite else 2
    row_data = [
        (row1_imgs, row1_masks, row1_colors, "Sprayed + synthetic"),
        (row2_imgs, [None] * N_COLS, [None] * N_COLS, "Annex + synthetic"),
    ]
    if has_offsite:
        row_data.append(
            (row3_imgs, [None] * N_COLS, [None] * N_COLS, "Offsite + synthetic"),
        )

    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(N_COLS * 3, n_rows * 3))
    if n_rows == 1:
        axes = [axes]

    for ri, (imgs, masks, colors, label) in enumerate(row_data):
        for ci in range(N_COLS):
            ax = axes[ri][ci] if n_rows > 1 else axes[ci]
            if ci < len(imgs):
                ax.imshow(imgs[ci])
                if ci < len(masks) and masks[ci] is not None and ci < len(colors) and colors[ci]:
                    draw_gt_bbox(ax, masks[ci], colors[ci])
            else:
                ax.set_facecolor("lightgray")
            ax.set_xticks([])
            ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(label, fontsize=10)

    # Column headers
    for ci, header in enumerate(["Red"] * (N_COLS // 2) + ["Blue"] * (N_COLS // 2)):
        axes[0][ci].set_title(header, fontsize=10)

    fig.suptitle(f"Synthetic overlay examples (seed={SEED})", fontsize=14)
    plt.tight_layout()

    out = output_path or os.path.join(OUTPUT_DIR, f"examples_seed{SEED}.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate example tile visualizations")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    plot(args.output)
