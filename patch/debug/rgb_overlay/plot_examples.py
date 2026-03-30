"""
Visualize CIELAB KNN-matched overlay with Gaussian-feathered edges.

Per-patch L*a*b* matching, bilinear interpolation, feathered edges.

Usage:
  python -m patch.debug.rgb_overlay.build_table   # first time only
  python -m patch.debug.rgb_overlay.plot_examples
  python -m patch.debug.rgb_overlay.plot_examples --config offsite
"""

import argparse
import math
import random

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from scipy.ndimage import gaussian_filter, zoom
from scipy.spatial import cKDTree
from skimage.color import rgb2lab, lab2rgb

from patch.utils.config import (
    BLOB_SIZE_RANGE_PX,
    DYE_COLORS,
    GRID_DIM,
    HF_REPO,
    PRECROP_SIZE,
    VIT_PATCH_SIZE,
)
from patch.utils.augmentations import patch_aligned_crop

SEED = 42
TABLE_PATH = "patch/debug/rgb_overlay/lab_knn_deltas.npz"
OUTPUT_DIR = "patch/debug/rgb_overlay"
BLOB_SIGMA = 3.0


def _make_wobbly_circle(center_y, center_x, base_radius, H, W):
    mask = np.zeros((H, W), dtype=bool)
    n_harmonics = random.randint(2, 3)
    harmonics = []
    for _ in range(n_harmonics):
        n_lobes = random.randint(3, 7)
        amplitude = random.uniform(0.08, 0.20) / n_harmonics
        phase = random.uniform(0, 2 * math.pi)
        harmonics.append((n_lobes, amplitude, phase))

    r_max = base_radius * 1.25
    y_lo = max(0, int(center_y - r_max))
    y_hi = min(H, int(center_y + r_max) + 1)
    x_lo = max(0, int(center_x - r_max))
    x_hi = min(W, int(center_x + r_max) + 1)
    if y_hi <= y_lo or x_hi <= x_lo:
        return mask

    ys = np.arange(y_lo, y_hi, dtype=np.float32) - center_y
    xs = np.arange(x_lo, x_hi, dtype=np.float32) - center_x
    dy, dx = np.meshgrid(ys, xs, indexing="ij")
    dist = np.sqrt(dy * dy + dx * dx)
    theta = np.arctan2(dy, dx)

    wobble = np.zeros_like(theta)
    for n_lobes, amp, phase in harmonics:
        wobble += amp * np.sin(n_lobes * theta + phase)

    r_boundary = base_radius * (1.0 + wobble)
    mask[y_lo:y_hi, x_lo:x_hi] = dist <= r_boundary
    return mask


def _patch_mean_lab(lab_image, pr, pc, shadow_L=0.0):
    """Mean L*a*b* of a ViT patch, excluding shadow pixels (L* < shadow_L)."""
    y0 = pr * VIT_PATCH_SIZE
    x0 = pc * VIT_PATCH_SIZE
    block = lab_image[y0:y0 + VIT_PATCH_SIZE, x0:x0 + VIT_PATCH_SIZE]
    pixels = block.reshape(-1, 3)

    lit = pixels[:, 0] >= shadow_L
    if not lit.any():
        lit = np.ones(len(pixels), dtype=bool)

    return pixels[lit].mean(axis=0)


def _apply_overlay(image_float, color_name, kd_tree, delta_table):
    """Apply one blob with per-patch NN-matched Lab delta, bilinearly interpolated."""
    H, W = image_float.shape[:2]
    label_mask = np.zeros((GRID_DIM, GRID_DIM), dtype=np.int8)
    color_label = 1 if color_name == "red" else 2
    min_r = BLOB_SIZE_RANGE_PX[0] / 2
    max_r = BLOB_SIZE_RANGE_PX[1] / 2

    overlaid = image_float.copy()
    cy = random.uniform(0, H - 1)
    cx = random.uniform(0, W - 1)
    base_radius = random.uniform(min_r, max_r)

    blob_mask = _make_wobbly_circle(cy, cx, base_radius, H, W)
    if not blob_mask.any():
        return overlaid, label_mask

    soft_mask = gaussian_filter(blob_mask.astype(np.float32), sigma=BLOB_SIGMA)

    # Convert image to Lab
    lab_image = rgb2lab(overlaid)
    shadow_L = np.percentile(lab_image[:, :, 0].ravel(), 5)

    # Find covered patches
    fm = blob_mask[:GRID_DIM * VIT_PATCH_SIZE, :GRID_DIM * VIT_PATCH_SIZE]
    fm = fm.reshape(GRID_DIM, VIT_PATCH_SIZE, GRID_DIM, VIT_PATCH_SIZE)
    patch_hits = fm.any(axis=(1, 3))

    # Per-patch NN lookup in Lab space
    delta_grid = np.zeros((GRID_DIM, GRID_DIM, 3), dtype=np.float32)
    for pr in range(GRID_DIM):
        for pc in range(GRID_DIM):
            if not patch_hits[pr, pc]:
                continue
            patch_lab = _patch_mean_lab(lab_image, pr, pc, shadow_L)
            _, idx = kd_tree.query(patch_lab)
            delta_grid[pr, pc] = delta_table[idx]

    # Bilinear interpolate to pixel resolution
    delta_field = zoom(delta_grid, (VIT_PATCH_SIZE, VIT_PATCH_SIZE, 1), order=1)

    # Apply Lab delta with feathering
    ys, xs = np.where(blob_mask)
    ys_c = np.clip(ys, 0, delta_field.shape[0] - 1)
    xs_c = np.clip(xs, 0, delta_field.shape[1] - 1)

    strength = soft_mask[ys, xs]
    lab_image[ys, xs, 0] += delta_field[ys_c, xs_c, 0] * strength
    lab_image[ys, xs, 1] += delta_field[ys_c, xs_c, 1] * strength
    lab_image[ys, xs, 2] += delta_field[ys_c, xs_c, 2] * strength

    # Convert back to RGB, clip to valid range
    overlaid = np.clip(lab2rgb(lab_image), 0, 1).astype(np.float32)

    label_mask[patch_hits] = color_label
    return overlaid, label_mask


def main(hf_config: str, seed: int):
    random.seed(seed)
    np.random.seed(seed)

    # Load lookup table
    data = np.load(TABLE_PATH)
    trees = {}
    delta_tables = {}
    for color in ("red", "blue"):
        veg = data[f"{color}_veg"]
        delta = data[f"{color}_delta"]
        trees[color] = cKDTree(veg)
        delta_tables[color] = delta
        print(f"{color}: {len(veg)} pairs, KD-tree built")

    # Load tiles — 3 red + 3 blue
    ds = load_dataset(HF_REPO, hf_config, split="train")
    indices = random.sample(range(len(ds)), min(6, len(ds)))
    colors = ["red"] * 3 + ["blue"] * 3
    n = 6

    fig, axes = plt.subplots(n, 3, figsize=(12, 3.5 * n))

    for row_idx, (tile_idx, color_name) in enumerate(zip(indices, colors)):
        tile = ds[tile_idx]
        img_pil = tile["image"].convert("RGB")

        img_512 = img_pil.resize((PRECROP_SIZE, PRECROP_SIZE))
        img_np = np.array(img_512, dtype=np.uint8)
        crop_np, _ = patch_aligned_crop(img_np)
        crop_float = crop_np.astype(np.float32) / 255.0

        overlaid, label_mask = _apply_overlay(
            crop_float, color_name, trees[color_name], delta_tables[color_name]
        )

        axes[row_idx, 0].imshow(crop_float)
        axes[row_idx, 0].set_title(f"Original ({tile.get('month', '?')})", fontsize=9)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(overlaid)
        axes[row_idx, 1].set_title(f"CIELAB overlay ({color_name})", fontsize=9)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(label_mask,
                                cmap="Reds" if color_name == "red" else "Blues",
                                vmin=0, vmax=2, interpolation="nearest")
        dye_count = int((label_mask > 0).sum())
        axes[row_idx, 2].set_title(f"Mask ({dye_count} patches)", fontsize=9)
        axes[row_idx, 2].axis("off")

    fig.suptitle(f"CIELAB KNN Overlay — {hf_config}", fontsize=11, fontweight="bold")
    plt.tight_layout()

    out_path = f"{OUTPUT_DIR}/examples_{hf_config}_seed{seed}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="unsprayed_annex",
                        choices=["sprayed", "unsprayed_annex", "offsite"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.config, args.seed)
