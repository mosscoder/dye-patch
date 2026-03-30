"""
Build patch-level CIELAB delta lookup table.

For each 0.5m sprayed tile, pairs a jittered 16x16 dye box (within 10cm
of spray center) with each border-ring vegetation patch. Shadow pixels
(L* < 10th percentile of image) excluded from patch mean calculations.

Output: patch/debug/rgb_overlay/lab_knn_deltas.npz
  - red_veg: (N, 3) mean L*a*b* of vegetation patches
  - red_delta: (N, 3) Lab delta (dye - veg)
  - blue_veg / blue_delta: same for blue

Usage:
  python -m patch.debug.rgb_overlay.build_table
"""

import math
import random

import numpy as np
from datasets import load_dataset
from skimage.color import rgb2lab

from patch.utils.config import GSD_M, GRID_DIM, HF_REPO, PRECROP_SIZE, VIT_PATCH_SIZE

SEED = 42
SPRAY_SIZE_M = 0.5
SPRAY_JITTER_M = 0.10
OUTPUT_PATH = "patch/debug/rgb_overlay/lab_knn_deltas.npz"


def _image_to_lab(img_float):
    """Convert full image from RGB [0-1] to CIELAB. Returns (H, W, 3)."""
    return rgb2lab(img_float)


def _shadow_threshold(lab_image):
    """10th percentile of L* across entire image."""
    return np.percentile(lab_image[:, :, 0].ravel(), 5)


def _block_mean_lab(lab_image, y0, x0, shadow_L):
    """Mean L*a*b* of a 16x16 block, excluding shadow pixels (L* < shadow_L).

    Returns None if all pixels are shadow.
    """
    block = lab_image[y0:y0 + VIT_PATCH_SIZE, x0:x0 + VIT_PATCH_SIZE]
    pixels = block.reshape(-1, 3)

    lit = pixels[:, 0] >= shadow_L
    if not lit.any():
        return None

    return pixels[lit].mean(axis=0)


def _jittered_box_lab(lab_image, center_px, jitter_px, shadow_L):
    """Mean L*a*b* of a 16x16 box randomly jittered around center."""
    angle = random.uniform(0, 2 * math.pi)
    dist = random.uniform(0, jitter_px)
    cy = int(center_px + dist * math.sin(angle))
    cx = int(center_px + dist * math.cos(angle))

    H, W = lab_image.shape[:2]
    y0 = max(0, min(cy - VIT_PATCH_SIZE // 2, H - VIT_PATCH_SIZE))
    x0 = max(0, min(cx - VIT_PATCH_SIZE // 2, W - VIT_PATCH_SIZE))
    return _block_mean_lab(lab_image, y0, x0, shadow_L)


def _get_veg_patches(center_px, spray_radius_px):
    """Ring of patches just outside the spray boundary."""
    inner = spray_radius_px + VIT_PATCH_SIZE
    outer = spray_radius_px + 2 * VIT_PATCH_SIZE
    patches = []
    for pr in range(GRID_DIM):
        for pc in range(GRID_DIM):
            py = pr * VIT_PATCH_SIZE + VIT_PATCH_SIZE // 2
            px = pc * VIT_PATCH_SIZE + VIT_PATCH_SIZE // 2
            dist = ((py - center_px) ** 2 + (px - center_px) ** 2) ** 0.5
            if inner < dist <= outer:
                patches.append((pr, pc))
    return patches


def build():
    random.seed(SEED)
    np.random.seed(SEED)

    ds = load_dataset(HF_REPO, "sprayed", split="train")
    indices = [i for i in range(len(ds))
               if abs(ds[i].get("spray_size_m", 0) - SPRAY_SIZE_M) < 0.01]

    center_px = PRECROP_SIZE // 2
    spray_radius_px = int(SPRAY_SIZE_M / (2 * GSD_M))
    jitter_px = int(SPRAY_JITTER_M / GSD_M)

    veg_patches = _get_veg_patches(center_px, spray_radius_px)

    print(f"Tiles: {len(indices)}, Spray jitter: {jitter_px}px, Border veg patches: {len(veg_patches)}")

    results = {"red": {"veg": [], "delta": []}, "blue": {"veg": [], "delta": []}}

    for i, idx in enumerate(indices):
        row = ds[idx]
        color = row.get("color", "none")
        if color not in ("red", "blue"):
            continue

        img_float = np.array(row["image"].convert("RGB").resize((PRECROP_SIZE, PRECROP_SIZE)),
                             dtype=np.float32) / 255.0
        lab_image = _image_to_lab(img_float)
        shadow_L = _shadow_threshold(lab_image)

        for vp_r, vp_c in veg_patches:
            dye_lab = _jittered_box_lab(lab_image, center_px, jitter_px, shadow_L)
            if dye_lab is None:
                continue

            veg_lab = _block_mean_lab(lab_image, vp_r * VIT_PATCH_SIZE, vp_c * VIT_PATCH_SIZE, shadow_L)
            if veg_lab is None:
                continue

            delta = dye_lab - veg_lab
            results[color]["veg"].append(veg_lab)
            results[color]["delta"].append(delta)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(indices)} tiles")

    for color in ("red", "blue"):
        n = len(results[color]["veg"])
        if n > 0:
            deltas = np.array(results[color]["delta"])
            print(f"{color}: {n} pairs — "
                  f"dL: [{deltas[:,0].min():.1f}, {deltas[:,0].max():.1f}], "
                  f"da: [{deltas[:,1].min():.1f}, {deltas[:,1].max():.1f}], "
                  f"db: [{deltas[:,2].min():.1f}, {deltas[:,2].max():.1f}]")
        else:
            print(f"{color}: no pairs")

    np.savez(
        OUTPUT_PATH,
        red_veg=np.array(results["red"]["veg"], dtype=np.float32),
        red_delta=np.array(results["red"]["delta"], dtype=np.float32),
        blue_veg=np.array(results["blue"]["veg"], dtype=np.float32),
        blue_delta=np.array(results["blue"]["delta"], dtype=np.float32),
    )
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
