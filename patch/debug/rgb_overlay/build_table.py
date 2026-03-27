"""
Build patch-level HSV delta lookup table with KNN matching.

For each 0.5m sprayed tile, pairs a jittered 16x16 dye box (within 10cm
of spray center) with each border-ring vegetation patch. Stores
(veg_hsv_mean, dh, ds, dv) per pair.

Output: patch/debug/rgb_overlay/hsv_knn_deltas.npz
  - red_veg: (N, 3) mean HSV of vegetation patches (float32)
  - red_delta: (N, 3) HSV delta (dye - veg), hue wrapped to [-0.5, 0.5]
  - blue_veg: (N, 3)
  - blue_delta: (N, 3)

Usage:
  python -m patch.debug.rgb_overlay.build_table
"""

import colorsys
import math
import random

import numpy as np
from datasets import load_dataset

from patch.utils.config import GSD_M, GRID_DIM, HF_REPO, PRECROP_SIZE, VIT_PATCH_SIZE

SEED = 42
SPRAY_SIZE_M = 0.5
SPRAY_JITTER_M = 0.10
OUTPUT_PATH = "patch/debug/rgb_overlay/hsv_knn_deltas.npz"


def _block_mean_hsv(img_uint8, y0, x0):
    """Mean HSV of a 16x16 block. Circular mean for hue."""
    block = img_uint8[y0:y0 + VIT_PATCH_SIZE, x0:x0 + VIT_PATCH_SIZE]
    pixels = block.reshape(-1, 3)
    hsv = np.array([colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0) for r, g, b in pixels])

    hue_angles = hsv[:, 0] * 2 * np.pi
    mean_hue = (np.arctan2(np.mean(np.sin(hue_angles)), np.mean(np.cos(hue_angles))) / (2 * np.pi)) % 1.0
    mean_s = hsv[:, 1].mean()
    mean_v = hsv[:, 2].mean()
    return np.array([mean_hue, mean_s, mean_v])


def _jittered_box_hsv(img_uint8, center_px, jitter_px):
    """Mean HSV of a 16x16 box randomly jittered around center."""
    angle = random.uniform(0, 2 * math.pi)
    dist = random.uniform(0, jitter_px)
    cy = int(center_px + dist * math.sin(angle))
    cx = int(center_px + dist * math.cos(angle))

    H, W = img_uint8.shape[:2]
    y0 = max(0, min(cy - VIT_PATCH_SIZE // 2, H - VIT_PATCH_SIZE))
    x0 = max(0, min(cx - VIT_PATCH_SIZE // 2, W - VIT_PATCH_SIZE))
    return _block_mean_hsv(img_uint8, y0, x0)


def _patch_mean_hsv(img_uint8, patch_row, patch_col):
    """Mean HSV of a ViT-grid-aligned patch."""
    y0 = patch_row * VIT_PATCH_SIZE
    x0 = patch_col * VIT_PATCH_SIZE
    return _block_mean_hsv(img_uint8, y0, x0)


def _get_veg_patches(center_px, spray_radius_px):
    """Get the ring of patches just outside the spray boundary."""
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

    for idx in indices:
        row = ds[idx]
        color = row.get("color", "none")
        if color not in ("red", "blue"):
            continue

        img = np.array(row["image"].convert("RGB").resize((PRECROP_SIZE, PRECROP_SIZE)),
                       dtype=np.uint8)

        for vp_r, vp_c in veg_patches:
            dye_hsv = _jittered_box_hsv(img, center_px, jitter_px)
            veg_hsv = _patch_mean_hsv(img, vp_r, vp_c)

            # Hue delta wrapped to [-0.5, 0.5]
            dh = dye_hsv[0] - veg_hsv[0]
            if dh > 0.5:
                dh -= 1.0
            elif dh < -0.5:
                dh += 1.0

            delta = np.array([dh, dye_hsv[1] - veg_hsv[1], dye_hsv[2] - veg_hsv[2]])
            results[color]["veg"].append(veg_hsv)
            results[color]["delta"].append(delta)

    for color in ("red", "blue"):
        n = len(results[color]["veg"])
        print(f"{color}: {n} pairs")

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
