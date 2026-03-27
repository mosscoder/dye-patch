"""
Build patch-level HSV delta lookup table.

For each 0.5m sprayed tile, pairs a jittered 16x16 dye box (within 10cm
of spray center) with each border-ring vegetation patch. Shadow pixels
(V < 5th percentile of image) excluded from patch mean calculations.

Output: patch/debug/rgb_overlay/hsv_knn_deltas.npz
  - red_veg: (N, 3) mean HSV of vegetation patches
  - red_delta: (N, 3) HSV delta (dye - veg), hue wrapped
  - blue_veg / blue_delta: same for blue

Usage:
  python -m patch.debug.rgb_overlay.build_table
"""

import math
import random

import numpy as np
from datasets import load_dataset

from patch.utils.config import GSD_M, GRID_DIM, HF_REPO, PRECROP_SIZE, VIT_PATCH_SIZE

SEED = 42
SPRAY_SIZE_M = 0.5
SPRAY_JITTER_M = 0.10
OUTPUT_PATH = "patch/debug/rgb_overlay/hsv_knn_deltas.npz"


def _rgb_to_hsv_vectorized(rgb):
    """Vectorized RGB (float 0-1) to HSV. Input (N, 3), output (N, 3)."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    diff = maxc - minc
    s = np.where(maxc > 0, diff / maxc, 0.0)

    h = np.zeros_like(maxc)
    mask = diff > 0
    rc = np.where(mask, (maxc - r) / np.where(mask, diff, 1), 0)
    gc = np.where(mask, (maxc - g) / np.where(mask, diff, 1), 0)
    bc = np.where(mask, (maxc - b) / np.where(mask, diff, 1), 0)

    h = np.where(mask & (r == maxc), bc - gc, h)
    h = np.where(mask & (g == maxc), 2.0 + rc - bc, h)
    h = np.where(mask & (b == maxc), 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0

    return np.column_stack([h, s, v])


def _block_mean_hsv(img_float, y0, x0, shadow_v):
    """Mean HSV of a 16x16 block, excluding shadow pixels. Returns None if all shadow."""
    block = img_float[y0:y0 + VIT_PATCH_SIZE, x0:x0 + VIT_PATCH_SIZE]
    pixels = block.reshape(-1, 3)
    hsv = _rgb_to_hsv_vectorized(pixels)

    lit = hsv[:, 2] >= shadow_v
    if not lit.any():
        return None

    hsv_lit = hsv[lit]
    hue_angles = hsv_lit[:, 0] * 2 * np.pi
    mean_hue = (np.arctan2(np.mean(np.sin(hue_angles)), np.mean(np.cos(hue_angles))) / (2 * np.pi)) % 1.0
    return np.array([mean_hue, hsv_lit[:, 1].mean(), hsv_lit[:, 2].mean()])


def _jittered_box_hsv(img_float, center_px, jitter_px, shadow_v):
    """Mean HSV of a 16x16 box randomly jittered around center, shadow-excluded."""
    angle = random.uniform(0, 2 * math.pi)
    dist = random.uniform(0, jitter_px)
    cy = int(center_px + dist * math.sin(angle))
    cx = int(center_px + dist * math.cos(angle))

    H, W = img_float.shape[:2]
    y0 = max(0, min(cy - VIT_PATCH_SIZE // 2, H - VIT_PATCH_SIZE))
    x0 = max(0, min(cx - VIT_PATCH_SIZE // 2, W - VIT_PATCH_SIZE))
    return _block_mean_hsv(img_float, y0, x0, shadow_v)


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

        img = np.array(row["image"].convert("RGB").resize((PRECROP_SIZE, PRECROP_SIZE)),
                       dtype=np.float32) / 255.0
        shadow_v = np.percentile(img.max(axis=-1).ravel(), 10)

        for vp_r, vp_c in veg_patches:
            dye_hsv = _jittered_box_hsv(img, center_px, jitter_px, shadow_v)
            if dye_hsv is None:
                continue

            veg_hsv = _block_mean_hsv(img, vp_r * VIT_PATCH_SIZE, vp_c * VIT_PATCH_SIZE, shadow_v)
            if veg_hsv is None:
                continue

            dh = dye_hsv[0] - veg_hsv[0]
            if dh > 0.5:
                dh -= 1.0
            elif dh < -0.5:
                dh += 1.0

            delta = np.array([dh, dye_hsv[1] - veg_hsv[1], dye_hsv[2] - veg_hsv[2]])
            results[color]["veg"].append(veg_hsv)
            results[color]["delta"].append(delta)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(indices)} tiles")

    for color in ("red", "blue"):
        n = len(results[color]["veg"])
        if n > 0:
            deltas = np.array(results[color]["delta"])
            print(f"{color}: {n} pairs — "
                  f"dh: [{deltas[:,0].min():.3f}, {deltas[:,0].max():.3f}], "
                  f"ds: [{deltas[:,1].min():.3f}, {deltas[:,1].max():.3f}], "
                  f"dv: [{deltas[:,2].min():.3f}, {deltas[:,2].max():.3f}]")
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
