"""
Empirical HSV delta estimation from sprayed train tiles.

For each sprayed tile (0.5m patches only):
  1. Compute mean HSV of the central ViT patch (16x16 px, dye present)
  2. Compute mean HSV of a random peripheral ViT patch (outside spray bounds)
  3. Delta = dye patch mean - vegetation patch mean

One delta per tile. Bootstrap (1000 resamples) across tiles per color to
estimate mean and std of the delta distribution.

Output: tuning/results/overlay/hsv_deltas.json
"""

import argparse
import colorsys
import json
import os
import random

import numpy as np
from datasets import load_dataset

from patch.utils.config import GSD_M, GRID_DIM, HF_REPO, PRECROP_SIZE, VIT_PATCH_SIZE
RESULTS_DIR = "patch/tuning/results/overlay"
SEED = 42

SPRAY_SIZE_M = 0.5
BUFFER_INWARD_M = 0.125  # dye patch must be within this distance of centroid
N_BOOTSTRAPS = 1000


def _patch_mean_hsv(img: np.ndarray, patch_row: int, patch_col: int) -> np.ndarray:
    """Compute mean HSV of a single ViT patch (16x16 pixels).

    Uses circular mean for hue to handle the 0/1 boundary correctly
    (e.g., red hues near 0.95 and 0.05 average to ~0.0, not ~0.5).

    img: uint8 RGB, shape (H, W, 3).
    Returns: array [h, s, v] with values in [0, 1].
    """
    y0 = patch_row * VIT_PATCH_SIZE
    x0 = patch_col * VIT_PATCH_SIZE
    block = img[y0:y0 + VIT_PATCH_SIZE, x0:x0 + VIT_PATCH_SIZE]  # (16, 16, 3)

    # Convert each pixel to HSV
    hsv = np.array([
        colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        for r, g, b in block.reshape(-1, 3)
    ])

    # Circular mean for hue (convert to angles, average sin/cos, convert back)
    hue_angles = hsv[:, 0] * 2 * np.pi
    mean_sin = np.mean(np.sin(hue_angles))
    mean_cos = np.mean(np.cos(hue_angles))
    mean_hue = (np.arctan2(mean_sin, mean_cos) / (2 * np.pi)) % 1.0

    # Arithmetic mean for saturation and value
    mean_s = hsv[:, 1].mean()
    mean_v = hsv[:, 2].mean()

    return np.array([mean_hue, mean_s, mean_v])


def _get_spray_patches(center_px: int, spray_radius_px: int, buffer_px: int) -> list[tuple[int, int]]:
    """Get ViT patch indices that are within the buffered spray region."""
    patches = []
    for pr in range(GRID_DIM):
        for pc in range(GRID_DIM):
            # Patch center in pixel coords
            py = pr * VIT_PATCH_SIZE + VIT_PATCH_SIZE // 2
            px = pc * VIT_PATCH_SIZE + VIT_PATCH_SIZE // 2
            dist = ((py - center_px) ** 2 + (px - center_px) ** 2) ** 0.5
            if dist <= buffer_px:
                patches.append((pr, pc))
    return patches


def _get_veg_patches(center_px: int, spray_radius_px: int) -> list[tuple[int, int]]:
    """Get ViT patch indices that are fully outside the spray region."""
    margin_px = spray_radius_px + VIT_PATCH_SIZE  # ensure no overlap
    patches = []
    for pr in range(GRID_DIM):
        for pc in range(GRID_DIM):
            py = pr * VIT_PATCH_SIZE + VIT_PATCH_SIZE // 2
            px = pc * VIT_PATCH_SIZE + VIT_PATCH_SIZE // 2
            dist = ((py - center_px) ** 2 + (px - center_px) ** 2) ** 0.5
            if dist > margin_px:
                patches.append((pr, pc))
    return patches


def _bootstrap_stats(deltas: np.ndarray, n_boot: int = N_BOOTSTRAPS) -> dict:
    """Bootstrap q25, median, and q75 from an array of deltas.

    deltas: shape (N, 3) for [dh, ds, dv].
    All quantiles are bootstrapped: resample N tiles with replacement 1000 times,
    compute q25/median/q75 of each resample, report the median of those bootstrapped
    quantiles. The blob drawer samples uniformly from [q25, q75].
    """
    n = len(deltas)
    boot_q25 = np.zeros((n_boot, 3))
    boot_median = np.zeros((n_boot, 3))
    boot_q75 = np.zeros((n_boot, 3))

    for b in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        sample = deltas[idx]
        boot_q25[b] = np.percentile(sample, 25, axis=0)
        boot_median[b] = np.median(sample, axis=0)
        boot_q75[b] = np.percentile(sample, 75, axis=0)

    # Use median of bootstrapped quantiles as the stable estimate
    q25 = np.median(boot_q25, axis=0)
    median = np.median(boot_median, axis=0)
    q75 = np.median(boot_q75, axis=0)

    return {
        "dh": {"q25": float(q25[0]), "median": float(median[0]), "q75": float(q75[0])},
        "ds": {"q25": float(q25[1]), "median": float(median[1]), "q75": float(q75[1])},
        "dv": {"q25": float(q25[2]), "median": float(median[2]), "q75": float(q75[2])},
        "n_tiles": n,
        "bootstrap_se": {
            "dh": float(np.std(boot_median[:, 0])),
            "ds": float(np.std(boot_median[:, 1])),
            "dv": float(np.std(boot_median[:, 2])),
        },
    }


def compute_deltas():
    """Compute patch-level HSV deltas from sprayed train tiles."""
    random.seed(SEED)
    np.random.seed(SEED)

    print("Loading sprayed train tiles...")
    ds = load_dataset(HF_REPO, "sprayed", split="train")

    # Filter to 0.5m patches only
    indices = [i for i in range(len(ds))
               if abs(ds[i].get("spray_size_m", 0) - SPRAY_SIZE_M) < 0.01]
    print(f"Found {len(indices)} tiles with spray_size_m={SPRAY_SIZE_M}")

    # Geometry
    center_px = PRECROP_SIZE // 2  # 256
    spray_radius_px = int(SPRAY_SIZE_M / (2 * GSD_M))  # ~36px
    buffer_px = int(BUFFER_INWARD_M / GSD_M)  # ~18px

    spray_patches = _get_spray_patches(center_px, spray_radius_px, buffer_px)
    veg_patches = _get_veg_patches(center_px, spray_radius_px)

    print(f"Spray patches (within {BUFFER_INWARD_M}m of centroid): {len(spray_patches)}")
    print(f"Vegetation patches (outside spray): {len(veg_patches)}")

    if not spray_patches or not veg_patches:
        raise ValueError("No valid spray or vegetation patches found")

    deltas = {"red": [], "blue": []}

    for idx in indices:
        row = ds[idx]
        color = row.get("color", "none")
        if color not in ("red", "blue"):
            continue

        img = np.array(row["image"].convert("RGB"), dtype=np.uint8)

        # Pick a random spray patch and compute its mean HSV
        sp_r, sp_c = random.choice(spray_patches)
        dye_hsv = _patch_mean_hsv(img, sp_r, sp_c)

        # Pick a random vegetation patch and compute its mean HSV
        vp_r, vp_c = random.choice(veg_patches)
        veg_hsv = _patch_mean_hsv(img, vp_r, vp_c)

        # Delta (circular-aware for hue)
        dh = dye_hsv[0] - veg_hsv[0]
        # Wrap hue delta to [-0.5, 0.5] (shortest arc on the hue circle)
        if dh > 0.5:
            dh -= 1.0
        elif dh < -0.5:
            dh += 1.0
        delta = np.array([dh, dye_hsv[1] - veg_hsv[1], dye_hsv[2] - veg_hsv[2]])
        deltas[color].append(delta)

    # Bootstrap per color
    results = {}
    for color in ("red", "blue"):
        arr = np.array(deltas[color])
        if len(arr) == 0:
            print(f"WARNING: no deltas for {color}")
            continue

        stats = _bootstrap_stats(arr)
        results[color] = stats

        print(f"\n{color} (n={stats['n_tiles']} tiles, {N_BOOTSTRAPS} bootstraps):")
        for ch in ("dh", "ds", "dv"):
            s = stats[ch]
            print(f"  {ch}: median={s['median']:.4f}  Q25={s['q25']:.4f}  Q75={s['q75']:.4f}  "
                  f"bootstrap_se={stats['bootstrap_se'][ch]:.4f}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "hsv_deltas.json")

    output = {
        "method": "patch_level_nested_bootstrap",
        "n_bootstraps": N_BOOTSTRAPS,
        "spray_size_m": SPRAY_SIZE_M,
        "buffer_inward_m": BUFFER_INWARD_M,
        "patch_size_px": VIT_PATCH_SIZE,
        "seed": SEED,
        **results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    return results


if __name__ == "__main__":
    compute_deltas()
