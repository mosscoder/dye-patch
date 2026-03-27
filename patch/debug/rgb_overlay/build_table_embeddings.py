"""
Build HSV delta lookup table with DINOv3 embedding-based matching.

Same paired measurements as build_table.py, but stores DINOv3 patch
embeddings of the vegetation patches for KNN matching instead of raw HSV.

The idea: match synthetic blob targets by semantic similarity (what the
model sees) rather than pixel color.

Output: patch/debug/rgb_overlay/dino_knn_deltas.npz
  - red_embed: (N, D) DINOv3 embeddings of vegetation patches
  - red_delta: (N, 3) HSV delta (dye - veg)
  - blue_embed: (N, D)
  - blue_delta: (N, 3)

Usage:
  python -m patch.debug.rgb_overlay.build_table_embeddings
"""

import colorsys
import math
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoImageProcessor

from patch.utils.config import GSD_M, GRID_DIM, HF_REPO, MODEL_NAME, PRECROP_SIZE, VIT_PATCH_SIZE

SEED = 42
SPRAY_SIZE_M = 0.5
SPRAY_JITTER_M = 0.10
OUTPUT_PATH = "patch/debug/rgb_overlay/dino_knn_deltas.npz"


def _block_mean_hsv(img_uint8, y0, x0):
    """Mean HSV of a 16x16 block. Circular mean for hue."""
    block = img_uint8[y0:y0 + VIT_PATCH_SIZE, x0:x0 + VIT_PATCH_SIZE]
    pixels = block.reshape(-1, 3)
    hsv = np.array([colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0) for r, g, b in pixels])

    hue_angles = hsv[:, 0] * 2 * np.pi
    mean_hue = (np.arctan2(np.mean(np.sin(hue_angles)), np.mean(np.cos(hue_angles))) / (2 * np.pi)) % 1.0
    return np.array([mean_hue, hsv[:, 1].mean(), hsv[:, 2].mean()])


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


@torch.no_grad()
def _extract_patch_embeddings(model, processor, img_pil, device):
    """Extract per-patch DINOv3 embeddings for a single image.

    Returns: (24, 24, D) array of patch embeddings.
    """
    inputs = processor(images=img_pil, return_tensors="pt", size={"height": 384, "width": 384}).to(device)
    outputs = model(**inputs)

    # Drop CLS + register tokens
    n_reg = getattr(model.config, "num_register_tokens", 0)
    patch_tokens = outputs.last_hidden_state[:, 1 + n_reg:, :]  # (1, 576, D)
    patch_tokens = patch_tokens.squeeze(0)  # (576, D)

    # Reshape to spatial grid
    embeddings = patch_tokens.reshape(GRID_DIM, GRID_DIM, -1)  # (24, 24, D)
    return embeddings.cpu().numpy()


def build():
    random.seed(SEED)
    np.random.seed(SEED)

    # Load model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, token=os.environ.get("HF_TOKEN"))
    model = model.to(device).eval()
    embed_dim = model.config.hidden_size
    print(f"  Loaded. embed_dim={embed_dim}")

    ds = load_dataset(HF_REPO, "sprayed", split="train")
    indices = [i for i in range(len(ds))
               if abs(ds[i].get("spray_size_m", 0) - SPRAY_SIZE_M) < 0.01]

    center_px = PRECROP_SIZE // 2
    spray_radius_px = int(SPRAY_SIZE_M / (2 * GSD_M))
    jitter_px = int(SPRAY_JITTER_M / GSD_M)

    veg_patches = _get_veg_patches(center_px, spray_radius_px)

    print(f"Tiles: {len(indices)}, Border veg patches: {len(veg_patches)}")

    results = {
        "red": {"embed": [], "delta": []},
        "blue": {"embed": [], "delta": []},
    }

    for i, idx in enumerate(indices):
        row = ds[idx]
        color = row.get("color", "none")
        if color not in ("red", "blue"):
            continue

        img_pil = row["image"].convert("RGB").resize((PRECROP_SIZE, PRECROP_SIZE))
        img_uint8 = np.array(img_pil, dtype=np.uint8)

        # Extract DINOv3 embeddings for all patches
        # Processor expects the model's native input size
        embeddings = _extract_patch_embeddings(model, processor, img_pil, device)

        for vp_r, vp_c in veg_patches:
            # HSV delta (same as build_table.py)
            dye_hsv = _jittered_box_hsv(img_uint8, center_px, jitter_px)
            veg_hsv = _patch_mean_hsv(img_uint8, vp_r, vp_c)

            dh = dye_hsv[0] - veg_hsv[0]
            if dh > 0.5:
                dh -= 1.0
            elif dh < -0.5:
                dh += 1.0
            delta = np.array([dh, dye_hsv[1] - veg_hsv[1], dye_hsv[2] - veg_hsv[2]])

            # DINOv3 embedding of the veg patch
            veg_embed = embeddings[vp_r, vp_c]  # (D,)

            results[color]["embed"].append(veg_embed)
            results[color]["delta"].append(delta)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(indices)} tiles")

    for color in ("red", "blue"):
        n = len(results[color]["embed"])
        print(f"{color}: {n} pairs (embed_dim={embed_dim})")

    np.savez(
        OUTPUT_PATH,
        red_embed=np.array(results["red"]["embed"], dtype=np.float32),
        red_delta=np.array(results["red"]["delta"], dtype=np.float32),
        blue_embed=np.array(results["blue"]["embed"], dtype=np.float32),
        blue_delta=np.array(results["blue"]["delta"], dtype=np.float32),
    )
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
