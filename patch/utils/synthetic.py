"""
Synthetic dye overlay: wobbly circles with per-patch KNN-matched HSV deltas.

Each call places 1-10 blobs at random coordinates. Each blob:
- Is a roughly circular shape with sinusoidal boundary wobble
- Has a random diameter between 0.1 and 1.0 meters
- Gets per-patch HSV deltas matched by nearest-neighbor to a lookup table
  built from real sprayed tiles in the training fold
- Deltas are bilinearly interpolated to pixel resolution for smooth transitions
- Edges are Gaussian-feathered (sigma=3px)

The lookup table is built at init from the train fold's 0.5m sprayed tiles,
pairing jittered dye boxes with border-ring vegetation patches.
"""

import colorsys
import math
import random

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.spatial import cKDTree

from patch.utils.config import (
    BLOB_SIZE_RANGE_PX,
    COLOR_TO_LABEL,
    DYE_COLORS,
    GSD_M,
    GRID_DIM,
    PRECROP_SIZE,
    VIT_PATCH_SIZE,
)

SPRAY_SIZE_M = 0.5
SPRAY_JITTER_M = 0.10
BLOB_SIGMA = 3.0


def _make_wobbly_circle(
    center_y: float,
    center_x: float,
    base_radius: float,
    H: int,
    W: int,
) -> np.ndarray:
    """Rasterize a wobbly circle into a boolean mask (vectorized)."""
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


def _block_mean_hsv(img_uint8, y0, x0):
    """Mean HSV of a 16x16 pixel block with circular hue mean."""
    block = img_uint8[y0:y0 + VIT_PATCH_SIZE, x0:x0 + VIT_PATCH_SIZE]
    pixels = block.reshape(-1, 3)
    hsv = np.array([colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0) for r, g, b in pixels])

    hue_angles = hsv[:, 0] * 2 * np.pi
    mean_hue = (np.arctan2(np.mean(np.sin(hue_angles)), np.mean(np.cos(hue_angles))) / (2 * np.pi)) % 1.0
    return np.array([mean_hue, hsv[:, 1].mean(), hsv[:, 2].mean()])


def _patch_mean_hsv_float(image_float, pr, pc):
    """Mean HSV of a ViT patch from a float32 [0-1] RGB image."""
    y0 = pr * VIT_PATCH_SIZE
    x0 = pc * VIT_PATCH_SIZE
    block = image_float[y0:y0 + VIT_PATCH_SIZE, x0:x0 + VIT_PATCH_SIZE]
    pixels = block.reshape(-1, 3)
    hsv = np.array([colorsys.rgb_to_hsv(r, g, b) for r, g, b in pixels])

    hue_angles = hsv[:, 0] * 2 * np.pi
    mean_hue = (np.arctan2(np.mean(np.sin(hue_angles)), np.mean(np.cos(hue_angles))) / (2 * np.pi)) % 1.0
    return np.array([mean_hue, hsv[:, 1].mean(), hsv[:, 2].mean()])


def _build_lookup_table(sprayed_tiles):
    """Build per-color HSV delta lookup tables from 0.5m sprayed tiles.

    Returns dict: {"red": (kd_tree, delta_array), "blue": (kd_tree, delta_array)}
    """
    center_px = PRECROP_SIZE // 2
    spray_radius_px = int(SPRAY_SIZE_M / (2 * GSD_M))
    jitter_px = int(SPRAY_JITTER_M / GSD_M)

    # Border ring: one patch width outside spray boundary
    inner = spray_radius_px + VIT_PATCH_SIZE
    outer = spray_radius_px + 2 * VIT_PATCH_SIZE
    veg_patches = []
    for pr in range(GRID_DIM):
        for pc in range(GRID_DIM):
            py = pr * VIT_PATCH_SIZE + VIT_PATCH_SIZE // 2
            px = pc * VIT_PATCH_SIZE + VIT_PATCH_SIZE // 2
            dist = ((py - center_px) ** 2 + (px - center_px) ** 2) ** 0.5
            if inner < dist <= outer:
                veg_patches.append((pr, pc))

    results = {"red": {"veg": [], "delta": []}, "blue": {"veg": [], "delta": []}}

    for row in sprayed_tiles:
        color = row.get("color", "none")
        if color not in ("red", "blue"):
            continue

        img = np.array(row["image"].convert("RGB").resize((PRECROP_SIZE, PRECROP_SIZE)),
                       dtype=np.uint8)

        for vp_r, vp_c in veg_patches:
            # Jittered dye box
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(0, jitter_px)
            cy = int(center_px + dist * math.sin(angle))
            cx = int(center_px + dist * math.cos(angle))
            H, W = img.shape[:2]
            y0 = max(0, min(cy - VIT_PATCH_SIZE // 2, H - VIT_PATCH_SIZE))
            x0 = max(0, min(cx - VIT_PATCH_SIZE // 2, W - VIT_PATCH_SIZE))
            dye_hsv = _block_mean_hsv(img, y0, x0)

            # Veg patch
            vy0 = vp_r * VIT_PATCH_SIZE
            vx0 = vp_c * VIT_PATCH_SIZE
            veg_hsv = _block_mean_hsv(img, vy0, vx0)

            # HSV delta with circular hue wrapping
            dh = dye_hsv[0] - veg_hsv[0]
            if dh > 0.5:
                dh -= 1.0
            elif dh < -0.5:
                dh += 1.0
            delta = np.array([dh, dye_hsv[1] - veg_hsv[1], dye_hsv[2] - veg_hsv[2]])

            results[color]["veg"].append(veg_hsv)
            results[color]["delta"].append(delta)

    tables = {}
    for color in ("red", "blue"):
        veg = np.array(results[color]["veg"], dtype=np.float32)
        delta = np.array(results[color]["delta"], dtype=np.float32)
        if len(veg) > 0:
            tables[color] = (cKDTree(veg), delta)
        else:
            tables[color] = None

    return tables


class SyntheticDyeOverlay:
    """Synthetic dye overlay with per-patch KNN-matched HSV deltas.

    Parameters
    ----------
    sprayed_dataset : HF Dataset or list
        Sprayed 0.5m tiles from the training fold. Used to build the
        HSV delta lookup table at init.
    min_blobs, max_blobs : int
        Range for number of blobs per image (default 1-10).
    """

    def __init__(
        self,
        sprayed_dataset,
        min_blobs: int = 1,
        max_blobs: int = 10,
    ):
        # Filter to 0.5m sprayed tiles
        tiles = [r for r in sprayed_dataset
                 if abs(r.get("spray_size_m", 0) - SPRAY_SIZE_M) < 0.01
                 and r.get("color", "none") in ("red", "blue")]

        self.tables = _build_lookup_table(tiles)
        self.min_blobs = min_blobs
        self.max_blobs = max_blobs
        self.min_radius_px = BLOB_SIZE_RANGE_PX[0] / 2
        self.max_radius_px = BLOB_SIZE_RANGE_PX[1] / 2

        n_red = len(self.tables["red"][1]) if self.tables["red"] else 0
        n_blue = len(self.tables["blue"][1]) if self.tables["blue"] else 0
        print(f"  Overlay table built: red={n_red}, blue={n_blue} pairs")

    def __call__(self, image: np.ndarray, label_mask: np.ndarray, color_name: str | None = None):
        """Apply synthetic dye overlay with per-patch KNN-matched HSV deltas.

        Parameters
        ----------
        image : np.ndarray
            RGB image, shape (H, W, 3), dtype uint8 or float32 [0-1].
        label_mask : np.ndarray
            Existing 24x24 label mask (0=none, 1=red, 2=blue).
        color_name : str or None
            "red" or "blue". If None, picks randomly.

        Returns
        -------
        image, label_mask : modified copies.
        """
        H, W = image.shape[:2]

        if color_name is None:
            color_name = random.choice(DYE_COLORS)

        table = self.tables.get(color_name)
        if table is None:
            return image, label_mask
        kd_tree, delta_table = table

        is_float = image.dtype == np.float32
        if not is_float:
            image = image.astype(np.float32) / 255.0

        n_blobs = random.randint(self.min_blobs, self.max_blobs)
        final_mask = np.zeros((H, W), dtype=bool)

        for _ in range(n_blobs):
            cy = random.uniform(0, H - 1)
            cx = random.uniform(0, W - 1)
            base_radius = random.uniform(self.min_radius_px, self.max_radius_px)

            blob_mask = _make_wobbly_circle(cy, cx, base_radius, H, W)
            blob_mask &= ~final_mask
            if not blob_mask.any():
                continue

            # Soft feathered edges
            soft_mask = gaussian_filter(blob_mask.astype(np.float32), sigma=BLOB_SIGMA)

            # Find covered ViT patches
            fm = blob_mask[:GRID_DIM * VIT_PATCH_SIZE, :GRID_DIM * VIT_PATCH_SIZE]
            fm_r = fm.reshape(GRID_DIM, VIT_PATCH_SIZE, GRID_DIM, VIT_PATCH_SIZE)
            patch_hits = fm_r.any(axis=(1, 3))

            # Per-patch KNN lookup
            delta_grid = np.zeros((GRID_DIM, GRID_DIM, 3), dtype=np.float32)
            for pr in range(GRID_DIM):
                for pc in range(GRID_DIM):
                    if not patch_hits[pr, pc]:
                        continue
                    patch_hsv = _patch_mean_hsv_float(image, pr, pc)
                    _, idx = kd_tree.query(patch_hsv)
                    delta_grid[pr, pc] = delta_table[idx]

            # Bilinear interpolate to pixel resolution
            delta_field = zoom(delta_grid, (VIT_PATCH_SIZE, VIT_PATCH_SIZE, 1), order=1)

            # Apply interpolated HSV delta with feathering
            ys, xs = np.where(blob_mask)
            ys_c = np.clip(ys, 0, delta_field.shape[0] - 1)
            xs_c = np.clip(xs, 0, delta_field.shape[1] - 1)

            pixels = image[ys, xs]
            hsv = np.array([colorsys.rgb_to_hsv(r, g, b) for r, g, b in pixels])

            strength = soft_mask[ys, xs]
            hsv[:, 0] = (hsv[:, 0] + delta_field[ys_c, xs_c, 0] * strength) % 1.0
            hsv[:, 1] = np.clip(hsv[:, 1] + delta_field[ys_c, xs_c, 1] * strength, 0, 1)
            hsv[:, 2] = np.clip(hsv[:, 2] + delta_field[ys_c, xs_c, 2] * strength, 0, 1)

            rgb = np.array([colorsys.hsv_to_rgb(h, s, v) for h, s, v in hsv])
            image[ys, xs] = rgb.astype(np.float32)

            final_mask |= blob_mask

        # Derive patch labels
        fm = final_mask[:GRID_DIM * VIT_PATCH_SIZE, :GRID_DIM * VIT_PATCH_SIZE]
        fm = fm.reshape(GRID_DIM, VIT_PATCH_SIZE, GRID_DIM, VIT_PATCH_SIZE)
        patch_hits = fm.any(axis=(1, 3))
        label_mask[patch_hits] = COLOR_TO_LABEL[color_name]

        if not is_float:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        return image, label_mask
