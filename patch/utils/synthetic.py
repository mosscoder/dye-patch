"""
Synthetic dye overlay: wobbly circles from random dart throws.

Each call places 1-10 blobs at random coordinates.  Each blob is a roughly
circular shape defined in polar coordinates with low-frequency sinusoidal
wobble, resembling what a weed technician would spray.

Each blob applies an HSV delta (shift) to the underlying pixels, sampled from
empirically derived mean/var distributions.  No alpha-blending or opacity —
the delta naturally captures how dye modifies pixel appearance.
"""

import colorsys
import json
import math
import os
import random

import numpy as np

from patch.utils.config import (
    BLOB_SIZE_RANGE_PX,
    COLOR_TO_LABEL,
    DYE_COLORS,
    GRID_DIM,
    VIT_PATCH_SIZE,
)

# Default deltas used if no empirical hsv_deltas.json exists
# Per-blob, sample each channel uniformly from [q10, q90]
DEFAULT_DELTAS = {
    "red":  {"dh": {"q10": -0.05, "q90": 0.05}, "ds": {"q10": -0.05, "q90": 0.25}, "dv": {"q10": -0.20, "q90": 0.10}},
    "blue": {"dh": {"q10": -0.05, "q90": 0.05}, "ds": {"q10": -0.05, "q90": 0.25}, "dv": {"q10": -0.20, "q90": 0.10}},
}

HSV_DELTAS_PATH = "patch/tuning/results/overlay/hsv_deltas.json"


def _sample_range(q25: float, q75: float) -> float:
    """Sample uniformly from [q25, q75]."""
    return random.uniform(q25, q75)


def _load_deltas(path: str = HSV_DELTAS_PATH) -> dict:
    """Load empirical HSV deltas, falling back to defaults."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return DEFAULT_DELTAS


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


class SyntheticDyeOverlay:
    """Generate wobbly-circle synthetic dye blobs with empirical HSV deltas.

    Each call places 1-10 blobs.  Each blob:
    - Is a roughly circular shape with sinusoidal boundary wobble
    - Has a random diameter between 0.1 and 1.0 meters
    - Applies an HSV delta sampled from empirical distributions
    - Can overlap real dye (same color, no label conflict)

    Parameters
    ----------
    deltas : dict or None
        Per-color HSV deltas as ``{"red": {"dh": {"mean":, "var":}, ...}, "blue": {...}}``.
        If None, loads from hsv_deltas.json or uses defaults.
    min_blobs, max_blobs : int
        Range for number of blobs per image (default 1-10).
    """

    def __init__(
        self,
        deltas: dict | None = None,
        min_blobs: int = 1,
        max_blobs: int = 10,
    ):
        self.deltas = deltas or _load_deltas()
        self.min_blobs = min_blobs
        self.max_blobs = max_blobs
        self.min_radius_px = BLOB_SIZE_RANGE_PX[0] / 2
        self.max_radius_px = BLOB_SIZE_RANGE_PX[1] / 2

    def _sample_delta(self, color_name: str) -> tuple[float, float, float]:
        """Sample (dh, ds, dv) for one blob from the empirical q10-q90 range."""
        d = self.deltas.get(color_name, DEFAULT_DELTAS.get(color_name))
        dh = _sample_range(d["dh"]["q10"], d["dh"]["q90"])
        ds = _sample_range(d["ds"]["q10"], d["ds"]["q90"])
        dv = _sample_range(d["dv"]["q10"], d["dv"]["q90"])
        return dh, ds, dv

    def _apply_hsv_delta(self, image: np.ndarray, blob_mask: np.ndarray,
                         dh: float, ds: float, dv: float):
        """Apply HSV delta to blob pixels in-place.

        image: float32 [0-1], shape (H, W, 3) RGB.
        """
        ys, xs = np.where(blob_mask)
        if len(ys) == 0:
            return

        # Extract blob pixels and convert to HSV
        pixels = image[ys, xs]  # (N, 3)
        hsv = np.array([colorsys.rgb_to_hsv(r, g, b) for r, g, b in pixels])

        # Apply deltas (hue wraps circularly, S and V clamp)
        hsv[:, 0] = (hsv[:, 0] + dh) % 1.0
        hsv[:, 1] = np.clip(hsv[:, 1] + ds, 0, 1)
        hsv[:, 2] = np.clip(hsv[:, 2] + dv, 0, 1)

        # Convert back to RGB
        rgb = np.array([colorsys.hsv_to_rgb(h, s, v) for h, s, v in hsv])
        image[ys, xs] = rgb.astype(np.float32)

    def __call__(self, image: np.ndarray, label_mask: np.ndarray, color_name: str | None = None):
        """Apply synthetic dye overlay via HSV deltas.

        Parameters
        ----------
        image : np.ndarray
            RGB image, shape (H, W, 3), dtype uint8 or float32 [0-1].
        label_mask : np.ndarray
            Existing 24x24 label mask (0=none, 1=red, 2=blue).
        color_name : str or None
            "red" or "blue" — selects which HSV deltas to use.
            If None, picks randomly. Labels are ternary (0/1/2).

        Returns
        -------
        image, label_mask : modified copies.
        """
        H, W = image.shape[:2]

        if color_name is None:
            color_name = random.choice(DYE_COLORS)

        is_float = image.dtype == np.float32
        if not is_float:
            image = image.astype(np.float32) / 255.0

        # Place blobs via dart throws
        n_blobs = random.randint(self.min_blobs, self.max_blobs)
        final_mask = np.zeros((H, W), dtype=bool)

        for _ in range(n_blobs):
            cy = random.uniform(0, H - 1)
            cx = random.uniform(0, W - 1)
            base_radius = random.uniform(self.min_radius_px, self.max_radius_px)

            blob_mask = _make_wobbly_circle(cy, cx, base_radius, H, W)
            blob_mask &= ~final_mask  # no overlap between blobs

            if not blob_mask.any():
                continue

            # Per-blob HSV delta
            dh, ds, dv = self._sample_delta(color_name)
            self._apply_hsv_delta(image, blob_mask, dh, ds, dv)
            final_mask |= blob_mask

        # Derive patch labels (vectorized)
        fm = final_mask[:GRID_DIM * VIT_PATCH_SIZE, :GRID_DIM * VIT_PATCH_SIZE]
        fm = fm.reshape(GRID_DIM, VIT_PATCH_SIZE, GRID_DIM, VIT_PATCH_SIZE)
        patch_hits = fm.any(axis=(1, 3))
        label_mask[patch_hits] = COLOR_TO_LABEL[color_name]

        if not is_float:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        return image, label_mask
