"""
Dataset class and data loading for patch-level dye classification.

Tiles are stored as 512px pre-crops in HuggingFace.  On-the-fly processing:
  1. Pre-overlay augmentations (PIL)
  2. Random crop to 384px (128px jitter range)
  3. Compute 24x24 label mask from spray metadata + crop offset
  4. Synthetic dye overlay (respects existing labels)
  5. ToTensor + Normalize
"""

import random

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from patch.utils.config import (
    COLOR_TO_LABEL,
    GSD_M,
    GRID_DIM,
    MODEL_INPUT_SIZE,
    MONTHS,
    PRECROP_SIZE,
    TUNING_POINTS_PER_SET,
    TUNING_TEST_FRAC,
    VIT_PATCH_SIZE,
)
from patch.utils.augmentations import (
    create_eval_transform,
    create_post_overlay_transform,
    create_pre_overlay_transform,
    patch_aligned_crop,
)
from PIL import Image as PILImage
from patch.utils.synthetic import SyntheticDyeOverlay


def generate_patch_labels(
    crop_offset: tuple[int, int],
    spray_size_m: float,
    spray_color: str,
    gsd_m: float = GSD_M,
    patch_size: int = VIT_PATCH_SIZE,
    grid_dim: int = GRID_DIM,
) -> np.ndarray:
    """Compute 24x24 label mask from spray metadata and crop offset.

    Spray is always centered at (PRECROP_SIZE//2, PRECROP_SIZE//2) in the
    512px pre-crop by construction.

    Parameters
    ----------
    crop_offset : (row, col)
        Top-left corner of the 384px crop within the 512px pre-crop.
    spray_size_m : float
        Physical side length of the spray patch (0.1 or 0.5 m).
    spray_color : str
        "red", "blue", or "none". Determines the label value:
        0=none, 1=red, 2=blue.
    """
    mask = np.zeros((grid_dim, grid_dim), dtype=np.int8)

    if spray_size_m <= 0 or spray_color not in COLOR_TO_LABEL:
        return mask

    label = COLOR_TO_LABEL[spray_color]
    half_px = spray_size_m / (2 * gsd_m)

    # Spray is always centered in the 512px pre-crop
    center = PRECROP_SIZE // 2

    # Spray bounds in 512px coordinates
    spray_top = center - half_px
    spray_bottom = center + half_px
    spray_left = center - half_px
    spray_right = center + half_px

    # Convert to 384px crop coordinates
    spray_top -= crop_offset[0]
    spray_bottom -= crop_offset[0]
    spray_left -= crop_offset[1]
    spray_right -= crop_offset[1]

    # Check which ViT patches overlap with spray square
    for r in range(grid_dim):
        patch_top = r * patch_size
        patch_bottom = (r + 1) * patch_size
        for c in range(grid_dim):
            patch_left = c * patch_size
            patch_right = (c + 1) * patch_size

            # Any overlap?
            if (patch_bottom > spray_top and patch_top < spray_bottom and
                    patch_right > spray_left and patch_left < spray_right):
                mask[r, c] = label

    return mask


class DyePatchDataset(Dataset):
    """Dataset for patch-level dye classification.

    Wraps a HuggingFace dataset and applies on-the-fly augmentation,
    random cropping, label mask generation, and synthetic dye overlay.

    Parameters
    ----------
    hf_dataset : datasets.Dataset
        HuggingFace dataset with columns: image, spray_size_m, month, color,
        concentration, tile_type, latitude, longitude, point_name.
    overlay : SyntheticDyeOverlay or None
        Synthetic overlay generator.  None disables overlay (eval mode).
    training : bool
        If True, apply augmentations and random crop.  If False, centre crop.
    """

    def __init__(self, hf_dataset, overlay: SyntheticDyeOverlay | None = None, training: bool = True):
        self.data = hf_dataset
        self.overlay = overlay
        self.training = training

        if training:
            self.pre_transform = create_pre_overlay_transform()
        else:
            self.pre_transform = None

        self.post_transform = create_post_overlay_transform()
        self.eval_transform = create_eval_transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        image = row["image"]  # PIL Image

        metadata = {
            k: row[k] for k in row if k != "image"
        }

        if not self.training:
            # Eval: centre crop, no overlay
            tensor = self.eval_transform(image)

            # Centre crop offset for label mask
            top = (PRECROP_SIZE - MODEL_INPUT_SIZE) // 2
            left = (PRECROP_SIZE - MODEL_INPUT_SIZE) // 2
            crop_offset = (top, left)

            mask = generate_patch_labels(
                crop_offset=crop_offset,
                spray_size_m=row.get("spray_size_m", 0.0),
                spray_color=row.get("color", "none"),
            )
            return tensor, torch.from_numpy(mask).long(), metadata

        # --- Training path ---
        # 1. Pre-overlay augmentations (PIL)
        image = self.pre_transform(image)

        # 2. Random crop to 384px, snapped to ViT patch grid
        image_np = np.array(image, dtype=np.uint8)
        crop_np, crop_offset = patch_aligned_crop(image_np)
        image = PILImage.fromarray(crop_np)

        # 3. Compute label mask from spray metadata + crop offset
        mask = generate_patch_labels(
            crop_offset=crop_offset,
            spray_size_m=row.get("spray_size_m", 0.0),
            spray_color=row.get("color", "none"),
        )

        # 4. Synthetic dye overlay (50% chance)
        if self.overlay is not None and random.random() < 0.5:
            image_np = np.array(image, dtype=np.float32) / 255.0

            # Determine overlay colour: match real dye for spray tiles
            tile_type = row.get("tile_type", "")
            real_color = row.get("color", "none")
            overlay_color = real_color if real_color in ("red", "blue") else None

            image_np, mask = self.overlay(image_np, mask, color_name=overlay_color)
            image = PILImage.fromarray((np.clip(image_np, 0, 1) * 255).astype(np.uint8))

        # 5. ToTensor + Normalize
        tensor = self.post_transform(image)

        return tensor, torch.from_numpy(mask).long(), metadata


# =============================================================================
# Split functions
# =============================================================================

def _get_point_strata(hf_dataset) -> dict[str, str]:
    """Build point_name → stratum mapping (color-concentration-size).

    Stratum is defined at the point level (same across months).
    """
    point_strata = {}
    for r in hf_dataset:
        pn = str(r.get("point_name", ""))
        if pn not in point_strata:
            point_strata[pn] = (
                f"{r.get('color', '')}-{r.get('concentration', '')}-{r.get('spray_size_m', 0)}"
            )
    return point_strata


def _point_indices(hf_dataset, points: set) -> list[int]:
    """Return row indices for all tiles belonging to the given point set."""
    return [i for i, r in enumerate(hf_dataset) if str(r.get("point_name", "")) in points]


def stratified_split(hf_dataset, test_frac: float = TUNING_TEST_FRAC, seed: int = 0):
    """70/30 train/test split at the POINT level.

    All months of a given point go to the same split.
    Stratified by (color, concentration, size) — point-level attributes.

    Returns (train_dataset, test_dataset).
    """
    point_strata = _get_point_strata(hf_dataset)
    points = list(point_strata.keys())
    strata = [point_strata[p] for p in points]

    train_points, test_points = train_test_split(
        points, test_size=test_frac, stratify=strata, random_state=seed
    )

    train_idx = _point_indices(hf_dataset, set(train_points))
    test_idx = _point_indices(hf_dataset, set(test_points))

    return hf_dataset.select(train_idx), hf_dataset.select(test_idx)


def temporal_holdout_split(hf_dataset, holdout_month: str):
    """Leave-one-month-out split.

    Train = rows from the two non-held-out months.
    Test  = rows from the held-out month.

    Filters by month from an already point-level-split dataset.
    """
    other_months = [m for m in MONTHS if m != holdout_month]
    train_indices = [i for i, r in enumerate(hf_dataset) if r.get("month", "") in other_months]
    test_indices = [i for i, r in enumerate(hf_dataset) if r.get("month", "") == holdout_month]
    return hf_dataset.select(train_indices), hf_dataset.select(test_indices)


def tuning_split(train_dataset, seed: int = 0):
    """Split train set into tuning-train and tuning-val at the POINT level.

    Randomly samples TUNING_POINTS_PER_SET points for train and the same
    count for val (40 points × 3 months = 120 tiles per set by default).
    All months of a point stay in the same set.
    Falls back to random (non-stratified) split when all points share a
    single stratum (e.g., offsite or annex tiles with no color/size variation).

    Returns (tuning_train, tuning_val) as HF Dataset objects.
    """
    point_strata = _get_point_strata(train_dataset)
    points = list(point_strata.keys())
    strata = [point_strata[p] for p in points]
    can_stratify = len(set(strata)) > 1

    n_per_set = min(TUNING_POINTS_PER_SET, len(points) // 2)
    n_points_needed = n_per_set * 2
    n_points_needed = min(n_points_needed, len(points))

    # Subsample points if we have more than needed
    if n_points_needed < len(points):
        points, _ = train_test_split(
            points, train_size=n_points_needed,
            stratify=strata if can_stratify else None,
            random_state=seed,
        )
        strata = [point_strata[p] for p in points]
        can_stratify = len(set(strata)) > 1

    # Split selected points into equal train/val sets
    train_points, val_points = train_test_split(
        points, train_size=n_per_set,
        stratify=strata if can_stratify else None,
        random_state=seed + 1,
    )

    train_idx = _point_indices(train_dataset, set(train_points))
    val_idx = _point_indices(train_dataset, set(val_points))

    return train_dataset.select(train_idx), train_dataset.select(val_idx)
