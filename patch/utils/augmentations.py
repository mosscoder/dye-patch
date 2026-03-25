"""
Augmentation pipelines matching wolverines (excluding grayscale).

Two-stage design: pre-overlay transforms run on PIL before synthetic dye is
applied, post-overlay transforms convert to tensor and normalise.  This keeps
the 24x24 label mask aligned with pixel content.
"""

import torchvision.transforms as T
from PIL import Image

import random as _random

import numpy as np

from patch.utils.config import (
    AUG_BRIGHTNESS,
    AUG_BLUR_KERNEL,
    AUG_BLUR_SIGMA,
    AUG_CONTRAST,
    AUG_PROB,
    AUG_ROTATION_DEGREES,
    MODEL_INPUT_SIZE,
    NORM_MEAN,
    NORM_STD,
    PRECROP_SIZE,
    VIT_PATCH_SIZE,
)


def create_pre_overlay_transform():
    """Geometric + colour augmentations applied to PIL image before crop/overlay.

    Each augmentation fires with probability AUG_PROB (0.5).
    Grayscale is intentionally excluded — colour is the dye signal.
    """
    return T.Compose([
        T.Resize(size=(PRECROP_SIZE, PRECROP_SIZE), interpolation=Image.LANCZOS),
        T.RandomHorizontalFlip(p=AUG_PROB),
        T.RandomApply(
            [T.RandomRotation(degrees=AUG_ROTATION_DEGREES)], p=AUG_PROB
        ),
        T.RandomApply(
            [T.GaussianBlur(kernel_size=AUG_BLUR_KERNEL, sigma=AUG_BLUR_SIGMA)],
            p=AUG_PROB,
        ),
        T.RandomApply(
            [T.ColorJitter(
                brightness=AUG_BRIGHTNESS,
                contrast=AUG_CONTRAST,
            )],
            p=AUG_PROB,
        ),
    ])


def patch_aligned_crop(tile_512: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Random crop 512 → 384, snapped to ViT patch grid.

    Offset is a multiple of VIT_PATCH_SIZE (16px) so GT bbox is consistent.
    Returns (cropped_array, (row_offset, col_offset)).
    """
    max_steps = (PRECROP_SIZE - MODEL_INPUT_SIZE) // VIT_PATCH_SIZE
    row_off = _random.randint(0, max_steps) * VIT_PATCH_SIZE
    col_off = _random.randint(0, max_steps) * VIT_PATCH_SIZE
    crop = tile_512[row_off:row_off + MODEL_INPUT_SIZE, col_off:col_off + MODEL_INPUT_SIZE].copy()
    return crop, (row_off, col_off)


def create_post_overlay_transform():
    """ToTensor + Normalize, applied after synthetic overlay."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def create_eval_transform():
    """Deterministic eval pipeline: resize → centre crop → tensor → normalise."""
    return T.Compose([
        T.Resize(size=(PRECROP_SIZE, PRECROP_SIZE), interpolation=Image.LANCZOS),
        T.CenterCrop(size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
