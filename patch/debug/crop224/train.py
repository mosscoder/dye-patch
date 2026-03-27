"""
Test 224px center crop across real_only, hybrid, synth_local configs.

224px input → 14×14 = 196 patch grid (vs 384px → 24×24 = 576).
Spray covers more of the image proportionally → less class imbalance.
Matches drone-lsr's working setup.

9 jobs = 3 configs × 3 seeds
SLURM array: --array=0-8
Index mapping: divmod(idx, 3) → (config_idx, seed)

Usage:
  python -u -m patch.debug.crop224.train --idx 0  # real_only, seed=0
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image as PILImage

import torchvision.transforms as T

from patch.utils.config import (
    COLOR_TO_LABEL,
    GSD_M,
    HF_REPO,
    NORM_MEAN,
    NORM_STD,
    NO_DYE_NEG_SAMPLE,
    PRECROP_SIZE,
)
from patch.utils.models import create_model
from patch.utils.train import collate_fn, compute_spray_metrics, save_results, set_seed
from patch.utils.synthetic import SyntheticDyeOverlay

RESULTS_DIR = "patch/debug/crop224/results"
MODEL_NAME_LARGE = "facebook/dinov3-vitl16-pretrain-sat493m"
LR = 5e-4
N_EPOCHS = 50
NEG_MULT = 8

# 224px overrides
MODEL_INPUT = 224
VIT_PATCH = 16
GRID = MODEL_INPUT // VIT_PATCH  # 14
EVAL_OFFSET = (PRECROP_SIZE - MODEL_INPUT) // 2  # 144

CONFIGS = ["real_only", "hybrid", "synth_local"]


def _generate_labels_224(crop_offset, spray_size_m, spray_color):
    """24x24 → 14x14 label generation for 224px crop."""
    mask = np.zeros((GRID, GRID), dtype=np.int8)
    if spray_size_m <= 0 or spray_color not in COLOR_TO_LABEL:
        return mask

    label = COLOR_TO_LABEL[spray_color]
    half_px = spray_size_m / (2 * GSD_M)
    center = PRECROP_SIZE // 2

    spray_top = center - half_px - crop_offset[0]
    spray_bottom = center + half_px - crop_offset[0]
    spray_left = center - half_px - crop_offset[1]
    spray_right = center + half_px - crop_offset[1]

    for r in range(GRID):
        patch_top = r * VIT_PATCH
        patch_bottom = (r + 1) * VIT_PATCH
        for c in range(GRID):
            patch_left = c * VIT_PATCH
            patch_right = (c + 1) * VIT_PATCH
            if (patch_bottom > spray_top and patch_top < spray_bottom and
                    patch_right > spray_left and patch_left < spray_right):
                mask[r, c] = label
    return mask


class Dataset224(Dataset):
    """Dataset with 224px center crop."""

    def __init__(self, hf_dataset, overlay=None, training=True, suppress_real_labels=False):
        self.data = hf_dataset
        self.overlay = overlay
        self.training = training
        self.suppress_real_labels = suppress_real_labels

        self.pre_transform = T.Compose([
            T.Resize((PRECROP_SIZE, PRECROP_SIZE), interpolation=PILImage.LANCZOS),
            T.RandomHorizontalFlip(0.5),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.15)], p=0.5),
        ]) if training else None

        self.post_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])
        self.eval_transform = T.Compose([
            T.Resize((PRECROP_SIZE, PRECROP_SIZE), interpolation=PILImage.LANCZOS),
            T.CenterCrop(MODEL_INPUT),
            T.ToTensor(),
            T.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        image = row["image"]
        metadata = {k: row[k] for k in row if k != "image"}

        if not self.training:
            tensor = self.eval_transform(image)
            mask = _generate_labels_224(
                (EVAL_OFFSET, EVAL_OFFSET),
                row.get("spray_size_m", 0.0),
                row.get("color", "none"),
            )
            return tensor, torch.from_numpy(mask).long(), metadata

        image = self.pre_transform(image)

        # Center crop to 224 (no random jitter for simplicity)
        img_np = np.array(image, dtype=np.uint8)
        y0 = (PRECROP_SIZE - MODEL_INPUT) // 2
        x0 = (PRECROP_SIZE - MODEL_INPUT) // 2
        crop_np = img_np[y0:y0 + MODEL_INPUT, x0:x0 + MODEL_INPUT].copy()
        crop_offset = (y0, x0)
        image = PILImage.fromarray(crop_np)

        if self.suppress_real_labels:
            mask = _generate_labels_224(crop_offset, 0.0, "none")
        else:
            mask = _generate_labels_224(
                crop_offset,
                row.get("spray_size_m", 0.0),
                row.get("color", "none"),
            )

        if self.overlay is not None:
            image_np = np.array(image, dtype=np.float32) / 255.0
            real_color = row.get("color", "none")
            overlay_color = real_color if real_color in ("red", "blue") else None
            # Overlay expects specific grid dim — pass mask and let it derive labels
            # But overlay uses GRID_DIM from config (24). We need custom label derivation.
            # Apply overlay on pixels, derive 14×14 labels ourselves.
            from patch.utils.synthetic import _make_wobbly_circle, BLOB_SIZE_RANGE_PX
            import math

            H, W = image_np.shape[:2]
            color_name = overlay_color or random.choice(["red", "blue"])

            n_blobs = random.randint(1, 10)
            final_mask = np.zeros((H, W), dtype=bool)

            for _ in range(n_blobs):
                cy = random.uniform(0, H - 1)
                cx = random.uniform(0, W - 1)
                base_r = random.uniform(BLOB_SIZE_RANGE_PX[0] / 2, BLOB_SIZE_RANGE_PX[1] / 2)
                blob_mask = _make_wobbly_circle(cy, cx, base_r, H, W)
                blob_mask &= ~final_mask
                if not blob_mask.any():
                    continue

                # Apply HSV delta from overlay's table
                if hasattr(self.overlay, 'tables') and self.overlay.tables.get(color_name):
                    from scipy.spatial import cKDTree
                    from scipy.ndimage import gaussian_filter
                    import colorsys

                    kd_tree, delta_table = self.overlay.tables[color_name]
                    soft_mask = gaussian_filter(blob_mask.astype(np.float32), sigma=3.0)

                    ys, xs = np.where(blob_mask)
                    pixels = image_np[ys, xs]

                    # Vectorized RGB→HSV
                    r_, g_, b_ = pixels[:, 0], pixels[:, 1], pixels[:, 2]
                    maxc = np.maximum(np.maximum(r_, g_), b_)
                    minc = np.minimum(np.minimum(r_, g_), b_)
                    v = maxc
                    diff = maxc - minc
                    s = np.where(maxc > 0, diff / maxc, 0.0)
                    h = np.zeros_like(maxc)
                    m = diff > 0
                    rc = np.where(m, (maxc - r_) / np.where(m, diff, 1), 0)
                    gc = np.where(m, (maxc - g_) / np.where(m, diff, 1), 0)
                    bc = np.where(m, (maxc - b_) / np.where(m, diff, 1), 0)
                    h = np.where(m & (r_ == maxc), bc - gc, h)
                    h = np.where(m & (g_ == maxc), 2.0 + rc - bc, h)
                    h = np.where(m & (b_ == maxc), 4.0 + gc - rc, h)
                    h = (h / 6.0) % 1.0
                    hsv = np.column_stack([h, s, v])

                    # Mean HSV for blob → NN lookup
                    hue_angles = h * 2 * np.pi
                    mean_h = (np.arctan2(np.mean(np.sin(hue_angles)), np.mean(np.cos(hue_angles))) / (2 * np.pi)) % 1.0
                    blob_hsv = np.array([mean_h, s.mean(), v.mean()])
                    _, didx = kd_tree.query(blob_hsv)
                    delta = delta_table[didx]

                    strength = soft_mask[ys, xs]
                    hsv[:, 0] = (hsv[:, 0] + delta[0] * strength) % 1.0
                    hsv[:, 1] = np.clip(hsv[:, 1] + delta[1] * strength, 0, 1)
                    hsv[:, 2] = np.clip(hsv[:, 2] + delta[2] * strength, 0, 1)

                    # Vectorized HSV→RGB
                    h2, s2, v2 = hsv[:, 0], hsv[:, 1], hsv[:, 2]
                    i_val = (h2 * 6.0).astype(int) % 6
                    f_val = h2 * 6.0 - np.floor(h2 * 6.0)
                    p = v2 * (1.0 - s2)
                    q = v2 * (1.0 - s2 * f_val)
                    t = v2 * (1.0 - s2 * (1.0 - f_val))
                    rgb_out = np.zeros_like(hsv)
                    for iv, (rv, gv, bv) in enumerate([(v2, t, p), (q, v2, p), (p, v2, t), (p, q, v2), (t, p, v2), (v2, p, q)]):
                        im = i_val == iv
                        rgb_out[im, 0] = rv[im]
                        rgb_out[im, 1] = gv[im]
                        rgb_out[im, 2] = bv[im]
                    image_np[ys, xs] = rgb_out.astype(np.float32)

                final_mask |= blob_mask

            # Derive 14×14 labels from blob mask
            fm = final_mask[:GRID * VIT_PATCH, :GRID * VIT_PATCH]
            fm = fm.reshape(GRID, VIT_PATCH, GRID, VIT_PATCH)
            patch_hits = fm.any(axis=(1, 3))
            mask[patch_hits] = COLOR_TO_LABEL[color_name]

            image = PILImage.fromarray((np.clip(image_np, 0, 1) * 255).astype(np.uint8))

        tensor = self.post_transform(image)
        return tensor, torch.from_numpy(mask).long(), metadata


def _balanced_mask_224(targets, neg_mult=NEG_MULT):
    B = targets.shape[0]
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for i in range(B):
        dye_idx = (targets[i] > 0).nonzero(as_tuple=False)
        n_dye = len(dye_idx)
        if n_dye > 0:
            mask[i, dye_idx[:, 0], dye_idx[:, 1]] = True
            bg_idx = (targets[i] == 0).nonzero(as_tuple=False)
            if len(bg_idx) > 0:
                n_sample = min(n_dye * neg_mult, len(bg_idx))
                perm = torch.randperm(len(bg_idx), device=targets.device)[:n_sample]
                mask[i, bg_idx[perm, 0], bg_idx[perm, 1]] = True
        else:
            bg_idx = (targets[i] == 0).nonzero(as_tuple=False)
            if len(bg_idx) > 0:
                n_sample = min(NO_DYE_NEG_SAMPLE, len(bg_idx))
                perm = torch.randperm(len(bg_idx), device=targets.device)[:n_sample]
                mask[i, bg_idx[perm, 0], bg_idx[perm, 1]] = True
    return mask


def _get_data(config):
    if config == "real_only":
        return load_dataset(HF_REPO, "sprayed", split="train")
    elif config == "hybrid":
        s = load_dataset(HF_REPO, "sprayed", split="train")
        a = load_dataset(HF_REPO, "unsprayed_annex", split="train")
        return concatenate_datasets([s, a])
    elif config == "synth_local":
        s = load_dataset(HF_REPO, "sprayed", split="train")
        a = load_dataset(HF_REPO, "unsprayed_annex", split="train")
        return concatenate_datasets([s, a])


def run(config: str, seed: int):
    print(f"224px crop: Config={config} LR={LR} Seed={seed}")
    set_seed(seed)

    from patch.utils.dataset import stratified_split
    ds = _get_data(config)
    tune_train, tune_val = stratified_split(ds, test_frac=0.2, seed=seed)

    # Val always sprayed only
    sprayed_val = load_dataset(HF_REPO, "sprayed", split="train")
    _, val_hf = stratified_split(sprayed_val, test_frac=0.2, seed=seed)

    # Build overlay from sprayed tiles
    overlay = None
    if config != "real_only":
        overlay = SyntheticDyeOverlay(sprayed_val)

    suppress = config == "synth_local"
    train_ds = Dataset224(tune_train, overlay=overlay, training=True, suppress_real_labels=suppress)
    val_ds = Dataset224(val_hf, overlay=None, training=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(model_name=MODEL_NAME_LARGE, device=device)
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=LR, weight_decay=0.01)

    train_history = []
    val_history = []
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        model.backbone.eval()
        total_loss = 0.0

        for images, masks, _ in tqdm(train_loader, desc="  train", leave=False):
            images = images.to(device)
            masks = masks.to(device).long()

            logits = model(images)
            ce = F.cross_entropy(logits, masks, reduction="none")
            loss_mask = _balanced_mask_224(masks)
            selected = ce[loss_mask]
            loss = selected.mean() if len(selected) > 0 else ce.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        train_loss = total_loss / max(len(train_ds), 1)
        train_history.append({"loss": train_loss})

        # Validate with super-patch F1
        model.eval()
        val_loss_total = 0.0
        all_preds = []
        all_metadata = []

        with torch.no_grad():
            for images, masks, metadata in tqdm(val_loader, desc="  val", leave=False):
                images = images.to(device)
                masks = masks.to(device).long()

                logits = model(images)
                ce = F.cross_entropy(logits, masks, reduction="none")
                loss_mask = _balanced_mask_224(masks)
                selected = ce[loss_mask]
                val_loss_total += (selected.mean() if len(selected) > 0 else ce.mean()).item() * images.size(0)

                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_metadata.extend(metadata)

        val_loss = val_loss_total / max(len(val_ds), 1)
        preds_cat = torch.cat(all_preds, dim=0)

        # compute_spray_metrics uses EVAL_CROP_OFFSET from config (64px for 384).
        # We need 14×14 grid metrics. Compute manually.
        tp = fn = fp = tn = 0
        for i, meta in enumerate(all_metadata):
            pred = preds_cat[i].numpy()
            gt = _generate_labels_224(
                (EVAL_OFFSET, EVAL_OFFSET),
                meta.get("spray_size_m", 0.0),
                meta.get("color", "none"),
            )
            spray = gt > 0
            periph = ~spray
            if spray.any():
                if (pred[spray] > 0).any():
                    tp += 1
                else:
                    fn += 1
            fp += int((pred[periph] > 0).sum())
            tn += int((pred[periph] == 0).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        val_record = {"loss": val_loss, "f1": f1, "precision": precision, "recall": recall}
        val_history.append(val_record)

        is_best = f1 > best_val_f1
        if is_best:
            best_val_f1 = f1
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.classifier.state_dict().items()}

        star = " *" if is_best else ""
        print(
            f"Epoch {epoch:>3}/{N_EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"F1={f1:.4f}  "
            f"P={precision:.4f}  "
            f"R={recall:.4f}{star}"
        )

    if best_state is not None:
        model.classifier.load_state_dict(best_state)

    print(f"Best epoch: {best_epoch} (F1={best_val_f1:.4f})")

    results = {
        "train_history": train_history,
        "val_history": val_history,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "lr": LR,
        "seed": seed,
        "config": config,
        "input_size": MODEL_INPUT,
        "grid_dim": GRID,
    }
    out_path = os.path.join(RESULTS_DIR, f"{config}_seed={seed}.json")
    save_results(results, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True)
    args = parser.parse_args()
    config_idx, seed = divmod(args.idx, 3)
    config = CONFIGS[config_idx]
    run(config=config, seed=seed)
