"""
Loss function sweep: focal vs CE with 10x neg ratio.

Ternary classification (none=0, red=1, blue=2), all spray patches as positives,
10x negatives, dinov3-vitl16 (Large), 30 epochs, 80/20 stratified split.
F1 eval is color-agnostic (pred > 0 = dye).

8 jobs = 2 loss fns × 2 LRs × 2 seeds
SLURM array: --array=0-7
Index mapping: divmod(idx, 4) → (seed, config_idx)

Usage:
  python -u -m patch.debug.pos_patch_centered --idx 0  # seed=0, CE/1e-4
"""

import argparse
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from patch.utils.config import FOCAL_ALPHA, FOCAL_GAMMA
from patch.utils.dataset import DyePatchDataset, stratified_split
from patch.utils.models import create_model
from patch.utils.train import compute_spray_metrics, save_results, set_seed

HF_REPO = "mpg-ranch/dye-patch"
RESULTS_DIR = "patch/debug/results/loss_sweep"
N_EPOCHS = 30
N_NEG_RATIO = 10
NUM_CLASSES = 3
MODEL_NAME_LARGE = "facebook/dinov3-vitl16-pretrain-sat493m"

CONFIGS = [
    ("ce", 1e-4),
    ("ce", 5e-4),
    ("focal", 1e-4),
    ("focal", 5e-4),
]
COLOR_TO_LABEL = {"red": 1, "blue": 2}


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    metadata = [b[2] for b in batch]
    return images, masks, metadata


def _relabel_ternary(masks, metadata_batch):
    """Replace binary dye label (1) with color-specific labels (red=1, blue=2)."""
    masks = masks.clone()
    for i, meta in enumerate(metadata_batch):
        color = meta.get("color", "none")
        if color in COLOR_TO_LABEL:
            dye_mask = masks[i] == 1
            masks[i][dye_mask] = COLOR_TO_LABEL[color]
    return masks


def _build_mask(targets):
    """All positive spray patches + 10x random negatives."""
    B = targets.shape[0]
    mask = torch.zeros_like(targets, dtype=torch.bool)

    for i in range(B):
        dye_idx = (targets[i] > 0).nonzero(as_tuple=False)

        if len(dye_idx) > 0:
            mask[i, dye_idx[:, 0], dye_idx[:, 1]] = True
            n_pos = len(dye_idx)

            bg_idx = (targets[i] == 0).nonzero(as_tuple=False)
            if len(bg_idx) > 0:
                n_sample = min(n_pos * N_NEG_RATIO, len(bg_idx))
                perm = torch.randperm(len(bg_idx), device=targets.device)[:n_sample]
                sampled = bg_idx[perm]
                mask[i, sampled[:, 0], sampled[:, 1]] = True
        else:
            bg_idx = (targets[i] == 0).nonzero(as_tuple=False)
            if len(bg_idx) > 0:
                n_sample = min(10, len(bg_idx))
                perm = torch.randperm(len(bg_idx), device=targets.device)[:n_sample]
                sampled = bg_idx[perm]
                mask[i, sampled[:, 0], sampled[:, 1]] = True

    return mask


def compute_loss(logits, targets, loss_fn):
    """CE or focal loss on masked patches."""
    ce = F.cross_entropy(logits, targets, reduction="none")

    if loss_fn == "focal":
        pt = torch.exp(-ce)
        alpha_t = torch.where(targets > 0, FOCAL_ALPHA, 1 - FOCAL_ALPHA)
        per_patch = alpha_t * (1 - pt) ** FOCAL_GAMMA * ce
    else:
        per_patch = ce

    mask = _build_mask(targets)
    selected = per_patch[mask]
    if len(selected) == 0:
        return per_patch.mean()
    return selected.mean()


def run(seed: int, loss_fn: str, lr: float):
    print(f"Loss sweep: loss={loss_fn} LR={lr} Seed={seed}")
    set_seed(seed)

    ds = load_dataset(HF_REPO, "sprayed", split="train")
    tune_train, tune_val = stratified_split(ds, test_frac=0.2, seed=seed)

    train_ds = DyePatchDataset(tune_train, overlay=None, training=True)
    val_ds = DyePatchDataset(tune_val, overlay=None, training=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(model_name=MODEL_NAME_LARGE, num_classes=NUM_CLASSES, device=device)
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=lr, weight_decay=0.01)

    train_history = []
    val_history = []
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        model.backbone.eval()
        total_loss = 0.0

        for images, masks, metadata in tqdm(train_loader, desc="  train", leave=False):
            images = images.to(device)
            masks = masks.to(device).long()
            masks = _relabel_ternary(masks, metadata)

            logits = model(images)
            loss = compute_loss(logits, masks, loss_fn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        train_loss = total_loss / max(len(train_ds), 1)
        train_history.append({"loss": train_loss})

        model.eval()
        val_loss_total = 0.0
        all_preds = []
        all_metadata = []

        with torch.no_grad():
            for images, masks, metadata in tqdm(val_loader, desc="  val", leave=False):
                images = images.to(device)
                masks = masks.to(device).long()
                masks = _relabel_ternary(masks, metadata)

                logits = model(images)
                loss = compute_loss(logits, masks, loss_fn)
                val_loss_total += loss.item() * images.size(0)

                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_metadata.extend(metadata)

        val_loss = val_loss_total / max(len(val_ds), 1)
        preds_cat = torch.cat(all_preds, dim=0)
        metrics = compute_spray_metrics(preds_cat, all_metadata)

        val_record = {"loss": val_loss, "f1": metrics["f1"],
                      "precision": metrics["precision"], "recall": metrics["recall"]}
        val_history.append(val_record)

        is_best = metrics["f1"] > best_val_f1
        if is_best:
            best_val_f1 = metrics["f1"]
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.classifier.state_dict().items()}

        star = " *" if is_best else ""
        print(
            f"Epoch {epoch:>3}/{N_EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"P={metrics['precision']:.4f}  "
            f"R={metrics['recall']:.4f}{star}"
        )

    if best_state is not None:
        model.classifier.load_state_dict(best_state)

    print(f"Best epoch: {best_epoch} (F1={best_val_f1:.4f})")

    results = {
        "train_history": train_history,
        "val_history": val_history,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "lr": lr,
        "seed": seed,
        "loss_fn": loss_fn,
        "model": "large",
    }
    out_path = os.path.join(RESULTS_DIR, f"{loss_fn}_lr={lr}_seed={seed}.json")
    save_results(results, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True)
    args = parser.parse_args()
    seed, config_idx = divmod(args.idx, len(CONFIGS))
    loss_fn, lr = CONFIGS[config_idx]
    run(seed=seed, loss_fn=loss_fn, lr=lr)
