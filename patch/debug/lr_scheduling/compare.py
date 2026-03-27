"""
Compare LR schedules on real_only config.

9 jobs = 3 schedules × 3 seeds
  idx 0-2: fixed LR=5e-4, seeds 0-2
  idx 3-5: cosine 1e-3 → 1e-5, seeds 0-2
  idx 6-8: cosine with 5-epoch warmup 1e-5 → 1e-3 → 1e-5, seeds 0-2

Ternary CE, 8x neg samples, Large model, 100 epochs, 80/20 stratified split.

SLURM array: --array=0-8

Usage:
  python -u -m patch.debug.lr_scheduling.compare --idx 0  # fixed, seed=0
  python -u -m patch.debug.lr_scheduling.compare --idx 3  # cosine, seed=0
  python -u -m patch.debug.lr_scheduling.compare --idx 6  # warmup_cosine, seed=0
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from patch.utils.config import HF_REPO, NO_DYE_NEG_SAMPLE
from patch.utils.dataset import DyePatchDataset, stratified_split
from patch.utils.models import create_model
from patch.utils.train import collate_fn, compute_spray_metrics, save_results, set_seed

RESULTS_DIR = "patch/debug/lr_scheduling/results"
MODEL_NAME_LARGE = "facebook/dinov3-vitl16-pretrain-sat493m"
NEG_MULT = 8
N_EPOCHS = 100
WARMUP_EPOCHS = 5
SCHEDULES = ["fixed", "cosine", "warmup_cosine"]

# LR configs per schedule
SCHEDULE_CONFIG = {
    "fixed":          {"start_lr": 5e-4, "eta_min": 5e-4},
    "cosine":         {"start_lr": 1e-3, "eta_min": 1e-5},
    "warmup_cosine":  {"start_lr": 1e-3, "eta_min": 1e-5},
}


def _balanced_mask(targets, neg_multiplier):
    B = targets.shape[0]
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for i in range(B):
        dye_idx = (targets[i] > 0).nonzero(as_tuple=False)
        n_dye = len(dye_idx)
        if n_dye > 0:
            mask[i, dye_idx[:, 0], dye_idx[:, 1]] = True
            bg_idx = (targets[i] == 0).nonzero(as_tuple=False)
            if len(bg_idx) > 0:
                n_sample = min(n_dye * neg_multiplier, len(bg_idx))
                perm = torch.randperm(len(bg_idx), device=targets.device)[:n_sample]
                mask[i, bg_idx[perm, 0], bg_idx[perm, 1]] = True
        else:
            bg_idx = (targets[i] == 0).nonzero(as_tuple=False)
            if len(bg_idx) > 0:
                n_sample = min(NO_DYE_NEG_SAMPLE, len(bg_idx))
                perm = torch.randperm(len(bg_idx), device=targets.device)[:n_sample]
                mask[i, bg_idx[perm, 0], bg_idx[perm, 1]] = True
    return mask


def compute_loss(logits, targets):
    ce = F.cross_entropy(logits, targets, reduction="none")
    mask = _balanced_mask(targets, NEG_MULT)
    selected = ce[mask]
    return selected.mean() if len(selected) > 0 else ce.mean()


def run(seed: int, schedule: str):
    cfg = SCHEDULE_CONFIG[schedule]
    start_lr = cfg["start_lr"]
    eta_min = cfg["eta_min"]

    print(f"Schedule={schedule} start_lr={start_lr} eta_min={eta_min} Seed={seed} NegMult={NEG_MULT}")
    set_seed(seed)

    ds = load_dataset(HF_REPO, "sprayed", split="train")
    tune_train, tune_val = stratified_split(ds, test_frac=0.2, seed=seed)

    train_ds = DyePatchDataset(tune_train, overlay=None, training=True)
    val_ds = DyePatchDataset(tune_val, overlay=None, training=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(model_name=MODEL_NAME_LARGE, device=device)

    if schedule == "warmup_cosine":
        # Start at eta_min, warmup will ramp to start_lr
        optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=eta_min, weight_decay=0.01)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=start_lr / eta_min, total_iters=WARMUP_EPOCHS
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=N_EPOCHS - WARMUP_EPOCHS, eta_min=eta_min
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS]
        )
    elif schedule == "cosine":
        optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=start_lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=N_EPOCHS, eta_min=eta_min
        )
    else:  # fixed
        optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=start_lr, weight_decay=0.01)
        scheduler = None

    train_history = []
    val_history = []
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, N_EPOCHS + 1):
        # Train
        model.train()
        model.backbone.eval()
        total_loss = 0.0

        for images, masks, _ in tqdm(train_loader, desc="  train", leave=False):
            images = images.to(device)
            masks = masks.to(device).long()

            logits = model(images)
            loss = compute_loss(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        if scheduler:
            scheduler.step()

        train_loss = total_loss / max(len(train_ds), 1)
        train_history.append({"loss": train_loss})

        # Validate
        model.eval()
        val_loss_total = 0.0
        all_preds = []
        all_metadata = []

        with torch.no_grad():
            for images, masks, metadata in tqdm(val_loader, desc="  val", leave=False):
                images = images.to(device)
                masks = masks.to(device).long()

                logits = model(images)
                loss = compute_loss(logits, masks)
                val_loss_total += loss.item() * images.size(0)

                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_metadata.extend(metadata)

        val_loss = val_loss_total / max(len(val_ds), 1)
        preds_cat = torch.cat(all_preds, dim=0)
        metrics = compute_spray_metrics(preds_cat, all_metadata)

        current_lr = scheduler.get_last_lr()[0] if scheduler else start_lr
        val_record = {"loss": val_loss, "f1": metrics["f1"],
                      "precision": metrics["precision"], "recall": metrics["recall"],
                      "lr": current_lr}
        val_history.append(val_record)

        is_best = metrics["f1"] > best_val_f1
        if is_best:
            best_val_f1 = metrics["f1"]
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.classifier.state_dict().items()}

        star = " *" if is_best else ""
        print(
            f"Epoch {epoch:>3}/{N_EPOCHS}  "
            f"lr={current_lr:.6f}  "
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
        "start_lr": start_lr,
        "eta_min": eta_min,
        "schedule": schedule,
        "seed": seed,
        "neg_multiplier": NEG_MULT,
    }
    out_path = os.path.join(RESULTS_DIR, f"{schedule}_seed={seed}.json")
    save_results(results, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True)
    args = parser.parse_args()
    sched_idx, seed = divmod(args.idx, 3)
    schedule = SCHEDULES[sched_idx]
    run(seed=seed, schedule=schedule)
