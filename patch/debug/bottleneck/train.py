"""
Bottleneck projection head: Linear(1024, 32) → ReLU → Linear(32, 3).

Forces a 32-dim compressed representation before classification.
Limits memorization capacity in the high-dimensional embedding space.

6 jobs = 2 configs (baseline, bottleneck) × 3 seeds
SLURM array: --array=0-5

Usage:
  python -u -m patch.debug.bottleneck.train --idx 0  # baseline, seed=0
  python -u -m patch.debug.bottleneck.train --idx 3  # bottleneck, seed=0
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from patch.utils.config import GRID_DIM, HF_REPO, NO_DYE_NEG_SAMPLE, NUM_CLASSES
from patch.utils.dataset import DyePatchDataset, stratified_split
from patch.utils.models import create_model
from patch.utils.train import collate_fn, compute_spray_metrics, save_results, set_seed

RESULTS_DIR = "patch/debug/bottleneck/results"
MODEL_NAME_LARGE = "facebook/dinov3-vitl16-pretrain-sat493m"
LR = 5e-4
N_EPOCHS = 50
NEG_MULT = 8
BOTTLENECK_DIM = 32
CONFIGS = ["baseline", "bottleneck"]


def create_bottleneck_model(use_bottleneck=False, device="cuda"):
    """Create model with optional bottleneck head."""
    from transformers import AutoModel
    from dotenv import load_dotenv
    from patch.utils.models import DyePatchModel
    load_dotenv()

    backbone = AutoModel.from_pretrained(MODEL_NAME_LARGE, token=os.environ.get("HF_TOKEN"))
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    hidden_size = backbone.config.hidden_size  # 1024

    if use_bottleneck:
        classifier = nn.Sequential(
            nn.Linear(hidden_size, BOTTLENECK_DIM),
            nn.ReLU(),
            nn.Linear(BOTTLENECK_DIM, NUM_CLASSES),
        )
    else:
        classifier = nn.Sequential(nn.Linear(hidden_size, NUM_CLASSES))

    model = DyePatchModel(backbone, classifier)

    if isinstance(device, str):
        if device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    return model.to(device)


def _balanced_mask(targets):
    B = targets.shape[0]
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for i in range(B):
        dye_idx = (targets[i] > 0).nonzero(as_tuple=False)
        n_dye = len(dye_idx)
        if n_dye > 0:
            mask[i, dye_idx[:, 0], dye_idx[:, 1]] = True
            bg_idx = (targets[i] == 0).nonzero(as_tuple=False)
            if len(bg_idx) > 0:
                n_sample = min(n_dye * NEG_MULT, len(bg_idx))
                perm = torch.randperm(len(bg_idx), device=targets.device)[:n_sample]
                mask[i, bg_idx[perm, 0], bg_idx[perm, 1]] = True
        else:
            bg_idx = (targets[i] == 0).nonzero(as_tuple=False)
            if len(bg_idx) > 0:
                n_sample = min(NO_DYE_NEG_SAMPLE, len(bg_idx))
                perm = torch.randperm(len(bg_idx), device=targets.device)[:n_sample]
                mask[i, bg_idx[perm, 0], bg_idx[perm, 1]] = True
    return mask


def run(seed: int, use_bottleneck: bool):
    config_name = "bottleneck" if use_bottleneck else "baseline"
    print(f"Bottleneck: config={config_name} dim={BOTTLENECK_DIM if use_bottleneck else 'N/A'} "
          f"LR={LR} Seed={seed}")
    set_seed(seed)

    ds = load_dataset(HF_REPO, "sprayed", split="train")
    tune_train, tune_val = stratified_split(ds, test_frac=0.2, seed=seed)

    train_ds = DyePatchDataset(tune_train, overlay=None, training=True)
    val_ds = DyePatchDataset(tune_val, overlay=None, training=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_bottleneck_model(use_bottleneck=use_bottleneck, device=device)
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=LR, weight_decay=0.01)

    n_params = sum(p.numel() for p in model.get_trainable_parameters())
    print(f"  Trainable params: {n_params:,}")

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
            loss_mask = _balanced_mask(masks)
            selected = ce[loss_mask]
            loss = selected.mean() if len(selected) > 0 else ce.mean()
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
                logits = model(images)
                ce = F.cross_entropy(logits, masks, reduction="none")
                loss_mask = _balanced_mask(masks)
                selected = ce[loss_mask]
                val_loss_total += (selected.mean() if len(selected) > 0 else ce.mean()).item() * images.size(0)
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
        print(f"Epoch {epoch:>3}/{N_EPOCHS}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  F1={metrics['f1']:.4f}  "
              f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}{star}")

    if best_state is not None:
        model.classifier.load_state_dict(best_state)
    print(f"Best epoch: {best_epoch} (F1={best_val_f1:.4f})")

    results = {"train_history": train_history, "val_history": val_history,
               "best_epoch": best_epoch, "best_val_f1": best_val_f1,
               "lr": LR, "seed": seed, "config": config_name,
               "bottleneck_dim": BOTTLENECK_DIM if use_bottleneck else None}
    out_path = os.path.join(RESULTS_DIR, f"{config_name}_seed={seed}.json")
    save_results(results, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True)
    args = parser.parse_args()
    config_idx, seed = divmod(args.idx, 3)
    run(seed=seed, use_bottleneck=config_idx == 1)
