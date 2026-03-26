"""
Centroid-only training experiment.

For sprayed tiles: 1 positive patch (closest to spray center) + 1 random
negative from the peripheral zone (fully outside spray bounds + margin).

For no-dye tiles: 1 random background patch.

Binary cross-entropy, dinov3-vitl16 (Large), LR=5e-3, seeds 0-2, 30 epochs.
1 centroid positive + 10 random negatives from non-spray area.
SLURM array: --array=0-2 (one job per seed)

Usage:
  python -u -m patch.debug.pos_patch_centered --idx 0  # seed=0
"""

import argparse
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from patch.utils.config import MODEL_NAME
from patch.utils.dataset import DyePatchDataset, tuning_split
from patch.utils.models import create_model
from patch.utils.train import compute_spray_metrics, save_results, set_seed

HF_REPO = "mpg-ranch/dye-patch"
RESULTS_DIR = "patch/debug/results/centroid"
N_EPOCHS = 30
MODEL_NAME_LARGE = "facebook/dinov3-vitl16-pretrain-sat493m"


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    metadata = [b[2] for b in batch]
    return images, masks, metadata


N_NEG = 10


def _build_mask(targets):
    """Build centroid-only loss mask from the targets tensor directly.

    Sprayed tiles: 1 patch closest to centroid of dye region + N_NEG random
    negatives from all non-spray patches.
    No-dye tiles: N_NEG random background patches.
    """
    B = targets.shape[0]
    mask = torch.zeros_like(targets, dtype=torch.bool)

    for i in range(B):
        dye_idx = (targets[i] == 1).nonzero(as_tuple=False)

        if len(dye_idx) > 0:
            # 1 positive: patch closest to centroid
            centroid_r = dye_idx[:, 0].float().mean()
            centroid_c = dye_idx[:, 1].float().mean()
            dists = (dye_idx[:, 0].float() - centroid_r) ** 2 + (dye_idx[:, 1].float() - centroid_c) ** 2
            best = dists.argmin()
            mask[i, dye_idx[best, 0], dye_idx[best, 1]] = True

            # N_NEG random negatives from all non-spray patches
            bg_idx = (targets[i] == 0).nonzero(as_tuple=False)
            if len(bg_idx) > 0:
                n_sample = min(N_NEG, len(bg_idx))
                perm = torch.randperm(len(bg_idx), device=targets.device)[:n_sample]
                sampled = bg_idx[perm]
                mask[i, sampled[:, 0], sampled[:, 1]] = True
        else:
            # No-dye tile: N_NEG random background patches
            bg_idx = (targets[i] == 0).nonzero(as_tuple=False)
            if len(bg_idx) > 0:
                n_sample = min(N_NEG, len(bg_idx))
                perm = torch.randperm(len(bg_idx), device=targets.device)[:n_sample]
                sampled = bg_idx[perm]
                mask[i, sampled[:, 0], sampled[:, 1]] = True

    return mask


def balanced_bce(logits, targets):
    """Binary cross-entropy on centroid + 1 peripheral patch only."""
    ce = F.cross_entropy(logits, targets, reduction="none")
    mask = _build_mask(targets)
    selected = ce[mask]
    if len(selected) == 0:
        return ce.mean()
    return selected.mean()


def run(seed: int, lr: float):
    print(f"Centroid training: Model=large LR={lr} Seed={seed}")
    set_seed(seed)

    from patch.utils.dataset import stratified_split
    ds = load_dataset(HF_REPO, "sprayed", split="train")
    tune_train, tune_val = stratified_split(ds, test_frac=0.2, seed=seed)

    train_ds = DyePatchDataset(tune_train, overlay=None, training=True)
    val_ds = DyePatchDataset(tune_val, overlay=None, training=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(model_name=MODEL_NAME_LARGE, device=device)
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=lr, weight_decay=0.01)

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

        for images, masks, metadata in tqdm(train_loader, desc="  train", leave=False):
            images = images.to(device)
            masks = masks.to(device).long()

            logits = model(images)
            loss = balanced_bce(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        train_loss = total_loss / max(len(train_ds), 1)
        train_history.append({"loss": train_loss})

        # Validate (standard super-patch F1 on all patches)
        model.eval()
        val_loss_total = 0.0
        all_preds = []
        all_metadata = []

        with torch.no_grad():
            for images, masks, metadata in tqdm(val_loader, desc="  val", leave=False):
                images = images.to(device)
                masks = masks.to(device).long()

                logits = model(images)
                loss = balanced_bce(logits, masks)
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
        "model": "large",
    }
    out_path = os.path.join(RESULTS_DIR, f"large_lr={lr}_seed={seed}.json")
    save_results(results, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True)
    args = parser.parse_args()
    seed = args.idx
    run(seed=seed, lr=5e-4)
