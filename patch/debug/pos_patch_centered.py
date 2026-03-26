"""
Centroid-only training experiment.

For sprayed tiles: 1 positive patch (closest to spray center) + 1 random
negative from the peripheral zone (fully outside spray bounds + margin).

For no-dye tiles: 1 random background patch.

Binary cross-entropy, LR=1e-4, seeds 0-2, 15 epochs, two model sizes.
SLURM array: --array=0-5 (3 seeds × 2 models)
Index mapping: divmod(idx, 2) → (seed, model_idx)
  model_idx 0 = dinov3-vit7b16 (7B)
  model_idx 1 = dinov3-vitl16 (Large)

Usage:
  python -u -m patch.debug.pos_patch_centered --idx 0  # seed=0, 7B
  python -u -m patch.debug.pos_patch_centered --idx 1  # seed=0, Large
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
LR = 1e-4
N_EPOCHS = 15
MODEL_NAME_LARGE = "facebook/dinov3-vitl16-pretrain-sat493m"
MODELS = [MODEL_NAME, MODEL_NAME_LARGE]
MODEL_LABELS = ["7b", "large"]


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    metadata = [b[2] for b in batch]
    return images, masks, metadata


def _build_mask(targets):
    """Build centroid-only loss mask from the targets tensor directly.

    Sprayed tiles: 1 patch closest to centroid of dye region + 1 random
    peripheral negative (outside spray extent + 1 patch margin).
    No-dye tiles: 1 random background patch.
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

            # 1 negative: random from peripheral zone
            spray_radius = dists.max().sqrt() + 1.0
            margin = spray_radius + 1.0

            bg_idx = (targets[i] == 0).nonzero(as_tuple=False)
            if len(bg_idx) > 0:
                bg_dists = ((bg_idx[:, 0].float() - centroid_r) ** 2 +
                            (bg_idx[:, 1].float() - centroid_c) ** 2).sqrt()
                peripheral = bg_idx[bg_dists > margin]

                if len(peripheral) > 0:
                    idx = torch.randint(len(peripheral), (1,), device=targets.device)
                    mask[i, peripheral[idx, 0], peripheral[idx, 1]] = True
                else:
                    idx = torch.randint(len(bg_idx), (1,), device=targets.device)
                    mask[i, bg_idx[idx, 0], bg_idx[idx, 1]] = True
        else:
            # No-dye tile: 1 random background patch
            bg_idx = (targets[i] == 0).nonzero(as_tuple=False)
            if len(bg_idx) > 0:
                idx = torch.randint(len(bg_idx), (1,), device=targets.device)
                mask[i, bg_idx[idx, 0], bg_idx[idx, 1]] = True

    return mask


def balanced_bce(logits, targets):
    """Binary cross-entropy on centroid + 1 peripheral patch only."""
    ce = F.cross_entropy(logits, targets, reduction="none")
    mask = _build_mask(targets)
    selected = ce[mask]
    if len(selected) == 0:
        return ce.mean()
    return selected.mean()


def run(seed: int, model_idx: int):
    model_name = MODELS[model_idx]
    model_label = MODEL_LABELS[model_idx]
    print(f"Centroid training: Model={model_label} LR={LR} Seed={seed}")
    set_seed(seed)

    ds = load_dataset(HF_REPO, "sprayed", split="train")
    tune_train, tune_val = tuning_split(ds, seed=seed)

    train_ds = DyePatchDataset(tune_train, overlay=None, training=True)
    val_ds = DyePatchDataset(tune_val, overlay=None, training=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(model_name=model_name, device=device)
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=LR, weight_decay=0.01)

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
        "lr": LR,
        "seed": seed,
        "model": model_label,
    }
    out_path = os.path.join(RESULTS_DIR, f"{model_label}_seed={seed}.json")
    save_results(results, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True)
    args = parser.parse_args()
    seed, model_idx = divmod(args.idx, len(MODELS))
    run(seed=seed, model_idx=model_idx)
