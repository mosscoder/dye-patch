"""
Training loop for patch-level dye classification.
"""

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn

from patch.utils.config import EVAL_CROP_OFFSET, WEIGHT_DECAY
from patch.utils.dataset import generate_patch_labels


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_spray_metrics(preds, metadata_list):
    """Compute super-patch F1 from predictions and metadata dicts.

    Same logic as data_source.compute_spray_metrics but accepts a list of
    metadata dicts (from DataLoader collate) instead of an HF dataset.

    Super-patch (spray zone): one binary verdict per sprayed tile.
      TP: ANY patch in spray bounds predicts dye.
      FN: NO patch in spray bounds predicts dye.

    Peripheral patches (outside spray bounds on all tiles):
      FP: individual patches predicting dye where there is none.
      TN: individual patches correctly predicting none.
    """
    tp = fn = 0
    fp = tn = 0
    center_offset = EVAL_CROP_OFFSET

    for i, meta in enumerate(metadata_list):
        pred = preds[i].numpy() if hasattr(preds[i], "numpy") else preds[i]

        mask = generate_patch_labels(
            crop_offset=(center_offset, center_offset),
            spray_size_m=meta.get("spray_size_m", 0.0),
            spray_color=meta.get("color", "none"),
        )
        spray_patches = mask > 0
        peripheral_patches = ~spray_patches

        if spray_patches.any():
            if (pred[spray_patches] > 0).any():
                tp += 1
            else:
                fn += 1

        periph_preds = pred[peripheral_patches]
        fp += int((periph_preds > 0).sum())
        tn += int((periph_preds == 0).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
    }


class PatchTrainer:
    """Train a DyePatchModel with per-patch cross-entropy loss.

    Parameters
    ----------
    model : DyePatchModel
    lr : float
    weight_decay : float
    device : torch.device or str
    """

    def __init__(self, model, lr: float, weight_decay: float = WEIGHT_DECAY, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.optimizer = torch.optim.AdamW(
            model.get_trainable_parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, loader):
        """Run one training epoch.

        Returns dict with 'loss' and 'accuracy' (patch-level).
        """
        self.model.train()
        # Keep backbone in eval mode (frozen BatchNorm etc.)
        self.model.backbone.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, masks, _ in loader:
            images = images.to(self.device)
            masks = masks.to(self.device).long()

            logits = self.model(images)  # [B, C, 24, 24]
            loss = self.criterion(logits, masks)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)  # [B, 24, 24]
            correct += (preds == masks).sum().item()
            total += masks.numel()

        n = len(loader.dataset)
        return {"loss": total_loss / max(n, 1), "accuracy": correct / max(total, 1)}

    @torch.no_grad()
    def validate_epoch(self, loader):
        """Run one validation epoch.

        Returns dict with 'loss', 'accuracy', per-tile predictions/labels,
        and collected metadata for super-patch F1 computation.
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_metadata = []

        for images, masks, metadata in loader:
            images = images.to(self.device)
            masks = masks.to(self.device).long()

            logits = self.model(images)
            loss = self.criterion(logits, masks)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == masks).sum().item()
            total += masks.numel()

            all_preds.append(preds.cpu())
            all_labels.append(masks.cpu())
            all_metadata.extend(metadata)

        n = len(loader.dataset)
        preds_cat = torch.cat(all_preds, dim=0)

        spray_metrics = compute_spray_metrics(preds_cat, all_metadata)

        return {
            "loss": total_loss / max(n, 1),
            "accuracy": correct / max(total, 1),
            "f1": spray_metrics["f1"],
            "precision": spray_metrics["precision"],
            "recall": spray_metrics["recall"],
            "preds": preds_cat,
            "labels": torch.cat(all_labels, dim=0),
        }

    def train(self, train_loader, val_loader, epochs: int, verbose: bool = True):
        """Full training loop with best-model tracking by super-patch F1.

        Returns
        -------
        dict with 'train_history', 'val_history', 'best_epoch', 'best_val_f1'.
        """
        train_history = []
        val_history = []
        best_val_f1 = -1.0
        best_epoch = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)

            # Drop non-serialisable tensors from history
            val_record = {k: v for k, v in val_metrics.items() if k not in ("preds", "labels")}

            train_history.append(train_metrics)
            val_history.append(val_record)

            is_best = val_metrics["f1"] > best_val_f1
            if is_best:
                best_val_f1 = val_metrics["f1"]
                best_epoch = epoch
                best_state = {
                    k: v.clone() for k, v in self.model.classifier.state_dict().items()
                }

            if verbose:
                star = " *" if is_best else ""
                print(
                    f"Epoch {epoch:>3}/{epochs}  "
                    f"train_loss={train_metrics['loss']:.4f}  "
                    f"val_loss={val_metrics['loss']:.4f}  "
                    f"F1={val_metrics['f1']:.4f}  "
                    f"P={val_metrics['precision']:.4f}  "
                    f"R={val_metrics['recall']:.4f}{star}"
                )

        # Restore best model
        if best_state is not None:
            self.model.classifier.load_state_dict(best_state)

        if verbose:
            print(f"Best epoch: {best_epoch} (F1={best_val_f1:.4f})")

        return {
            "train_history": train_history,
            "val_history": val_history,
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
        }


def save_results(results: dict, path: str):
    """Save training results to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(path: str) -> dict:
    """Load training results from JSON."""
    with open(path) as f:
        return json.load(f)
