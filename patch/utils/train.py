"""
Training loop for patch-level dye classification.
"""

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from patch.utils.config import WEIGHT_DECAY


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

        Returns dict with 'loss', 'accuracy', and per-tile predictions/labels.
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for images, masks, _ in loader:
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

        n = len(loader.dataset)
        return {
            "loss": total_loss / max(n, 1),
            "accuracy": correct / max(total, 1),
            "preds": torch.cat(all_preds, dim=0),
            "labels": torch.cat(all_labels, dim=0),
        }

    def train(self, train_loader, val_loader, epochs: int, verbose: bool = True):
        """Full training loop with best-model tracking.

        Returns
        -------
        dict with 'train_history', 'val_history', 'best_epoch', 'best_val_loss'.
        """
        train_history = []
        val_history = []
        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)

            # Drop non-serialisable tensors from history
            val_record = {k: v for k, v in val_metrics.items() if k not in ("preds", "labels")}

            train_history.append(train_metrics)
            val_history.append(val_record)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                best_state = {
                    k: v.clone() for k, v in self.model.classifier.state_dict().items()
                }

            if verbose:
                print(
                    f"Epoch {epoch}/{epochs}  "
                    f"train_loss={train_metrics['loss']:.4f}  "
                    f"val_loss={val_metrics['loss']:.4f}  "
                    f"val_acc={val_metrics['accuracy']:.4f}"
                )

        # Restore best model
        if best_state is not None:
            self.model.classifier.load_state_dict(best_state)

        return {
            "train_history": train_history,
            "val_history": val_history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
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
