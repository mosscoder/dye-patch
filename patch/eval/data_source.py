"""
Train final models and evaluate all four configs on the test set.

SLURM array: --array=0-3 (4 configs, seed=0)
"""

import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from patch.utils.config import CONFIGS, EVAL_SEED, HF_REPO
from patch.utils.dataset import DyePatchDataset
from patch.utils.models import create_model, save_head
from patch.utils.synthetic import SyntheticDyeOverlay
from patch.utils.train import PatchTrainer, compute_spray_metrics, save_results, set_seed
from patch.utils.dataset import get_train_data_for_config
from patch.tuning.sweep_epochs import load_best_overlay, select_best_epoch
from patch.utils.train import collate_fn
from patch.tuning.sweep_lr import select_best_lr
from patch.tuning.sweep_neg import select_best_neg

RESULTS_DIR = "patch/eval/results/data_source"



def run_eval(idx: int):
    """Train and evaluate one config."""
    config = CONFIGS[idx]
    seed = EVAL_SEED
    print(f"Config={config} Seed={seed}")

    set_seed(seed)

    lr = select_best_lr(config)
    neg_mult = select_best_neg()
    overlay = load_best_overlay(config)
    n_epochs = select_best_epoch(config)

    # Train on full training split
    train_hf = get_train_data_for_config(config)
    train_ds = DyePatchDataset(train_hf, overlay=overlay, training=True)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(device=device)
    trainer = PatchTrainer(model, lr=lr, neg_multiplier=neg_mult, device=device)

    # Train (no val loader for final training, use all train data)
    for epoch in range(1, n_epochs + 1):
        metrics = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch:>3}/{n_epochs}  train_loss={metrics['loss']:.4f}")

    # Evaluate on test set
    test_hf = load_dataset(HF_REPO, "sprayed", split="test")
    test_ds = DyePatchDataset(test_hf, overlay=None, training=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

    val_results = trainer.validate_epoch(test_loader)
    all_preds = val_results["preds"]  # [N, 24, 24]

    spray_metrics = compute_spray_metrics(all_preds, test_hf)

    results = {
        "config": config,
        "seed": seed,
        "lr": lr,
        "n_epochs": n_epochs,
        "test_loss": val_results["loss"],
        "test_accuracy": val_results["accuracy"],
        "spray_metrics": spray_metrics,
    }

    # Save per-tile predictions with full metadata for downstream visualizations
    per_tile = []
    for i in range(len(test_hf)):
        row = test_hf[i]
        per_tile.append({
            "pred": all_preds[i].numpy().tolist(),
            "tile_type": row.get("tile_type", ""),
            "color": row.get("color", "none"),
            "concentration": row.get("concentration", "none"),
            "spray_size_m": float(row.get("spray_size_m", 0.0)),
            "month": row.get("month", "none"),
            "point_name": str(row.get("point_name", "")),
        })
    results["per_tile"] = per_tile

    out_path = os.path.join(RESULTS_DIR, f"{config}.json")
    save_results(results, out_path)
    save_head(model, os.path.join(RESULTS_DIR, f"{config}.pt"))
    print(f"Config={config} F1={spray_metrics['f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True, help="SLURM_ARRAY_TASK_ID")
    args = parser.parse_args()
    run_eval(args.idx)
