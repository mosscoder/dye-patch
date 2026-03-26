"""
Learning rate grid search on real_only data.

SLURM array: --array=0-14 (5 LRs × 3 seeds = 15 jobs)
Index mapping: divmod(idx, 3) → (lr_idx, seed)
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from patch.utils.config import CONFIGS, LR_GRID, LR_SEEDS
from patch.utils.dataset import DyePatchDataset, get_train_data_for_config, tuning_split
from patch.utils.models import create_model, save_head
from patch.utils.synthetic import SyntheticDyeOverlay
from patch.utils.train import PatchTrainer, collate_fn, save_results, set_seed
from patch.tuning.sweep_neg import select_best_neg

RESULTS_DIR = "patch/tuning/results/lr"
N_EPOCHS = 50


def get_overlay_for_config(config: str) -> SyntheticDyeOverlay | None:
    """Return overlay if config uses synthetic data, else None."""
    if config in ("hybrid", "synth_local", "synth_offsite"):
        return SyntheticDyeOverlay()
    return None


def get_split_for_config(config: str, seed: int = 0):
    """Load and split data appropriate for the training config."""
    ds = get_train_data_for_config(config)
    return tuning_split(ds, seed=seed)


def run_sweep(idx: int):
    """Run one (lr, seed) combination on real data only.

    LR is established from real sprayed data and shared across all configs.
    """
    lr_idx, seed_idx = divmod(idx, len(LR_SEEDS))

    config = "real_only"
    lr = LR_GRID[lr_idx]
    seed = LR_SEEDS[seed_idx]

    print(f"Config={config} LR={lr} Seed={seed}")
    set_seed(seed)

    tune_train, tune_val = get_split_for_config(config, seed=seed)

    train_ds = DyePatchDataset(tune_train, overlay=None, training=True)
    val_ds = DyePatchDataset(tune_val, overlay=None, training=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    neg_mult = select_best_neg()
    model = create_model(device=device)
    trainer = PatchTrainer(model, lr=lr, neg_multiplier=neg_mult, device=device)

    results = trainer.train(train_loader, val_loader, epochs=N_EPOCHS)
    results["config"] = config
    results["lr"] = lr
    results["seed"] = seed

    out_dir = os.path.join(RESULTS_DIR, config)
    out_path = os.path.join(out_dir, f"lr={lr}_seed={seed}.json")
    save_results(results, out_path)
    save_head(model, os.path.join(out_dir, f"lr={lr}_seed={seed}.pt"))
    print(f"Saved to {out_path}")


def select_best_lr(config: str = "real_only") -> float:
    """Select best LR by cross-seed mean super-patch F1.

    LR is established from real_only data and shared across all configs.
    The config arg is accepted for backward compatibility but ignored.
    """
    result_dir = os.path.join(RESULTS_DIR, "real_only")
    best_lr = None
    best_f1 = -1.0

    for lr in LR_GRID:
        f1s = []
        for seed in LR_SEEDS:
            path = os.path.join(result_dir, f"lr={lr}_seed={seed}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                r = json.load(f)
            f1s.append(r["best_val_f1"])

        if f1s:
            mean_f1 = sum(f1s) / len(f1s)
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_lr = lr

    print(f"Best LR: {best_lr} (mean val F1: {best_f1:.4f})")
    return best_lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True, help="SLURM_ARRAY_TASK_ID")
    args = parser.parse_args()
    run_sweep(args.idx)
