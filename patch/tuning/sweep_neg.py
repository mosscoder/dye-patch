"""
Negative sampling multiplier sweep.

Sweeps NEG_MULTIPLIERS at fixed best LR to find optimal neg:pos ratio.

SLURM array: --array=0-11 (4 multipliers × 3 seeds = 12 jobs)
Index mapping: divmod(idx, 3) → (mult_idx, seed)
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from patch.utils.config import LR_SEEDS, NEG_MULTIPLIERS
from patch.utils.dataset import DyePatchDataset, tuning_split
from patch.utils.models import create_model, save_head
from patch.utils.train import PatchTrainer, save_results, set_seed
from patch.tuning.sweep_lr import collate_fn

RESULTS_DIR = "patch/tuning/results/neg"
HF_REPO = "mpg-ranch/dye-patch"
N_EPOCHS = 30
FIXED_LR = 0.0005


def run_sweep(idx: int):
    """Run one (multiplier, seed) combination on real data."""
    mult_idx, seed_idx = divmod(idx, len(LR_SEEDS))

    neg_mult = NEG_MULTIPLIERS[mult_idx]
    seed = LR_SEEDS[seed_idx]
    lr = FIXED_LR

    print(f"NegMult={neg_mult} LR={lr} Seed={seed}")
    set_seed(seed)

    ds = load_dataset(HF_REPO, "sprayed", split="train")
    tune_train, tune_val = tuning_split(ds, seed=seed)

    train_ds = DyePatchDataset(tune_train, overlay=None, training=True)
    val_ds = DyePatchDataset(tune_val, overlay=None, training=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(device=device)
    trainer = PatchTrainer(model, lr=lr, neg_multiplier=neg_mult, device=device)

    results = trainer.train(train_loader, val_loader, epochs=N_EPOCHS)
    results["neg_multiplier"] = neg_mult
    results["lr"] = lr
    results["seed"] = seed

    out_dir = os.path.join(RESULTS_DIR, f"neg={neg_mult}")
    out_path = os.path.join(out_dir, f"seed={seed}.json")
    save_results(results, out_path)
    save_head(model, os.path.join(out_dir, f"seed={seed}.pt"))
    print(f"Saved to {out_path}")


def select_best_neg() -> int:
    """Select best negative multiplier by cross-seed mean F1."""
    best_mult = None
    best_f1 = -1.0

    for neg_mult in NEG_MULTIPLIERS:
        f1s = []
        for seed in LR_SEEDS:
            path = os.path.join(RESULTS_DIR, f"neg={neg_mult}", f"seed={seed}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                r = json.load(f)
            f1s.append(r["best_val_f1"])

        if f1s:
            mean_f1 = sum(f1s) / len(f1s)
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_mult = neg_mult

    print(f"Best neg multiplier: {best_mult} (mean val F1: {best_f1:.4f})")
    return best_mult


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True, help="SLURM_ARRAY_TASK_ID")
    args = parser.parse_args()
    run_sweep(args.idx)
