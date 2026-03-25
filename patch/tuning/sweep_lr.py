"""
Learning rate grid search across configs and seeds.

SLURM array: --array=0-79 (4 configs × 4 LRs × 5 seeds = 80 jobs)
Index mapping: divmod(idx, 5) → (config_lr_idx, seed)
               divmod(config_lr_idx, 4) → (config_idx, lr_idx)
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from patch.utils.config import CONFIGS, LR_GRID, LR_SEEDS
from patch.utils.dataset import DyePatchDataset, tuning_split
from patch.utils.models import create_model, save_head
from patch.utils.synthetic import SyntheticDyeOverlay
from patch.utils.train import PatchTrainer, save_results, set_seed

RESULTS_DIR = "patch/tuning/results/lr"
HF_REPO = "mpg-ranch/dye-patch"
N_EPOCHS = 10


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    metadata = [b[2] for b in batch]
    return images, masks, metadata


def get_overlay_for_config(config: str) -> SyntheticDyeOverlay | None:
    """Return overlay if config uses synthetic data, else None."""
    if config in ("hybrid", "synth_local", "synth_offsite"):
        return SyntheticDyeOverlay()
    return None


def get_split_for_config(config: str, seed: int = 0):
    """Load and split data appropriate for the training config."""
    if config == "real_only":
        ds = load_dataset(HF_REPO, "sprayed", split="train")
    elif config == "hybrid":
        from datasets import concatenate_datasets
        sprayed = load_dataset(HF_REPO, "sprayed", split="train")
        annex = load_dataset(HF_REPO, "unsprayed_annex", split="train")
        ds = concatenate_datasets([sprayed, annex])
    elif config == "synth_local":
        ds = load_dataset(HF_REPO, "unsprayed_annex", split="train")
    elif config == "synth_offsite":
        offsite_ds = load_dataset(HF_REPO, "offsite", split="train")
        return tuning_split(offsite_ds, seed=seed)

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
    model = create_model(device=device)
    trainer = PatchTrainer(model, lr=lr, device=device)

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
    """Select best LR by cross-seed mean validation loss.

    LR is established from real_only data and shared across all configs.
    The config arg is accepted for backward compatibility but ignored.
    """
    result_dir = os.path.join(RESULTS_DIR, "real_only")
    best_lr = None
    best_loss = float("inf")

    for lr in LR_GRID:
        losses = []
        for seed in LR_SEEDS:
            path = os.path.join(result_dir, f"lr={lr}_seed={seed}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                r = json.load(f)
            losses.append(r["best_val_loss"])

        if losses:
            mean_loss = sum(losses) / len(losses)
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_lr = lr

    print(f"Best LR: {best_lr} (mean val loss: {best_loss:.4f})")
    return best_lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True, help="SLURM_ARRAY_TASK_ID")
    args = parser.parse_args()
    run_sweep(args.idx)
