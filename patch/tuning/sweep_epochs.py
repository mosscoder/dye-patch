"""
Epoch count tuning via k-fold cross-validation.

Two modes:
  Full-data (for data_source.py):
    SLURM --array=0-19 (4 configs × 5 folds)
    python -m patch.tuning.sweep_epochs --mode full --idx $SLURM_ARRAY_TASK_ID

  Temporal holdout (for temporal_holdout.py):
    SLURM --array=0-59 (4 configs × 3 months × 5 folds)
    python -m patch.tuning.sweep_epochs --mode temporal --idx $SLURM_ARRAY_TASK_ID
"""

import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader

from patch.utils.config import CONFIGS, MONTHS
from patch.utils.dataset import DyePatchDataset, _get_point_strata, _point_indices
from patch.utils.models import create_model
from patch.utils.synthetic import SyntheticDyeOverlay
from patch.utils.train import PatchTrainer, save_results, set_seed
from patch.tuning.sweep_lr import collate_fn, select_best_lr

RESULTS_DIR = "patch/tuning/results/epochs"
HF_REPO = "mpg-ranch/dye-patch"
MAX_EPOCHS = 50
N_FOLDS = 5
N_FOLDS_TEMPORAL = 4


def load_best_overlay(config: str) -> SyntheticDyeOverlay | None:
    """Load empirical HSV deltas, or None for real_only."""
    if config == "real_only":
        return None
    return SyntheticDyeOverlay()


def get_train_data_for_config(config: str):
    """Load full training split for a config."""
    if config == "real_only":
        return load_dataset(HF_REPO, "sprayed", split="train")
    elif config == "hybrid":
        from datasets import concatenate_datasets
        sprayed = load_dataset(HF_REPO, "sprayed", split="train")
        annex = load_dataset(HF_REPO, "unsprayed_annex", split="train")
        return concatenate_datasets([sprayed, annex])
    elif config == "synth_local":
        return load_dataset(HF_REPO, "unsprayed_annex", split="train")
    elif config == "synth_offsite":
        return load_dataset(HF_REPO, "offsite", split="train")
    return load_dataset(HF_REPO, "sprayed", split="train")


def _kfold_split(ds, fold_idx: int):
    """K-fold at the point level. Returns (fold_train, fold_val) datasets."""
    point_strata = _get_point_strata(ds)
    points = list(point_strata.keys())
    strata = [point_strata[p] for p in points]

    if len(set(strata)) > 1:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
        splits = list(skf.split(points, strata))
    else:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
        splits = list(kf.split(points))

    train_point_idx, val_point_idx = splits[fold_idx]
    train_points = set(points[i] for i in train_point_idx)
    val_points = set(points[i] for i in val_point_idx)

    return ds.select(_point_indices(ds, train_points)), ds.select(_point_indices(ds, val_points))


def _train_fold(ds, fold_idx: int, config: str, out_dir: str, extra_meta: dict = None):
    """Train one fold, save results."""
    set_seed(fold_idx)

    lr = select_best_lr(config)
    overlay = load_best_overlay(config)

    fold_train, fold_val = _kfold_split(ds, fold_idx)

    train_ds = DyePatchDataset(fold_train, overlay=overlay, training=True)
    val_ds = DyePatchDataset(fold_val, overlay=None, training=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(device=device)
    trainer = PatchTrainer(model, lr=lr, device=device)

    results = trainer.train(train_loader, val_loader, epochs=MAX_EPOCHS)
    results["config"] = config
    results["fold"] = fold_idx
    if extra_meta:
        results.update(extra_meta)

    out_path = os.path.join(out_dir, f"fold={fold_idx}.json")
    save_results(results, out_path)
    print(f"Saved to {out_path}")


# =========================================================================
# Full-data mode (for data_source.py)
# =========================================================================

def run_full(idx: int):
    """Run one (config, fold) for full-data epoch tuning."""
    config_idx, fold_idx = divmod(idx, N_FOLDS)
    config = CONFIGS[config_idx]

    print(f"[full] Config={config} Fold={fold_idx}")
    ds = get_train_data_for_config(config)

    out_dir = os.path.join(RESULTS_DIR, "full", config)
    _train_fold(ds, fold_idx, config, out_dir)


def select_best_epoch(config: str, train_month: str | None = None) -> int:
    """Aggregate folds and select optimal epoch count.

    If train_month is None, uses full-data results.
    If train_month is provided, uses temporal holdout results for that month.
    """
    if train_month is None:
        result_dir = os.path.join(RESULTS_DIR, "full", config)
    else:
        result_dir = os.path.join(RESULTS_DIR, "temporal", config, train_month)

    n_folds = N_FOLDS_TEMPORAL if train_month is not None else N_FOLDS
    all_val_f1s = []
    for fold in range(n_folds):
        path = os.path.join(result_dir, f"fold={fold}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            r = json.load(f)
        f1s = [h["f1"] for h in r["val_history"]]
        all_val_f1s.append(f1s)

    if not all_val_f1s:
        print(f"WARNING: no fold results found in {result_dir}, using {MAX_EPOCHS}")
        return MAX_EPOCHS

    min_len = min(len(l) for l in all_val_f1s)
    mean_f1s = [
        np.mean([l[i] for l in all_val_f1s]) for i in range(min_len)
    ]
    best_epoch = int(np.argmax(mean_f1s)) + 1
    label = f"{config}" if train_month is None else f"{config}/train_{train_month}"
    print(f"Best epoch for {label}: {best_epoch} (mean val F1: {mean_f1s[best_epoch-1]:.4f})")
    return best_epoch


# =========================================================================
# Temporal holdout mode (for temporal_holdout.py)
# =========================================================================

def _get_single_month_data(config: str, train_month: str):
    """Load training data for a single month, filtered by config."""
    if config == "synth_offsite":
        # Offsite has no months — use all offsite data
        return load_dataset(HF_REPO, "offsite", split="train")

    ds = get_train_data_for_config(config)
    # Filter to the single training month
    indices = [i for i, r in enumerate(ds) if r.get("month", "") == train_month]
    return ds.select(indices)


def run_temporal(idx: int):
    """Run one (config, month, fold) for temporal holdout epoch tuning."""
    config_month_idx, fold_idx = divmod(idx, N_FOLDS_TEMPORAL)
    config_idx, month_idx = divmod(config_month_idx, len(MONTHS))
    config = CONFIGS[config_idx]
    train_month = MONTHS[month_idx]

    print(f"[temporal] Config={config} Month={train_month} Fold={fold_idx}")
    ds = _get_single_month_data(config, train_month)

    out_dir = os.path.join(RESULTS_DIR, "temporal", config, train_month)
    _train_fold(ds, fold_idx, config, out_dir, extra_meta={"train_month": train_month})


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "temporal"], required=True,
                        help="full = all-months training, temporal = single-month training")
    parser.add_argument("--idx", type=int, required=True, help="SLURM_ARRAY_TASK_ID")
    args = parser.parse_args()

    if args.mode == "full":
        run_full(args.idx)
    elif args.mode == "temporal":
        run_temporal(args.idx)
