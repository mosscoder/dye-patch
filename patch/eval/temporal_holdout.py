"""
Temporal holdout: train on ONE season, evaluate on the other two.

Simulates a practitioner with only one season of imagery, evaluating
on new seasons.

SLURM array: --array=0-11 (4 configs × 3 train_months, seed=0)
Index mapping: divmod(idx, 3) → (config_idx, month_idx)
"""

import argparse
import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from patch.utils.config import CONFIGS, EVAL_SEED, MONTHS
from patch.utils.dataset import DyePatchDataset
from patch.utils.models import create_model, save_head
from patch.utils.train import PatchTrainer, save_results, set_seed
from patch.tuning.sweep_epochs import load_best_overlay, select_best_epoch
from patch.tuning.sweep_lr import collate_fn, select_best_lr
from patch.eval.data_source import compute_spray_metrics

RESULTS_DIR = "patch/eval/results/temporal"
HF_REPO = "mpg-ranch/dye_patch"


def get_train_data_for_config_and_month(config: str, train_month: str):
    """Load data for one config filtered to a single month."""
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
        return load_dataset(HF_REPO, "offsite", split="train")

    # Filter to the single training month
    ds = ds.filter(lambda r: r["month"] == train_month)
    return ds


def get_eval_data_for_holdout(config: str, train_month: str):
    """Load test split for the two held-out months."""
    ds = load_dataset(HF_REPO, "sprayed", split="test")
    held_out_months = [m for m in MONTHS if m != train_month]

    # Filter to held-out months
    ds = ds.filter(lambda r: r["month"] in held_out_months)
    return ds


def run_holdout(idx: int):
    """Train on one season, evaluate on the other two."""
    config_idx, month_idx = divmod(idx, len(MONTHS))
    config = CONFIGS[config_idx]
    train_month = MONTHS[month_idx]
    seed = EVAL_SEED

    print(f"Config={config} TrainMonth={train_month} Seed={seed}")
    set_seed(seed)

    lr = select_best_lr(config)
    overlay = load_best_overlay(config)
    n_epochs = select_best_epoch(config, train_month=train_month)

    # Train on single month's train split
    train_hf = get_train_data_for_config_and_month(config, train_month)

    # Eval on the other two months' test splits
    eval_hf = get_eval_data_for_holdout(config, train_month)

    train_ds = DyePatchDataset(train_hf, overlay=overlay, training=True)
    eval_ds = DyePatchDataset(eval_hf, overlay=None, training=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    eval_loader = DataLoader(eval_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(device=device)
    trainer = PatchTrainer(model, lr=lr, device=device)

    for epoch in range(1, n_epochs + 1):
        metrics = trainer.train_epoch(train_loader)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{n_epochs} loss={metrics['loss']:.4f}")

    val_results = trainer.validate_epoch(eval_loader)
    spray_metrics = compute_spray_metrics(val_results["preds"], eval_hf)

    # Per-tile predictions with metadata for visualization
    per_tile = []
    for i in range(len(eval_hf)):
        row = eval_hf[i]
        per_tile.append({
            "pred": val_results["preds"][i].numpy().tolist(),
            "tile_type": row.get("tile_type", ""),
            "color": row.get("color", "none"),
            "month": row.get("month", "none"),
            "point_name": str(row.get("point_name", "")),
        })

    results = {
        "config": config,
        "train_month": train_month,
        "seed": seed,
        "lr": lr,
        "n_epochs": n_epochs,
        "test_loss": val_results["loss"],
        "test_accuracy": val_results["accuracy"],
        "spray_metrics": spray_metrics,
        "per_tile": per_tile,
    }

    out_dir = os.path.join(RESULTS_DIR, config)
    out_path = os.path.join(out_dir, f"train_{train_month}.json")
    save_results(results, out_path)
    print(f"Config={config} TrainMonth={train_month} F1={spray_metrics['f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True, help="SLURM_ARRAY_TASK_ID")
    args = parser.parse_args()
    run_holdout(args.idx)
