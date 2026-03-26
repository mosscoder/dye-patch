"""
Quick comparison of loss functions: focal, bce, bce_smooth.

Runs one seed (0), one LR (0.0001), 15 epochs for each loss function.
Saves results to patch/debug/results/loss_fn/{loss_fn}.json.

SLURM array: --array=0-2 (one job per loss function)

Usage:
  python -u -m patch.debug.check_loss_fn --idx 0  # focal
  python -u -m patch.debug.check_loss_fn --idx 1  # bce
  python -u -m patch.debug.check_loss_fn --idx 2  # bce_smooth
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from patch.utils.dataset import DyePatchDataset, tuning_split
from patch.utils.models import create_model
from patch.utils.train import PatchTrainer, save_results, set_seed

HF_REPO = "mpg-ranch/dye-patch"
RESULTS_DIR = "patch/debug/results/loss_fn"
LR = 0.0001
SEED = 0
N_EPOCHS = 15
LOSS_FNS = ["focal", "bce", "bce_smooth"]


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    metadata = [b[2] for b in batch]
    return images, masks, metadata


def run(idx: int):
    loss_fn = LOSS_FNS[idx]
    print(f"Loss={loss_fn} LR={LR} Seed={SEED}")
    set_seed(SEED)

    ds = load_dataset(HF_REPO, "sprayed", split="train")
    tune_train, tune_val = tuning_split(ds, seed=SEED)

    train_ds = DyePatchDataset(tune_train, overlay=None, training=True)
    val_ds = DyePatchDataset(tune_val, overlay=None, training=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(device=device)
    trainer = PatchTrainer(model, lr=LR, loss_fn=loss_fn, device=device)

    results = trainer.train(train_loader, val_loader, epochs=N_EPOCHS)
    results["loss_fn"] = loss_fn
    results["lr"] = LR
    results["seed"] = SEED

    out_path = os.path.join(RESULTS_DIR, f"{loss_fn}.json")
    save_results(results, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True, help="0=focal, 1=bce, 2=bce_smooth")
    args = parser.parse_args()
    run(args.idx)
