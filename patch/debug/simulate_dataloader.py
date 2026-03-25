"""
Simulate data loading for all 4 training configs across 3 epochs.

Times each batch and reports per-epoch and per-config statistics.
No GPU or model needed — just measures the data pipeline.

Usage:
  python -m patch.debug.simulate_dataloader
  python -m patch.debug.simulate_dataloader --num-workers 0
"""

import argparse
import time

import torch
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

from patch.utils.dataset import DyePatchDataset, tuning_split
from patch.utils.synthetic import SyntheticDyeOverlay

HF_REPO = "mpg-ranch/dye-patch"
BATCH_SIZE = 32
N_EPOCHS = 3
SEED = 0


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    metadata = [b[2] for b in batch]
    return images, masks, metadata


def get_data_for_config(config: str):
    if config == "real_only":
        return load_dataset(HF_REPO, "sprayed", split="train")
    elif config == "hybrid":
        sprayed = load_dataset(HF_REPO, "sprayed", split="train")
        annex = load_dataset(HF_REPO, "unsprayed_annex", split="train")
        return concatenate_datasets([sprayed, annex])
    elif config == "synth_local":
        return load_dataset(HF_REPO, "unsprayed_annex", split="train")
    elif config == "synth_offsite":
        return load_dataset(HF_REPO, "offsite", split="train")


def run_config(config: str, num_workers: int):
    print(f"\n{'='*60}")
    print(f"Config: {config}")
    print(f"{'='*60}")

    hf_data = get_data_for_config(config)
    train_hf, val_hf = tuning_split(hf_data, seed=SEED)

    overlay = SyntheticDyeOverlay() if config != "real_only" else None

    train_ds = DyePatchDataset(train_hf, overlay=overlay, training=True)
    val_ds = DyePatchDataset(val_hf, overlay=None, training=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    print(f"  Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_ds)} samples, {len(val_loader)} batches")
    print(f"  Overlay: {overlay is not None}")
    print(f"  Workers: {num_workers}")

    for epoch in range(1, N_EPOCHS + 1):
        epoch_start = time.perf_counter()

        # Train
        for images, masks, metadata in tqdm(train_loader, desc=f"  Epoch {epoch}/{N_EPOCHS} train", leave=False):
            pass
        train_done = time.perf_counter()

        # Val
        for images, masks, metadata in tqdm(val_loader, desc=f"  Epoch {epoch}/{N_EPOCHS} val  ", leave=False):
            pass
        val_done = time.perf_counter()

        train_time = train_done - epoch_start
        val_time = val_done - train_done
        total = val_done - epoch_start

        print(
            f"  Epoch {epoch}/{N_EPOCHS}  "
            f"train={train_time:.1f}s ({train_time/len(train_loader):.2f}s/batch)  "
            f"val={val_time:.1f}s ({val_time/len(val_loader):.2f}s/batch)  "
            f"total={total:.1f}s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--configs", nargs="+",
                        default=["real_only", "hybrid", "synth_local", "synth_offsite"])
    args = parser.parse_args()

    print(f"Batch size: {BATCH_SIZE}, Epochs: {N_EPOCHS}")

    for config in args.configs:
        run_config(config, args.num_workers)

    print("\nDone.")
