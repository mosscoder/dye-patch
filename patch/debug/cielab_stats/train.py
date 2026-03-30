"""
Test CIELAB patch statistics concatenated with DINO embeddings.

Per-patch mean L*a*b* (z-scored) is concatenated with the 1024-dim DINO
embedding → classifier input is 1027-dim. Compares with/without Lab features.

6 jobs = 2 configs (baseline, +lab) × 3 seeds
SLURM array: --array=0-5
Index mapping: divmod(idx, 3) → (config_idx, seed)

Usage:
  python -u -m patch.debug.cielab_stats.train --idx 0  # baseline, seed=0
  python -u -m patch.debug.cielab_stats.train --idx 3  # +lab, seed=0
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.color import rgb2lab

from patch.utils.config import (
    GRID_DIM,
    HF_REPO,
    MODEL_INPUT_SIZE,
    NO_DYE_NEG_SAMPLE,
    NUM_CLASSES,
    VIT_PATCH_SIZE,
)
from patch.utils.dataset import DyePatchDataset, stratified_split
from patch.utils.models import create_model
from patch.utils.train import collate_fn, compute_spray_metrics, save_results, set_seed

RESULTS_DIR = "patch/debug/cielab_stats/results"
MODEL_NAME_LARGE = "facebook/dinov3-vitl16-pretrain-sat493m"
LR = 5e-4
N_EPOCHS = 50
NEG_MULT = 8
CONFIGS = ["baseline", "lab"]


class DyePatchModelLab(nn.Module):
    """DyePatchModel with optional CIELAB patch statistics concatenation.

    When use_lab=True, per-patch mean L*a*b* (z-scored) is concatenated with
    the DINO embedding before classification. Requires denormalized images to
    be passed alongside normalized tensors.
    """

    def __init__(self, backbone, classifier, grid_dim=GRID_DIM, use_lab=False,
                 lab_mean=None, lab_std=None):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.grid_dim = grid_dim
        self.use_lab = use_lab

        if use_lab and lab_mean is not None:
            self.register_buffer("lab_mean", torch.tensor(lab_mean, dtype=torch.float32))
            self.register_buffer("lab_std", torch.tensor(lab_std, dtype=torch.float32))

    def _compute_patch_lab(self, images_raw):
        """Compute per-patch mean L*a*b* from raw [0-1] RGB images.

        images_raw: [B, 3, H, W] float tensor (NOT normalized)
        Returns: [B, num_patches, 3]
        """
        B = images_raw.shape[0]
        # Move to numpy for skimage
        imgs_np = images_raw.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, 3]

        all_lab = []
        for i in range(B):
            lab = rgb2lab(imgs_np[i])  # [H, W, 3]
            # Reshape to patch grid and take means
            lab_patches = lab[:self.grid_dim * VIT_PATCH_SIZE, :self.grid_dim * VIT_PATCH_SIZE]
            lab_patches = lab_patches.reshape(self.grid_dim, VIT_PATCH_SIZE,
                                              self.grid_dim, VIT_PATCH_SIZE, 3)
            patch_means = lab_patches.mean(axis=(1, 3))  # [grid_dim, grid_dim, 3]
            all_lab.append(patch_means.reshape(-1, 3))  # [num_patches, 3]

        lab_tensor = torch.tensor(np.stack(all_lab), dtype=torch.float32,
                                  device=images_raw.device)  # [B, num_patches, 3]

        # Z-score normalize
        lab_tensor = (lab_tensor - self.lab_mean) / (self.lab_std + 1e-8)
        return lab_tensor

    def forward(self, x, x_raw=None):
        with torch.no_grad():
            outputs = self.backbone(x)
            n_reg = getattr(self.backbone.config, "num_register_tokens", 0)
            patch_tokens = outputs.last_hidden_state[:, 1 + n_reg:, :]  # [B, 576, 1024]

        if self.use_lab and x_raw is not None:
            lab_features = self._compute_patch_lab(x_raw)  # [B, 576, 3]
            patch_tokens = torch.cat([patch_tokens, lab_features], dim=-1)  # [B, 576, 1027]

        logits = self.classifier(patch_tokens)  # [B, 576, C]
        B, N, C = logits.shape
        logits = logits.permute(0, 2, 1).reshape(B, C, self.grid_dim, self.grid_dim)
        return logits

    def get_trainable_parameters(self):
        return self.classifier.parameters()


def create_lab_model(use_lab=False, lab_mean=None, lab_std=None, device="cuda"):
    """Create model with optional Lab feature concatenation."""
    from transformers import AutoModel
    from dotenv import load_dotenv
    load_dotenv()

    backbone = AutoModel.from_pretrained(MODEL_NAME_LARGE, token=os.environ.get("HF_TOKEN"))
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    hidden_size = backbone.config.hidden_size  # 1024
    input_dim = hidden_size + 3 if use_lab else hidden_size
    classifier = nn.Sequential(nn.Linear(input_dim, NUM_CLASSES))

    model = DyePatchModelLab(backbone, classifier, use_lab=use_lab,
                             lab_mean=lab_mean, lab_std=lab_std)

    if isinstance(device, str):
        if device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    return model.to(device)


def compute_lab_stats(hf_dataset):
    """Compute mean and std of per-patch L*a*b* across a dataset for z-scoring."""
    from patch.utils.config import NORM_MEAN, NORM_STD, PRECROP_SIZE
    import torchvision.transforms as T
    from PIL import Image

    transform = T.Compose([
        T.Resize((PRECROP_SIZE, PRECROP_SIZE), interpolation=Image.LANCZOS),
        T.CenterCrop(MODEL_INPUT_SIZE),
    ])

    all_lab = []
    for i in range(len(hf_dataset)):
        img = hf_dataset[i]["image"]
        img = transform(img)
        img_np = np.array(img, dtype=np.float32) / 255.0
        lab = rgb2lab(img_np)  # [H, W, 3]
        lab_patches = lab[:GRID_DIM * VIT_PATCH_SIZE, :GRID_DIM * VIT_PATCH_SIZE]
        lab_patches = lab_patches.reshape(GRID_DIM, VIT_PATCH_SIZE,
                                          GRID_DIM, VIT_PATCH_SIZE, 3)
        patch_means = lab_patches.mean(axis=(1, 3))  # [24, 24, 3]
        all_lab.append(patch_means.reshape(-1, 3))

    all_lab = np.concatenate(all_lab, axis=0)  # [N*576, 3]
    return all_lab.mean(axis=0), all_lab.std(axis=0)


def _balanced_mask(targets, neg_mult=NEG_MULT):
    B = targets.shape[0]
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for i in range(B):
        dye_idx = (targets[i] > 0).nonzero(as_tuple=False)
        n_dye = len(dye_idx)
        if n_dye > 0:
            mask[i, dye_idx[:, 0], dye_idx[:, 1]] = True
            bg_idx = (targets[i] == 0).nonzero(as_tuple=False)
            if len(bg_idx) > 0:
                n_sample = min(n_dye * neg_mult, len(bg_idx))
                perm = torch.randperm(len(bg_idx), device=targets.device)[:n_sample]
                mask[i, bg_idx[perm, 0], bg_idx[perm, 1]] = True
        else:
            bg_idx = (targets[i] == 0).nonzero(as_tuple=False)
            if len(bg_idx) > 0:
                n_sample = min(NO_DYE_NEG_SAMPLE, len(bg_idx))
                perm = torch.randperm(len(bg_idx), device=targets.device)[:n_sample]
                mask[i, bg_idx[perm, 0], bg_idx[perm, 1]] = True
    return mask


def _denormalize(tensor):
    """Undo ImageNet normalization to get [0-1] RGB."""
    from patch.utils.config import NORM_MEAN, NORM_STD
    mean = torch.tensor(NORM_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=tensor.device).view(1, 3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def run(seed: int, use_lab: bool):
    config_name = "lab" if use_lab else "baseline"
    print(f"CIELAB stats: config={config_name} LR={LR} Seed={seed}")
    set_seed(seed)

    ds = load_dataset(HF_REPO, "sprayed", split="train")
    tune_train, tune_val = stratified_split(ds, test_frac=0.2, seed=seed)

    # Compute Lab z-score stats from training data
    lab_mean, lab_std = None, None
    if use_lab:
        print("  Computing Lab z-score stats from training data...")
        lab_mean, lab_std = compute_lab_stats(tune_train)
        print(f"  Lab mean: {lab_mean}, std: {lab_std}")

    train_ds = DyePatchDataset(tune_train, overlay=None, training=True)
    val_ds = DyePatchDataset(tune_val, overlay=None, training=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_lab_model(use_lab=use_lab, lab_mean=lab_mean, lab_std=lab_std, device=device)
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=LR, weight_decay=0.01)

    train_history = []
    val_history = []
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        model.backbone.eval()
        total_loss = 0.0

        for images, masks, _ in tqdm(train_loader, desc="  train", leave=False):
            images = images.to(device)
            masks = masks.to(device).long()

            x_raw = _denormalize(images) if use_lab else None
            logits = model(images, x_raw=x_raw)

            ce = F.cross_entropy(logits, masks, reduction="none")
            loss_mask = _balanced_mask(masks)
            selected = ce[loss_mask]
            loss = selected.mean() if len(selected) > 0 else ce.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        train_loss = total_loss / max(len(train_ds), 1)
        train_history.append({"loss": train_loss})

        model.eval()
        val_loss_total = 0.0
        all_preds = []
        all_metadata = []

        with torch.no_grad():
            for images, masks, metadata in tqdm(val_loader, desc="  val", leave=False):
                images = images.to(device)
                masks = masks.to(device).long()

                x_raw = _denormalize(images) if use_lab else None
                logits = model(images, x_raw=x_raw)

                ce = F.cross_entropy(logits, masks, reduction="none")
                loss_mask = _balanced_mask(masks)
                selected = ce[loss_mask]
                val_loss_total += (selected.mean() if len(selected) > 0 else ce.mean()).item() * images.size(0)

                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_metadata.extend(metadata)

        val_loss = val_loss_total / max(len(val_ds), 1)
        preds_cat = torch.cat(all_preds, dim=0)
        metrics = compute_spray_metrics(preds_cat, all_metadata)

        val_record = {"loss": val_loss, "f1": metrics["f1"],
                      "precision": metrics["precision"], "recall": metrics["recall"]}
        val_history.append(val_record)

        is_best = metrics["f1"] > best_val_f1
        if is_best:
            best_val_f1 = metrics["f1"]
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.classifier.state_dict().items()}

        star = " *" if is_best else ""
        print(
            f"Epoch {epoch:>3}/{N_EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"P={metrics['precision']:.4f}  "
            f"R={metrics['recall']:.4f}{star}"
        )

    if best_state is not None:
        model.classifier.load_state_dict(best_state)

    print(f"Best epoch: {best_epoch} (F1={best_val_f1:.4f})")

    results = {
        "train_history": train_history,
        "val_history": val_history,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "lr": LR,
        "seed": seed,
        "config": config_name,
        "use_lab": use_lab,
    }
    out_path = os.path.join(RESULTS_DIR, f"{config_name}_seed={seed}.json")
    save_results(results, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True)
    args = parser.parse_args()
    config_idx, seed = divmod(args.idx, 3)
    use_lab = config_idx == 1
    run(seed=seed, use_lab=use_lab)
