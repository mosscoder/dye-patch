"""
LoRA: rank=4, alpha=8 on backbone attention layers.

Low-rank adaptation allows the frozen backbone to learn dye-specific
feature adjustments with minimal extra parameters (~50K).

6 jobs = 2 configs (baseline, +lora) × 3 seeds
SLURM array: --array=0-5

Usage:
  python -u -m patch.debug.lora.train --idx 0  # baseline, seed=0
  python -u -m patch.debug.lora.train --idx 3  # +lora, seed=0
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from patch.utils.config import GRID_DIM, HF_REPO, NO_DYE_NEG_SAMPLE, NUM_CLASSES
from patch.utils.dataset import DyePatchDataset, stratified_split
from patch.utils.train import collate_fn, compute_spray_metrics, save_results, set_seed

RESULTS_DIR = "patch/debug/lora/results"
MODEL_NAME_LARGE = "facebook/dinov3-vitl16-pretrain-sat493m"
LR = 5e-4
N_EPOCHS = 50
NEG_MULT = 8
LORA_R = 4
LORA_ALPHA = 8
CONFIGS = ["baseline", "lora"]


class DyePatchModelLoRA(nn.Module):
    """DyePatchModel with optional LoRA on backbone attention."""

    def __init__(self, backbone, classifier, grid_dim=GRID_DIM, use_lora=False):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.grid_dim = grid_dim
        self.use_lora = use_lora

    def forward(self, x):
        if self.use_lora:
            # LoRA params need gradients — no torch.no_grad()
            outputs = self.backbone(x)
        else:
            with torch.no_grad():
                outputs = self.backbone(x)

        n_reg = getattr(self.backbone.config, "num_register_tokens", 0)
        patch_tokens = outputs.last_hidden_state[:, 1 + n_reg:, :]

        logits = self.classifier(patch_tokens)
        B, N, C = logits.shape
        logits = logits.permute(0, 2, 1).reshape(B, C, self.grid_dim, self.grid_dim)
        return logits

    def get_trainable_parameters(self):
        """Return classifier params + LoRA params (if any)."""
        params = list(self.classifier.parameters())
        if self.use_lora:
            for name, param in self.backbone.named_parameters():
                if param.requires_grad:
                    params.append(param)
        return params


def create_lora_model(use_lora=False, device="cuda"):
    """Create model with optional LoRA adapters."""
    from transformers import AutoModel
    from dotenv import load_dotenv
    load_dotenv()

    backbone = AutoModel.from_pretrained(MODEL_NAME_LARGE, token=os.environ.get("HF_TOKEN"))

    if use_lora:
        from peft import LoraConfig, get_peft_model

        # Find all Linear layers in attention blocks for LoRA targets
        target_modules = set()
        for name, module in backbone.named_modules():
            if isinstance(module, nn.Linear) and "attention" in name:
                target_modules.add(name)

        if not target_modules:
            # Fallback: print all module names for debugging
            print("WARNING: No attention Linear layers found. Module names:")
            for name, mod in backbone.named_modules():
                if isinstance(mod, nn.Linear):
                    print(f"  {name}")
            raise ValueError("Could not find LoRA target modules")

        print(f"  LoRA targets: {sorted(target_modules)[:5]}... ({len(target_modules)} total)")

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=list(target_modules),
            lora_dropout=0.0,
            bias="none",
        )
        backbone = get_peft_model(backbone, lora_config)
        backbone.print_trainable_parameters()
    else:
        # Freeze all backbone params
        for param in backbone.parameters():
            param.requires_grad = False

    backbone.eval()

    hidden_size = backbone.config.hidden_size if hasattr(backbone.config, 'hidden_size') else 1024
    classifier = nn.Sequential(nn.Linear(hidden_size, NUM_CLASSES))

    model = DyePatchModelLoRA(backbone, classifier, use_lora=use_lora)

    if isinstance(device, str):
        if device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    return model.to(device)


def _balanced_mask(targets):
    B = targets.shape[0]
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for i in range(B):
        dye_idx = (targets[i] > 0).nonzero(as_tuple=False)
        n_dye = len(dye_idx)
        if n_dye > 0:
            mask[i, dye_idx[:, 0], dye_idx[:, 1]] = True
            bg_idx = (targets[i] == 0).nonzero(as_tuple=False)
            if len(bg_idx) > 0:
                n_sample = min(n_dye * NEG_MULT, len(bg_idx))
                perm = torch.randperm(len(bg_idx), device=targets.device)[:n_sample]
                mask[i, bg_idx[perm, 0], bg_idx[perm, 1]] = True
        else:
            bg_idx = (targets[i] == 0).nonzero(as_tuple=False)
            if len(bg_idx) > 0:
                n_sample = min(NO_DYE_NEG_SAMPLE, len(bg_idx))
                perm = torch.randperm(len(bg_idx), device=targets.device)[:n_sample]
                mask[i, bg_idx[perm, 0], bg_idx[perm, 1]] = True
    return mask


def run(seed: int, use_lora: bool):
    config_name = "lora" if use_lora else "baseline"
    print(f"LoRA: config={config_name} r={LORA_R} alpha={LORA_ALPHA} LR={LR} Seed={seed}")
    set_seed(seed)

    ds = load_dataset(HF_REPO, "sprayed", split="train")
    tune_train, tune_val = stratified_split(ds, test_frac=0.2, seed=seed)

    train_ds = DyePatchDataset(tune_train, overlay=None, training=True)
    val_ds = DyePatchDataset(tune_val, overlay=None, training=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_lora_model(use_lora=use_lora, device=device)

    n_params = sum(p.numel() for p in model.get_trainable_parameters())
    print(f"  Trainable params: {n_params:,}")

    if use_lora:
        lora_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": model.classifier.parameters(), "lr": LR},
            {"params": lora_params, "lr": 1e-5},
        ], weight_decay=0.01)
        print(f"  Differential LR: classifier={LR}, LoRA=1e-5")
    else:
        optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=LR, weight_decay=0.01)

    train_history = []
    val_history = []
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        if not use_lora:
            model.backbone.eval()
        total_loss = 0.0

        for images, masks, _ in tqdm(train_loader, desc="  train", leave=False):
            images = images.to(device)
            masks = masks.to(device).long()
            logits = model(images)
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
                logits = model(images)
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
        print(f"Epoch {epoch:>3}/{N_EPOCHS}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  F1={metrics['f1']:.4f}  "
              f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}{star}")

    if best_state is not None:
        model.classifier.load_state_dict(best_state)
    print(f"Best epoch: {best_epoch} (F1={best_val_f1:.4f})")

    results = {"train_history": train_history, "val_history": val_history,
               "best_epoch": best_epoch, "best_val_f1": best_val_f1,
               "lr": LR, "seed": seed, "config": config_name,
               "lora_r": LORA_R if use_lora else None,
               "lora_alpha": LORA_ALPHA if use_lora else None,
               "n_trainable_params": n_params}
    out_path = os.path.join(RESULTS_DIR, f"{config_name}_seed={seed}.json")
    save_results(results, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True)
    args = parser.parse_args()
    config_idx, seed = divmod(args.idx, 3)
    run(seed=seed, use_lora=config_idx == 1)
