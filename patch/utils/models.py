"""
DINOv3-sat frozen backbone with shared linear patch-level classification head.
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModel
from dotenv import load_dotenv

from patch.utils.config import MODEL_NAME, NUM_CLASSES, GRID_DIM

load_dotenv()


class DyePatchModel(nn.Module):
    """Frozen DINOv3-sat backbone with per-patch linear classifier.

    Input:  [B, 3, 384, 384]
    Output: [B, NUM_CLASSES, 24, 24]  (spatial binary predictions: 0=none, 1=dye)
    """

    def __init__(self, backbone, classifier, grid_dim=GRID_DIM):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.grid_dim = grid_dim

    def forward(self, x):
        with torch.no_grad():
            outputs = self.backbone(x)
            # last_hidden_state: [B, 1 + num_patches, hidden_dim]
            patch_tokens = outputs.last_hidden_state[:, 1:, :]  # drop CLS

        # Shared linear applied to each token: [B, num_patches, num_classes]
        logits = self.classifier(patch_tokens)

        # Reshape to spatial grid: [B, num_classes, H, W]
        B, N, C = logits.shape
        logits = logits.permute(0, 2, 1).reshape(B, C, self.grid_dim, self.grid_dim)
        return logits

    def get_trainable_parameters(self):
        return self.classifier.parameters()


def create_model(
    model_name: str = MODEL_NAME,
    num_classes: int = NUM_CLASSES,
    dropout: float = 0.0,
    device: str = "cuda",
) -> DyePatchModel:
    """Create and initialize the dye patch classifier."""

    backbone = AutoModel.from_pretrained(
        model_name, token=os.environ.get("HF_TOKEN")
    )

    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    hidden_size = backbone.config.hidden_size
    layers = []
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_size, num_classes))
    classifier = nn.Sequential(*layers)

    model = DyePatchModel(backbone, classifier)

    if isinstance(device, str):
        if device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif device == "gpu":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")

    model = model.to(device)
    return model


def save_head(model: DyePatchModel, path: str):
    """Save only the classifier head state dict."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.classifier.state_dict(), path)


def load_head(model: DyePatchModel, path: str):
    """Load classifier head state dict."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.classifier.load_state_dict(state)
