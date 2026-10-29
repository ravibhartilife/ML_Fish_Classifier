# =============================================================
# model.py  —  Build EfficientNetB0 for fish classification
# =============================================================
# What this file does:
#   1. Loads EfficientNetB0 pretrained on ImageNet
#   2. Freezes all layers (keep what it already learned)
#   3. Replaces the last layer for our 31 fish species
#   4. Only the last layer trains initially
# =============================================================

import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes=31):
    """
    Loads pretrained EfficientNetB0 and adapts it for fish classification.
    
    Args:
        num_classes: number of fish species (31 in our case)
    
    Returns:
        model ready for training
    """

    # Step 1: Load EfficientNetB0 pretrained on ImageNet
    # First run will download ~20MB weights automatically
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    # Step 2: Freeze all layers
    # These layers already know edges, textures, shapes — don't retrain them
    for param in model.parameters():
        param.requires_grad = False

    # Step 3: Replace the classifier (last layer)
    # Original: 1000 ImageNet classes
    # Ours    : 31 fish species
    in_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, num_classes)
    # Only this new layer will train

    return model


# ── Quick test ─────────────────────────────────────────────────
# Run: python src/model.py
if __name__ == "__main__":
    model = build_model(num_classes=31)
    print(model.classifier)  # should show Linear(1280 → 31)

    # Test with a dummy image batch
    dummy = torch.randn(2, 3, 224, 224)  # 2 fake images
    output = model(dummy)
    print(f"Output shape: {output.shape}")  # should be [2, 31]