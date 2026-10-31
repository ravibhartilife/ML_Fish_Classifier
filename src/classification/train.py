# =============================================================
# train.py  —  Train EfficientNetB0 on fish images
# =============================================================
# What this file does:
#   1. Loads data (dataset.py) and model (model.py)
#   2. Trains for N epochs
#   3. Validates after each epoch
#   4. Saves the best model (highest val accuracy)
#   5. Prints progress after each epoch
# =============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from dataset import get_dataloaders
from model import build_model

# ── Configuration ─────────────────────────────────────────────
EPOCHS     = 20        # number of training cycles
LR         = 0.001     # learning rate
MODEL_DIR  = r"C:\Users\ravib\ML_Fish_Classifier\models"
DEVICE     = torch.device("cuda:0")  # RTX 5050

def train():

    # ── Load data ──────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)

    # ── Build model ────────────────────────────────────────────
    model = build_model(num_classes=num_classes)
    model = model.to(DEVICE)  # move model to GPU

    # ── Loss & Optimizer ───────────────────────────────────────
    # CrossEntropyLoss: standard for multiclass classification
    criterion = nn.CrossEntropyLoss()

    # Only optimize the last layer (frozen rest)
    optimizer = optim.Adam(
        model.classifier[1].parameters(),
        lr=LR
    )

    # Reduce LR if val accuracy stops improving
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ── Training loop ──────────────────────────────────────────
    best_val_acc = 0.0
    os.makedirs(MODEL_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        # ── Train phase ────────────────────────────────────────
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()          # clear previous gradients
            outputs = model(images)        # forward pass
            loss = criterion(outputs, labels)  # compute loss
            loss.backward()               # backpropagation
            optimizer.step()              # update weights

            train_loss    += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        # ── Validation phase ───────────────────────────────────
        model.eval()
        val_loss, val_correct = 0.0, 0

        with torch.no_grad():  # no gradient needed for validation
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        # ── Compute metrics ────────────────────────────────────
        train_acc = train_correct / len(train_loader.dataset)
        val_acc   = val_correct   / len(val_loader.dataset)
        train_loss = train_loss   / len(train_loader.dataset)
        val_loss   = val_loss     / len(val_loader.dataset)

        scheduler.step()

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc*100:.2f}%  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc*100:.2f}%")

        # ── Save best model ────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, "best_model.pth"))
            print(f"  → Best model saved (val acc: {val_acc*100:.2f}%)")

    # ── Save class names ───────────────────────────────────────
    with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    print(f"\nTraining complete. Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"Model saved → {MODEL_DIR}/best_model.pth")


if __name__ == "__main__":
    train()