# =============================================================
# evaluate.py  —  Test model on full test set
# =============================================================
# What this file does:
#   1. Loads best saved model
#   2. Runs inference on all test images
#   3. Prints accuracy + per-class report
#   4. Shows confusion matrix
# =============================================================

import torch
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import get_dataloaders
from model import build_model

# ── Configuration ─────────────────────────────────────────────
MODEL_PATH   = r"C:\Users\ravib\ML_Fish_Classifier\models\best_model.pth"
CLASSES_PATH = r"C:\Users\ravib\ML_Fish_Classifier\models\class_names.json"
DEVICE       = torch.device("cuda:0")

def evaluate():
    # Load class names
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)

    # Load model
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # Load test data
    _, _, test_loader, _ = get_dataloaders()

    # Run inference
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    acc = (all_preds == all_labels).mean()
    print(f"\nTest Accuracy: {acc*100:.2f}%")

    # Per-class report
    print("\nPer-Class Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(r"C:\Users\ravib\ML_Fish_Classifier\models\confusion_matrix.png")
    print("\nConfusion matrix saved → models/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()