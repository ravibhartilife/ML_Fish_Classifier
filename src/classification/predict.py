# =============================================================
# predict.py  —  Identify a fish from a single image
# =============================================================
# Usage:
#   python src/predict.py --image "path/to/fish.jpg"
# =============================================================

import torch
import json
import argparse
from PIL import Image
from torchvision import transforms
from model import build_model

# ── Configuration ─────────────────────────────────────────────
MODEL_PATH  = r"C:\Users\ravib\ML_Fish_Classifier\models\best_model.pth"
CLASSES_PATH = r"C:\Users\ravib\ML_Fish_Classifier\models\class_names.json"
DEVICE      = torch.device("cuda:0")

# ── Same transform as validation (no augmentation) ────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

def load_model(num_classes):
    model = build_model(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()  # set to inference mode
    return model

def predict(image_path):
    # Load class names
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)

    # Load model
    model = load_model(num_classes=len(class_names))

    # Load & preprocess image
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)  # add batch dim

    # Predict
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]  # convert to probabilities
        top5    = torch.topk(probs, k=5)            # top 5 predictions

    print(f"\nImage: {image_path}")
    print(f"\nTop 5 Predictions:")
    print(f"{'Rank':<6} {'Species':<30} {'Confidence'}")
    print("-" * 50)
    for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
        print(f"  {i+1}    {class_names[idx]:<30} {prob*100:.2f}%")

    print(f"\n→ Predicted: {class_names[top5.indices[0]]} ({top5.values[0]*100:.2f}% confidence)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to fish image")
    args = parser.parse_args()
    predict(args.image)