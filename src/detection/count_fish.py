# =============================================================
# count_fish.py  —  Detect fish + identify species
# =============================================================
# Pipeline:
#   Image → YOLOv8 (detect & count) → crop each fish
#         → EfficientNet (identify species) → final report
# =============================================================

import torch
import json
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'classification'))
from model import build_model
from collections import Counter

# ── Configuration ─────────────────────────────────────────────
YOLO_MODEL    = r"C:\Users\ravib\ML_Fish_Classifier\models\yolo_fish\weights\best.pt"
EFFNET_MODEL  = r"C:\Users\ravib\ML_Fish_Classifier\models\best_model.pth"
CLASSES_PATH  = r"C:\Users\ravib\ML_Fish_Classifier\models\class_names.json"
OUTPUT_DIR    = r"C:\Users\ravib\ML_Fish_Classifier\models\combined"
DEVICE        = torch.device("cuda:0")
CONF          = 0.25   # YOLOv8 confidence threshold

# ── EfficientNet transform ────────────────────────────────────
effnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

def load_effnet(num_classes):
    """Load EfficientNet species classifier."""
    model = build_model(num_classes=num_classes)
    model.load_state_dict(torch.load(EFFNET_MODEL, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def identify_species(crop, effnet_model, class_names):
    """
    Given a cropped fish image, identify its species.
    
    Args:
        crop: numpy array (BGR image from OpenCV)
        effnet_model: loaded EfficientNet model
        class_names: list of species names
    
    Returns:
        species name, confidence
    """
    # Convert BGR (OpenCV) to RGB (PIL)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img  = Image.fromarray(crop_rgb)

    # Preprocess
    tensor = effnet_transform(pil_img).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        outputs = effnet_model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        top1    = torch.argmax(probs).item()
        conf    = probs[top1].item()

    return class_names[top1], conf

def count_fish(image_path):
    """
    Main pipeline: detect fish → identify species → count.
    
    Args:
        image_path: path to input image
    """
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load class names ───────────────────────────────────────
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)

    # ── Load models ───────────────────────────────────────────
    yolo_model   = YOLO(YOLO_MODEL)
    effnet_model = load_effnet(num_classes=len(class_names))

    # ── Step 1: YOLOv8 detection ──────────────────────────────
    print(f"\nProcessing: {image_path}")
    image_bgr = cv2.imread(image_path)
    results   = yolo_model.predict(source=image_path, conf=CONF, verbose=False)
    result    = results[0]
    boxes     = result.boxes

    print(f"Fish detected by YOLOv8: {len(boxes)}")

    # ── Step 2: Crop each fish + identify species ──────────────
    species_list = []

    for i, box in enumerate(boxes):
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Add padding to crop
        pad = 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image_bgr.shape[1], x2 + pad)
        y2 = min(image_bgr.shape[0], y2 + pad)

        # Crop fish from image
        crop = image_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # Identify species
        species, conf = identify_species(crop, effnet_model, class_names)
        species_list.append(species)

        # Draw box and label on image
        color = (0, 255, 0)  # green
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{species} {conf*100:.0f}%"
        cv2.putText(image_bgr, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # ── Step 3: Count per species ──────────────────────────────
    species_count = Counter(species_list)

    print(f"\n{'='*40}")
    print(f"FISH COUNT REPORT")
    print(f"{'='*40}")
    for species, count in species_count.most_common():
        print(f"  {species:<30} : {count}")
    print(f"{'─'*40}")
    print(f"  {'TOTAL':<30} : {len(species_list)}")
    print(f"{'='*40}")

    # ── Step 4: Save result image ──────────────────────────────
    import os
    output_path = os.path.join(OUTPUT_DIR,
                  os.path.basename(image_path))
    cv2.imwrite(output_path, image_bgr)
    print(f"\nResult saved → {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to fish image")
    args = parser.parse_args()
    count_fish(args.image)