# =============================================================
# integrated_report.py  —  Fish detection + Water quality
# =============================================================
# What this file does:
#   1. Runs fish detection + species identification
#   2. Calculates fish density
#   3. Runs water quality prediction (safe/unsafe + pH)
#   4. Combines both into one final report
# =============================================================

import sys
import os
import torch
import json
import cv2
import numpy as np
import joblib
import math
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from collections import Counter

# Add classification to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'classification'))
from model import build_model

# Add water quality models path
WATER_QUALITY_DIR = r"C:\Users\ravib\ML_Fish_Classifier\water_quality"
sys.path.append(WATER_QUALITY_DIR)

from xgboost import XGBClassifier, XGBRegressor

# ── Configuration ─────────────────────────────────────────────
YOLO_MODEL   = r"C:\Users\ravib\ML_Fish_Classifier\models\yolo_fish\weights\best.pt"
EFFNET_MODEL = r"C:\Users\ravib\ML_Fish_Classifier\models\best_model.pth"
CLASSES_PATH = r"C:\Users\ravib\ML_Fish_Classifier\models\class_names.json"
DEVICE       = torch.device("cuda:0")
CONF         = 0.25

# Water quality model paths
CLF_MODEL    = os.path.join(WATER_QUALITY_DIR, "model_clf.json")
REG_MODEL    = os.path.join(WATER_QUALITY_DIR, "model_reg.json")
SCALER_CLF   = os.path.join(WATER_QUALITY_DIR, "scaler_clf.pkl")
SCALER_REG   = os.path.join(WATER_QUALITY_DIR, "scaler_reg.pkl")

# Median values for missing features
MEDIAN_VALUES = [
    196.0, 500.0, 7.1, 250.0, 426.0,
    14.0, 66.0, 3.9, 75.0, 28.0,
    50.0, 10.0, 150.0, 20.0, 0.9, 0.2, 120.0
]

# ── EfficientNet transform ─────────────────────────────────────
effnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

def load_fish_models():
    """Load YOLOv8 and EfficientNet models."""
    yolo = YOLO(YOLO_MODEL)
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)
    effnet = build_model(num_classes=len(class_names))
    effnet.load_state_dict(torch.load(EFFNET_MODEL, map_location=DEVICE))
    effnet = effnet.to(DEVICE)
    effnet.eval()
    return yolo, effnet, class_names

def load_water_models():
    """Load water quality classifier and regressor."""
    clf = XGBClassifier()
    clf.load_model(CLF_MODEL)
    reg = XGBRegressor()
    reg.load_model(REG_MODEL)
    scaler_clf = joblib.load(SCALER_CLF)
    scaler_reg = joblib.load(SCALER_REG)
    return clf, reg, scaler_clf, scaler_reg

def predict_water_quality(turbidity, hardness, tds,
                          clf, reg, scaler_clf, scaler_reg):
    """Predict water safety and pH."""
    features      = MEDIAN_VALUES.copy()
    features[0]   = hardness
    features[1]   = tds
    features[7]   = turbidity
    sample        = np.array([features])
    sample_clf    = scaler_clf.transform(sample)
    sample_reg    = scaler_reg.transform(sample)
    potability    = int(clf.predict(sample_clf)[0])
    ph            = float(reg.predict(sample_reg)[0])
    return potability, round(ph, 2)

def detect_and_count(image_path, yolo, effnet, class_names):
    """Detect fish and identify species."""
    image_bgr = cv2.imread(image_path)
    results   = yolo.predict(source=image_path,
                             conf=CONF, verbose=False)
    boxes     = results[0].boxes
    species_list = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = image_bgr[max(0,y1):y2, max(0,x1):x2]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(crop_rgb)
        tensor   = effnet_transform(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out   = effnet(tensor)
            probs = torch.softmax(out, dim=1)[0]
            idx   = torch.argmax(probs).item()
            conf  = probs[idx].item()
        species = class_names[idx] if conf >= 0.50 else "Unknown"
        species_list.append(species)

    return species_list

def calculate_density(fish_count, depth_m, fov_deg, pond_area_m2):
    """Calculate fish density and estimated total."""
    fov_rad  = math.radians(fov_deg)
    width    = 2 * depth_m * math.tan(fov_rad / 2)
    height   = width / (4/3)
    cam_area = width * height
    density  = fish_count / cam_area
    est_total = int(density * pond_area_m2)
    return cam_area, density, est_total

def integrated_report(image_path, turbidity, hardness,
                      tds, depth_m, fov_deg, pond_area_m2):
    """Full integrated report."""

    print("\nLoading models...")
    yolo, effnet, class_names = load_fish_models()
    clf, reg, scaler_clf, scaler_reg = load_water_models()

    # ── Fish detection ─────────────────────────────────────────
    print("Running fish detection...")
    species_list  = detect_and_count(image_path, yolo,
                                     effnet, class_names)
    species_count = Counter(species_list)
    total_fish    = len(species_list)

    # ── Density ────────────────────────────────────────────────
    cam_area, density, est_total = calculate_density(
        total_fish, depth_m, fov_deg, pond_area_m2)

    # ── Water quality ──────────────────────────────────────────
    print("Running water quality prediction...")
    potability, ph = predict_water_quality(
        turbidity, hardness, tds,
        clf, reg, scaler_clf, scaler_reg)

    safe_msg = "Safe ✅" if potability == 1 else "Not Safe ❌"

    # ── Print report ───────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"   INTEGRATED FISH & WATER QUALITY REPORT")
    print(f"{'='*55}")

    print(f"\n  📷 IMAGE: {os.path.basename(image_path)}")

    print(f"\n  🐟 FISH DETECTION:")
    print(f"     Total fish detected : {total_fish}")
    for sp, cnt in species_count.most_common():
        print(f"     {sp:<30} : {cnt}")

    print(f"\n  📊 DENSITY ESTIMATION:")
    print(f"     Camera area         : {cam_area:.2f} m²")
    print(f"     Areal density       : {density:.4f} fish/m²")
    print(f"     Est. total in pond  : {est_total} fish")

    print(f"\n  💧 WATER QUALITY:")
    print(f"     Turbidity           : {turbidity} NTU")
    print(f"     Hardness            : {hardness} mg/L")
    print(f"     TDS                 : {tds} mg/L")
    print(f"     Predicted pH        : {ph}")
    print(f"     Water safety        : {safe_msg}")

    print(f"\n{'='*55}\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True)
    parser.add_argument("--turbidity",  type=float, default=3.9)
    parser.add_argument("--hardness",   type=float, default=196.0)
    parser.add_argument("--tds",        type=float, default=500.0)
    parser.add_argument("--depth",      type=float, default=2.0)
    parser.add_argument("--fov",        type=float, default=90.0)
    parser.add_argument("--pond_area",  type=float, default=2000.0)
    args = parser.parse_args()

    integrated_report(
        image_path   = args.image,
        turbidity    = args.turbidity,
        hardness     = args.hardness,
        tds          = args.tds,
        depth_m      = args.depth,
        fov_deg      = args.fov,
        pond_area_m2 = args.pond_area
    )