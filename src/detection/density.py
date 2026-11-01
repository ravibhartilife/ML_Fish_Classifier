# =============================================================
# density.py  —  Estimate fish population density
# =============================================================
# What this file does:
#   1. Takes fish count from YOLOv8 detection
#   2. Takes camera parameters from user
#   3. Calculates camera coverage area
#   4. Estimates fish density (fish/m²)
#   5. Estimates total fish population in pond
# =============================================================

import math
import argparse
import sys
import os
from collections import Counter

# Add classification folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'classification'))

# ── Configuration ─────────────────────────────────────────────
YOLO_MODEL   = r"C:\Users\ravib\ML_Fish_Classifier\models\yolo_fish\weights\best.pt"
EFFNET_MODEL = r"C:\Users\ravib\ML_Fish_Classifier\models\best_model.pth"
CLASSES_PATH = r"C:\Users\ravib\ML_Fish_Classifier\models\class_names.json"
DEVICE       = "cuda:0"
CONF         = 0.25

def calculate_camera_area(depth_m, fov_deg, aspect_ratio=4/3):
    """
    Calculate the area (m²) visible to the camera.

    Args:
        depth_m      : distance from camera to fish (meters)
        fov_deg      : camera field of view in degrees
        aspect_ratio : width/height ratio (default 4:3)

    Returns:
        area in m²
    """
    fov_rad = math.radians(fov_deg)
    width   = 2 * depth_m * math.tan(fov_rad / 2)
    height  = width / aspect_ratio
    area    = width * height
    return area, width, height

def estimate_density(image_path, depth_m, fov_deg, pond_area_m2):
    """
    Full density estimation pipeline.

    Args:
        image_path   : path to fish pond image
        depth_m      : camera depth in meters
        fov_deg      : camera FOV in degrees
        pond_area_m2 : total pond area in m²
    """
    import torch
    import json
    import cv2
    from PIL import Image
    from torchvision import transforms
    from ultralytics import YOLO
    from model import build_model   # noqa: E402

    # ── Load models ───────────────────────────────────────────
    yolo_model = YOLO(YOLO_MODEL)

    with open(CLASSES_PATH) as f:
        class_names = json.load(f)

    effnet = build_model(num_classes=len(class_names))
    import torch
    effnet.load_state_dict(torch.load(EFFNET_MODEL,
                           map_location=DEVICE))
    effnet = effnet.to(DEVICE)
    effnet.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

    # ── Step 1: Detect fish ───────────────────────────────────
    results   = yolo_model.predict(source=image_path,
                                   conf=CONF, verbose=False)
    result    = results[0]
    boxes     = result.boxes
    image_bgr = cv2.imread(image_path)

    # ── Step 2: Identify species ──────────────────────────────
    species_list = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(crop_rgb)
        tensor   = transform(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out   = effnet(tensor)
            probs = torch.softmax(out, dim=1)[0]
            idx   = torch.argmax(probs).item()
        species_list.append(class_names[idx])

    # ── Step 3: Calculate camera coverage area ────────────────
    cam_area, width, height = calculate_camera_area(depth_m, fov_deg)

    # ── Step 4: Calculate density ─────────────────────────────
    total_fish     = len(species_list)
    density        = total_fish / cam_area          # fish/m²
    est_total_fish = int(density * pond_area_m2)    # estimated total in pond

    species_count  = Counter(species_list)

    # ── Step 5: Print report ──────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  FISH DENSITY ESTIMATION REPORT")
    print(f"{'='*50}")
    print(f"\n  Camera Parameters:")
    print(f"    Depth          : {depth_m} m")
    print(f"    FOV            : {fov_deg}°")
    print(f"    Coverage       : {width:.2f}m × {height:.2f}m")
    print(f"    Camera Area    : {cam_area:.2f} m²")
    print(f"\n  Pond Information:")
    print(f"    Total Area     : {pond_area_m2} m²")
    print(f"\n  Detection Results:")
    print(f"    Fish Detected  : {total_fish}")
    print(f"\n  Density Metrics:")
    print(f"    Areal Density  : {density:.4f} fish/m²")
    print(f"    Est. Total Fish: {est_total_fish} fish in pond")

    print(f"\n  Per-Species Density:")
    print(f"  {'Species':<30} {'Count':<8} {'Density (fish/m²)'}")
    print(f"  {'-'*55}")
    for species, count in species_count.most_common():
        sp_density = count / cam_area
        print(f"  {species:<30} {count:<8} {sp_density:.4f}")

    print(f"\n{'='*50}")
    print(f"  NOTE: Density is estimated from a single frame.")
    print(f"  For accurate results use multiple frames.")
    print(f"{'='*50}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",     required=True,  help="Path to fish image")
    parser.add_argument("--depth",     type=float, default=2.0,   help="Camera depth in meters")
    parser.add_argument("--fov",       type=float, default=90.0,  help="Camera FOV in degrees")
    parser.add_argument("--pond_area", type=float, default=2000.0,help="Total pond area in m²")
    args = parser.parse_args()

    estimate_density(
        image_path   = args.image,
        depth_m      = args.depth,
        fov_deg      = args.fov,
        pond_area_m2 = args.pond_area
    )