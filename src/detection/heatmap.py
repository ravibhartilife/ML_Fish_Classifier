# =============================================================
# heatmap.py  —  Spatial fish distribution heat map
# =============================================================
# What this file does:
#   1. Detects fish using YOLOv8
#   2. Gets center point of each detected fish
#   3. Creates a 2D density grid
#   4. Applies Gaussian blur to smooth distribution
#   5. Overlays colored heat map on original image
#   6. Saves result
# =============================================================

import cv2
import numpy as np
import argparse
import os
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────
YOLO_MODEL = r"C:\Users\ravib\ML_Fish_Classifier\models\yolo_fish\weights\best.pt"
OUTPUT_DIR = r"C:\Users\ravib\ML_Fish_Classifier\models\heatmap"
CONF       = 0.25

def generate_heatmap(image_path):
    """
    Generate spatial fish distribution heat map.

    Args:
        image_path: path to fish pond image
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Load image ────────────────────────────────────
    image_bgr = cv2.imread(image_path)
    h, w      = image_bgr.shape[:2]

    # ── Step 2: Detect fish ───────────────────────────────────
    model   = YOLO(YOLO_MODEL)
    results = model.predict(source=image_path,
                            conf=CONF, verbose=False)
    boxes   = results[0].boxes

    print(f"\nImage : {image_path}")
    print(f"Fish detected : {len(boxes)}")

    # ── Step 3: Create density grid ───────────────────────────
    # Empty grid same size as image
    density_map = np.zeros((h, w), dtype=np.float32)

    # Plot each fish center point on the grid
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Center of bounding box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Add a dot at fish center
        # We use a circle instead of a single pixel
        # so the heat map looks smoother
        radius = max(w, h) // 20  # radius scales with image size
        cv2.circle(density_map, (cx, cy), radius, 1.0, -1)

    # ── Step 4: Gaussian blur ─────────────────────────────────
    # Blur spreads the heat around each fish location
    # Higher sigma = more spread out heat
    sigma       = max(w, h) // 15
    density_map = cv2.GaussianBlur(density_map,
                                   (0, 0), sigmaX=sigma)

    # ── Step 5: Normalize to 0-255 ────────────────────────────
    if density_map.max() > 0:
        density_map = density_map / density_map.max()
    density_map_uint8 = (density_map * 255).astype(np.uint8)

    # ── Step 6: Apply color map ───────────────────────────────
    # COLORMAP_JET: blue=low, green=medium, red=high density
    heatmap_color = cv2.applyColorMap(density_map_uint8,
                                      cv2.COLORMAP_JET)

    # ── Step 7: Overlay on original image ─────────────────────
    alpha  = 0.5   # transparency of heat map overlay
    result = cv2.addWeighted(image_bgr,    1 - alpha,
                             heatmap_color, alpha, 0)

    # ── Step 8: Draw fish centers as dots ─────────────────────
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(result, (cx, cy), 4, (255, 255, 255), -1)

    # ── Step 9: Add legend ────────────────────────────────────
    cv2.putText(result, f"Fish count: {len(boxes)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    cv2.putText(result, "Blue=Low  Green=Med  Red=High",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

    # ── Step 10: Save ─────────────────────────────────────────
    output_path = os.path.join(OUTPUT_DIR,
                  "heatmap_" + os.path.basename(image_path))
    cv2.imwrite(output_path, result)
    print(f"Heat map saved → {output_path}")

    # Also save pure heat map without original image
    heatmap_only_path = os.path.join(OUTPUT_DIR,
                        "heatmap_only_" + os.path.basename(image_path))
    cv2.imwrite(heatmap_only_path, heatmap_color)
    print(f"Heat map only → {heatmap_only_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True,
                        help="Path to fish pond image")
    args = parser.parse_args()
    generate_heatmap(args.image)