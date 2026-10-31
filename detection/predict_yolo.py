# =============================================================
# predict_yolo.py  —  Detect and count fish in an image
# =============================================================
# What this file does:
#   1. Loads trained YOLOv8 model
#   2. Runs detection on a single image
#   3. Draws bounding boxes around each fish
#   4. Counts total fish detected
#   5. Saves result image with boxes drawn
# =============================================================

from ultralytics import YOLO
import cv2

# ── Configuration ─────────────────────────────────────────────
MODEL_PATH  = r"C:\Users\ravib\ML_Fish_Classifier\models\yolo_fish\weights\best.pt"
OUTPUT_DIR  = r"C:\Users\ravib\ML_Fish_Classifier\models\yolo_fish"
CONF        = 0.25   # confidence threshold (0-1)
                     # lower = detect more fish (may include false positives)
                     # higher = only confident detections

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to fish image")
    args = parser.parse_args()

    # ── Load model ─────────────────────────────────────────────
    model = YOLO(MODEL_PATH)

    # ── Run detection ──────────────────────────────────────────
    results = model.predict(
        source  = args.image,
        conf    = CONF,
        save    = True,           # saves image with boxes drawn
        project = OUTPUT_DIR,
        name    = "predictions",
        exist_ok= True
    )

    # ── Count fish ─────────────────────────────────────────────
    for result in results:
        fish_count = len(result.boxes)
        print(f"\nImage : {args.image}")
        print(f"Fish detected : {fish_count}")
        print(f"\nDetections:")
        for i, box in enumerate(result.boxes):
            conf = box.conf[0].item()
            print(f"  Fish {i+1}: confidence {conf*100:.1f}%")

    print(f"\nResult image saved → {OUTPUT_DIR}/predictions/")