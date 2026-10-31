# =============================================================
# train_yolo.py  —  Train YOLOv8 for fish detection
# =============================================================
# What this file does:
#   1. Loads YOLOv8 pretrained model
#   2. Trains on DePondFi/Etroplus dataset
#   3. Saves best model weights
# =============================================================

from ultralytics import YOLO

DATA_YAML  = r"C:\Users\ravib\ML_Fish_Classifier\detection\data.yaml"
MODEL_DIR  = r"C:\Users\ravib\ML_Fish_Classifier\models"
EPOCHS     = 50
IMG_SIZE   = 640
BATCH_SIZE = 16
DEVICE     = 0

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")

    results = model.train(
        data    = DATA_YAML,
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = BATCH_SIZE,
        device  = DEVICE,
        project = MODEL_DIR,
        name    = "yolo_fish",
        exist_ok= True
    )

    print("\nTraining complete!")
    print(f"Best model saved → {MODEL_DIR}/yolo_fish/weights/best.pt")