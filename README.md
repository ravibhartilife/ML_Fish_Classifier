# ML Fish Classifier

An AI-based system for **fish species identification, detection, counting, density estimation and spatial heat mapping** in Indian freshwater ecosystems.

---

## Results Summary

| Module | Model | Metric | Value |
|---|---|---|---|
| Species Classification | EfficientNetB0 | Test Accuracy | 95.00% |
| Fish Detection | YOLOv8n | mAP@50 | 92.3% |
| Density Estimation | Formula-based | — | fish/m² |
| Heat Map | OpenCV | — | spatial overlay |

---

## Dataset

| Split | Images |
|---|---|
| Train | 24,373 |
| Val | 5,416 |
| Test | 3,400 |
| **Total** | **33,189** |

**73 Indian freshwater fish species**

**Sources:**
- [Kaggle - Identify the Fish](https://www.kaggle.com/code/ammarbhilwarawala/identify-the-fish/input) — 31 species
- [AS-FishFresh Augmented Dataset (Mendeley)](https://data.mendeley.com/datasets/sdhp9bmrz5/2) — 48 Assam freshwater species

---

## Pretrained Models
Download from [Google Drive](https://drive.google.com/drive/folders/1M9j-u3C7IiUD4yURx-mrlKcZWbPN2lrv?usp=sharing):
- `best_model.pth` → EfficientNetB0 classification model
- `class_names.json` → species list

Place both in `models/` folder.

---

## Project Structure
ML_Fish_Classifier/
├── detection/
│ └── data.yaml # YOLOv8 dataset config
├── models/
│ ├── best_model.pth # EfficientNetB0 weights
│ ├── class_names.json # 73 species names
│ ├── confusion_matrix.png # evaluation results
│ └── yolo_fish/ # YOLOv8 training results
├── src/
│ ├── classification/
│ │ ├── dataset.py # data loading & augmentation
│ │ ├── model.py # EfficientNetB0 architecture
│ │ ├── train.py # training loop
│ │ ├── predict.py # single image inference
│ │ └── evaluate.py # test set evaluation
│ ├── detection/
│ │ ├── train_yolo.py # YOLOv8 training
│ │ ├── predict_yolo.py # fish detection + counting
│ │ ├── count_fish.py # detect + identify species
│ │ ├── density.py # fish density estimation
│ │ └── heatmap.py # spatial distribution map
│ └── merge_datasets.py # combine multiple datasets
├── requirements.txt
├── CONTRIBUTING.md
└── README.md

---

## Setup

```bash
conda create -n fish_classifier python=3.10
conda activate fish_classifier
```

Install PyTorch based on your GPU:
```bash
# NVIDIA RTX 50xx (Blackwell/CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# NVIDIA RTX 30xx/40xx (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio
```

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Species Classification
```bash
python src/classification/predict.py --image "path/to/fish.jpg"
```

### 2. Fish Detection & Counting
```bash
python src/detection/predict_yolo.py --image "path/to/pond.jpg"
```

### 3. Detect + Identify Species
```bash
python src/detection/count_fish.py --image "path/to/pond.jpg"
```

### 4. Density Estimation
```bash
python src/detection/density.py --image "path/to/pond.jpg" --depth 2 --fov 90 --pond_area 2000
```
- `--depth` : camera depth in meters
- `--fov` : camera field of view in degrees
- `--pond_area` : total pond area in m²

### 5. Heat Map
```bash
python src/detection/heatmap.py --image "path/to/pond.jpg"
```

### 6. Train from Scratch
```bash
# Classification
python src/classification/train.py

# Detection
python src/detection/train_yolo.py
```

---

## Pipeline Flow
Input Image/Video
↓
YOLOv8 → Detect fish + draw boxes + count
↓
EfficientNetB0 → Identify species per fish
↓
Density Module → fish/m² + estimated pond population
↓
Heat Map → spatial distribution overlay

## Sample Output

**Species Count:**
Glass Perchlet : 9 → 0.75 fish/m²
Gold Fish : 4 → 0.33 fish/m²
Freshwater Eel : 4 → 0.33 fish/m²
─────────────────────────────────────
TOTAL : 27

**Density Report:**
Camera Coverage : 4.00m × 3.00m = 12 m²
Areal Density : 2.25 fish/m²
Est. Total Fish : 4,500 in pond

---

## Roadmap
- [x] Species classification (73 species)
- [x] Fish detection (YOLOv8)
- [x] Species identification per detection
- [x] Density estimation
- [x] Spatial heat map
- [x] Fish tracking (ByteTrack) — avoid double counting in video
- [ ] Water quality IoT integration
- [ ] GIS mapping

### 6. Fish Tracking (Video)
```bash
python src/detection/track_fish.py --video "path/to/video.mp4"
```

**Output:**
Video : video.mp4
FPS : 29
Frames : 457

Frame 30/457 — Fish in frame: 21 — Unique total: 47
Frame 60/457 — Fish in frame: 18 — Unique total: 58
...
=============================================
FISH TRACKING REPORT
Total frames processed : 454
Avg fish per frame : 21.1
Max fish in one frame : 31
Unique fish tracked : 368


**How it works:**
- YOLOv8 detects fish in every frame
- ByteTrack assigns unique ID to each fish
- Same fish keeps same ID across frames
- Only new IDs counted → no double counting