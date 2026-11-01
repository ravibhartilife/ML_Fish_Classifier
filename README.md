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

```text
ML_Fish_Classifier/
├── detection/
│   └── data.yaml
├── models/
│   ├── best_model.pth
│   ├── class_names.json
│   ├── confusion_matrix.png
│   └── yolo_fish/
├── src/
│   ├── classification/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   ├── detection/
│   │   ├── train_yolo.py
│   │   ├── predict_yolo.py
│   │   ├── count_fish.py
│   │   ├── density.py
│   │   └── heatmap.py
│   └── merge_datasets.py
├── requirements.txt
├── CONTRIBUTING.md
└── README.md
```

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

### Roadmap
- [x] Species classification (73 species)
- [x] Fish detection (YOLOv8)
- [x] Species identification per detection
- [x] Unknown species handling
- [x] Density estimation
- [x] Spatial heat map
- [x] Fish tracking (ByteTrack)
- [x] Water quality integration
- [x] Integrated report
- [x] GIS mapping
- [ ] REST API (FastAPI)
- [ ] Streamlit frontend
- [ ] Mobile app


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

### 7. GIS Mapping
```bash
python src/gis_map.py \
  --lat 23.4120 \
  --lon 85.3094 \
  --location "Kanke Dam, Ranchi" \
  --total 27 \
  --density 2.25 \
  --ph 7.59 \
  --water_safe 1 \
  --species_json '{"Catfish":3,"Goby":3,"Glass Perchlet":9}'
```
Serve the output map:
```bash
cd models/gis
python -m http.server 8080
# Open: http://localhost:8080/fish_gis_map.html
```

### 8. Integrated Report (All-in-One)
```bash
python src/integrated_report.py \
  --image "path/to/pond.jpg" \
  --turbidity 3.9 \
  --hardness 196 \
  --tds 500 \
  --depth 2 \
  --fov 90 \
  --pond_area 2000
```

---

## Building a Frontend / UI

### What data the backend returns

**Fish Classification** (`src/classification/predict.py`):
```json
{
  "predicted": "Catfish",
  "confidence": 96.83,
  "top5": [
    {"species": "Catfish",       "confidence": 96.83},
    {"species": "Mudfish",       "confidence": 1.36},
    {"species": "Perch",         "confidence": 0.58},
    {"species": "Silver Perch",  "confidence": 0.49},
    {"species": "Freshwater Eel","confidence": 0.25}
  ]
}
```

**Fish Detection + Count** (`src/detection/count_fish.py`):
```json
{
  "total_fish": 27,
  "species_count": {
    "Glass Perchlet": 9,
    "Gold Fish": 4,
    "Goby": 3,
    "Unknown": 8
  }
}
```

**Density Estimation** (`src/detection/density.py`):
```json
{
  "camera_area_m2": 12.0,
  "areal_density": 2.25,
  "estimated_total_fish": 4500,
  "species_density": {
    "Glass Perchlet": 0.75,
    "Gold Fish": 0.33
  }
}
```

**Water Quality** (`water_quality/main.py`):
```json
{
  "safe": true,
  "potability": 1,
  "predicted_ph": 7.59,
  "message": "Safe to drink ✅"
}
```

---

### How to expose as REST API

Install FastAPI:
```bash
pip install fastapi uvicorn python-multipart
```

Create `api.py` in project root:
```python
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, sys
sys.path.append("src/detection")
sys.path.append("src/classification")
from count_fish import count_fish
from density import estimate_density

app = FastAPI(title="ML Fish Classifier API")

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"])

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Save uploaded image
    path = f"temp_{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # Run detection
    result = count_fish(path)
    os.remove(path)
    return result

@app.post("/density")
async def density(file: UploadFile = File(...),
                  depth: float = 2.0,
                  fov: float = 90.0,
                  pond_area: float = 2000.0):
    path = f"temp_{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    result = estimate_density(path, depth, fov, pond_area)
    os.remove(path)
    return result
```

Run API:
```bash
uvicorn api:app --reload
# API docs at: http://localhost:8000/docs
```

---

### How to call from frontend (JavaScript)

**Single image:**
```javascript
const formData = new FormData();
formData.append("file", imageFile);  // from <input type="file">

const response = await fetch("http://localhost:8000/detect", {
    method: "POST",
    body: formData
});
const data = await response.json();
console.log(data.total_fish);
console.log(data.species_count);
```

**With density:**
```javascript
const response = await fetch(
    "http://localhost:8000/density?depth=2&fov=90&pond_area=2000",
    { method: "POST", body: formData }
);
```

---

### Recommended Frontend Stack

| Purpose | Tool |
|---|---|
| Quick demo UI | Streamlit |
| Web app | React + Axios |
| Mobile app | Flutter |
| Dashboard | Grafana |

**Streamlit (easiest — 30 lines):**
```bash
pip install streamlit
```
```python
import streamlit as st
import requests

st.title("🐟 Fish Classifier")

uploaded = st.file_uploader("Upload fish image", 
                             type=["jpg","png","jpeg"])
if uploaded:
    res = requests.post("http://localhost:8000/detect",
                        files={"file": uploaded})
    data = res.json()
    st.write(f"Total fish: {data['total_fish']}")
    st.bar_chart(data['species_count'])
```
Run:
```bash
streamlit run app.py
```

---

### Input formats supported

| Input | Script | How to pass |
|---|---|---|
| Single image | `predict.py`, `count_fish.py`, `density.py` | `--image path/to/fish.jpg` |
| Video file | `track_fish.py` | `--video path/to/video.mp4` |
| Water params | `integrated_report.py` | `--turbidity 3.9 --hardness 196 --tds 500` |
| GPS location | `gis_map.py` | `--lat 23.41 --lon 85.30 --location "Pond name"` |
| Species data | `gis_map.py` | `--species_json '{"Catfish":9}'` |

---
