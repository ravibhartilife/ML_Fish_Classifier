# Contributing & Extensions

## Running the Project
1. Clone the repo
   ```bash
   git clone https://github.com/ravibhartilife/ML_Fish_Classifier.git
   cd ML_Fish_Classifier
   ```
2. Create environment
   ```bash
   conda create -n fish_classifier python=3.10
   conda activate fish_classifier
   ```
3. Install PyTorch (choose based on your GPU)
   ```bash
   # NVIDIA RTX 50xx (Blackwell)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   # NVIDIA RTX 30xx/40xx
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   # CPU only
   pip install torch torchvision torchaudio
   ```
4. Install other dependencies
   ```bash
   pip install -r requirements.txt
   ```
5. Download model files from [Google Drive](https://drive.google.com/drive/folders/1M9j-u3C7IiUD4yURx-mrlKcZWbPN2lrv?usp=sharing)
   and place in `models/` folder

6. Predict
   ```bash
   python src/predict.py --image "path/to/fish.jpg"
   ```

## Adding a Frontend

### Option 1 — Streamlit (Recommended, Easy)
```bash
pip install streamlit
```
Create `app.py` in project root:
```python
import streamlit as st
from src.predict import predict

st.title("Fish Classifier")
uploaded = st.file_uploader("Upload a fish image", type=["jpg","png","jpeg"])
if uploaded:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded.read())
    predict("temp.jpg")
```
Run:
```bash
streamlit run app.py
```

### Option 2 — Flask (Custom UI, Medium)
```bash
pip install flask
```
- Create `app.py` with Flask routes
- Accept image upload via POST request
- Call `predict()` from `src/predict.py`
- Return JSON result to frontend

### Option 3 — FastAPI (REST API, Advanced)
```bash
pip install fastapi uvicorn python-multipart
```
- Build REST API endpoint
- Connect any frontend (React, Vue, etc.)
- Deploy on cloud (Render, Railway, HuggingFace Spaces)

## Improving the Model
- Add more images per species (500+ recommended)
- Unfreeze more layers for fine-tuning
- Try EfficientNetB3 or ResNet50 for higher accuracy
- Add more Indian fish species

## Model Output

When you call `predict("fish.jpg")`, the model returns top 5 predictions:

Top 5 Predictions:
Rank Species Confidence

1 Catfish 96.83%
2 Mudfish 1.36%
3 Perch 0.58%
4 Silver Perch 0.49%
5 Freshwater Eel 0.25%

→ Predicted: Catfish (96.83% confidence)


### For Frontend Integration
If you need raw values instead of printed output, modify `predict()` in `src/predict.py` to return a dict:

```python
return {
    "predicted": class_names[top5.indices[0]],
    "confidence": round(top5.values[0].item() * 100, 2),
    "top5": [
        {
            "species": class_names[idx],
            "confidence": round(prob.item() * 100, 2)
        }
        for prob, idx in zip(top5.values, top5.indices)
    ]
}
```

### Example JSON response:
```json
{
  "predicted": "Catfish",
  "confidence": 96.83,
  "top5": [
    {"species": "Catfish", "confidence": 96.83},
    {"species": "Mudfish", "confidence": 1.36},
    {"species": "Perch", "confidence": 0.58},
    {"species": "Silver Perch", "confidence": 0.49},
    {"species": "Freshwater Eel", "confidence": 0.25}
  ]
}
```
## All Scripts Reference

| Script | Location | Usage |
|---|---|---|
| Species classification | `src/classification/predict.py` | `python src/classification/predict.py --image "path"` |
| Fish detection | `src/detection/predict_yolo.py` | `python src/detection/predict_yolo.py --image "path"` |
| Detect + identify species | `src/detection/count_fish.py` | `python src/detection/count_fish.py --image "path"` |
| Density estimation | `src/detection/density.py` | `python src/detection/density.py --image "path" --depth 2 --fov 90 --pond_area 2000` |
| Heat map | `src/detection/heatmap.py` | `python src/detection/heatmap.py --image "path"` |
| Fish tracking (video) | `src/detection/track_fish.py` | `python src/detection/track_fish.py --video "path"` |
| Integrated report | `src/integrated_report.py` | `python src/integrated_report.py --image "path" --turbidity 3.9 --hardness 196 --tds 500` |
| GIS map | `src/gis_map.py` | `python src/gis_map.py --lat 23.41 --lon 85.30 --location "Pond name"` |
| Train classification | `src/classification/train.py` | `python src/classification/train.py` |
| Train detection | `src/detection/train_yolo.py` | `python src/detection/train_yolo.py` |
| Merge datasets | `src/merge_datasets.py` | `python src/merge_datasets.py` |

## Unknown Species Handling

If a fish species is not in the 73 known species, EfficientNet will label it as **"Unknown"** when confidence is below 50%.

To add new species:
1. Collect 400+ images of the new species
2. Add to `C:\Users\ravib\ML_Data_Set\Fish_Image_Data\train\<species_name>\`
3. Run `python src/classification/train.py`

## Water Quality API

Start the water quality API:
```bash
cd water_quality
pip install fastapi uvicorn
uvicorn main:app --reload
```
API runs at: `http://localhost:8000`
Docs at: `http://localhost:8000/docs`

Input:
```json
{
  "Turbidity": 3.9,
  "Hardness": 196.0,
  "TDS": 500.0
}
```

Output:
```json
{
  "safe": true,
  "potability": 1,
  "predicted_ph": 7.59,
  "message": "Safe to drink ✅"
}
```

## GIS Map Usage

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

Serve the map:
```bash
cd models/gis
python -m http.server 8080
# Open: http://localhost:8080/fish_gis_map.html
```

## Common Errors

| Error | Fix |
|---|---|
| `CUDA not available` | Install correct PyTorch CUDA version |
| `No module named 'model'` | Run scripts from project root |
| `FileNotFoundError: best_model.pth` | Download from Google Drive |
| `Unknown species` | Confidence < 50%, fish not in 73 species |
| `GIS map blank` | Serve via `python -m http.server 8080` |