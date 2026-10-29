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