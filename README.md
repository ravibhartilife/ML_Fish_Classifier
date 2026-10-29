# ML Fish Classifier

Fish species classifier for 31 Indian fish species using EfficientNetB0 transfer learning.

## Results
- Val Accuracy  : 91.86%
- Test Accuracy : 90.91%

## Dataset
- 31 species, 13,304 images
- Train: 8793 | Val: 2751 | Test: 1760
- Source: [Kaggle - Identify the Fish](https://www.kaggle.com/code/ammarbhilwarawala/identify-the-fish/input)

## Pretrained Model
Download `best_model.pth` and `class_names.json` from:
[Google Drive](https://drive.google.com/drive/folders/1M9j-u3C7IiUD4yURx-mrlKcZWbPN2lrv?usp=sharing)

Place them in the `models/` folder.

## Setup
```bash
conda create -n fish_classifier python=3.10
conda activate fish_classifier
pip install -r requirements.txt
```

## Usage
```bash
# Train from scratch
python src/train.py

# Predict single image
python src/predict.py --image "path/to/fish.jpg"

# Evaluate on test set
python src/evaluate.py
```

## Project Structure
```
ML_Fish_Classifier/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── models/
│   ├── best_model.pth      # download from Google Drive
│   └── class_names.json    # download from Google Drive
├── requirements.txt
└── README.md
```