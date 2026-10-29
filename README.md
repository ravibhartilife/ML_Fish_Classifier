# ML Fish Classifier

Fish species classifier for 31 Indian fish species using EfficientNetB0 transfer learning.

## Results
- Val Accuracy : 91.86%
- Test Accuracy : 90.91%

## Dataset
- 31 species, 13,304 images
- Train: 8793 | Val: 2751 | Test: 1760

## Setup
```bash
conda create -n fish_classifier python=3.10
conda activate fish_classifier
pip install -r requirements.txt
```

## Usage
```bash
# Train
python src/train.py

# Predict
python src/predict.py --image "path/to/fish.jpg"

# Evaluate
python src/evaluate.py
```

## Structure
```
ML_Fish_Classifier/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── models/
│   └── best_model.pth
├── data/
├── requirements.txt
└── README.md
```