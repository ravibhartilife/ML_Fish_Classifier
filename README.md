# ML Fish Classifier

Fish species classifier for **73 Indian fish species** using EfficientNetB0 transfer learning.  
Covers species from both pan-Indian waters and freshwater ecosystems of Assam, North-East India.

## Results
| Metric | Value |
|---|---|
| Species | 73 |
| Val Accuracy | 95.62% |
| Test Accuracy | 95.00% |
| Precision (macro) | 0.96 |
| Recall (macro) | 0.95 |
| F1-Score (macro) | 0.95 |

## Dataset
| Split | Images |
|---|---|
| Train | 24,373 |
| Val | 5,416 |
| Test | 3,400 |
| **Total** | **33,189** |

**Sources:**
- [Kaggle - Identify the Fish](https://www.kaggle.com/code/ammarbhilwarawala/identify-the-fish/input) — 31 species
- [AS-FishFresh Augmented Dataset (Mendeley)](https://data.mendeley.com/datasets/sdhp9bmrz5/2) — 48 Assam freshwater species

## Pretrained Model
Download `best_model.pth` and `class_names.json` from:  
[Google Drive](https://drive.google.com/drive/folders/1M9j-u3C7IiUD4yURx-mrlKcZWbPN2lrv?usp=sharing)

Place both files in the `models/` folder.

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

Install other dependencies:
```bash
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

# Merge datasets (if adding new data)
python src/merge_datasets.py
```

## Project Structure
ML_Fish_Classifier/
├── src/
│ ├── dataset.py # data loading & augmentation
│ ├── model.py # EfficientNetB0 architecture
│ ├── train.py # training loop
│ ├── predict.py # single image inference
│ ├── evaluate.py # test set evaluation
│ └── merge_datasets.py # combine multiple datasets
├── models/
│ ├── best_model.pth # download from Google Drive
│ └── class_names.json # download from Google Drive
├── requirements.txt
├── CONTRIBUTING.md
└── README.md


## Model Output
Top 5 Predictions:
Rank Species Confidence

1 Catfish 96.83%
2 Mudfish 1.36%
3 Perch 0.58%
4 Silver Perch 0.49%
5 Freshwater Eel 0.25%

→ Predicted: Catfish (96.83% confidence)