# =============================================================
# dataset.py  —  Load & prepare fish image data
# =============================================================
# What this file does:
#   1. Reads images from train / val / test folders
#   2. Resizes all images to 224x224 (EfficientNet input size)
#   3. Applies data augmentation on training images
#      (flip, rotate, zoom) to help the model generalize
#   4. Returns DataLoader objects PyTorch uses during training
# =============================================================

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Configuration ─────────────────────────────────────────────
DATA_DIR   = r"C:\Users\ravib\ML_Data_Set\Fish_Image_Data"  # root folder
BATCH_SIZE = 32       # images per batch (lower if GPU runs out of memory)
IMG_SIZE   = 224      # EfficientNetB0 expects 224x224
NUM_WORKERS = 2       # parallel data loading threads

# ── Transforms ────────────────────────────────────────────────
# Training: augment to prevent overfitting
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),       # resize to 224x224
    transforms.RandomHorizontalFlip(),             # randomly flip left-right
    transforms.RandomRotation(15),                 # randomly rotate ±15°
    transforms.ColorJitter(brightness=0.2,         # slight color variation
                           contrast=0.2,
                           saturation=0.2),
    transforms.ToTensor(),                         # convert to tensor [0,1]
    transforms.Normalize(                          # normalize with ImageNet stats
        mean=[0.485, 0.456, 0.406],                # (EfficientNet pretrained on it)
        std =[0.229, 0.224, 0.225]
    )
])

# Validation & Test: NO augmentation, just resize + normalize
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# ── Dataset loader function ────────────────────────────────────
def get_dataloaders():
    """
    Reads images from:
        DATA_DIR/train/  — for training
        DATA_DIR/val/    — for validation during training
        DATA_DIR/test/   — for final evaluation

    Folder structure expected:
        Fish_Image_Data/
            train/
                Catfish/  img1.jpg  img2.jpg ...
                Goby/     img1.jpg  ...
                ...
            val/
                Catfish/  ...
            test/
                Catfish/  ...

    Returns:
        train_loader, val_loader, test_loader, class_names
    """

    # Load datasets from folders
    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "train"),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "val"),
        transform=val_transform
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "test"),
        transform=val_transform
    )

    # Wrap in DataLoader for batching
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,              # shuffle training data each epoch
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # Class names = folder names (e.g. ['Catfish', 'Goby', ...])
    class_names = train_dataset.classes
    print(f"Classes  : {len(class_names)}")
    print(f"Train    : {len(train_dataset)} images")
    print(f"Val      : {len(val_dataset)} images")
    print(f"Test     : {len(test_dataset)} images")

    return train_loader, val_loader, test_loader, class_names


# ── Quick test ─────────────────────────────────────────────────
# Run this file directly to verify data loads correctly:
#   python src/dataset.py
if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    print(f"\nClass names:\n{class_names}")

    # Load one batch and print shape
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape : {images.shape}")   # [32, 3, 224, 224]
    print(f"Labels shape: {labels.shape}")     # [32]
