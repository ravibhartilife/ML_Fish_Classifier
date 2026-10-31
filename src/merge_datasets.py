# =============================================================
# merge_datasets.py  —  Combine existing + Assam fish datasets
# =============================================================
# What this file does:
#   1. Reads new Assam species folders
#   2. Skips duplicates with existing dataset
#   3. Randomly picks 485 images per new species
#   4. Splits into train(380)/val(65)/test(40)
#   5. Copies into existing dataset structure
# =============================================================

import os
import shutil
import random

# ── Paths ─────────────────────────────────────────────────────
ASSAM_DIR    = r"C:\Users\ravib\ML_Data_Set\AS-FishFresh_Augmented_dataset"
EXISTING_DIR = r"C:\Users\ravib\ML_Data_Set\Fish_Image_Data"

# ── Split sizes ───────────────────────────────────────────────
TRAIN_COUNT = 380
VAL_COUNT   = 65
TEST_COUNT  = 40

# ── Known duplicates to skip ──────────────────────────────────
# Assam folder name : existing dataset folder name
DUPLICATES = {
    "GRASS CARP"  : "Grass Carp",
    "silvercarp"  : "Silver Carp",
    "bighead"     : "Big Head Carp",
    "rou"         : "Indian Carp",
    "basa"        : "Pangasius",
    "magur"       : "Catfish",
}

# ── Naming fixes ──────────────────────────────────────────────
# Assam folder name : clean name to use in combined dataset
NAME_FIXES = {
    "aari"        : "Aari",
    "aari2"       : "Aari",        # merge with aari
    "bami"        : "Bami",
    "bhokua"      : "Bhokua",
    "borali"      : "Borali",
    "bordua"      : "Bordua",
    "boriola"     : "Boriola",
    "chanda"      : "Chanda",
    "chanda nama" : "Chanda Nama",
    "commom carp" : "Common Carp",  # fix typo
    "gedgedi"     : "Gedgedi",
    "goroi"       : "Goroi",
    "ILISH"       : "Ilish",
    "indian gagata": "Indian Gagata",
    "kajoli"      : "Kajoli",
    "kanduli"     : "Kanduli",
    "KAWOI"       : "Kawoi",
    "kholihona"   : "Kholihona",
    "KOKILA"      : "Kokila",
    "koliajora"   : "Koliajora",
    "KOROTI"      : "Koroti",
    "KOS"         : "Kos",
    "kurhi"       : "Kurhi",
    "kusia"       : "Kusia",
    "lachim"      : "Lachim",
    "mirika"      : "Mirika",
    "misa"        : "Misa",
    "moa"         : "Moa",
    "PATI MUTURA" : "Pati Mutura",
    "pavo"        : "Pavo",
    "puthi"       : "Puthi",
    "rupchanda"   : "Rupchanda",
    "selkoni"     : "Selkoni",
    "sengeli"     : "Sengeli",
    "singi"       : "Singi",
    "sital"       : "Sital",
    "tura"        : "Tura",
    "turi"        : "Turi",
    "vangon"      : "Vangon",
    "xilghoria"   : "Xilghoria",
    "xingora"     : "Xingora",
    "xol"         : "Xol",
}

def get_images(folder):
    """Get all image files from a folder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts]

def copy_images(src_folder, images, dest_folder):
    """Copy list of images to destination folder."""
    os.makedirs(dest_folder, exist_ok=True)
    for img in images:
        shutil.copy2(
            os.path.join(src_folder, img),
            os.path.join(dest_folder, img)
        )

def merge():
    random.seed(42)  # reproducible splits

    assam_folders = os.listdir(ASSAM_DIR)
    print(f"Assam species folders found: {len(assam_folders)}")

    # Track merged species (handle aari + aari2)
    merged_species = {}  # clean_name → list of images already copied

    skipped  = []
    added    = []

    for folder in assam_folders:
        folder_path = os.path.join(ASSAM_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        # ── Skip duplicates ────────────────────────────────────
        if folder in DUPLICATES:
            print(f"  SKIP (duplicate): {folder}")
            skipped.append(folder)
            continue

        # ── Get clean name ─────────────────────────────────────
        clean_name = NAME_FIXES.get(folder, folder.title())

        # ── Get images ─────────────────────────────────────────
        images = get_images(folder_path)
        print(f"  Processing: {folder} → {clean_name} ({len(images)} images)")

        # ── Handle aari + aari2 merge ──────────────────────────
        if clean_name in merged_species:
            merged_species[clean_name].extend(
                [os.path.join(folder_path, img) for img in images]
            )
            continue
        else:
            merged_species[clean_name] = [
                os.path.join(folder_path, img) for img in images
            ]

    # ── Now split and copy each species ───────────────────────
    for clean_name, all_paths in merged_species.items():
        random.shuffle(all_paths)
        total_needed = TRAIN_COUNT + VAL_COUNT + TEST_COUNT

        if len(all_paths) < total_needed:
            print(f"  WARNING: {clean_name} has only {len(all_paths)} images, skipping")
            continue

        train_imgs = all_paths[:TRAIN_COUNT]
        val_imgs   = all_paths[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT]
        test_imgs  = all_paths[TRAIN_COUNT + VAL_COUNT:total_needed]

        for split, imgs in [("train", train_imgs),
                             ("val",   val_imgs),
                             ("test",  test_imgs)]:
            dest = os.path.join(EXISTING_DIR, split, clean_name)
            os.makedirs(dest, exist_ok=True)
            for src_path in imgs:
                shutil.copy2(src_path, os.path.join(dest,
                             os.path.basename(src_path)))

        added.append(clean_name)
        print(f"  Added: {clean_name} → train:{TRAIN_COUNT} val:{VAL_COUNT} test:{TEST_COUNT}")

    print(f"\nDone.")
    print(f"  Skipped (duplicates) : {len(skipped)} — {skipped}")
    print(f"  New species added    : {len(added)}")
    print(f"  Total species now    : {31 + len(added)}")

if __name__ == "__main__":
    merge()