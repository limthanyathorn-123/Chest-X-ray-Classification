import os
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ======== EDIT PATHS HERE ========
TRAIN_LIST = "/home/tlimsuva/cse591/presentation_classification/Xray14_train_official.txt"
VAL_LIST   = "/home/tlimsuva/cse591/presentation_classification/Xray14_val_official.txt"
TEST_LIST  = "/home/tlimsuva/cse591/presentation_classification/Xray14_test_official.txt"

SRC_DIR = "/scratch/tlimsuva/chestxray14/images"
DST_DIR = "/scratch/tlimsuva/chestxray14/images_224"
SIZE = 224
# ==================================

def collect_image_names(list_path):
    names = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_name = line.split()[0]
            names.append(os.path.basename(img_name))
    return names

def main():
    os.makedirs(DST_DIR, exist_ok=True)

    # collect unique images from all splits
    all_names = []
    for lst in [TRAIN_LIST, VAL_LIST, TEST_LIST]:
        all_names.extend(collect_image_names(lst))

    all_names = sorted(set(all_names))
    print(f"Total unique images: {len(all_names)}")

    missing = 0

    for name in tqdm(all_names, desc="Resizing images"):
        src = os.path.join(SRC_DIR, name)
        dst = os.path.join(DST_DIR, name)

        if not os.path.exists(src):
            missing += 1
            continue

        if os.path.exists(dst):
            continue  # skip if already resized

        try:
            img = Image.open(src).convert("RGB")
            img = img.resize((SIZE, SIZE), Image.BILINEAR)
            img.save(dst, optimize=True)
        except Exception as e:
            print(f"Error processing {name}: {e}")

    print("Done.")
    print("Missing files:", missing)

if __name__ == "__main__":
    main()