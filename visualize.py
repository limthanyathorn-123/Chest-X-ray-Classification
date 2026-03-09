#!/usr/bin/env python3

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import densenet121

# -----------------------
# CONFIG (EDIT PATHS)
# -----------------------
TEST_LIST = "/home/tlimsuva/cse591/presentation_classification/Xray14_test_official.txt"
DATA_DIR  = "/scratch/tlimsuva/chestxray14/images_224"   # resized images
MODEL_PATH = "./runs_chestxray14_a100/best_round2.pt" # best model
OUT_DIR = "./runs_chestxray14_a100/visual_predictions"
BATCH_SIZE = 64
NUM_WORKERS = 8
TOPK = 5

# Only these files will be saved
PICK_FILES = {"00000013_000.png","00000013_001.png", "00000013_004.png"}
# -----------------------

CLASSES_14 = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

# -----------------------
# Dataset
# -----------------------
def parse_line(line):
    parts = line.strip().split()
    img = parts[0]
    labels = np.array([float(x) for x in parts[1:15]], dtype=np.float32)
    return img, labels


class XrayDataset(Dataset):
    def __init__(self, list_path, data_dir, pick_files=None):
        self.data_dir = data_dir
        self.items = []
        with open(list_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                img, y = parse_line(line)
                base = os.path.basename(img)
                if pick_files is None or base in pick_files:
                    self.items.append((img, y))

        self.tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_name, y = self.items[idx]
        img_path = os.path.join(self.data_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        x = self.tfm(img)
        return x, y, os.path.basename(img_name)


# -----------------------
# Load model
# -----------------------
def build_model():
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 14)
    return model


# -----------------------
# Text wrapping + panel creation
# -----------------------
def wrap_text(draw, text, font, max_width):
    """
    Wrap text to fit within max_width (pixels).
    Returns list of lines.
    """
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = w if cur == "" else (cur + " " + w)
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def add_text_panel(img, lines, font, pad=10, line_spacing=4):
    """
    Create a white panel below the image and write wrapped lines there.
    Panel height grows to fit text.
    """
    w, h = img.size
    # measure line height
    tmp = Image.new("RGB", (w, 10), "white")
    dtmp = ImageDraw.Draw(tmp)
    bbox = dtmp.textbbox((0, 0), "Ag", font=font)
    line_h = (bbox[3] - bbox[1])

    panel_h = pad * 2 + len(lines) * line_h + (len(lines) - 1) * line_spacing
    out = Image.new("RGB", (w, h + panel_h), "white")
    out.paste(img, (0, 0))

    draw = ImageDraw.Draw(out)
    y = h + pad
    for line in lines:
        draw.text((pad, y), line, fill=(0, 0, 0), font=font)
        y += line_h + line_spacing
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = XrayDataset(TEST_LIST, DATA_DIR, pick_files=PICK_FILES)
    if len(dataset) == 0:
        raise RuntimeError(f"No matching files found in list for: {sorted(PICK_FILES)}")

    loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)),
                        shuffle=False, num_workers=NUM_WORKERS)

    model = build_model().to(device)
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Better font if available, else default
    try:
        # adjust if your system has a different TTF
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()

    with torch.no_grad():
        for x, y_true, names in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()

            for i in range(len(names)):
                fname = names[i]
                yt = np.array(y_true[i], dtype=np.float32)
                yp = probs[i]

                actual = [CLASSES_14[j] for j in range(14) if yt[j] > 0.5]
                actual_str = ", ".join(actual) if actual else "None"

                top_idx = np.argsort(-yp)[:TOPK]
                pred_str = ", ".join([f"{CLASSES_14[j]}:{yp[j]:.3f}" for j in top_idx])

                # Load image
                img_path = os.path.join(DATA_DIR, fname)
                img = Image.open(img_path).convert("RGB")

                # Create wrapped text lines (no cropping)
                tmp_draw = ImageDraw.Draw(img)
                max_text_width = img.width - 20  # padding*2

                text1 = f"File: {fname}"
                text2 = f"Actual: {actual_str}"
                text3 = f"Pred top{TOPK}: {pred_str}"

                lines = []
                lines += wrap_text(tmp_draw, text1, font, max_text_width)
                lines += wrap_text(tmp_draw, text2, font, max_text_width)
                lines += wrap_text(tmp_draw, text3, font, max_text_width)

                out_img = add_text_panel(img, lines, font, pad=10, line_spacing=4)

                out_path = os.path.join(OUT_DIR, fname)
                out_img.save(out_path)

    print("Saved visual predictions to:", OUT_DIR)
    print("Files:", ", ".join(sorted(PICK_FILES)))


if __name__ == "__main__":
    main()