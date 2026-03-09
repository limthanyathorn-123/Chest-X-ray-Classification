#!/usr/bin/env python3
"""
ChestXray14 multi-label classification (14 diseases) with:
- Uses official list files (img + 14 binary labels OR disease strings)
- Runs on A100 (AMP + pinned memory)
- Computes ROC-AUC per class + micro + macro WITHOUT sklearn/scipy
- Plots ROC curves and saves JSON metrics
- Plots TRAIN + TEST ROC curves (separate) + overlay (micro/macro)
- Saves 1 image per class:
    (a) raw example per class (actual positive)
    (b) pred_vs_actual per class (actual positive + predicted prob text)

IMPORTANT:
- If you preprocessed and resized images to 224, set data_dir to that folder
  and DO NOT use transforms.Resize in the pipeline (we don't).

Paths (edit in __main__):
  train_list, val_list, test_list
  data_dir (recommend: /scratch/.../images_224)
"""

import os
import re
import json
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights

import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True


# -----------------------------
# Config
# -----------------------------
CLASSES_15 = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia", "None"
]
SEED = 42


# -----------------------------
# NumPy-only ROC/AUC (no sklearn/scipy)
# -----------------------------
def roc_curve_np(y_true: np.ndarray, y_score: np.ndarray):
    """
    Binary ROC curve without sklearn/scipy.
    y_true: (N,) in {0,1}
    y_score: (N,) float scores
    Returns (fpr, tpr) or (None, None) if undefined.
    """
    y_true = y_true.astype(np.int32)
    order = np.argsort(-y_score)  # descending
    y_true = y_true[order]
    y_score = y_score[order]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    if P == 0 or N == 0:
        return None, None

    tp = 0
    fp = 0
    tpr = [0.0]
    fpr = [0.0]

    prev_score = None
    for yt, ys in zip(y_true, y_score):
        if prev_score is None:
            prev_score = ys
        elif ys != prev_score:
            tpr.append(tp / P)
            fpr.append(fp / N)
            prev_score = ys

        if yt == 1:
            tp += 1
        else:
            fp += 1

    tpr.append(tp / P)
    fpr.append(fp / N)
    return np.array(fpr, dtype=np.float64), np.array(tpr, dtype=np.float64)


def auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr = roc_curve_np(y_true, y_score)
    if fpr is None:
        return float("nan")
    return auc_trapz(fpr, tpr)


# -----------------------------
# Reproducibility
# -----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Faster on A100; deterministic=False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# -----------------------------
# List parsing
# -----------------------------
def _is_numeric_token(tok: str) -> bool:
    return re.fullmatch(r"[-+]?\d+(\.\d+)?", tok) is not None


def parse_list_line(line: str, class_to_idx: Dict[str, int]) -> Optional[Tuple[str, np.ndarray]]:
    """
    Supports:
      1) img.png 0 1 0 ... ( nums)
      2) img.png Atelectasis|Effusion
      3) img.png "Atelectasis|Effusion"
      4) img.png, Atelectasis|Effusion
    Returns (img_name_or_path, multi_hot[15]) or None.
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split()
    img = parts[0]

    # Case A: numeric labels
    nums = []
    for tok in parts[1:]:
        if _is_numeric_token(tok):
            nums.append(float(tok))
        else:
            nums = []
            break

    if len(nums) >= len(CLASSES_15):
        vec = np.array(nums[:len(CLASSES_15)], dtype=np.float32)
        vec = (vec > 0.5).astype(np.float32)
        return img, vec

    # Case B: string labels
    label_str = " ".join(parts[1:]).strip().strip('"').strip("'").strip()
    if not label_str:
        return img, np.zeros(len(CLASSES_15), dtype=np.float32)

    if label_str.startswith(","):
        label_str = label_str[1:].strip()

    raw_labels = re.split(r"[|,;]+", label_str)
    raw_labels = [x.strip() for x in raw_labels if x.strip()]

    vec = np.zeros(len(CLASSES_15), dtype=np.float32)
    for lab in raw_labels:
        if lab.lower() in ["no finding", "nofinding", "normal"]:
            continue
        lab2 = lab.replace(" ", "_")
        if lab2 in class_to_idx:
            vec[class_to_idx[lab2]] = 1.0
        elif lab in class_to_idx:
            vec[class_to_idx[lab]] = 1.0

    return img, vec


def load_list(list_path: str) -> List[Tuple[str, np.ndarray]]:
    class_to_idx = {c: i for i, c in enumerate(CLASSES_15)}
    items: List[Tuple[str, np.ndarray]] = []
    with open(list_path, "r") as f:
        for line in f:
            parsed = parse_list_line(line, class_to_idx)
            if parsed is not None:
                items.append(parsed)
    return items


# -----------------------------
# Dataset
# -----------------------------
class ChestXray14ListDataset(Dataset):
    def __init__(self, items: List[Tuple[str, np.ndarray]], data_dir: str, tfm):
        self.items = items
        self.data_dir = data_dir
        self.tfm = tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_name, y = self.items[idx]
        img_path = img_name if os.path.isabs(img_name) else os.path.join(self.data_dir, img_name)

        if not os.path.exists(img_path):
            img_path2 = os.path.join(self.data_dir, os.path.basename(img_name))
            if os.path.exists(img_path2):
                img_path = img_path2

        img = Image.open(img_path).convert("RGB")
        x = self.tfm(img)
        y = torch.from_numpy(y.astype(np.float32))
        return x, y, os.path.basename(img_path)


# -----------------------------
# Model
# -----------------------------
def build_model(num_classes: int):
    m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    in_features = m.classifier.in_features
    m.classifier = nn.Linear(in_features, num_classes)
    return m


# -----------------------------
# Train/Eval
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_targets = []
    all_names = []
    for x, y, name in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(y.cpu().numpy())
        all_names.extend(list(name))
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return all_targets, all_probs, all_names


def train_one_epoch(model, loader, optimizer, scaler, device, criterion):
    model.train()
    running = 0.0
    n = 0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        running += loss.item() * bs
        n += bs
    return running / max(1, n)


# -----------------------------
# AUC + ROC plotting
# -----------------------------
def compute_auc_roc(y_true: np.ndarray, y_prob: np.ndarray):
    curves = {}
    per_class_auc = {}

    for i, cls in enumerate(CLASSES_15):
        yt = y_true[:, i]
        yp = y_prob[:, i]

        if len(np.unique(yt)) < 2:
            per_class_auc[cls] = float("nan")
            continue

        fpr, tpr = roc_curve_np(yt, yp)
        if fpr is None:
            per_class_auc[cls] = float("nan")
            continue

        a = roc_auc_score_np(yt, yp)
        per_class_auc[cls] = float(a)
        curves[cls] = (fpr, tpr)

    # micro
    yt_micro = y_true.ravel()
    yp_micro = y_prob.ravel()
    micro_auc = float("nan")
    micro_curve = None
    if len(np.unique(yt_micro)) >= 2:
        fpr, tpr = roc_curve_np(yt_micro, yp_micro)
        if fpr is not None:
            micro_auc = float(roc_auc_score_np(yt_micro, yp_micro))
            micro_curve = (fpr, tpr)

    # macro AUC = mean per-class AUC (ignore nan)
    aucs = [v for v in per_class_auc.values() if not math.isnan(v)]
    macro_auc = float(np.mean(aucs)) if len(aucs) else float("nan")

    # macro ROC curve
    valid = [(cls, curves[cls]) for cls in CLASSES_15 if cls in curves]
    macro_curve = None
    if len(valid) > 0:
        all_fpr = np.unique(np.concatenate([c[0] for _, c in valid]))
        mean_tpr = np.zeros_like(all_fpr)
        for _, (fpr, tpr) in valid:
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= len(valid)
        macro_curve = (all_fpr, mean_tpr)

    curves["_micro"] = micro_curve
    curves["_macro"] = macro_curve
    return per_class_auc, micro_auc, macro_auc, curves


def plot_roc(curves, per_class_auc, micro_auc, macro_auc, out_path, title="ChestXray14 ROC Curves"):
    plt.figure(figsize=(10, 8))

    for cls in CLASSES_15:
        if cls in curves:
            fpr, tpr = curves[cls]
            plt.plot(fpr, tpr, label=f"{cls} (AUC={per_class_auc[cls]:.3f})")

    if curves.get("_micro") is not None and not math.isnan(micro_auc):
        fpr, tpr = curves["_micro"]
        plt.plot(fpr, tpr, linewidth=3, label=f"micro-average (AUC={micro_auc:.3f})")

    if curves.get("_macro") is not None and not math.isnan(macro_auc):
        fpr, tpr = curves["_macro"]
        plt.plot(fpr, tpr, linewidth=3, label=f"macro-average (AUC={macro_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_roc_train_vs_test(curves_train, curves_test, train_micro, train_macro, test_micro, test_macro, out_path):
    """
    Overlay only micro and macro ROC curves for Train vs Test to keep the plot readable.
    """
    plt.figure(figsize=(8, 7))

    # Train
    if curves_train.get("_micro") is not None and not math.isnan(train_micro):
        fpr, tpr = curves_train["_micro"]
        plt.plot(fpr, tpr, linewidth=3, label=f"Train micro (AUC={train_micro:.3f})")
    if curves_train.get("_macro") is not None and not math.isnan(train_macro):
        fpr, tpr = curves_train["_macro"]
        plt.plot(fpr, tpr, linewidth=3, label=f"Train macro (AUC={train_macro:.3f})")

    # Test
    if curves_test.get("_micro") is not None and not math.isnan(test_micro):
        fpr, tpr = curves_test["_micro"]
        plt.plot(fpr, tpr, linewidth=3, linestyle="--", label=f"Test micro (AUC={test_micro:.3f})")
    if curves_test.get("_macro") is not None and not math.isnan(test_macro):
        fpr, tpr = curves_test["_macro"]
        plt.plot(fpr, tpr, linewidth=3, linestyle="--", label=f"Test macro (AUC={test_macro:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ChestXray14 ROC: Train vs Test (micro/macro)")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Save images per class
# -----------------------------
def save_one_example_per_class(items: List[Tuple[str, np.ndarray]], data_dir: str, out_dir: str):
    """
    Saves 1 raw image per class where that class is actually positive.
    """
    os.makedirs(out_dir, exist_ok=True)
    picked = {c: None for c in CLASSES_15}

    for img_name, y in items:
        for i, cls in enumerate(CLASSES_15):
            if picked[cls] is None and y[i] > 0.5:
                picked[cls] = img_name
        if all(v is not None for v in picked.values()):
            break

    for cls, img_name in picked.items():
        if img_name is None:
            continue
        src = img_name if os.path.isabs(img_name) else os.path.join(data_dir, img_name)
        if not os.path.exists(src):
            src2 = os.path.join(data_dir, os.path.basename(img_name))
            if os.path.exists(src2):
                src = src2
        img = Image.open(src).convert("RGB")
        img.save(os.path.join(out_dir, f"{cls}.png"), optimize=True)


def save_pred_vs_actual_per_class(
    test_items: List[Tuple[str, np.ndarray]],
    data_dir: str,
    out_dir: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    names: List[str],
):
    """
    Saves 1 image per class where actual label=1, and titles it with:
      - Actual positives (all diseases that are 1)
      - Predicted prob for this class
      - Top-3 predicted diseases
    """
    os.makedirs(out_dir, exist_ok=True)

    name_to_idx = {names[i]: i for i in range(len(names))}
    saved = {c: False for c in CLASSES_15}

    for img_name, y in test_items:
        base = os.path.basename(img_name)
        if base not in name_to_idx:
            continue
        i = name_to_idx[base]
        yt = y_true[i]
        yp = y_prob[i]

        actual_pos = [CLASSES_15[k] for k in range(len(CLASSES_15)) if yt[k] > 0.5]
        topk = np.argsort(-yp)[:3]
        topk_str = ", ".join([f"{CLASSES_15[k]}:{yp[k]:.2f}" for k in topk])

        for ci, cls in enumerate(CLASSES_15):
            if saved[cls]:
                continue
            if yt[ci] < 0.5:
                continue

            src = img_name if os.path.isabs(img_name) else os.path.join(data_dir, img_name)
            if not os.path.exists(src):
                src2 = os.path.join(data_dir, base)
                if os.path.exists(src2):
                    src = src2
            img = Image.open(src).convert("RGB")

            title = f"Actual: {', '.join(actual_pos)} | Pred({cls})={yp[ci]:.3f} | Top3: {topk_str}"

            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.axis("off")
            plt.title(title, fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{cls}.png"), dpi=200)
            plt.close()

            saved[cls] = True

        if all(saved.values()):
            break

    missing = [c for c, v in saved.items() if not v]
    if missing:
        print("No positive example found in test set for:", missing)


# -----------------------------
# Main
# -----------------------------
@dataclass
class Args:
    train_list: str
    val_list: str
    test_list: str
    data_dir: str
    out_dir: str = "./runs_chestxray14"
    epochs: int = 8
    batch_size: int = 64
    lr: float = 3e-4
    num_workers: int = 8


def main(args: Args):
    seed_everything(SEED)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_items = load_list(args.train_list)
    val_items = load_list(args.val_list)
    test_items = load_list(args.test_list)

    # Save raw example per class (actual positive) from test set
    save_one_example_per_class(
        items=test_items,
        data_dir=args.data_dir,
        out_dir=os.path.join(args.out_dir, "one_image_per_class"),
    )

    # IMPORTANT: no Resize here (assumes preprocessed 224x224)
    train_tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ChestXray14ListDataset(train_items, args.data_dir, train_tfm)
    val_ds = ChestXray14ListDataset(val_items, args.data_dir, eval_tfm)
    test_ds = ChestXray14ListDataset(test_items, args.data_dir, eval_tfm)

    # Deterministic train evaluation loader (no augmentation)
    train_eval_ds = ChestXray14ListDataset(train_items, args.data_dir, eval_tfm)

    # Faster dataloading on A100
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=4
    )
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )

    model = build_model(len(CLASSES_15)).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_macro = -1.0
    best_path = os.path.join(args.out_dir, "best_round3.pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion)

        yv_true, yv_prob, _ = evaluate(model, val_loader, device)
        _, micro_auc, macro_auc, _ = compute_auc_roc(yv_true, yv_prob)

        print(f"[Epoch {epoch:02d}] train_loss={tr_loss:.4f}  val_microAUC={micro_auc:.4f}  val_macroAUC={macro_auc:.4f}")

        if not math.isnan(macro_auc) and macro_auc > best_val_macro:
            best_val_macro = macro_auc
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)

    # -----------------------------
    # Load best and evaluate TRAIN + TEST
    # -----------------------------
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)

    # Train (deterministic) + Test
    yt_tr, yp_tr, _ = evaluate(model, train_eval_loader, device)
    per_auc_tr, micro_auc_tr, macro_auc_tr, curves_tr = compute_auc_roc(yt_tr, yp_tr)

    yt_te, yp_te, names = evaluate(model, test_loader, device)
    per_auc_te, micro_auc_te, macro_auc_te, curves_te = compute_auc_roc(yt_te, yp_te)

    # Save metrics
    metrics_path = os.path.join(args.out_dir, "auc_train_test.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "train": {
                "per_class_auc": per_auc_tr,
                "micro_auc": micro_auc_tr,
                "macro_auc": macro_auc_tr
            },
            "test": {
                "per_class_auc": per_auc_te,
                "micro_auc": micro_auc_te,
                "macro_auc": macro_auc_te
            }
        }, f, indent=2)

    # Plot ROC curves (separate)
    roc_train_path = os.path.join(args.out_dir, "roc_curves_train.png")
    plot_roc(curves_tr, per_auc_tr, micro_auc_tr, macro_auc_tr, roc_train_path,
             title="ChestXray14 ROC Curves - TRAIN (per-class + micro + macro)")

    roc_test_path = os.path.join(args.out_dir, "roc_curves_test.png")
    plot_roc(curves_te, per_auc_te, micro_auc_te, macro_auc_te, roc_test_path,
             title="ChestXray14 ROC Curves - TEST (per-class + micro + macro)")

    # Overlay Train vs Test (micro/macro)
    roc_overlay_path = os.path.join(args.out_dir, "roc_train_vs_test_micro_macro.png")
    plot_roc_train_vs_test(curves_tr, curves_te, micro_auc_tr, macro_auc_tr, micro_auc_te, macro_auc_te, roc_overlay_path)

    # Save pred vs actual per class (use TEST)
    pred_dir = os.path.join(args.out_dir, "pred_vs_actual")
    save_pred_vs_actual_per_class(
        test_items=test_items,
        data_dir=args.data_dir,
        out_dir=pred_dir,
        y_true=yt_te,
        y_prob=yp_te,
        names=names
    )

    print("\n=== Train AUC ===")
    print(f"train_micro_auc: {micro_auc_tr}")
    print(f"train_macro_auc: {macro_auc_tr}")

    print("\n=== Test AUC ===")
    for c in CLASSES_15:
        print(f"{c:18s}: {per_auc_te[c]}")
    print(f"test_micro_auc: {micro_auc_te}")
    print(f"test_macro_auc: {macro_auc_te}")

    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {roc_train_path}")
    print(f"Saved: {roc_test_path}")
    print(f"Saved: {roc_overlay_path}")
    print(f"Saved examples: {os.path.join(args.out_dir, 'one_image_per_class')}")
    print(f"Saved pred-vs-actual: {pred_dir}")


if __name__ == "__main__":
    # Correct split mapping
    train_list = "/home/tlimsuva/cse591/presentation_classification/Xray14_train_official.txt"
    val_list   = "/home/tlimsuva/cse591/presentation_classification/Xray14_val_official.txt"
    test_list  = "/home/tlimsuva/cse591/presentation_classification/Xray14_test_official.txt"

    # Use resized images (224x224) stored in scratch
    data_dir = "/scratch/tlimsuva/chestxray14/images_224"

    args = Args(
        train_list=train_list,
        val_list=val_list,
        test_list=test_list,
        data_dir=data_dir,
        out_dir="./runs_chestxray14_a100",
        epochs=30,
        batch_size=32,
        lr=3e-4,
        num_workers=8,
    )
    print(args)

    main(args)
