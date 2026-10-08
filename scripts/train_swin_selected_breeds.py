"""Train and Evaluate High-Capacity Swin-Transformer Selected-Breed AI Model.

Backbone: Swin-T (28.3M parameters, Hierarchical Shifted-Window Vision Transformer)
Dataset:
- Training: 153 REAL + 800 SYNTHETIC = 953 active images
- Validation: 34 REAL images only
- Locked Test: 33 REAL images only (evaluated strictly at the end)

Features:
- Domain-weighted loss with 3.5x Real domain boost & inverse-frequency class balancing
- Context-preserving 15% padding crop (center 90% crop)
- Morphology-safe augmentations
- 2-phase training: Transformer Head warmup + Stage 3-4 fine-tuning
- Multi-scale / Flip Test-Time Augmentation (TTA)
"""

import os
import sys
import time
import json
import copy
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

SELECTED_CLASSES = [
    {"class_id": 0, "breed_id": "cow_dangi", "breed_name": "Dangi", "species": "cattle", "display_name": "Dangi Cattle"},
    {"class_id": 1, "breed_id": "cow_gaolao", "breed_name": "Gaolao", "species": "cattle", "display_name": "Gaolao Cattle"},
    {"class_id": 2, "breed_id": "cow_gir", "breed_name": "Gir", "species": "cattle", "display_name": "Gir Cattle"},
    {"class_id": 3, "breed_id": "cow_ongole", "breed_name": "Ongole", "species": "cattle", "display_name": "Ongole Cattle"},
    {"class_id": 4, "breed_id": "cow_sahiwal", "breed_name": "Sahiwal", "species": "cattle", "display_name": "Sahiwal Cattle"},
    {"class_id": 5, "breed_id": "buffalo_manda", "breed_name": "Manda", "species": "buffalo", "display_name": "Manda Buffalo"},
    {"class_id": 6, "breed_id": "buffalo_surti", "breed_name": "Surti", "species": "buffalo", "display_name": "Surti Buffalo"},
    {"class_id": 7, "breed_id": "buffalo_toda", "breed_name": "Toda", "species": "buffalo", "display_name": "Toda Buffalo"},
]

BREED_TO_IDX = {item["breed_name"].lower(): item["class_id"] for item in SELECTED_CLASSES}
BREED_ID_TO_IDX = {item["breed_id"].lower(): item["class_id"] for item in SELECTED_CLASSES}
IDX_TO_BREED = {item["class_id"]: item["breed_name"] for item in SELECTED_CLASSES}
IDX_TO_SPECIES = {item["class_id"]: item["species"] for item in SELECTED_CLASSES}

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
TARGET_SIZE = (224, 224)

class BovineDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, sample_weights=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.sample_weights = sample_weights
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        try:
            with Image.open(img_path) as im:
                im_rgb = im.convert("RGB")
                w, h = im_rgb.size
                crop_box = (int(0.05 * w), int(0.05 * h), int(0.95 * w), int(0.95 * h))
                cropped = im_rgb.crop(crop_box)
        except Exception:
            cropped = Image.new("RGB", TARGET_SIZE, (128, 128, 128))

        if self.transform:
            tensor_img = self.transform(cropped)
        else:
            base_t = transforms.Compose([
                transforms.Resize(TARGET_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
            ])
            tensor_img = base_t(cropped)

        b_name = row["breed"].lower() if "breed" in row else row["breed_name"].lower()
        class_id = BREED_TO_IDX[b_name] if b_name in BREED_TO_IDX else BREED_ID_TO_IDX[row["breed_id"].lower()]

        weight = self.sample_weights[idx] if self.sample_weights is not None else 1.0
        return tensor_img, class_id, weight, idx

def verify_and_log_integrity(train_df, val_df, test_df, locked_df):
    print(">>> Verifying Dataset Integrity & Splits...")
    assert len(train_df) == 953, f"Expected 953 train, got {len(train_df)}"
    assert (train_df["source_type"] == "REAL").sum() == 153, "Train real count mismatch"
    assert (train_df["source_type"] == "SYNTHETIC").sum() == 800, "Train synthetic count mismatch"
    assert len(val_df) == 34, f"Expected 34 val, got {len(val_df)}"
    assert (val_df["source_type"] == "REAL").sum() == 34, "Val must be 100% REAL"
    assert len(test_df) == 33, f"Expected 33 test, got {len(test_df)}"
    assert (test_df["source_type"] == "REAL").sum() == 33, "Test must be 100% REAL"
    assert len(locked_df) == 33, "Locked test count mismatch"

    train_shas = set(train_df["sha256"])
    val_shas = set(val_df["sha256"])
    test_shas = set(test_df["sha256"])
    locked_shas = set(locked_df["sha256"])
    syn_train_shas = set(train_df[train_df["source_type"] == "SYNTHETIC"]["sha256"])

    assert len(train_shas.intersection(test_shas)) == 0, "Train/Test leakage detected!"
    assert len(val_shas.intersection(test_shas)) == 0, "Val/Test leakage detected!"
    assert len(train_shas.intersection(val_shas)) == 0, "Train/Val leakage detected!"
    assert len(syn_train_shas.intersection(test_shas)) == 0, "Synthetic in Test detected!"
    assert len(syn_train_shas.intersection(val_shas)) == 0, "Synthetic in Val detected!"
    assert len(test_shas.symmetric_difference(locked_shas)) == 0, "Locked test set modified!"

    print("  [PASSED] All integrity constraints satisfied! 0% Leakage.")

def compute_sample_weights(train_df: pd.DataFrame, real_boost: float = 3.5) -> np.ndarray:
    weights = np.zeros(len(train_df), dtype=np.float32)
    real_counts = {}
    for b in SELECTED_CLASSES:
        b_name = b["breed_name"]
        cnt = len(train_df[(train_df["breed"] == b_name) & (train_df["source_type"] == "REAL")])
        real_counts[b_name] = max(1, cnt)

    mean_real = np.mean(list(real_counts.values()))

    for i, (_, row) in enumerate(train_df.iterrows()):
        b_name = row["breed"]
        if row["source_type"] == "REAL":
            class_factor = mean_real / real_counts[b_name]
            weights[i] = real_boost * class_factor
        else:
            weights[i] = 1.0

    weights = weights / weights.mean()
    return weights

def evaluate_loader(model, loader, device="cpu") -> Dict[str, Any]:
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    inference_times = []

    with torch.no_grad():
        for x, y, _, _ in loader:
            x = x.to(device)
            t0 = time.time()
            logits = model(x)
            inference_times.append((time.time() - t0) / len(x))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_targets.extend(y.numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    top1 = accuracy_score(all_targets, all_preds)
    top3 = 0.0
    for i in range(len(all_targets)):
        if all_targets[i] in np.argsort(all_probs[i])[-3:]:
            top3 += 1.0
    top3 /= len(all_targets)

    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)

    cattle_mask = np.array([IDX_TO_SPECIES[int(t)] == "cattle" for t in all_targets])
    buffalo_mask = np.array([IDX_TO_SPECIES[int(t)] == "buffalo" for t in all_targets])
    cattle_acc = accuracy_score(all_targets[cattle_mask], all_preds[cattle_mask]) if cattle_mask.sum() > 0 else 0.0
    buffalo_acc = accuracy_score(all_targets[buffalo_mask], all_preds[buffalo_mask]) if buffalo_mask.sum() > 0 else 0.0

    latency_ms = float(np.mean(inference_times) * 1000) if len(inference_times) > 0 else 0.0

    return {
        "top1_acc": float(top1),
        "top3_acc": float(top3),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "weighted_precision": float(w_prec),
        "weighted_recall": float(w_rec),
        "weighted_f1": float(w_f1),
        "cattle_acc": float(cattle_acc),
        "buffalo_acc": float(buffalo_acc),
        "correct_count": int((all_targets == all_preds).sum()),
        "total_count": len(all_targets),
        "latency_ms": latency_ms,
        "targets": all_targets,
        "predictions": all_preds,
        "probabilities": all_probs
    }

def evaluate_with_tta(model, df: pd.DataFrame, device="cpu") -> Dict[str, Any]:
    """5-view Test-Time Augmentation (Original, HFlip, slight zoom crops)."""
    model.eval()
    all_targets = []
    all_probs = []

    t_orig = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    t_flip = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    t_crop1 = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.CenterCrop(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    t_crop2 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    t_crop3 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(TARGET_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    tta_transforms = [t_orig, t_flip, t_crop1, t_crop2, t_crop3]

    with torch.no_grad():
        for _, row in df.iterrows():
            img_path = row["image_path"]
            try:
                with Image.open(img_path) as im:
                    im_rgb = im.convert("RGB")
                    w, h = im_rgb.size
                    crop_box = (int(0.05 * w), int(0.05 * h), int(0.95 * w), int(0.95 * h))
                    cropped = im_rgb.crop(crop_box)
            except Exception:
                cropped = Image.new("RGB", TARGET_SIZE, (128, 128, 128))

            b_name = row["breed"].lower() if "breed" in row else row["breed_name"].lower()
            class_id = BREED_TO_IDX[b_name] if b_name in BREED_TO_IDX else BREED_ID_TO_IDX[row["breed_id"].lower()]
            all_targets.append(class_id)

            sample_probs = np.zeros(8, dtype=np.float32)
            for t in tta_transforms:
                inp = t(cropped).unsqueeze(0).to(device)
                logits = model(inp)
                p = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                sample_probs += p

            sample_probs /= len(tta_transforms)
            all_probs.append(sample_probs)

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_preds = np.argmax(all_probs, axis=1)

    top1 = accuracy_score(all_targets, all_preds)
    top3 = 0.0
    for i in range(len(all_targets)):
        if all_targets[i] in np.argsort(all_probs[i])[-3:]:
            top3 += 1.0
    top3 /= len(all_targets)

    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)

    cattle_mask = np.array([IDX_TO_SPECIES[int(t)] == "cattle" for t in all_targets])
    buffalo_mask = np.array([IDX_TO_SPECIES[int(t)] == "buffalo" for t in all_targets])
    cattle_acc = accuracy_score(all_targets[cattle_mask], all_preds[cattle_mask]) if cattle_mask.sum() > 0 else 0.0
    buffalo_acc = accuracy_score(all_targets[buffalo_mask], all_preds[buffalo_mask]) if buffalo_mask.sum() > 0 else 0.0

    return {
        "top1_acc": float(top1),
        "top3_acc": float(top3),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "weighted_precision": float(w_prec),
        "weighted_recall": float(w_rec),
        "weighted_f1": float(w_f1),
        "cattle_acc": float(cattle_acc),
        "buffalo_acc": float(buffalo_acc),
        "correct_count": int((all_targets == all_preds).sum()),
        "total_count": len(all_targets),
        "targets": all_targets,
        "predictions": all_preds,
        "probabilities": all_probs
    }

def main():
    print("=" * 80)
    print("STARTING SWIN-TRANSFORMER TRAINING: SELECTED-BREED BACKBONE EXPERIMENT")
    print("CONFIGURATION: 153 REAL + 800 SYNTHETIC = 953 ACTIVE TRAINING IMAGES")
    print("=" * 80)

    train_df = pd.read_csv("splits/selected_breeds/train.csv")
    val_df = pd.read_csv("splits/selected_breeds/validation.csv")
    test_df = pd.read_csv("splits/selected_breeds/test.csv")
    locked_df = pd.read_csv("splits/selected_breeds/locked_test_manifest.csv")

    verify_and_log_integrity(train_df, val_df, test_df, locked_df)

    train_transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    sample_weights = compute_sample_weights(train_df, real_boost=3.5)
    train_ds = BovineDataset(train_df, transform=train_transform, sample_weights=sample_weights, is_train=True)
    val_ds = BovineDataset(val_df, transform=eval_transform, is_train=False)
    test_ds = BovineDataset(test_df, transform=eval_transform, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    print("\nLoading Swin-T with ImageNet-1K pretrained weights (28.3M parameters)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.Swin_T_Weights.DEFAULT
    model = models.swin_t(weights=weights)

    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 8)
    )
    model.to(device)

    # Warmup Phase (3 Epochs): Freeze features
    print("\n--- Phase 1: Swin Transformer Head Warmup (3 Epochs) ---")
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    optimizer_head = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none")

    for ep in range(3):
        t0 = time.time()
        model.train()
        ep_loss = 0.0
        for x, y, w, _ in train_loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            optimizer_head.zero_grad()
            out = model(x)
            loss_unreduced = criterion(out, y)
            loss = (loss_unreduced * w).mean()
            loss.backward()
            optimizer_head.step()
            ep_loss += loss.item() * len(y)

        val_m = evaluate_loader(model, val_loader, device=device)
        print(f"  Swin Warmup Epoch {ep+1}/3 - Loss: {ep_loss / len(train_ds):.4f} - Val Top-1: {val_m['top1_acc']*100:.2f}% ({val_m['correct_count']}/34) - Time: {time.time()-t0:.1f}s")

    # Phase 2: Fine-Tuning Swin Stage 3 and 4 (10 Epochs)
    print("\n--- Phase 2: Top Swin Stages Fine-Tuning (10 Epochs) ---")
    for i, layer in enumerate(model.features):
        if i >= 5:  # Stage 3 (384-dim) and Stage 4 (768-dim)
            for param in layer.parameters():
                param.requires_grad = True
        else:
            for param in layer.parameters():
                param.requires_grad = False

    for param in model.norm.parameters():
        param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW([
        {"params": [p for i, layer in enumerate(model.features) if i >= 5 for p in layer.parameters()], "lr": 1.0e-4},
        {"params": model.norm.parameters(), "lr": 2.0e-4},
        {"params": model.head.parameters(), "lr": 5.0e-4}
    ], weight_decay=1e-4)

    total_epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-5)

    best_val_score = -1.0
    best_weights = None
    best_epoch = 0
    training_history = []

    for ep in range(total_epochs):
        t0_ep = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x, y, w, _ in train_loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss_unreduced = criterion(out, y)
            loss = (loss_unreduced * w).mean()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            preds = out.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += len(y)

        scheduler.step()
        ep_train_loss = train_loss / train_total
        ep_train_acc = train_correct / train_total

        val_metrics = evaluate_loader(model, val_loader, device=device)
        val_top1 = val_metrics["top1_acc"]
        val_top3 = val_metrics["top3_acc"]
        val_f1 = val_metrics["macro_f1"]

        val_score = val_top1 + 0.1 * val_f1
        is_best = False
        if val_score > best_val_score:
            best_val_score = val_score
            best_val_top1 = val_top1
            best_epoch = ep + 1
            best_weights = copy.deepcopy(model.state_dict())
            is_best = True

        ep_duration = time.time() - t0_ep
        training_history.append({
            "epoch": ep + 1,
            "train_loss": ep_train_loss,
            "train_accuracy": ep_train_acc,
            "val_top1": val_top1,
            "val_top3": val_top3,
            "val_macro_f1": val_f1,
            "duration_s": ep_duration,
            "is_best": is_best
        })

        best_flag = " [BEST]" if is_best else ""
        print(f"Epoch {ep+1:02d}/{total_epochs:02d} | Train Loss: {ep_train_loss:.4f} | Train Acc: {ep_train_acc*100:.2f}% | Val Top-1: {val_top1*100:.2f}% ({val_metrics['correct_count']}/34) | Val Top-3: {val_top3*100:.2f}% | F1: {val_f1:.4f} | {ep_duration:.1f}s{best_flag}")

    print(f"\nLoading best Swin weights from Epoch {best_epoch} (Val Top-1: {best_val_top1*100:.2f}%)...")
    model.load_state_dict(best_weights)

    sys.stdout.reconfigure(line_buffering=True)
    out_dir = Path("models/selected_breeds/swin_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Path("reports/selected_breeds_swin")
    rep_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "best_swin_model.pth")
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)

    val_res = evaluate_loader(model, val_loader, device=device)
    val_tta = evaluate_with_tta(model, val_df, device=device)
    test_res = evaluate_loader(model, test_loader, device=device)
    test_tta = evaluate_with_tta(model, test_df, device=device)

    print("\n--- Swin-T Results ---")
    print(f"Validation: Top-1 = {val_res['top1_acc']*100:.2f}% ({val_res['correct_count']}/34), Top-3 = {val_res['top3_acc']*100:.2f}%")
    print(f"Val (TTA): Top-1 = {val_tta['top1_acc']*100:.2f}% ({val_tta['correct_count']}/34), Top-3 = {val_tta['top3_acc']*100:.2f}%")
    print(f"Locked Test: Top-1 = {test_res['top1_acc']*100:.2f}% ({test_res['correct_count']}/33), Top-3 = {test_res['top3_acc']*100:.2f}%")
    print(f"Locked Test (TTA): Top-1 = {test_tta['top1_acc']*100:.2f}% ({test_tta['correct_count']}/33), Top-3 = {test_tta['top3_acc']*100:.2f}%")

    report_md = f"""# Swin-Transformer (Swin-T) Backbone Evaluation Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Project**: AI-Breed-Recognition-Selected-Breeds  
**Backbone Tested**: Swin-T (28.3M parameters, Hierarchical Shifted-Window Vision Transformer)  
**Dataset**: 953 Active Training Images (153 Real + 800 Controlled Synthetic)  
**Validation**: 34 Real Photographs Only (0% synthetic)  
**Locked Test**: 33 Real Photographs Only (0% synthetic, immutable benchmark)  

---

## Performance Summary

| Split / Method | Top-1 Accuracy | Top-3 Accuracy | Macro F1 | Correct / Total |
| :--- | :---: | :---: | :---: | :---: |
| **Validation Standard** | **{val_res['top1_acc']*100:.2f}%** | {val_res['top3_acc']*100:.2f}% | {val_res['macro_f1']:.4f} | {val_res['correct_count']}/34 |
| **Validation (5-View TTA)** | **{val_tta['top1_acc']*100:.2f}%** | {val_tta['top3_acc']*100:.2f}% | {val_tta['macro_f1']:.4f} | {val_tta['correct_count']}/34 |
| **Locked Test Standard** | **{test_res['top1_acc']*100:.2f}%** | {test_res['top3_acc']*100:.2f}% | {test_res['macro_f1']:.4f} | {test_res['correct_count']}/33 |
| **Locked Test (5-View TTA)** | **{test_tta['top1_acc']*100:.2f}%** | {test_tta['top3_acc']*100:.2f}% | {test_tta['macro_f1']:.4f} | {test_tta['correct_count']}/33 |
"""
    with open(rep_dir / "swin_evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved Swin evaluation report to: {rep_dir / 'swin_evaluation_report.md'}")

if __name__ == "__main__":
    main()
