"""Hierarchical Two-Stage Selected-Breed AI Classifier with Test-Time Augmentation (TTA).

Architecture:
- Shared EfficientNet-B0 Backbone (300x300, ImageNet Pretrained)
- Stage 1: Species Head (Binary: Cattle vs Buffalo)
- Stage 2A: Cattle Expert Head (5 Classes: Dangi, Gaolao, Gir, Ongole, Sahiwal)
- Stage 2B: Buffalo Expert Head (3 Classes: Manda, Surti, Toda)
- Multi-Task Loss with Domain & Class Balancing (Real boost 3.5x)
- Joint Probability Formulation: P(Breed) = P(Species) * P(Breed | Species)
- Test-Time Augmentation (TTA) with 5 Canonical Views

Dataset:
- Training: 153 Real + 800 Synthetic = 953 images
- Validation: 34 Real images (for checkpoint selection)
- Locked Test: 33 Real images (frozen, evaluated once)
"""

import os
import sys
import time
import json
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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
IDX_TO_BREED = {item["class_id"]: item["breed_name"] for item in SELECTED_CLASSES}
IDX_TO_SPECIES = {item["class_id"]: item["species"] for item in SELECTED_CLASSES}
IDX_TO_DISPLAY = {item["class_id"]: item["display_name"] for item in SELECTED_CLASSES}

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
TARGET_SIZE = (300, 300)

class BovineHierarchicalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, sample_weights=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.sample_weights = sample_weights

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        try:
            with Image.open(img_path) as im:
                im_rgb = im.convert("RGB")
                w, h = im_rgb.size
                # 15% context-preserving crop (90% center)
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
        class_id = BREED_TO_IDX[b_name]

        # Species target: 0 = cattle, 1 = buffalo
        is_buffalo = 1 if class_id >= 5 else 0
        # Sub-class target: 0..4 for cattle, 0..2 for buffalo
        sub_class_id = (class_id - 5) if is_buffalo else class_id

        weight = self.sample_weights[idx] if self.sample_weights is not None else 1.0
        return tensor_img, class_id, is_buffalo, sub_class_id, weight, idx

class HierarchicalBovineModel(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        base = models.efficientnet_b0(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool
        in_f = base.classifier[1].in_features  # 1280

        # Stage 1: Species Classifier (Cattle vs Buffalo)
        self.species_head = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_f, 2)
        )

        # Stage 2A: Cattle Expert (5 classes: Dangi, Gaolao, Gir, Ongole, Sahiwal)
        self.cattle_head = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_f, 5)
        )

        # Stage 2B: Buffalo Expert (3 classes: Manda, Surti, Toda)
        self.buffalo_head = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_f, 3)
        )

        # Flat unified projection head (for standard backward-compatible 8-class logits)
        self.unified_head = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_f, 8)
        )

    def extract_features(self, x):
        f = self.features(x)
        f = self.avgpool(f)
        return torch.flatten(f, 1)

    def forward(self, x):
        feats = self.extract_features(x)
        species_logits = self.species_head(feats)
        cattle_logits = self.cattle_head(feats)
        buffalo_logits = self.buffalo_head(feats)
        unified_logits = self.unified_head(feats)

        # Compute joint hierarchical probabilities:
        # P(Breed_i) = P(Species) * P(Breed_i | Species)
        sp_probs = F.softmax(species_logits, dim=-1)
        cat_probs = F.softmax(cattle_logits, dim=-1)
        buf_probs = F.softmax(buffalo_logits, dim=-1)

        hier_probs = torch.cat([
            cat_probs * sp_probs[:, 0:1],
            buf_probs * sp_probs[:, 1:2]
        ], dim=-1)

        # Unified ensemble: blended with flat logits for optimal calibration
        flat_probs = F.softmax(unified_logits, dim=-1)
        blended_probs = 0.75 * hier_probs + 0.25 * flat_probs

        return {
            "species_logits": species_logits,
            "cattle_logits": cattle_logits,
            "buffalo_logits": buffalo_logits,
            "unified_logits": unified_logits,
            "hier_probs": hier_probs,
            "blended_probs": blended_probs
        }

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

    return weights / weights.mean()

def evaluate_with_tta(model, df: pd.DataFrame, device="cpu") -> Dict[str, Any]:
    """Evaluate dataset using 5-view Test-Time Augmentation (TTA)."""
    model.eval()
    t_base = transforms.Compose([
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

    all_preds, all_probs, all_targets = [], [], []
    inference_times = []

    with torch.no_grad():
        for _, row in df.iterrows():
            img_path = row["image_path"]
            with Image.open(img_path) as im:
                im_rgb = im.convert("RGB")
                w, h = im_rgb.size
                # 5 Views:
                # View 1: 15% context crop
                v1 = im_rgb.crop((int(0.05 * w), int(0.05 * h), int(0.95 * w), int(0.95 * h)))
                # View 2: Horizontal flip of View 1
                v2 = v1.transpose(Image.FLIP_LEFT_RIGHT)
                # View 3: Full image resized
                v3 = im_rgb
                # View 4: Tighter crop (10% border)
                v4 = im_rgb.crop((int(0.10 * w), int(0.10 * h), int(0.90 * w), int(0.90 * h)))
                # View 5: Slight scale crop (2.5% border)
                v5 = im_rgb.crop((int(0.025 * w), int(0.025 * h), int(0.975 * w), int(0.975 * h)))

            views_tensors = torch.stack([
                t_base(v1), t_base(v2), t_base(v3), t_base(v4), t_base(v5)
            ]).to(device)

            t0 = time.time()
            out = model(views_tensors)
            inference_times.append((time.time() - t0) / 5.0)

            # Average blended probabilities across the 5 TTA views
            probs_tta = out["blended_probs"].mean(dim=0).cpu().numpy()
            pred_tta = np.argmax(probs_tta)

            b_name = row["breed"].lower() if "breed" in row else row["breed_name"].lower()
            target_cls = BREED_TO_IDX[b_name]

            all_probs.append(probs_tta)
            all_preds.append(pred_tta)
            all_targets.append(target_cls)

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

    cat_mask = np.array([IDX_TO_SPECIES[int(t)] == "cattle" for t in all_targets])
    buf_mask = np.array([IDX_TO_SPECIES[int(t)] == "buffalo" for t in all_targets])
    cat_acc = accuracy_score(all_targets[cat_mask], all_preds[cat_mask]) if cat_mask.sum() > 0 else 0.0
    buf_acc = accuracy_score(all_targets[buf_mask], all_preds[buf_mask]) if buf_mask.sum() > 0 else 0.0

    return {
        "top1_acc": float(top1),
        "top3_acc": float(top3),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "weighted_precision": float(w_prec),
        "weighted_recall": float(w_rec),
        "weighted_f1": float(w_f1),
        "cattle_acc": float(cat_acc),
        "buffalo_acc": float(buf_acc),
        "correct_count": int((all_targets == all_preds).sum()),
        "total_count": len(all_targets),
        "latency_ms": float(np.mean(inference_times) * 1000),
        "targets": all_targets,
        "predictions": all_preds,
        "probabilities": all_probs
    }

def main():
    print("=" * 80)
    print("STARTING HIERARCHICAL TWO-STAGE SELECTED-BREED TRAINING")
    print("PIPELINE: SPECIES GATING + EXPERT CLASSIFIERS + TEST-TIME AUGMENTATION")
    print("=" * 80)

    train_df = pd.read_csv("splits/selected_breeds/train.csv")
    val_df = pd.read_csv("splits/selected_breeds/validation.csv")
    test_df = pd.read_csv("splits/selected_breeds/test.csv")

    train_transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    sample_weights = compute_sample_weights(train_df, real_boost=3.5)
    train_ds = BovineHierarchicalDataset(train_df, transform=train_transform, sample_weights=sample_weights)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalBovineModel().to(device)

    # 1. Warmup: freeze features, train heads (2 epochs)
    print("\n--- Phase 1: Head Warmup (2 Epochs) ---")
    for name, param in model.named_parameters():
        if "features" in name:
            param.requires_grad = False

    opt_heads = torch.optim.AdamW([
        {"params": model.species_head.parameters(), "lr": 1e-3},
        {"params": model.cattle_head.parameters(), "lr": 1e-3},
        {"params": model.buffalo_head.parameters(), "lr": 1e-3},
        {"params": model.unified_head.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)

    crit_smooth = nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none")

    for ep in range(2):
        model.train()
        ep_loss = 0.0
        for x, y, is_buf, sub_y, w, _ in train_loader:
            x, y, is_buf, sub_y, w = x.to(device), y.to(device), is_buf.to(device), sub_y.to(device), w.to(device)
            opt_heads.zero_grad()
            out = model(x)

            # Multi-task loss:
            l_sp = (crit_smooth(out["species_logits"], is_buf) * w).mean()
            l_un = (crit_smooth(out["unified_logits"], y) * w).mean()

            # Cattle expert on cattle samples
            cat_mask = (is_buf == 0)
            if cat_mask.sum() > 0:
                l_cat = (crit_smooth(out["cattle_logits"][cat_mask], sub_y[cat_mask]) * w[cat_mask]).mean()
            else:
                l_cat = torch.tensor(0.0, device=device)

            # Buffalo expert on buffalo samples
            buf_mask = (is_buf == 1)
            if buf_mask.sum() > 0:
                l_buf = (crit_smooth(out["buffalo_logits"][buf_mask], sub_y[buf_mask]) * w[buf_mask]).mean()
            else:
                l_buf = torch.tensor(0.0, device=device)

            loss = l_sp + 0.5 * l_un + l_cat + l_buf
            loss.backward()
            opt_heads.step()
            ep_loss += loss.item() * len(y)
        print(f"  Warmup Epoch {ep+1}/2 - Multi-Task Loss: {ep_loss / len(train_ds):.4f}")

    # 2. Fine-tuning full network with discriminative LR
    print("\n--- Phase 2: Full Network Hierarchical Fine-Tuning (14 Epochs) ---")
    for param in model.parameters():
        param.requires_grad = True

    opt = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": 1.5e-4},
        {"params": model.species_head.parameters(), "lr": 6.0e-4},
        {"params": model.cattle_head.parameters(), "lr": 6.0e-4},
        {"params": model.buffalo_head.parameters(), "lr": 6.0e-4},
        {"params": model.unified_head.parameters(), "lr": 6.0e-4},
    ], weight_decay=1e-4)

    total_epochs = 14
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs, eta_min=1e-5)

    best_val_top1 = 0.0
    best_weights = None
    best_ep = 0

    for ep in range(total_epochs):
        t0_ep = time.time()
        model.train()
        train_loss = 0.0

        for x, y, is_buf, sub_y, w, _ in train_loader:
            x, y, is_buf, sub_y, w = x.to(device), y.to(device), is_buf.to(device), sub_y.to(device), w.to(device)
            opt.zero_grad()
            out = model(x)

            l_sp = (crit_smooth(out["species_logits"], is_buf) * w).mean()
            l_un = (crit_smooth(out["unified_logits"], y) * w).mean()

            cat_mask = (is_buf == 0)
            l_cat = (crit_smooth(out["cattle_logits"][cat_mask], sub_y[cat_mask]) * w[cat_mask]).mean() if cat_mask.sum() > 0 else torch.tensor(0.0, device=device)

            buf_mask = (is_buf == 1)
            l_buf = (crit_smooth(out["buffalo_logits"][buf_mask], sub_y[buf_mask]) * w[buf_mask]).mean() if buf_mask.sum() > 0 else torch.tensor(0.0, device=device)

            loss = l_sp + 0.5 * l_un + l_cat + l_buf
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(y)

        scheduler.step()
        ep_duration = time.time() - t0_ep

        # Evaluate on Validation set (34 Real Images) with TTA
        val_res = evaluate_with_tta(model, val_df, device=device)
        v_top1 = val_res["top1_acc"]
        v_top3 = val_res["top3_acc"]
        v_f1 = val_res["macro_f1"]

        if v_top1 > best_val_top1 or (v_top1 == best_val_top1 and v_f1 > best_val_f1):
            best_val_top1 = v_top1
            best_val_f1 = v_f1
            best_ep = ep + 1
            best_weights = copy.deepcopy(model.state_dict())

        print(f"  Epoch {ep+1:2d}/{total_epochs} [{ep_duration:4.1f}s] - Loss: {train_loss/len(train_ds):.4f} | Val TTA Top-1: {v_top1*100:5.2f}%, Top-3: {v_top3*100:5.2f}%, F1: {v_f1:.4f}")

    print(f"\nBest Hierarchical Checkpoint Selected at Epoch {best_ep} (Val Top-1: {best_val_top1*100:.2f}%)")
    model.load_state_dict(best_weights)

    # 3. Final Evaluation on Validation & Locked Test Set
    val_final = evaluate_with_tta(model, val_df, device=device)
    print("\n" + "=" * 60)
    print("FINAL HIERARCHICAL VALIDATION (34 REAL IMAGES + TTA)")
    print("=" * 60)
    print(f"  Top-1 Accuracy:  {val_final['top1_acc']*100:.2f}% ({val_final['correct_count']}/34)")
    print(f"  Top-3 Accuracy:  {val_final['top3_acc']*100:.2f}%")
    print(f"  Macro F1-Score:  {val_final['macro_f1']:.4f}")

    print("\n" + "=" * 60)
    print("EVALUATING ON LOCKED REAL TEST SET (33 IMAGES + TTA) - ONE PASS")
    print("=" * 60)
    test_final = evaluate_with_tta(model, test_df, device=device)
    print(f"  Locked Test Top-1 Accuracy:  {test_final['top1_acc']*100:.2f}% ({test_final['correct_count']}/33)")
    print(f"  Locked Test Top-3 Accuracy:  {test_final['top3_acc']*100:.2f}%")
    print(f"  Locked Test Macro Precision: {test_final['macro_precision']:.4f}")
    print(f"  Locked Test Macro Recall:    {test_final['macro_recall']:.4f}")
    print(f"  Locked Test Macro F1-Score:  {test_final['macro_f1']:.4f}")
    print(f"  Locked Test Cattle Accuracy: {test_final['cattle_acc']*100:.2f}%")
    print(f"  Locked Test Buffalo Accuracy:{test_final['buffalo_acc']*100:.2f}%")
    print(f"  Inference Latency (TTA 5x):  {test_final['latency_ms']:.2f} ms/image")

    # Save per-class test report
    cm_test = confusion_matrix(test_final["targets"], test_final["predictions"], labels=list(range(8)))
    df_cm = pd.DataFrame(cm_test, index=[IDX_TO_BREED[i] for i in range(8)], columns=[IDX_TO_BREED[i] for i in range(8)])
    print("\nLocked Test Confusion Matrix:")
    print(df_cm.to_string())

    # Save updated artifacts to models and reports
    torch.save(model.state_dict(), "models/selected_breeds/selected_breeds_v2/hierarchical_best_model.pth")
    print("Saved hierarchical_best_model.pth successfully!")

if __name__ == "__main__":
    main()
