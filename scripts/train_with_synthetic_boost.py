"""Train and Evaluate Selected 8-Breed Model Incorporating the 800 New Synthetic Images.

Explores:
1. Pure Real Baseline (N=153)
2. Flat Real + 800 Synthetic (N=953)
3. Domain-Weighted Loss (Real weight=4.0, Synthetic weight=1.0)
4. Two-Stage Curriculum Learning:
   - Stage 1: Pretrain classifier on 800 synthetic crops
   - Stage 2: Fine-tune on 153 authentic real photographs

Evaluates all strategies on the Real Validation Set (34 images) and evaluates the champion on the locked Real Test Set (33 images).
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

# Class definitions
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

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

t_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

t_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# Feature extraction using EfficientNet-B0
backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
backbone.eval()

def extract_features(df, transform=t_eval):
    feats = []
    for _, row in df.iterrows():
        with Image.open(row["image_path"]) as im:
            im_rgb = im.convert("RGB")
            tensor = transform(im_rgb).unsqueeze(0)
            with torch.no_grad():
                f = backbone.avgpool(backbone.features(tensor))
                f = torch.flatten(f, 1)
                feats.append(f)
    return torch.cat(feats)

def compute_metrics(y_true, y_pred, y_probs=None, latency_ms=12.5):
    top1 = accuracy_score(y_true, y_pred)
    top3 = 0.0
    if y_probs is not None:
        for i in range(len(y_true)):
            if y_true[i] in np.argsort(y_probs[i])[-3:]:
                top3 += 1.0
        top3 /= len(y_true)
    else:
        top3 = top1
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cat_mask = np.array([IDX_TO_SPECIES[int(t)] == "cattle" for t in y_true])
    buf_mask = np.array([IDX_TO_SPECIES[int(t)] == "buffalo" for t in y_true])
    cat_acc = accuracy_score(y_true[cat_mask], y_pred[cat_mask]) if cat_mask.sum() > 0 else 0.0
    buf_acc = accuracy_score(y_true[buf_mask], y_pred[buf_mask]) if buf_mask.sum() > 0 else 0.0
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
        "correct_count": int((y_true == y_pred).sum()),
        "total_count": len(y_true),
        "latency_ms": latency_ms
    }

def main():
    print("=" * 70)
    print("TRAINING WITH 800 NEW SYNTHETIC IMAGES (100 PER BREED)")
    print("=" * 70)

    manifest_path = "dataset/selected_breeds/selected_breed_dataset_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    
    train_real = manifest[(manifest["split"] == "train") & (manifest["source_type"] == "REAL")].reset_index(drop=True)
    val_real = manifest[(manifest["split"] == "validation") & (manifest["source_type"] == "REAL")].reset_index(drop=True)
    test_real = manifest[(manifest["split"] == "test") & (manifest["source_type"] == "REAL")].reset_index(drop=True)
    synth_new = manifest[(manifest["split"] == "train") & (manifest["source_type"] == "SYNTHETIC") & (manifest["image_path"].str.contains("_synthetic_"))].reset_index(drop=True)

    print(f"Data Partitions: Train Real={len(train_real)}, Val Real={len(val_real)}, Test Real={len(test_real)}, New Synthetic={len(synth_new)}")

    val_y = np.array([BREED_TO_IDX[b.lower()] for b in val_real["breed"]])
    test_y = np.array([BREED_TO_IDX[b.lower()] for b in test_real["breed"]])
    train_real_y = np.array([BREED_TO_IDX[b.lower()] for b in train_real["breed"]])
    synth_y = np.array([BREED_TO_IDX[b.lower()] for b in synth_new["breed"]])

    print("Extracting validation features...")
    val_X = extract_features(val_real)
    print("Extracting test features...")
    test_X = extract_features(test_real)
    print("Extracting train real features...")
    train_real_X = extract_features(train_real)
    print("Extracting synthetic features (800 crops)...")
    synth_X = extract_features(synth_new)

    results = []

    # -------------------------------------------------------------
    # Method 1: S0 Pure Real Baseline
    # -------------------------------------------------------------
    print("\n--- Training Method 1: Pure Real Baseline (S0, N=153) ---")
    seed_everything(42)
    head_s0 = nn.Sequential(nn.Dropout(0.25), nn.Linear(1280, 8))
    opt_s0 = torch.optim.AdamW(head_s0.parameters(), lr=0.008, weight_decay=1e-3)
    sched_s0 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s0, T_max=80)
    crit_s0 = nn.CrossEntropyLoss(label_smoothing=0.05)
    y_tr_t = torch.tensor(train_real_y, dtype=torch.long)
    for ep in range(80):
        head_s0.train()
        opt_s0.zero_grad()
        loss = crit_s0(head_s0(train_real_X), y_tr_t)
        loss.backward()
        opt_s0.step()
        sched_s0.step()
    head_s0.eval()
    with torch.no_grad():
        v_logits = head_s0(val_X)
        v_preds = v_logits.argmax(dim=1).numpy()
        v_probs = F.softmax(v_logits, dim=1).numpy()
    m_s0 = compute_metrics(val_y, v_preds, v_probs)
    print(f"  Method 1 Val: Top-1={m_s0['top1_acc']*100:.2f}%, Top-3={m_s0['top3_acc']*100:.2f}%, Macro F1={m_s0['macro_f1']:.4f}")
    results.append({"method": "Method 1: S0 Pure Real", "model": head_s0, **m_s0})

    # -------------------------------------------------------------
    # Method 2: S3 Flat Real + 800 Synthetic Equal Weight (N=953)
    # -------------------------------------------------------------
    print("\n--- Training Method 2: Flat Real + 800 Synthetic Equal Weight (N=953) ---")
    seed_everything(42)
    X_all = torch.cat([train_real_X, synth_X])
    y_all = np.concatenate([train_real_y, synth_y])
    y_all_t = torch.tensor(y_all, dtype=torch.long)
    head_s3 = nn.Sequential(nn.Dropout(0.25), nn.Linear(1280, 8))
    opt_s3 = torch.optim.AdamW(head_s3.parameters(), lr=0.008, weight_decay=1e-3)
    sched_s3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s3, T_max=80)
    crit_s3 = nn.CrossEntropyLoss(label_smoothing=0.05)
    for ep in range(80):
        head_s3.train()
        opt_s3.zero_grad()
        loss = crit_s3(head_s3(X_all), y_all_t)
        loss.backward()
        opt_s3.step()
        sched_s3.step()
    head_s3.eval()
    with torch.no_grad():
        v_logits = head_s3(val_X)
        v_preds = v_logits.argmax(dim=1).numpy()
        v_probs = F.softmax(v_logits, dim=1).numpy()
    m_s3 = compute_metrics(val_y, v_preds, v_probs)
    print(f"  Method 2 Val: Top-1={m_s3['top1_acc']*100:.2f}%, Top-3={m_s3['top3_acc']*100:.2f}%, Macro F1={m_s3['macro_f1']:.4f}")
    results.append({"method": "Method 2: Flat Equal Weight", "model": head_s3, **m_s3})

    # -------------------------------------------------------------
    # Method 3: Domain-Weighted Loss (Real=4.0x, Synthetic=1.0x)
    # -------------------------------------------------------------
    print("\n--- Training Method 3: Domain-Weighted Loss (Real=4.0x, Synthetic=1.0x) ---")
    seed_everything(42)
    sample_weights = torch.tensor([4.0] * len(train_real) + [1.0] * len(synth_new), dtype=torch.float32)
    sample_weights = sample_weights / sample_weights.mean()  # normalize to mean 1.0
    head_dw = nn.Sequential(nn.Dropout(0.25), nn.Linear(1280, 8))
    opt_dw = torch.optim.AdamW(head_dw.parameters(), lr=0.008, weight_decay=1e-3)
    sched_dw = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dw, T_max=80)
    crit_dw = nn.CrossEntropyLoss(label_smoothing=0.05, reduction='none')
    for ep in range(80):
        head_dw.train()
        opt_dw.zero_grad()
        loss_unreduced = crit_dw(head_dw(X_all), y_all_t)
        weighted_loss = (loss_unreduced * sample_weights).mean()
        weighted_loss.backward()
        opt_dw.step()
        sched_dw.step()
    head_dw.eval()
    with torch.no_grad():
        v_logits = head_dw(val_X)
        v_preds = v_logits.argmax(dim=1).numpy()
        v_probs = F.softmax(v_logits, dim=1).numpy()
    m_dw = compute_metrics(val_y, v_preds, v_probs)
    print(f"  Method 3 Val: Top-1={m_dw['top1_acc']*100:.2f}%, Top-3={m_dw['top3_acc']*100:.2f}%, Macro F1={m_dw['macro_f1']:.4f}")
    results.append({"method": "Method 3: Domain-Weighted Loss", "model": head_dw, **m_dw})

    # -------------------------------------------------------------
    # Method 4: Two-Stage Curriculum Learning
    # Stage 1: Pretrain on 800 synthetic crops (general bovine feature head)
    # Stage 2: Fine-tune on 153 real photographs
    # -------------------------------------------------------------
    print("\n--- Training Method 4: Two-Stage Curriculum Learning ---")
    seed_everything(42)
    head_curr = nn.Sequential(nn.Dropout(0.25), nn.Linear(1280, 8))
    # Stage 1: Pretrain on 800 synthetic crops
    opt_curr1 = torch.optim.AdamW(head_curr.parameters(), lr=0.01, weight_decay=1e-3)
    sched_curr1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_curr1, T_max=40)
    crit_curr = nn.CrossEntropyLoss(label_smoothing=0.05)
    synth_y_t = torch.tensor(synth_y, dtype=torch.long)
    for ep in range(40):
        head_curr.train()
        opt_curr1.zero_grad()
        loss = crit_curr(head_curr(synth_X), synth_y_t)
        loss.backward()
        opt_curr1.step()
        sched_curr1.step()

    # Stage 2: Fine-tune on authentic real photographs
    opt_curr2 = torch.optim.AdamW(head_curr.parameters(), lr=0.003, weight_decay=1e-3)
    sched_curr2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_curr2, T_max=60)
    for ep in range(60):
        head_curr.train()
        opt_curr2.zero_grad()
        loss = crit_curr(head_curr(train_real_X), y_tr_t)
        loss.backward()
        opt_curr2.step()
        sched_curr2.step()
        
    head_curr.eval()
    with torch.no_grad():
        v_logits = head_curr(val_X)
        v_preds = v_logits.argmax(dim=1).numpy()
        v_probs = F.softmax(v_logits, dim=1).numpy()
    m_curr = compute_metrics(val_y, v_preds, v_probs)
    print(f"  Method 4 Val: Top-1={m_curr['top1_acc']*100:.2f}%, Top-3={m_curr['top3_acc']*100:.2f}%, Macro F1={m_curr['macro_f1']:.4f}")
    results.append({"method": "Method 4: Two-Stage Curriculum", "model": head_curr, **m_curr})

    # -------------------------------------------------------------
    # Select Best Model Based on REAL Validation Performance
    # -------------------------------------------------------------
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "model"} for r in results])
    print("\n=== Validation Results Summary ===")
    print(results_df[["method", "top1_acc", "top3_acc", "macro_f1", "cattle_acc", "buffalo_acc"]])

    # Pick the model with highest Top-1 and Top-3 on validation
    best_idx = results_df.sort_values(by=["top1_acc", "top3_acc", "macro_f1"], ascending=False).index[0]
    best_res = results[best_idx]
    champion_head = best_res["model"]
    print(f"\nChampion Model Selected: {best_res['method']} (Val Top-1: {best_res['top1_acc']*100:.2f}%, Top-3: {best_res['top3_acc']*100:.2f}%)")

    # -------------------------------------------------------------
    # Serialize Champion Full Model
    # -------------------------------------------------------------
    full_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = full_model.classifier[1].in_features
    full_model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 8)
    )
    full_model.classifier[1].weight.data.copy_(champion_head[1].weight.data)
    full_model.classifier[1].bias.data.copy_(champion_head[1].bias.data)

    model_path = "models/selected_breeds/best_selected_breed_model.pth"
    sd = full_model.state_dict()
    if not any(k.startswith("backbone.") for k in sd.keys()):
        sd = {f"backbone.{k}": v for k, v in sd.items()}
    torch.save(sd, model_path)
    print(f"Saved champion model to {model_path}!")

    # -------------------------------------------------------------
    # Evaluate Champion on Locked Real Test Partition (33 images)
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("EVALUATING CHAMPION ON LOCKED REAL TEST SET (33 IMAGES)")
    print("="*50)
    full_model.eval()
    with torch.no_grad():
        test_logits = full_model.classifier(test_X) if hasattr(test_X, 'shape') else full_model(test_X)
        test_probs = F.softmax(test_logits, dim=1).numpy()
        test_preds = test_logits.argmax(dim=1).numpy()

    test_metrics = compute_metrics(test_y, test_preds, test_probs)
    print(f"  Test Top-1 Accuracy:  {test_metrics['top1_acc']*100:.2f}% ({test_metrics['correct_count']}/{test_metrics['total_count']})")
    print(f"  Test Top-3 Accuracy:  {test_metrics['top3_acc']*100:.2f}%")
    print(f"  Test Macro F1:        {test_metrics['macro_f1']:.4f}")
    print(f"  Test Cattle Accuracy: {test_metrics['cattle_acc']*100:.2f}%")
    print(f"  Test Buffalo Accuracy:{test_metrics['buffalo_acc']*100:.2f}%")

    # Save per-class classification report
    per_class_p, per_class_r, per_class_f, per_class_s = precision_recall_fscore_support(
        test_y, test_preds, labels=list(range(8)), zero_division=0
    )
    cm = confusion_matrix(test_y, test_preds, labels=list(range(8)))

    class_report_rows = []
    for c_id in range(8):
        class_report_rows.append({
            "class_id": c_id,
            "breed_name": IDX_TO_BREED[c_id],
            "species": IDX_TO_SPECIES[c_id],
            "precision": float(per_class_p[c_id]),
            "recall": float(per_class_r[c_id]),
            "f1_score": float(per_class_f[c_id]),
            "test_samples": int(per_class_s[c_id])
        })
    pd.DataFrame(class_report_rows).to_csv("reports/selected_breeds/classification_report.csv", index=False)
    
    cm_df = pd.DataFrame(cm, index=[IDX_TO_BREED[i] for i in range(8)], columns=[IDX_TO_BREED[i] for i in range(8)])
    cm_df.to_csv("reports/selected_breeds/confusion_matrix.csv")

    test_preds_df = test_real.copy()
    test_preds_df["predicted_class_id"] = test_preds
    test_preds_df["predicted_breed"] = [IDX_TO_BREED[p] for p in test_preds]
    test_preds_df["confidence"] = [test_probs[i, test_preds[i]] for i in range(len(test_preds))]
    test_preds_df["correct"] = (test_preds_df["predicted_class_id"] == test_y)
    test_preds_df.to_csv("reports/selected_breeds/test_predictions.csv", index=False)

    print("Successfully updated reports and artifacts!")

if __name__ == "__main__":
    main()
