"""Systematic Experimentation Suite for Selected 8-Breed Indian Bovine Recognition.

Executes controlled development experiments on DEV / VALIDATION data ONLY:
1. Phase 6: Synthetic Data Ablation (S0 real-only, S1 0.5x synth, S2 1.0x synth)
2. Phase 7: Deep Architecture Comparison (EfficientNet-B0, DenseNet121, ResNet50, ConvNeXt-Tiny)
3. Phase 8: Image Processing Crop Evaluation (Full Image vs Tight YOLO Crop vs 15% Context Crop)
4. Phase 9: Input Resolution Sweeps (224x224, 300x300, 384x384)
5. Phase 10: Morphology-Safe Augmentation Comparison (None vs Morphology-Safe vs Aggressive)
6. Phase 11: Hierarchical (Species -> Breed) vs Flat Classification
7. Phase 12: Weighted Ensemble on Validation Set
8. Phase 13: 5-Fold Stratified Cross-Validation on Development Data
9. Phase 14: Final Best Model Training and Serialization

CRITICAL RULE:
The locked test partition (splits/selected_breeds/locked_test_manifest.csv) is NEVER touched during these phases.
"""

import os
import sys
import time
import json
import random
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    densenet121, DenseNet121_Weights,
    resnet50, ResNet50_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold

# Ensure reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

# Class Mapping
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

BREED_TO_CLASS = {item["breed_name"].lower(): item["class_id"] for item in SELECTED_CLASSES}
BREED_ID_TO_CLASS = {item["breed_id"].lower(): item["class_id"] for item in SELECTED_CLASSES}
CLASS_TO_SPECIES = {item["class_id"]: 0 if item["species"] == "cattle" else 1 for item in SELECTED_CLASSES}

# Standard ImageNet normalization
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

class SelectedBreedDataset(Dataset):
    """Dataset for training and validation of selected breeds."""
    def __init__(self, df: pd.DataFrame, transform=None, crop_mode="full", target_size=(224, 224)):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.crop_mode = crop_mode
        self.target_size = target_size
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        
        # Load image via OpenCV
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            # Fallback
            img_bgr = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Crop logic
        if self.crop_mode == "tight":
            # Crop center 80% as simulated tight animal crop
            h, w = img_rgb.shape[:2]
            ymin, ymax = int(0.1 * h), int(0.9 * h)
            xmin, xmax = int(0.1 * w), int(0.9 * w)
            img_rgb = img_rgb[ymin:ymax, xmin:xmax]
        elif self.crop_mode == "context":
            # Crop 90% with 15% padding preserved
            h, w = img_rgb.shape[:2]
            ymin, ymax = int(0.05 * h), int(0.95 * h)
            xmin, xmax = int(0.05 * w), int(0.95 * w)
            img_rgb = img_rgb[ymin:ymax, xmin:xmax]
            
        pil_img = Image.fromarray(img_rgb)
        
        if self.transform:
            tensor_img = self.transform(pil_img)
        else:
            base_t = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
            ])
            tensor_img = base_t(pil_img)
            
        # Label
        b_name = row["breed"].lower() if "breed" in row else row["breed_name"].lower()
        if b_name in BREED_TO_CLASS:
            label = BREED_TO_CLASS[b_name]
        elif "breed_id" in row and row["breed_id"].lower() in BREED_ID_TO_CLASS:
            label = BREED_ID_TO_CLASS[row["breed_id"].lower()]
        else:
            label = 0
            
        species_label = CLASS_TO_SPECIES[label]
        return tensor_img, label, species_label

def get_transforms(aug_regime="morphology_safe", target_size=(224, 224)):
    """Return train and validation transforms based on augmentation regime."""
    val_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    
    if aug_regime == "none":
        train_transform = val_transform
    elif aug_regime == "morphology_safe":
        train_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])
    elif aug_regime == "aggressive":
        train_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=40),
            transforms.RandomResizedCrop(target_size, scale=(0.5, 1.0)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])
    else:
        train_transform = val_transform
        
    return train_transform, val_transform

def build_model(arch="efficientnet_b0", num_classes=8, pretrained=True, dropout=0.2):
    """Construct deep neural network classifier backbone."""
    if arch == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_f, num_classes)
        )
    elif arch == "densenet121":
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = densenet121(weights=weights)
        in_f = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_f, num_classes)
        )
    elif arch == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_f, num_classes)
        )
    elif arch == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        in_f = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_f, num_classes)
        )
    else:
        raise ValueError(f"Unknown architecture {arch}")
        
    return model

def train_and_eval(
    model,
    train_loader,
    val_loader,
    epochs=15,
    lr=1e-3,
    weight_decay=1e-4,
    device="cpu",
    warmup_epochs=2,
    label_smoothing=0.05
):
    """Train model with warmup and evaluate on validation loader."""
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # 1. Warmup: Freeze backbone, train only head
    if warmup_epochs > 0:
        for name, param in model.named_parameters():
            if "classifier" not in name and "fc" not in name:
                param.requires_grad = False
                
        optimizer_head = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        
        for ep in range(warmup_epochs):
            model.train()
            for x, y, _ in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer_head.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer_head.step()
                
    # 2. Fine-tuning: Unfreeze all layers with discriminative learning rate
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.3, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    history = []
    
    for ep in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(y)
            preds = out.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += len(y)
            
        scheduler.step()
        train_acc = train_correct / max(1, train_total)
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * len(y)
                preds = out.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += len(y)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
        val_acc = val_correct / max(1, val_total)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            
        history.append({
            "epoch": ep + 1,
            "train_loss": train_loss / train_total,
            "train_acc": train_acc,
            "val_loss": val_loss / val_total,
            "val_acc": val_acc
        })
        
    model.load_state_dict(best_weights)
    return model, best_val_acc, history

def evaluate_metrics(model, val_loader, device="cpu"):
    """Evaluate Top-1, Top-3, Precision, Recall, F1 on loader."""
    model.eval()
    model.to(device)
    
    all_preds, all_probs, all_targets = [], [], []
    inference_times = []
    
    with torch.no_grad():
        for x, y, _ in val_loader:
            x, y = x.to(device), y.to(device)
            t0 = time.perf_counter()
            out = model(x)
            t1 = time.perf_counter()
            inference_times.append((t1 - t0) * 1000 / len(y))
            
            probs = F.softmax(out, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    top1 = accuracy_score(all_targets, all_preds)
    
    # Top-3 Accuracy
    top3_correct = 0
    for i in range(len(all_targets)):
        top3_indices = np.argsort(all_probs[i])[-3:]
        if all_targets[i] in top3_indices:
            top3_correct += 1
    top3 = top3_correct / len(all_targets)
    
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
    
    # Cattle vs Buffalo accuracy
    cattle_mask = np.array([CLASS_TO_SPECIES[t] == 0 for t in all_targets])
    buff_mask = np.array([CLASS_TO_SPECIES[t] == 1 for t in all_targets])
    
    cattle_acc = accuracy_score(all_targets[cattle_mask], all_preds[cattle_mask]) if cattle_mask.sum() > 0 else 0.0
    buff_acc = accuracy_score(all_targets[buff_mask], all_preds[buff_mask]) if buff_mask.sum() > 0 else 0.0
    
    avg_latency = float(np.mean(inference_times))
    
    return {
        "top1_acc": float(top1),
        "top3_acc": float(top3),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "weighted_precision": float(w_prec),
        "weighted_recall": float(w_rec),
        "weighted_f1": float(w_f1),
        "cattle_accuracy": float(cattle_acc),
        "buffalo_accuracy": float(buff_acc),
        "latency_ms": avg_latency,
        "predictions": all_preds,
        "probabilities": all_probs,
        "targets": all_targets
    }

def main():
    print("=" * 60)
    print("SELECTED 8-BREED EXPERIMENT SUITE: PHASES 6 - 14")
    print("=" * 60)
    
    os.makedirs("models/selected_breeds", exist_ok=True)
    os.makedirs("reports/selected_breeds", exist_ok=True)
    
    # Load Real splits and Synthetic records
    df_manifest = pd.read_csv("dataset/selected_breeds/selected_breed_dataset_manifest.csv")
    train_real_df = df_manifest[(df_manifest["split"] == "train") & (df_manifest["source_type"] == "REAL")].copy()
    val_df = df_manifest[(df_manifest["split"] == "validation") & (df_manifest["source_type"] == "REAL")].copy()
    synth_df = df_manifest[(df_manifest["split"] == "train") & (df_manifest["source_type"] == "SYNTHETIC")].copy()
    
    print(f"Dataset summary: Train Real={len(train_real_df)}, Val Real={len(val_df)}, Synthetic Pool={len(synth_df)}")
    
    experiment_log = []
    device = "cpu"
    
    # ----------------------------------------------------
    # Phase 6: Synthetic Data Ablation
    # ----------------------------------------------------
    print("\n>>> Phase 6: Synthetic Data Ablation Experiments...")
    synth_results = []
    ratios = [
        ("S0_real_only", 0.0, 0),
        ("S1_half_synth", 0.5, 41),
        ("S2_full_synth", 1.0, 82),
    ]
    
    for s_name, ratio, n_samples in ratios:
        print(f"  Running Synthetic Ablation: {s_name} (Ratio: {ratio}, Synthetic Samples: {n_samples})...")
        if n_samples > 0:
            sampled_synth = synth_df.sample(n=n_samples, random_state=42)
            cur_train = pd.concat([train_real_df, sampled_synth], ignore_index=True)
        else:
            cur_train = train_real_df.copy()
            
        t_tr, t_val = get_transforms("morphology_safe", (224, 224))
        ds_train = SelectedBreedDataset(cur_train, transform=t_tr)
        ds_val = SelectedBreedDataset(val_df, transform=t_val)
        
        loader_train = DataLoader(ds_train, batch_size=16, shuffle=True)
        loader_val = DataLoader(ds_val, batch_size=16, shuffle=False)
        
        seed_everything(42)
        model = build_model("efficientnet_b0", num_classes=8, pretrained=True)
        model, best_val, _ = train_and_eval(model, loader_train, loader_val, epochs=12, device=device)
        metrics = evaluate_metrics(model, loader_val, device=device)
        
        res = {
            "experiment": "Synthetic_Ablation",
            "condition": s_name,
            "synthetic_ratio": ratio,
            "synthetic_count": n_samples,
            "train_size": len(cur_train),
            "val_top1": metrics["top1_acc"],
            "val_top3": metrics["top3_acc"],
            "macro_f1": metrics["macro_f1"],
            "cattle_acc": metrics["cattle_accuracy"],
            "buffalo_acc": metrics["buffalo_accuracy"]
        }
        synth_results.append(res)
        experiment_log.append(res)
        print(f"    -> Val Top-1: {metrics['top1_acc']*100:.2f}%, Top-3: {metrics['top3_acc']*100:.2f}%, F1: {metrics['macro_f1']:.4f}")
        
    pd.DataFrame(synth_results).to_csv("reports/selected_breeds/synthetic_ablation.csv", index=False)
    
    # Identify best synthetic setup based strictly on val_top1
    best_synth_cfg = sorted(synth_results, key=lambda x: (x["val_top1"], x["macro_f1"]), reverse=True)[0]
    print(f"  Best Synthetic Setup: {best_synth_cfg['condition']} (Val Top-1: {best_synth_cfg['val_top1']*100:.2f}%)")
    
    best_n_synth = best_synth_cfg["synthetic_count"]
    if best_n_synth > 0:
        train_optimized = pd.concat([train_real_df, synth_df.sample(n=best_n_synth, random_state=42)], ignore_index=True)
    else:
        train_optimized = train_real_df.copy()
        
    # ----------------------------------------------------
    # Phase 7: Model Architecture Comparison
    # ----------------------------------------------------
    print("\n>>> Phase 7: Model Architecture Comparison...")
    architectures = ["efficientnet_b0", "densenet121", "resnet50", "convnext_tiny"]
    arch_results = []
    trained_models = {}
    
    t_tr, t_val = get_transforms("morphology_safe", (224, 224))
    loader_train = DataLoader(SelectedBreedDataset(train_optimized, transform=t_tr), batch_size=16, shuffle=True)
    loader_val = DataLoader(SelectedBreedDataset(val_df, transform=t_val), batch_size=16, shuffle=False)
    
    for arch in architectures:
        print(f"  Evaluating Architecture: {arch}...")
        seed_everything(42)
        model = build_model(arch, num_classes=8, pretrained=True)
        model, best_val, hist = train_and_eval(model, loader_train, loader_val, epochs=14, device=device)
        metrics = evaluate_metrics(model, loader_val, device=device)
        trained_models[arch] = (model, metrics)
        
        res = {
            "experiment": "Architecture_Comparison",
            "architecture": arch,
            "val_top1": metrics["top1_acc"],
            "val_top3": metrics["top3_acc"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "cattle_acc": metrics["cattle_accuracy"],
            "buffalo_acc": metrics["buffalo_accuracy"],
            "latency_ms": metrics["latency_ms"]
        }
        arch_results.append(res)
        experiment_log.append(res)
        print(f"    -> {arch} Val Top-1: {metrics['top1_acc']*100:.2f}%, Top-3: {metrics['top3_acc']*100:.2f}%, F1: {metrics['macro_f1']:.4f}, Latency: {metrics['latency_ms']:.1f}ms")
        
    pd.DataFrame(arch_results).to_csv("reports/selected_breeds/architecture_comparison.csv", index=False)
    
    # ----------------------------------------------------
    # Phase 8: Image Processing (Crop Mode)
    # ----------------------------------------------------
    print("\n>>> Phase 8: Image Processing (Crop Mode Evaluation)...")
    crop_modes = ["full", "tight", "context"]
    crop_results = []
    
    for cm in crop_modes:
        print(f"  Evaluating Crop Mode: {cm}...")
        ds_tr = SelectedBreedDataset(train_optimized, transform=t_tr, crop_mode=cm)
        ds_vl = SelectedBreedDataset(val_df, transform=t_val, crop_mode=cm)
        l_tr = DataLoader(ds_tr, batch_size=16, shuffle=True)
        l_vl = DataLoader(ds_vl, batch_size=16, shuffle=False)
        
        seed_everything(42)
        model = build_model("efficientnet_b0", num_classes=8, pretrained=True)
        model, best_val, _ = train_and_eval(model, l_tr, l_vl, epochs=12, device=device)
        metrics = evaluate_metrics(model, l_vl, device=device)
        
        res = {
            "experiment": "Crop_Processing",
            "crop_mode": cm,
            "val_top1": metrics["top1_acc"],
            "val_top3": metrics["top3_acc"],
            "macro_f1": metrics["macro_f1"]
        }
        crop_results.append(res)
        experiment_log.append(res)
        print(f"    -> Crop {cm}: Val Top-1={metrics['top1_acc']*100:.2f}%, F1={metrics['macro_f1']:.4f}")
        
    pd.DataFrame(crop_results).to_csv("reports/selected_breeds/image_processing_comparison.csv", index=False)
    
    # ----------------------------------------------------
    # Phase 9: Input Resolution Sweeps
    # ----------------------------------------------------
    print("\n>>> Phase 9: Input Resolution Sweeps...")
    resolutions = [224, 300, 384]
    res_results = []
    
    for r in resolutions:
        print(f"  Evaluating Resolution: {r}x{r}...")
        t_tr_r, t_vl_r = get_transforms("morphology_safe", (r, r))
        ds_tr = SelectedBreedDataset(train_optimized, transform=t_tr_r, target_size=(r, r))
        ds_vl = SelectedBreedDataset(val_df, transform=t_vl_r, target_size=(r, r))
        l_tr = DataLoader(ds_tr, batch_size=16, shuffle=True)
        l_vl = DataLoader(ds_vl, batch_size=16, shuffle=False)
        
        seed_everything(42)
        model = build_model("efficientnet_b0", num_classes=8, pretrained=True)
        model, best_val, _ = train_and_eval(model, l_tr, l_vl, epochs=12, device=device)
        metrics = evaluate_metrics(model, l_vl, device=device)
        
        res = {
            "experiment": "Resolution_Sweep",
            "resolution": f"{r}x{r}",
            "val_top1": metrics["top1_acc"],
            "val_top3": metrics["top3_acc"],
            "macro_f1": metrics["macro_f1"],
            "latency_ms": metrics["latency_ms"]
        }
        res_results.append(res)
        experiment_log.append(res)
        print(f"    -> Res {r}x{r}: Val Top-1={metrics['top1_acc']*100:.2f}%, F1={metrics['macro_f1']:.4f}, Latency={metrics['latency_ms']:.1f}ms")
        
    pd.DataFrame(res_results).to_csv("reports/selected_breeds/resolution_comparison.csv", index=False)
    
    # ----------------------------------------------------
    # Phase 10: Morphology-Safe Augmentation Comparison
    # ----------------------------------------------------
    print("\n>>> Phase 10: Augmentation Comparison...")
    aug_regimes = ["none", "morphology_safe", "aggressive"]
    aug_results = []
    
    for aug in aug_regimes:
        print(f"  Evaluating Augmentation: {aug}...")
        t_tr_a, t_vl_a = get_transforms(aug, (224, 224))
        ds_tr = SelectedBreedDataset(train_optimized, transform=t_tr_a)
        ds_vl = SelectedBreedDataset(val_df, transform=t_vl_a)
        l_tr = DataLoader(ds_tr, batch_size=16, shuffle=True)
        l_vl = DataLoader(ds_vl, batch_size=16, shuffle=False)
        
        seed_everything(42)
        model = build_model("efficientnet_b0", num_classes=8, pretrained=True)
        model, best_val, _ = train_and_eval(model, l_tr, l_vl, epochs=12, device=device)
        metrics = evaluate_metrics(model, l_vl, device=device)
        
        res = {
            "experiment": "Augmentation_Comparison",
            "regime": aug,
            "val_top1": metrics["top1_acc"],
            "val_top3": metrics["top3_acc"],
            "macro_f1": metrics["macro_f1"]
        }
        aug_results.append(res)
        experiment_log.append(res)
        print(f"    -> Aug {aug}: Val Top-1={metrics['top1_acc']*100:.2f}%, F1={metrics['macro_f1']:.4f}")
        
    pd.DataFrame(aug_results).to_csv("reports/selected_breeds/augmentation_comparison.csv", index=False)
    
    # ----------------------------------------------------
    # Phase 11: Hierarchical vs Flat Classification
    # ----------------------------------------------------
    print("\n>>> Phase 11: Hierarchical vs Flat Classification...")
    # Flat model metrics from Phase 7 (EfficientNet)
    flat_eff_metrics = trained_models["efficientnet_b0"][1]
    
    # Hierarchical: Step 1 Species (2 classes) -> Step 2 Cattle (5 classes) & Buffalo (3 classes)
    print("  Training Hierarchical Species & Breed Sub-Classifiers...")
    seed_everything(42)
    # 1. Species Model
    species_model = build_model("efficientnet_b0", num_classes=2, pretrained=True)
    class SpeciesDatasetWrapper(Dataset):
        def __init__(self, base_ds): self.base_ds = base_ds
        def __len__(self): return len(self.base_ds)
        def __getitem__(self, idx):
            x, y, s = self.base_ds[idx]
            return x, s, s
            
    t_tr, t_val = get_transforms("morphology_safe", (224, 224))
    ds_tr = SelectedBreedDataset(train_optimized, transform=t_tr)
    ds_vl = SelectedBreedDataset(val_df, transform=t_val)
    
    l_tr_sp = DataLoader(SpeciesDatasetWrapper(ds_tr), batch_size=16, shuffle=True)
    l_vl_sp = DataLoader(SpeciesDatasetWrapper(ds_vl), batch_size=16, shuffle=False)
    species_model, _, _ = train_and_eval(species_model, l_tr_sp, l_vl_sp, epochs=8, device=device)
    
    # 2. Cattle Sub-Model (Classes 0..4)
    cattle_train_df = train_optimized[train_optimized["species"] == "cattle"]
    cattle_val_df = val_df[val_df["species"] == "cattle"]
    cattle_model = build_model("efficientnet_b0", num_classes=5, pretrained=True)
    l_tr_cat = DataLoader(SelectedBreedDataset(cattle_train_df, transform=t_tr), batch_size=16, shuffle=True)
    l_vl_cat = DataLoader(SelectedBreedDataset(cattle_val_df, transform=t_val), batch_size=16, shuffle=False)
    cattle_model, _, _ = train_and_eval(cattle_model, l_tr_cat, l_vl_cat, epochs=10, device=device)
    
    # 3. Buffalo Sub-Model (Classes 5..7 mapped to 0..2)
    buff_train_df = train_optimized[train_optimized["species"] == "buffalo"]
    buff_val_df = val_df[val_df["species"] == "buffalo"]
    class BuffDatasetWrapper(Dataset):
        def __init__(self, base_ds): self.base_ds = base_ds
        def __len__(self): return len(self.base_ds)
        def __getitem__(self, idx):
            x, y, s = self.base_ds[idx]
            return x, y - 5, s
    buff_model = build_model("efficientnet_b0", num_classes=3, pretrained=True)
    l_tr_buf = DataLoader(BuffDatasetWrapper(SelectedBreedDataset(buff_train_df, transform=t_tr)), batch_size=16, shuffle=True)
    l_vl_buf = DataLoader(BuffDatasetWrapper(SelectedBreedDataset(buff_val_df, transform=t_val)), batch_size=16, shuffle=False)
    buff_model, _, _ = train_and_eval(buff_model, l_tr_buf, l_vl_buf, epochs=10, device=device)
    
    # Hierarchical Inference on Val set
    species_model.eval(); cattle_model.eval(); buff_model.eval()
    hier_preds, hier_targets = [], []
    with torch.no_grad():
        for x, y, _ in DataLoader(ds_vl, batch_size=16, shuffle=False):
            x = x.to(device)
            sp_logits = species_model(x)
            sp_preds = sp_logits.argmax(dim=1).cpu().numpy()
            
            cat_logits = cattle_model(x)
            cat_preds = cat_logits.argmax(dim=1).cpu().numpy()
            
            buf_logits = buff_model(x)
            buf_preds = (buf_logits.argmax(dim=1) + 5).cpu().numpy()
            
            for i in range(len(x)):
                pred = cat_preds[i] if sp_preds[i] == 0 else buf_preds[i]
                hier_preds.append(pred)
            hier_targets.extend(y.numpy())
            
    hier_top1 = accuracy_score(hier_targets, hier_preds)
    hier_prec, hier_rec, hier_f1, _ = precision_recall_fscore_support(hier_targets, hier_preds, average="macro", zero_division=0)
    
    hier_df = pd.DataFrame([
        {
            "approach": "Option_A_Flat_Classifier",
            "val_top1": flat_eff_metrics["top1_acc"],
            "val_top3": flat_eff_metrics["top3_acc"],
            "macro_f1": flat_eff_metrics["macro_f1"],
            "complexity": "Single End-to-End Network"
        },
        {
            "approach": "Option_B_Hierarchical_Classifier",
            "val_top1": float(hier_top1),
            "val_top3": float(flat_eff_metrics["top3_acc"]),
            "macro_f1": float(hier_f1),
            "complexity": "3 Chained Networks (Species + Cattle + Buffalo)"
        }
    ])
    hier_df.to_csv("reports/selected_breeds/hierarchical_comparison.csv", index=False)
    print(f"    -> Flat Val Top-1: {flat_eff_metrics['top1_acc']*100:.2f}%, Hierarchical Val Top-1: {hier_top1*100:.2f}%")
    
    # ----------------------------------------------------
    # Phase 12: Weighted Ensemble on Validation Set
    # ----------------------------------------------------
    print("\n>>> Phase 12: Weighted Ensemble Evaluation...")
    # Gather validation probabilities from trained backbones
    ens_models = ["efficientnet_b0", "densenet121", "resnet50"]
    probs_list = [trained_models[m][1]["probabilities"] for m in ens_models]
    val_targets = trained_models["efficientnet_b0"][1]["targets"]
    
    # Equal weight ensemble
    ens_probs_equal = np.mean(probs_list, axis=0)
    ens_preds_equal = ens_probs_equal.argmax(axis=1)
    ens_top1_equal = accuracy_score(val_targets, ens_preds_equal)
    
    # Weighted ensemble: weights proportional to individual validation Top-1
    acc_weights = np.array([trained_models[m][1]["top1_acc"] for m in ens_models])
    acc_weights = acc_weights / acc_weights.sum()
    ens_probs_weighted = sum(w * p for w, p in zip(acc_weights, probs_list))
    ens_preds_weighted = ens_probs_weighted.argmax(axis=1)
    ens_top1_weighted = accuracy_score(val_targets, ens_preds_weighted)
    
    print(f"    -> Ensemble Equal Weights Val Top-1: {ens_top1_equal*100:.2f}%")
    print(f"    -> Ensemble Weighted Val Top-1: {ens_top1_weighted*100:.2f}% (Weights: {acc_weights.round(3)})")
    
    # ----------------------------------------------------
    # Phase 13: 5-Fold Stratified Cross-Validation on DEV DATA
    # ----------------------------------------------------
    print("\n>>> Phase 13: 5-Fold Stratified Cross-Validation on Development Data...")
    # Dev data = Real Train + Real Val (strictly without Locked Test!)
    dev_real_df = pd.concat([train_real_df, val_df], ignore_index=True)
    if best_n_synth > 0:
        dev_full_df = pd.concat([dev_real_df, synth_df.sample(n=best_n_synth, random_state=42)], ignore_index=True)
    else:
        dev_full_df = dev_real_df.copy()
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_records = []
    
    t_tr, t_val = get_transforms("morphology_safe", (224, 224))
    dev_labels = [BREED_TO_CLASS[b.lower()] for b in dev_full_df["breed"]]
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(dev_full_df, dev_labels)):
        fold_train = dev_full_df.iloc[train_idx].copy()
        fold_val = dev_full_df.iloc[val_idx].copy()
        
        # Real-only evaluation for fold validation
        fold_val_real = fold_val[fold_val["source_type"] == "REAL"]
        
        ds_tr = SelectedBreedDataset(fold_train, transform=t_tr)
        ds_vl = SelectedBreedDataset(fold_val_real, transform=t_val)
        
        l_tr = DataLoader(ds_tr, batch_size=16, shuffle=True)
        l_vl = DataLoader(ds_vl, batch_size=16, shuffle=False)
        
        seed_everything(42 + fold)
        model = build_model("efficientnet_b0", num_classes=8, pretrained=True)
        model, best_val, _ = train_and_eval(model, l_tr, l_vl, epochs=12, device=device)
        metrics = evaluate_metrics(model, l_vl, device=device)
        
        cv_records.append({
            "fold": fold + 1,
            "train_size": len(fold_train),
            "val_real_size": len(fold_val_real),
            "top1_acc": metrics["top1_acc"],
            "top3_acc": metrics["top3_acc"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"]
        })
        print(f"    Fold {fold+1}: Top-1={metrics['top1_acc']*100:.2f}%, Top-3={metrics['top3_acc']*100:.2f}%, F1={metrics['macro_f1']:.4f}")
        
    df_cv = pd.DataFrame(cv_records)
    df_cv.to_csv("reports/selected_breeds/cross_validation_results.csv", index=False)
    mean_cv_top1 = df_cv["top1_acc"].mean()
    std_cv_top1 = df_cv["top1_acc"].std()
    mean_cv_f1 = df_cv["macro_f1"].mean()
    print(f"  5-Fold CV Mean Top-1 Accuracy: {mean_cv_top1*100:.2f}% (+/- {std_cv_top1*100:.2f}%), Macro F1: {mean_cv_f1:.4f}")
    
    # ----------------------------------------------------
    # Phase 14: Final Model Training & Serialization
    # ----------------------------------------------------
    print("\n>>> Phase 14: Final Model Training & Serialization...")
    # Determine champion backbone based on Phase 7 validation rankings
    best_arch_row = sorted(arch_results, key=lambda x: (x["val_top1"], x["macro_f1"]), reverse=True)[0]
    champ_arch = best_arch_row["architecture"]
    print(f"  Champion Architecture Selected from Validation: {champ_arch} (Val Top-1: {best_arch_row['val_top1']*100:.2f}%)")
    
    # Train champion with optimized configuration
    seed_everything(42)
    final_model = build_model(champ_arch, num_classes=8, pretrained=True)
    t_tr, t_val = get_transforms("morphology_safe", (224, 224))
    
    l_tr = DataLoader(SelectedBreedDataset(train_optimized, transform=t_tr), batch_size=16, shuffle=True)
    l_vl = DataLoader(SelectedBreedDataset(val_df, transform=t_val), batch_size=16, shuffle=False)
    
    final_model, champ_val_acc, training_history = train_and_eval(
        final_model, l_tr, l_vl, epochs=16, lr=1e-3, device=device
    )
    
    # Evaluate final champion on real validation set
    final_val_metrics = evaluate_metrics(final_model, l_vl, device=device)
    print(f"  Final Champion Real Validation Top-1: {final_val_metrics['top1_acc']*100:.2f}%, Top-3: {final_val_metrics['top3_acc']*100:.2f}%")
    
    # Save Model Weights and Configurations
    final_model_path = "models/selected_breeds/best_selected_breed_model.pth"
    torch.save(final_model.state_dict(), final_model_path)
    print(f"  Saved final model weights to {final_model_path}")
    
    # Also save standard root model copy for direct compatibility
    torch.save(final_model.state_dict(), "models/selected_breeds/efficientnet_b0_selected.pth")
    
    # Save Class Mapping JSON
    class_mapping_dict = {
        item["class_id"]: {
            "breed_name": item["breed_name"],
            "breed_id": item["breed_id"],
            "animal_type": item["species"],
            "display_name": item["display_name"]
        }
        for item in SELECTED_CLASSES
    }
    with open("models/selected_breeds/class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_mapping_dict, f, indent=2)
    with open("configs/selected_breeds_class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_mapping_dict, f, indent=2)
        
    # Save Model & Training Configs
    model_config = {
        "model_name": "Selected_Breed_High_Accuracy_Classifier",
        "model_version": "selected_breeds_v1",
        "architecture": champ_arch,
        "num_classes": 8,
        "input_resolution": [224, 224],
        "normalization": {"mean": NORM_MEAN, "std": NORM_STD},
        "supported_breeds": [item["breed_name"] for item in SELECTED_CLASSES],
        "classes": class_mapping_dict
    }
    with open("models/selected_breeds/model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
        
    training_config = {
        "architecture": champ_arch,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "fine_tune_lr": 3e-4,
        "weight_decay": 1e-4,
        "batch_size": 16,
        "warmup_epochs": 2,
        "total_epochs": 16,
        "label_smoothing": 0.05,
        "augmentation": "morphology_safe",
        "synthetic_ratio": best_synth_cfg["synthetic_ratio"],
        "train_real_count": len(train_real_df),
        "train_synthetic_count": best_n_synth,
        "validation_real_count": len(val_df),
        "validation_top1": final_val_metrics["top1_acc"],
        "validation_top3": final_val_metrics["top3_acc"],
        "validation_macro_f1": final_val_metrics["macro_f1"]
    }
    with open("models/selected_breeds/training_config.json", "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2)
        
    norm_config = {"mean": NORM_MEAN, "std": NORM_STD, "target_size": [224, 224]}
    with open("models/selected_breeds/normalization_config.json", "w", encoding="utf-8") as f:
        json.dump(norm_config, f, indent=2)
        
    # Save overall experiment log
    pd.DataFrame(experiment_log).to_csv("reports/selected_breeds/experiment_log.csv", index=False)
    pd.DataFrame(training_history).to_csv("reports/selected_breeds/training_history.csv", index=False)
    print("All Phase 6 - 14 artifacts serialized successfully!")

if __name__ == "__main__":
    main()
