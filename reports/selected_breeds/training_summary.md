# Selected-Breed Training and Experimentation Summary

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Cohort Scope**: Focused Selected-Breed High-Accuracy Version (8 Breeds)  
**Architecture**: EfficientNet-B0  
**Optimization Target**: $\ge 95\%$ Top-1 Accuracy  
**Date**: 2026-10-02  

---

## 1. Experimental Progression & Methodological Matrix

| Phase | Evaluation Dimension | Key Finding / Winner | Quantitative Evidence (Real Val) |
| :---: | :--- | :--- | :--- |
| **Phase 6** | Synthetic Supplementary Ratio | **S0 (Real Only)** | Val Top-1: 79.41%, Macro F1: **0.7919** (S2 synthetic dropped F1 to 0.7858) |
| **Phase 7** | Deep Backbone Architecture | **EfficientNet-B0** | Val Top-1: **79.41%**, Top-3: **94.12%**, Latency: **14.8 ms** (vs ResNet 22.7ms, DenseNet 25.4ms) |
| **Phase 8** | Image Crop Processing | **Context Crop (15% Padding)** | Val Top-1: 79.41%, Macro F1: **0.7979** (Tight crop dropped Top-1 to 70.59%) |
| **Phase 9** | Input Resolution | **300 x 300** | Val Top-1: **85.29%**, Macro F1: **0.8491** (Peak validation performance) |
| **Phase 10** | Augmentation Regime | **Morphology-Safe / None** | Aggressive augmentation distorted horn morphology, dropping accuracy to 70.59% |
| **Phase 11** | Hierarchical vs Flat | **Option A (Flat Classifier)** | Flat Top-1: **79.41%** vs Hierarchical Top-1: 73.53% (error cascade in two-stage) |
| **Phase 12** | Multi-Model Ensemble | **Weighted Ensemble** | Val Top-1: **82.35%** (EfficientNet + ResNet + DenseNet) |
| **Phase 13** | 5-Fold Stratified CV | **Robust Generalization** | Dev CV Mean Top-1: **75.42% $\pm$ 4.99%**, Top-3: up to **94.74%** |
| **Phase 15** | Final Locked Test Set | **100% Real Evaluation** | **63.64% Top-1**, **81.82% Top-3**, **0.6076 Macro F1** (33 unseen real images) |

---

## 2. Key Insights for MCA Project Viva & Demonstration

1. **Why Selected Breeds?**:
   - The 82-breed dataset suffered from heavy data sparsity (over 50 breeds had $\le 3$ real images).
   - Downscaling the scope to 8 well-represented breeds produced a stable, reliable system with $\ge 81.8\%$ Top-3 confidence and honest, zero-leakage reporting.
2. **Context-Preserving YOLO Crop**:
   - Indian bovine breeds are distinguished heavily by peripheral anatomy: horn angle, ear curvature, hump stature, and dewlap size.
   - Tight bounding box cropping clips these crucial breed markers. 15% contextual padding proved empirically superior.
3. **Scientific Honesty**:
   - The target was 95%. The genuine, unmanipulated locked test result is **63.64% Top-1** and **81.82% Top-3**.
   - No test images were leaked, tuned against, or modified.
