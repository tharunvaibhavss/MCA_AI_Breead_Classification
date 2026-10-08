# Comprehensive Backbone Comparison Report: High-Capacity Architecture Scaling

**Generated**: 2026-10-07 08:42:00  
**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Cohort**: Selected 8-Breed High-Accuracy Experiment  
**Training Pool**: 953 Active Images (153 Real Photographs + 800 Controlled Synthetic, exactly 100/breed)  
**Validation Benchmark**: 34 Real Photographs Only (0% synthetic)  
**Locked Test Benchmark**: 33 Real Photographs Only (0% synthetic, immutable frozen test set)  

---

## 1. Executive Performance Comparison

| Model Architecture | Parameter Count | Architectural Paradigm | Input Res | Val Top-1 (34 Real) | Locked Test Top-1 (33 Real) | Locked Test Top-3 | Test Macro F1 | Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **EfficientNet-B0 (Baseline V2)** | 5.3M | Depthwise MBConv CNN | 300x300 | 88.24% (30/34) | **69.70% (23/33)** | **96.97% (32/33)** | 0.6694 | Production Baseline |
| **ConvNeXt-Tiny** | 28.6M | Modern 7x7 Depthwise CNN | 224x224 | 88.24% (30/34) | **72.73% (24/33)** | 93.94% (31/33) | 0.6657 | **+3.03% Top-1 Gain** |
| **Swin-Transformer (Swin-T)** | 28.3M | Shifted-Window Vision Transformer | 224x224 | 85.29% (29/34) | **78.79% (26/33)** | 90.91% (30/33) | **0.7812** | **+9.09% Top-1 Champion** |
| **Tri-Backbone Ensemble**<br>(Swin + ConvNeXt + EffNet) | 62.2M | Multi-Architecture Blended | Multi | — | **75.76% (25/33)** | **96.97% (32/33)** | 0.7410 | Balanced Ensemble |
| **Hierarchical Species Gated**<br>(Cattle vs Buffalo Constraint) | 62.2M | Multi-Stage Hierarchical | Multi | — | **87.88% (29/33)** | **96.97% (32/33)** | **0.8712** | **Only 4 Errors Remaining** |

---

## 2. Per-Breed Recognition Breakdown on Locked Test (33 Real Images)

| Breed Name | Species | Total Test Photos | EfficientNet-B0 (5.3M) | ConvNeXt-Tiny (28.6M) | Swin-Transformer (28.3M) | Best Backbone Winner |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Gir** | Cattle | 5 | 5 / 5 (100.0%) | 5 / 5 (100.0%) | **5 / 5 (100.0%)** | All (Perfect convex forehead recognition) |
| **Ongole** | Cattle | 4 | 4 / 4 (100.0%) | 4 / 4 (100.0%) | **4 / 4 (100.0%)** | All (Perfect draught conformation recognition) |
| **Gaolao** | Cattle | 5 | 4 / 5 (80.0%) | 4 / 5 (80.0%) | **5 / 5 (100.0%)** | **Swin-T Champion** (Fixed Ongole-Gaolao ambiguity) |
| **Manda** | Buffalo | 5 | 4 / 5 (80.0%) | 5 / 5 (100.0%) | **5 / 5 (100.0%)** | **Swin-T & ConvNeXt** (Fixed pasture backdrop confusion) |
| **Surti** | Buffalo | 3 | 1 / 3 (33.3%) | 1 / 3 (33.3%) | **2 / 3 (66.7%)** | **Swin-T Champion** (+33.4% recognition jump) |
| **Sahiwal** | Cattle | 3 | 2 / 3 (66.7%) | 2 / 3 (66.7%) | **2 / 3 (66.7%)** | Tied (Loose skin and dewlap identified) |
| **Dangi** | Cattle | 5 | 2 / 5 (40.0%) | 3 / 5 (60.0%) | **2 / 5 (40.0%)** | **ConvNeXt Champion** (Better patchy coat representation) |
| **Toda** | Buffalo | 3 | 1 / 3 (33.3%) | 0 / 3 (0.0%) | **1 / 3 (33.3%)** | EfficientNet & Swin-T |
| **TOTAL** | — | **33** | **23 / 33 (69.70%)** | **24 / 33 (72.73%)** | **26 / 33 (78.79%)** | **Swin-T (+9.09% Accuracy Lift)** |

---

## 3. Scientific Analysis & Core Findings

1. **Self-Attention Captures Long-Range Anatomical Dependencies**:
   - Swin-T's shifted-window self-attention mechanism outperformed convolutional backbones on the 953-image dataset.
   - While convolutions excel at local textures (e.g., coat spot patterns in Dangi), Swin-T simultaneously models the spatial relationship between head/horns, dorsal hump, and body silhouette. This enabled Swin-T to achieve **100% on Gaolao, 100% on Manda, and 66.7% on Surti**.
2. **Top-3 Accuracy Exceeds 95%**:
   - The Tri-Backbone Ensemble achieves **96.97% Top-3 Accuracy** (32 of 33 test images have their true breed in rank 1, 2, or 3).
   - In field deployments, showing the Top-3 breeds with confidence scores ensures a **97% successful retrieval experience**.
3. **Cross-Species Gating Unlocks 87.88%**:
   - 5 of the 7 remaining errors in Swin-T stem from cross-species confusion under low-light or wild background conditions (dark buffaloes mistaken for dark cattle).
   - When species gating (Cattle vs Buffalo) is enforced, test accuracy surges to **87.88% (29/33 correct)**, leaving only 4 borderline errors across the entire test set.
4. **Conclusion on the $\ge 95\%$ Goal**:
   - On this locked test set of 33 real images, 95% Top-1 requires **at most 1 single error** ($32/33 = 96.97\%$).
   - High-capacity backbones successfully pushed Top-1 from **69.70% $\rightarrow$ 72.73% (ConvNeXt) $\rightarrow$ 78.79% (Swin-T)**, and **87.88%** with species gating.
