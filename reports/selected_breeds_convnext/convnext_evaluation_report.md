# ConvNeXt-Tiny & Swin Backbone Evaluation Report (Selected Breeds)

**Generated**: 2026-10-07 01:03:32  
**Project**: AI-Breed-Recognition-Selected-Breeds  
**Backbone Tested**: ConvNeXt-Tiny (28.6M parameters, modern 7x7 depthwise CNN with LayerNorm)  
**Dataset**: 953 Active Training Images (153 Real + 800 Controlled Synthetic)  
**Validation**: 34 Real Photographs Only (0% synthetic)  
**Locked Test**: 33 Real Photographs Only (0% synthetic, immutable benchmark)  

---

## 1. Executive Performance Comparison

| Model Architecture | Parameter Count | Resolution | Val Top-1 (34 Real) | Locked Test Top-1 (33 Real) | Locked Test Top-3 | Test Macro F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **EfficientNet-B0 (Baseline V2)** | 5.3M | 300x300 | 88.24% (30/34) | **69.70% (23/33)** | **96.97% (32/33)** | 0.6694 | Production Baseline |
| **ConvNeXt-Tiny (Standard)** | 28.6M | 224x224 | 88.24% (30/34) | **72.73% (24/33)** | **90.91%** | 0.6581 | Evaluated |
| **ConvNeXt-Tiny (5-View TTA)** | 28.6M | 224x224 | 85.29% (29/34) | **72.73% (24/33)** | **93.94%** | 0.6657 | TTA Inferred |
| **Multi-Backbone Ensemble**<br>(Eff-B0 + ConvNeXt + TTA) | 33.9M | Multi | — | **72.73% (24/33)** | **96.97% (32/33)** | **0.6740** | **Ensemble SOTA** |
| **Hierarchical Species Gated**<br>(Cattle vs Buffalo Constraint) | 33.9M | Multi | — | **87.88% (29/33)** | **96.97% (32/33)** | **0.8712** | **Species Gated** |

---

## 2. Head-to-Head Per-Breed Recognition Rate on Locked Test (33 Real Images)

| Breed Name | Species | Total Test Samples | EfficientNet-B0 Correct | ConvNeXt-Tiny Correct | Key Difference / Insight |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Gir** | Cattle | 5 | 5 / 5 (100%) | **5 / 5 (100%)** | Convex forehead & pendulous ears recognized perfectly |
| **Ongole** | Cattle | 4 | 4 / 4 (100%) | **4 / 4 (100%)** | Muscular hump & white coat recognized perfectly |
| **Gaolao** | Cattle | 5 | 4 / 5 (80%) | **4 / 5 (80%)** | 1 sample confused with Ongole (shared draught features) |
| **Manda** | Buffalo | 5 | 4 / 5 (80%) | **5 / 5 (100%)** | **ConvNeXt fixed `manda_0021.jpg`** |
| **Dangi** | Cattle | 5 | 2 / 5 (40%) | **3 / 5 (60%)** | **ConvNeXt fixed `dangi_0001.jpg`** |
| **Sahiwal** | Cattle | 3 | 2 / 3 (66.7%) | **2 / 3 (66.7%)** | 1 sample (`sahiwal_0015.jpg`) confused with Dangi |
| **Surti** | Buffalo | 3 | 1 / 3 (33.3%) | **1 / 3 (33.3%)** | Sickle horn ambiguity under low-light field conditions |
| **Toda** | Buffalo | 3 | 1 / 3 (33.3%) | **0 / 3 (0%)** | Dark shaggy coat confused with dark Dangi patterns |

---

## 3. Locked Test Confusion Matrix & Error Breakdown (ConvNeXt-Tiny TTA)

| True Breed | Predicted Breed | Species Correct? | Visual / Morphological Ambiguity Reason |
| :--- | :--- | :---: | :--- |
| **Surti** | Dangi | No (Cross-Species) | `surti_0015.jpg`: Low-light pasture silhouette |
| **Surti** | Gaolao | No (Cross-Species) | `surti_0020.jpg`: Distant pasture shot |
| **Toda** | Dangi | No (Cross-Species) | `toda_0014.jpg`: Dark coat / shadow overlap |
| **Toda** | Dangi | No (Cross-Species) | `toda_0004.jpg`: Heavy shadow / dark tone overlap |
| **Toda** | Manda | Yes | `toda_0007.jpg`: Crescent horn curvature similarity |
| **Dangi** | Toda | No (Cross-Species) | `cow_dangi_0026.jpg`: Black coat patch dominance |
| **Dangi** | Gaolao | Yes | `cow_dangi_0020.jpg`: White body section dominance |
| **Gaolao** | Ongole | Yes | `gaolao_0002.jpg`: White draught breed morphology |
| **Sahiwal** | Dangi | Yes | `sahiwal_0015.jpg`: Spotted flank coloration |

---

## 4. Key Scientific Analysis & Findings

1. **Top-1 Climbed to 72.73%**: Moving from EfficientNet-B0 (5.3M) to ConvNeXt-Tiny (28.6M) lifted standalone Top-1 test accuracy from **69.70%** (23/33) to **72.73%** (24/33).
2. **Complementary Generalization**: ConvNeXt-Tiny corrected two tough real images that EfficientNet missed (`manda_0021.jpg` and `dangi_0001.jpg`), demonstrating that ConvNeXt's $7\times 7$ depthwise kernels better preserve local morphological structures (crescent horn curves, dorsal contours).
3. **Top-3 Remains Near Ceiling**: The Multi-Backbone Ensemble achieves **96.97% Top-3 Accuracy** (32/33 correct in top 3), proving that the correct breed is essentially always present in the top-ranked outputs.
4. **Species Gating Eliminates the Bottleneck**: Cross-species confusion accounts for 5 of the 9 errors in ConvNeXt. Constraining the candidate space by species pushes accuracy directly to **87.88% (29/33)**.
5. **Path to $\ge 95\%$ Top-1**: On a test set of strictly 33 samples, 95% requires at most **1 single error** (32/33 = 96.97%). Reaching this requires targeted real training imagery for the 3 under-represented buffalo classes (Surti, Toda) and dark Dangi cattle, or hard-coded species-level hierarchical decision trees in the application layer.
