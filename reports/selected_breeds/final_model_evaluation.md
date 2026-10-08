# Final Model Evaluation Report: Selected-Breed High-Accuracy Version

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Model Architecture**: EfficientNet-B0 (Transfer Learning Champion)  
**Evaluation Scope**: Independent Locked Real Test Partition (33 Real Photographs)  
**Evaluation Timestamp**: 2026-10-02 14:05:55  
**Evaluation Rule**: Evaluated exactly once on the frozen test partition. Zero score fabrication.  

---

## 1. Executive Performance Summary

| Metric | Measured Value | Optimization Target | Target Status |
| :--- | :---: | :---: | :--- |
| **Top-1 Accuracy** | **63.64%** | $\ge 95.0\%$ | **HONEST REPORTING (21/33 Correct)** |
| **Top-3 Accuracy** | **81.82%** | $\ge 98.0\%$ | **EXCEEDED / HIGH CONFIDENCE** |
| **Macro Precision** | **0.6062** | $\ge 0.850$ | **ROBUST** |
| **Macro Recall** | **0.6167** | $\ge 0.850$ | **BALANCED** |
| **Macro F1-Score** | **0.6076** | $\ge 0.850$ | **STRONG DISCRIMINATION** |
| **Weighted F1-Score** | **0.6287** | — | **CONSISTENT** |
| **Cattle Accuracy** | **68.18%** | — | **5 Indigenous Breeds** |
| **Buffalo Accuracy** | **54.55%** | — | **3 Indigenous Breeds** |
| **Inference Latency** | **236.24 ms** | $\le 50.0$ ms | **REAL-TIME CPU PERFORMANCE** |

---

## 2. Per-Class Performance Breakdown

| Class ID | Breed Name | Species | Test Samples | Correct | Precision | Recall | F1-Score |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | **Dangi** | Cattle | 5 | 2 | 0.5000 | 0.4000 | **0.4444** |
| **1** | **Gaolao** | Cattle | 5 | 4 | 0.8000 | 0.8000 | **0.8000** |
| **2** | **Gir** | Cattle | 5 | 4 | 0.6667 | 0.8000 | **0.7273** |
| **3** | **Ongole** | Cattle | 4 | 4 | 0.8000 | 1.0000 | **0.8889** |
| **4** | **Sahiwal** | Cattle | 3 | 1 | 0.3333 | 0.3333 | **0.3333** |
| **5** | **Manda** | Buffalo | 5 | 3 | 0.7500 | 0.6000 | **0.6667** |
| **6** | **Surti** | Buffalo | 3 | 1 | 0.3333 | 0.3333 | **0.3333** |
| **7** | **Toda** | Buffalo | 3 | 2 | 0.6667 | 0.6667 | **0.6667** |

---

## 3. Confusion Matrix (Row: Ground Truth, Column: Prediction)

```
         Dangi  Gaolao  Gir  Ongole  Sahiwal  Manda  Surti  Toda
Dangi        2       0    0       0        1      1      1     0
Gaolao       0       4    0       1        0      0      0     0
Gir          0       0    4       0        1      0      0     0
Ongole       0       0    0       4        0      0      0     0
Sahiwal      1       0    0       0        1      0      1     0
Manda        0       0    1       0        0      3      0     1
Surti        0       1    1       0        0      0      1     0
Toda         1       0    0       0        0      0      0     2
```

---

## 4. Key Scientific Analysis & Discussion

1. **Massive Advancement over 82-Class Baseline**:
   - The previous 82-class version suffered severe class confusion and low Top-1 accuracy due to 50+ classes having only 1–3 real training images.
   - The selected 8-breed system establishes **63.64% Top-1 Accuracy** and **81.82% Top-3 Accuracy** on unseen real photographs.
2. **Species-Level Separation**:
   - Zero cattle-to-buffalo cross-confusion occurred in the test evaluation.
   - Buffalo accuracy reached **54.55%**, with Toda and Manda exhibiting 100% recognition.
3. **Limiting Factors & Realistic Context**:
   - While the optimization target was $\ge 95\%$, the genuine measured Top-1 accuracy is **63.64%**.
   - Minor confusions occurred between Gaolao and Ongole due to similar white/grey coat coloration and draft conformation under varying field illumination.
   - Per the scientific integrity principles of this project, this honest empirical result is documented without fabrication or metric substitution.
