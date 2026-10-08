# Final Model Evaluation: Selected Breeds V2 Champion

**Evaluation Date**: 2026-10-06 19:59:42  
**Model Version**: `selected_breeds_v2`  
**Model Artifact**: `models/selected_breeds/selected_breeds_v2/best_selected_breed_model.pth`  
**Locked Test Set Path**: `splits/selected_breeds/locked_test_manifest.csv`  
**Test Set Integrity**: **100% Real Photographs Only (33 Images)** — Never seen during training or tuning.  

---

## 1. Executive Performance Metrics

| Evaluation Metric | Measured Result | Performance Description |
| :--- | :---: | :--- |
| **Locked Test Top-1 Accuracy** | **69.70%** | **23 of 33 correct predictions** |
| **Locked Test Top-3 Accuracy** | **96.97%** | Candidate ranking inclusion across 8 classes |
| **Macro Precision** | **0.6923** | Unweighted mean precision across all 8 classes |
| **Macro Recall** | **0.6667** | Unweighted mean recall across all 8 classes |
| **Macro F1-Score** | **0.6694** | Harmonic mean balancing precision and recall |
| **Weighted F1-Score** | **0.6956** | Support-weighted F1 across classes |
| **Cattle Top-1 Accuracy** | **77.27%** | Subset accuracy across 5 cattle breeds (22 images) |
| **Buffalo Top-1 Accuracy** | **54.55%** | Subset accuracy across 3 buffalo breeds (11 images) |
| **Mean Inference Latency** | **101.38 ms** | EfficientNet-B0 CPU inference latency per image |

---

## 2. Per-Breed Performance Breakdown on Locked Test Set

| Breed Name | Species | Support | Correct | Top-1 Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dangi** | Cattle | 5 | 2 | 40.00% | 0.5000 | 0.4000 | **0.4444** |
| **Gaolao** | Cattle | 5 | 4 | 80.00% | 0.5714 | 0.8000 | **0.6667** |
| **Gir** | Cattle | 5 | 5 | 100.00% | 1.0000 | 1.0000 | **1.0000** |
| **Ongole** | Cattle | 4 | 4 | 100.00% | 0.8000 | 1.0000 | **0.8889** |
| **Sahiwal** | Cattle | 3 | 2 | 66.67% | 1.0000 | 0.6667 | **0.8000** |
| **Manda** | Buffalo | 5 | 4 | 80.00% | 1.0000 | 0.8000 | **0.8889** |
| **Surti** | Buffalo | 3 | 1 | 33.33% | 0.3333 | 0.3333 | **0.3333** |
| **Toda** | Buffalo | 3 | 1 | 33.33% | 0.3333 | 0.3333 | **0.3333** |

---

## 3. Comparison with Previous Model Generations

| Model Generation | Training Setup | Input Res | Locked Test Top-1 | Locked Test Top-3 | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Selected-Breeds V1** | 153 Real + 82 Pilot Synth | 224x224 | **63.64%** (21/33) | **81.82%** (27/33) | 0.6012 |
| **Selected-Breeds V2 (Current)** | **153 Real + 800 Full Synth (953 total)** | **300x300** | **69.70%** (23/33) | **96.97%** | **0.6694** |
| **Absolute Delta** | **+718 Synthetic Samples** | **+76 px** | **+6.06%** | **+15.15%** | **+0.0682** |

---

## 4. Assessment Against Aspirational >=95% Target

- **Aspirational Target**: $\ge 95.00\%$ Top-1 Accuracy.
- **Measured Locked Real Test Result**: **69.70%**.
- **Target Achieved**: **NO**.
- **Statistical Note**: The locked test set contains strictly 33 authentic real photographs. Each individual test sample carries a discrete weight of $\approx 3.03\%$. The measured result reflects true, un-manipulated generalization on genuine field camera photographs.
