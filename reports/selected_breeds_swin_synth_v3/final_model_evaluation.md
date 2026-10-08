# Final Model Evaluation on Locked Real Test Set

**Generated**: 2026-10-09 00:03:44  
**Model**: selected_breeds_swin_synth_v3  
**Locked Real Test Size**: 33 Real Photographs (0% synthetic)  

## 1. Performance Overview

| Metric | Measured Value | Target Value | Status |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | **75.76%** (25/33) | **>= 95.0%** (>= 32/33) | **TARGET NOT ACHIEVED** |
| **Top-3 Accuracy** | **100.00%** (33/33) | - | Strong coverage |
| **Macro Precision** | **0.7875** | - | - |
| **Macro Recall** | **0.7500** | - | - |
| **Macro F1** | **0.7326** | - | - |
| **Weighted F1** | **0.7551** | - | - |
| **Cattle Accuracy** | **72.73%** | - | - |
| **Buffalo Accuracy** | **81.82%** | - | - |
| **Inference Latency** | **136.31 ms** | - | Real-time |

---

## 2. Per-Breed Detailed Metrics

| Breed | Species | Support | Correct | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Dangi | cattle | 5 | 2 | 0.5000 | 0.4000 | 0.4444 |
| Gaolao | cattle | 5 | 4 | 1.0000 | 0.8000 | 0.8889 |
| Gir | cattle | 5 | 5 | 1.0000 | 1.0000 | 1.0000 |
| Ongole | cattle | 4 | 4 | 0.8000 | 1.0000 | 0.8889 |
| Sahiwal | cattle | 3 | 1 | 1.0000 | 0.3333 | 0.5000 |
| Manda | buffalo | 5 | 4 | 1.0000 | 0.8000 | 0.8889 |
| Surti | buffalo | 3 | 2 | 0.4000 | 0.6667 | 0.5000 |
| Toda | buffalo | 3 | 3 | 0.6000 | 1.0000 | 0.7500 |
