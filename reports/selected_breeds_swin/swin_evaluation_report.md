# Swin-Transformer (Swin-T) Backbone Evaluation Report

**Generated**: 2026-10-07 08:35:16  
**Project**: AI-Breed-Recognition-Selected-Breeds  
**Backbone Tested**: Swin-T (28.3M parameters, Hierarchical Shifted-Window Vision Transformer)  
**Dataset**: 953 Active Training Images (153 Real + 800 Controlled Synthetic)  
**Validation**: 34 Real Photographs Only (0% synthetic)  
**Locked Test**: 33 Real Photographs Only (0% synthetic, immutable benchmark)  

---

## Performance Summary

| Split / Method | Top-1 Accuracy | Top-3 Accuracy | Macro F1 | Correct / Total |
| :--- | :---: | :---: | :---: | :---: |
| **Validation Standard** | **85.29%** | 97.06% | 0.8461 | 29/34 |
| **Validation (5-View TTA)** | **85.29%** | 97.06% | 0.8491 | 29/34 |
| **Locked Test Standard** | **78.79%** | 90.91% | 0.7601 | 26/33 |
| **Locked Test (5-View TTA)** | **78.79%** | 90.91% | 0.7712 | 26/33 |
