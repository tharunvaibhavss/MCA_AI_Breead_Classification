# Training Summary Report: Selected Breeds V4

**Generated**: 2026-10-08 00:16:05  
**Project**: AI-Breed-Recognition-Selected-Breeds  
**Architecture**: EfficientNet-B4 (17,562,960 parameters)  
**Input Resolution**: 300x300  
**Training Pool**: 4,313 Images (153 Real + 4,160 Synthetic)  
**Optimizer**: AdamW with Cosine Annealing Schedule  
**Domain Loss Weighting**: 10.0x Real domain boost  
**Label Smoothing**: 0.05  
**Warmup Epochs**: 2 | **Fine-Tuning Epochs**: 6 | **Best Epoch**: 5  

---

## Epoch History

| Epoch | Train Loss | Train Acc (%) | Val Top-1 (%) | Val Top-3 (%) | Val Macro F1 | Duration (s) | Best? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.9040 | 92.88% | 47.06% | 97.06% | 0.4202 | 1316.1 | YES |
| 2 | 0.7195 | 94.25% | 67.65% | 88.24% | 0.6549 | 1303.6 | YES |
| 3 | 0.6344 | 95.94% | 67.65% | 88.24% | 0.6548 | 1295.9 | — |
| 4 | 0.5645 | 96.64% | 67.65% | 88.24% | 0.6461 | 1291.7 | — |
| 5 | 0.5371 | 96.99% | 67.65% | 88.24% | 0.6581 | 1366.1 | YES |
| 6 | 0.5330 | 96.94% | 67.65% | 88.24% | 0.6581 | 1361.7 | — |
