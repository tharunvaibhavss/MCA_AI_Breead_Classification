# Model Training Summary: Selected Breeds V2

**Execution Date**: 2026-10-06 19:59:42  
**Model Architecture**: EfficientNet-B0 (Pretrained ImageNet-1k Backbone)  
**Input Resolution**: 300 x 300  
**Active Training Dataset**: 153 Real + 800 Synthetic = **953 Total Images**  

---

## 1. Hyperparameter & Optimization Configuration

| Parameter | Specification | Purpose / Rationale |
| :--- | :--- | :--- |
| **Model Backbone** | EfficientNet-B0 | Champion architecture from Phase 7 comparison |
| **Input Resolution** | 300 x 300 | High spatial fidelity preserving zebu humps & horn curves |
| **Crop Strategy** | Context-Preserving Crop (15% padding) | Maintains bovine morphological profile and head contours |
| **Augmentation** | Morphology-Safe Augmentation | Subtle jitter, affine ($\pm 5\%$), rotation ($\pm 10^\circ$) preventing biological distortion |
| **Optimizer** | AdamW | Weight decay $1 \times 10^{-4}$, decoupled adaptive gradients |
| **Learning Rate Schedule** | 2-Epoch Head Warmup + Cosine Annealing (16 Epochs) | Stable feature adaptation without catastrophic forgetting |
| **Discriminative LR** | Backbone: $1.5 \times 10^{-4}$, Classifier: $6.0 \times 10^{-4}$ | Refines high-level head features faster than visual backbone |
| **Label Smoothing** | 0.05 | Prevents overconfidence on synthetic training samples |
| **Balancing Strategy** | Domain-Weighted & Class-Balanced Loss | Real boost 3.5x + Inverse Class Frequency weighting |

---

## 2. Validation Progression & Early Stopping Monitoring

- **Best Validation Epoch**: Epoch 9
- **Validation Top-1 Accuracy**: **88.24%** (30/34)
- **Validation Top-3 Accuracy**: **97.06%**
- **Validation Macro F1**: **0.8710**
- **Validation Weighted F1**: **0.8838**

---

## 3. Training History Log

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Top-1 (%) | Val Top-3 (%) | Val Macro F1 | Duration (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.8511 | 90.66% | 0.9874 | 70.59% | 91.18% | 0.7085 | 349.03s |
| 2 | 0.5051 | 96.75% | 0.8398 | 79.41% | 94.12% | 0.7871 | 343.73s |
| 3 | 0.3994 | 98.22% | 0.8443 | 76.47% | 94.12% | 0.7499 | 343.0s |
| 4 | 0.3391 | 99.16% | 0.8190 | 82.35% | 97.06% | 0.8039 | 342.8s |
| 5 | 0.3406 | 99.58% | 0.7144 | 85.29% | 97.06% | 0.8366 | 340.24s |
| 6 | 0.3254 | 99.69% | 0.7754 | 88.24% | 97.06% | 0.8657 | 337.73s |
| 7 | 0.3103 | 99.90% | 0.8204 | 79.41% | 97.06% | 0.7907 | 343.08s |
| 8 | 0.3106 | 99.58% | 0.7716 | 79.41% | 97.06% | 0.7742 | 349.36s |
| 9 | 0.3013 | 100.00% | 0.7687 | 88.24% | 97.06% | 0.8710 | 360.92s |
| 10 | 0.3034 | 99.79% | 0.7756 | 79.41% | 94.12% | 0.7859 | 354.26s |
| 11 | 0.2985 | 100.00% | 0.7442 | 82.35% | 94.12% | 0.8188 | 364.69s |
| 12 | 0.2945 | 100.00% | 0.7916 | 76.47% | 97.06% | 0.7592 | 364.54s |
| 13 | 0.2990 | 99.79% | 0.7651 | 79.41% | 97.06% | 0.7861 | 359.18s |
| 14 | 0.2939 | 100.00% | 0.7735 | 82.35% | 94.12% | 0.8174 | 359.33s |
| 15 | 0.2944 | 99.90% | 0.7477 | 85.29% | 94.12% | 0.8389 | 357.05s |
| 16 | 0.2920 | 100.00% | 0.7582 | 85.29% | 97.06% | 0.8389 | 360.68s |
