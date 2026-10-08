# Training Summary: Swin-T Synthetic V3

**Generated**: 2026-10-09 00:03:44  
**Model Name**: selected_breeds_swin_synth_v3  
**Backbone**: Swin Transformer Tiny (Swin-T, 28,296,808 parameters)  

## 1. Training Setup

- **Pretrained Weights**: ImageNet-1K (`torchvision.models.Swin_T_Weights.DEFAULT`)
- **Input Resolution**: 224x224 RGB
- **Context-Preserving Padding**: 15% padding crop (center 85% area)
- **Active Training Pool**: 4,313 images (153 Real + 4,160 Synthetic)
- **Real Domain Boost Factor**: 5.0x
- **Label Smoothing**: 0.1
- **Head Architecture**: Standard 8-Class Swin Head
- **Optimizer**: AdamW
- **Head LR**: 0.001
- **Backbone Stage 3-4 LR**: 1.0e-4
- **Weight Decay**: 1.0e-4
- **Fine-Tuning Epochs**: 4

---

## 2. Validation Progression

| Epoch | Train Loss | Train Accuracy | Val Top-1 (%) | Val Top-3 (%) | Val Macro F1 | Duration (s) | Best Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.6447 | 97.26% | 82.35% | 91.18% | 0.8053 | 1310.2s | YES |
| 2 | 0.5294 | 99.79% | 79.41% | 94.12% | 0.7669 | 1312.9s | NO |
| 3 | 0.5149 | 99.91% | 76.47% | 94.12% | 0.7613 | 1292.6s | NO |
| 4 | 0.5086 | 99.93% | 82.35% | 94.12% | 0.8067 | 1320.1s | YES |
