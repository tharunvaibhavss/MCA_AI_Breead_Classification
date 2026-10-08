# Systematic Optimization Report: Selected Breeds V4

**Generated**: 2026-10-08 00:16:05  
**Methodology**: Rigorous grid search over 48 candidate hyperparameter configurations, selected STRICTLY via the 34-image Real Validation Set without test feedback.  

---

## 1. Selected Champion Configuration

- **Architecture**: Standard EfficientNet-B4
- **Domain Boost**: 10.0x
- **Label Smoothing**: 0.05
- **Learning Rate**: 0.0005
- **Validation Top-1**: 52.94% (18/34)
- **Validation Top-3**: 85.29%
- **Validation Macro F1**: 0.5427

---

## 2. Parameter Sensitivity Findings

1. **Domain Loss Weighting**: Giving real photographic training images a strong domain boost factor (10.0x) is essential. Without domain weighting, the 4,160 synthetic images dominate gradient descent, reducing validation accuracy on real field photographs.
2. **Label Smoothing**: Label smoothing around 0.05 prevented overconfidence on synthetic textures.
3. **Capacity Scaling**: EfficientNet-B4 (19.3M params) demonstrates robust feature extraction over EfficientNet-B0 (5.3M params).
