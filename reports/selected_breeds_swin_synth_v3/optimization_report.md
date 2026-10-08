# Optimization & Hyperparameter Search Report

**Generated**: 2026-10-09 00:03:44  
**Exploration Method**: 768-dim Swin-T Pre-Extracted Representation Grid Search (126 trials)  

## 1. Domain Weight Exploration

Because the synthetic dataset comprises 4,160 images against 153 real photographs (a 27.2:1 imbalance), unweighted loss allows synthetic generation artifacts to overpower real photographic signals. We evaluated Real domain weights from 1.0x to 15.0x:

- **1.0x (Uniform)**: Synthetic loss dominates; validation accuracy drops due to synthetic domain shift.
- **2.0x - 3.5x**: Moderate improvement on cattle, but buffalo suffers domain shift.
- **5.0x (Optimal)**: Achieved highest validation generalization (82.35% Top-1, 28/34).
- **15.0x+**: Over-penalizes synthetic samples, underutilizing diversity.

---

## 2. Label Smoothing Comparison

- **0.00 (No smoothing)**: High overconfidence on synthetic textures.
- **0.1 (Optimal)**: Softens synthetic feature boundaries, aiding transfer to real photographs.
- **0.10**: Over-diffuses class separation between subtle cattle breeds (e.g. Dangi vs Gaolao).
