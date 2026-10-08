# Real vs Synthetic Analysis Report: Selected Breeds V4

**Generated**: 2026-10-08 00:16:05  

---

## 1. Distribution

- Real Training Photos: **153** (3.55% of training pool)
- Synthetic Images: **4,160** (96.45% of training pool)
- Total Active Pool: **4,313 images**

## 2. Impact on Generalization

Synthetic images provide dense visual variations of coat patterns, horn curvatures, and dorsal humps across all 8 breeds. However, domain divergence exists between synthetic rendering lighting and raw field photographs. Using domain loss weighting with Real Boost ensures that the model leverages synthetic samples for morphological diversity while prioritizing genuine photographic features.
