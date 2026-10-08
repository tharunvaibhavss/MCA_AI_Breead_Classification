# Dataset Analysis Report (Selected Breeds V4)

**Generated**: 2026-10-07 20:47:33  
**Project**: AI-Breed-Recognition-Selected-Breeds  
**Scope**: Full dataset audit across `dataset/selected_breeds/`  

---

## 1. Complete Dataset Breakdown

- **Total Image Files on Disk**: 4380 images
- **Real Photographic Dataset**: 220 images (153 Train, 34 Validation, 33 Locked Test)
- **Synthetic Dataset**: 4160 images across 8 target breeds
  - **Newly Generated Controlled Synthetic**: 4000 images (exactly 500 per breed across all 8 breeds)
  - **Legacy Synthetic (`SYNTH_` prefixes)**: 160 images
- **Non-Image Documentation Files**: 8 files (`README.txt`) excluded from training

## 2. Real vs Synthetic Ratio

- Real Training Share: 3.55% (153 images)
- Synthetic Training Share: 96.45% (4160 images)
- **Domain Weighting Requirement**: Because synthetic images comprise 96.5% of the V4 training pool, a domain loss weight ($w_{real} \gg 1.0$) is required during optimization to ensure real photographic domain representations dominate gradient updates.
