# Dataset Analysis Report: Selected Breeds Swin-T Synthetic V3

**Generated**: 2026-10-09 00:03:44  
**Project**: AI-Breed-Recognition-Selected-Breeds  

## 1. Executive Summary

This report documents the dataset audit and composition for the **Selected Breeds Swin-T Synthetic V3** training campaign. All available training data—comprising **153 real photographs** and **4,160 synthetic images** (totaling **4,313 active training images**)—was audited, verified, and mapped into the dedicated training manifest `splits/selected_breeds/train_swin_synth_v3.csv`.

Validation and test benchmarks remain strictly **100% real photographs** with **0% synthetic contamination**.

---

## 2. Partition Counts

| Partition | Real Images | Synthetic Images | Total Images | Synthetic Ratio | Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Training** | **153** | **4,160** | **4,313** | **27.19 : 1** | Model training & representation learning |
| **Validation** | **34** | **0** | **34** | **0.00 : 1** | Model selection, hyperparameter tuning & early stopping |
| **Locked Test** | **33** | **0** | **33** | **0.00 : 1** | Immutable independent benchmark (evaluated once) |
| **Total Pool** | **220** | **4,160** | **4,380** | **18.91 : 1** | Complete project dataset |

---

## 3. Breed Breakdown

| Class ID | Breed Name | Species | Real Train | Synthetic Train | Total Train | Real Val | Real Test | Total Real |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Dangi | cattle | 18 | 500 | 518 | 4 | 4 | 26 |
| 1 | Gaolao | cattle | 17 | 505 | 522 | 4 | 4 | 25 |
| 2 | Gir | cattle | 28 | 535 | 563 | 6 | 6 | 40 |
| 3 | Ongole | cattle | 19 | 520 | 539 | 4 | 4 | 27 |
| 4 | Sahiwal | cattle | 28 | 570 | 598 | 6 | 6 | 40 |
| 5 | Manda | buffalo | 14 | 500 | 514 | 3 | 3 | 20 |
| 6 | Surti | buffalo | 15 | 510 | 525 | 3 | 3 | 21 |
| 7 | Toda | buffalo | 14 | 520 | 534 | 4 | 3 | 21 |
| **Total** | **8 Breeds** | **2 Species** | **153** | **4,160** | **4,313** | **34** | **33** | **220** |

---

## 4. Synthetic Pool Composition

The 4,160 synthetic images on disk comprise:
- **v4_expanded_500**: 4,000 images (exactly 500 images per breed across 8 breeds)
- **legacy_synth_800**: 160 legacy `SYNTH_` images
- Non-image files excluded: 8 `README.txt` files (1 per breed folder)
- Corrupted images: 0
