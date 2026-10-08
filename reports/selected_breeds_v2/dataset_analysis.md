# Selected-Breeds Dataset Analysis (Version 2)

**Generated**: 2026-10-06 19:59:42  
**Target Scope**: 8 Selected Indian Bovine Breeds (5 Cattle, 3 Buffalo)  
**Total Images Inspected**: 1,020 Master Partition Images  

---

## 1. Complete Partition Breakdown

| Partition | Real Count | Synthetic Count | Total Count | Purity Rule |
| :--- | :---: | :---: | :---: | :--- |
| **Training** | 153 | 800 | **953** | Complete Active Training Pool |
| **Validation** | 34 | 0 | **34** | 100% Authentic Real Photographs |
| **Locked Test** | 33 | 0 | **33** | 100% Authentic Real Photographs |
| **Total Master** | **220** | **800** | **1,020** | Zero Contamination Across Splits |

---

## 2. Breed Class Distribution

| Class ID | Breed Name | Species | Real Train | Syn Train | Total Train | Val (Real) | Test (Real) | Total |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | **Dangi** | Cattle | 28 | 100 | **128** | 6 | 5 | 139 |
| 1 | **Gaolao** | Cattle | 23 | 100 | **123** | 5 | 5 | 133 |
| 2 | **Gir** | Cattle | 20 | 100 | **120** | 5 | 5 | 130 |
| 3 | **Ongole** | Cattle | 21 | 100 | **121** | 4 | 4 | 129 |
| 4 | **Sahiwal** | Cattle | 16 | 100 | **116** | 3 | 3 | 122 |
| 5 | **Manda** | Buffalo | 21 | 100 | **121** | 5 | 5 | 131 |
| 6 | **Surti** | Buffalo | 14 | 100 | **114** | 3 | 3 | 120 |
| 7 | **Toda** | Buffalo | 10 | 100 | **110** | 3 | 3 | 116 |

---

## 3. Data Quality & Pre-Training Observations
- **Image Integrity**: 100% of the 1,020 images decodable by PIL with verified SHA-256 hashes.
- **Corruptions**: 0 corrupted images.
- **Exact Duplicates**: 0 exact duplicates across real and synthetic images.
- **Perceptual Near-Duplicates**: 0 perceptual near-duplicates ($dHash \le 3$) between training and locked test sets.
- **Exclusion of Legacy Files**: The 32 legacy `SYNTH_` files remain safely on disk but excluded from the active training manifest to ensure exact 100-sample class balance per breed.
