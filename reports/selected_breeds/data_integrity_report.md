# Data Integrity & Split Verification Report

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Cohort Scope**: Selected 8-Breed High-Accuracy Application Version  
**Audit Timestamp**: 2026-10-02 13:07:12  
**Manifest Path**: `dataset/selected_breeds/selected_breed_dataset_manifest.csv`  
**Test Specification Lock**: `splits/selected_breeds/locked_test_spec.json`  

---

## 1. Split Partitioning & Source Verification

| Partition | Real Images | Synthetic Images | Total Images | Partition Purity | Verification Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Training** | 153 | 82 | **235** | Real + Controlled Supplementary Synthetic | **VERIFIED CLEAN** |
| **Validation** | 34 | 0 | **34** | **100% Real Photographs Only** | **STRICTLY ISOLATED** |
| **Test (Locked)** | 33 | 0 | **33** | **100% Real Photographs Only** | **FROZEN & LOCKED** |
| **Total** | **220** | **82** | **302** | — | — |

---

## 2. Partition Independence & Leakage Audit

Exhaustive cryptographic (SHA-256) and perceptual ($dHash$, 64-bit, Hamming distance $\le 3$) checks were executed across all partition pairs:

| Comparison Pair | Exact SHA-256 Duplicates | Perceptual Near-Duplicates ($dHash \le 3$) | Isolation Status |
| :--- | :---: | :---: | :--- |
| **Real Test $\leftrightarrow$ Real Train** | **0 / 33** | **0 / 33** | **100% CLEAN ISOLATION** |
| **Real Test $\leftrightarrow$ Real Validation** | **0 / 33** | **0 / 33** | **100% CLEAN ISOLATION** |
| **Real Validation $\leftrightarrow$ Real Train** | **0 / 34** | **0 / 34** | **100% CLEAN ISOLATION** |
| **Synthetic Train $\leftrightarrow$ Real Test** | **0 / 33** | **0 / 33** | **ZERO CONTAMINATION** |
| **Synthetic Train $\leftrightarrow$ Real Validation** | **0 / 34** | **0 / 34** | **ZERO CONTAMINATION** |

---

## 3. Class Representation in Locked Test Set

| Breed Name | Species | Real Test Samples | Real Val Samples | Real Train Samples | Synthetic Train | Total Samples |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Dangi** | Cattle | 5 | 6 | 28 | 0 | 39 |
| **Gaolao** | Cattle | 5 | 5 | 23 | 7 | 40 |
| **Gir** | Cattle | 5 | 5 | 20 | 9 | 39 |
| **Ongole** | Cattle | 4 | 4 | 21 | 9 | 38 |
| **Sahiwal** | Cattle | 3 | 3 | 16 | 14 | 36 |
| **Manda** | Buffalo | 5 | 5 | 21 | 9 | 40 |
| **Surti** | Buffalo | 3 | 3 | 14 | 15 | 35 |
| **Toda** | Buffalo | 3 | 3 | 10 | 19 | 35 |

---

## 4. Test Set Freezing Declaration

The 33 real photographs comprising the locked test set are cataloged with their immutable SHA-256 digests in `splits/selected_breeds/locked_test_spec.json`.
- **Condition**: This test partition is strictly withheld from model training, validation, hyperparameter sweeps, architecture comparison, and augmentation tuning.
- **Evaluation Rule**: The test partition will be evaluated exactly once when the final production model is frozen.
