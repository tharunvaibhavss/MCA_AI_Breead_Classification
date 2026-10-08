# Data Integrity & Split Verification Report (Selected Breeds V4)

**Generated**: 2026-10-07 20:47:33  
**Project**: AI-Breed-Recognition-Selected-Breeds  
**Cohort**: Selected 8-Breed High-Accuracy Campaign Version 4  

---

## 1. Partition Inventory

| Partition | Real Images | Synthetic Images | Total Images | Partition Purity | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Training (V4)** | **153** | **4160** | **4313** | Complete Dataset Pool (Real + Complete Synthetic) | **VERIFIED** |
| **Validation** | **34** | **0** | **34** | **100% Real Photographs Only** | **STRICTLY ISOLATED** |
| **Locked Test** | **33** | **0** | **33** | **100% Real Photographs Only** | **FROZEN & IMMUTABLE** |
| **Total Active Dataset** | **220** | **4160** | **4380** | — | — |

---

## 2. Cryptographic Leakage & Isolation Audit

| Audit Check | Overlap Count | Result | Integrity Status |
| :--- | :---: | :---: | :--- |
| **Synthetic / Locked Test SHA-256 Overlap** | **0 / 33** | **0** | **ZERO CONTAMINATION (PASSED)** |
| **Synthetic / Validation SHA-256 Overlap** | **0 / 34** | **0** | **ZERO CONTAMINATION (PASSED)** |
| **Real Train / Locked Test SHA-256 Overlap** | **0 / 33** | **0** | **ZERO LEAKAGE (PASSED)** |
| **Real Train / Validation SHA-256 Overlap** | **0 / 34** | **0** | **ZERO LEAKAGE (PASSED)** |
| **Validation / Locked Test SHA-256 Overlap** | **0 / 33** | **0** | **ZERO LEAKAGE (PASSED)** |
| **Locked Test Manifest Integrity Check** | **0** | **0** | **IDENTICAL & UNCHANGED (PASSED)** |

---

## 3. Training Partition Composition by Breed

| Breed Name | Species | Real Train Photos | New Synthetic | Legacy SYNTH_ | Total Synthetic | Total V4 Training Pool |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Dangi** | Cattle | 28 | 500 | 0 | 500 | **528** |
| **Gaolao** | Cattle | 23 | 500 | 5 | 505 | **528** |
| **Gir** | Cattle | 20 | 500 | 35 | 535 | **555** |
| **Ongole** | Cattle | 21 | 500 | 20 | 520 | **541** |
| **Sahiwal** | Cattle | 16 | 500 | 70 | 570 | **586** |
| **Manda** | Buffalo | 21 | 500 | 0 | 500 | **521** |
| **Surti** | Buffalo | 14 | 500 | 10 | 510 | **524** |
| **Toda** | Buffalo | 10 | 500 | 20 | 520 | **530** |

**Total V4 Training Pool**: **4313 images** (153 Real + 4160 Synthetic)
