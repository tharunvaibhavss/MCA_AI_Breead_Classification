# Data Integrity & Split Verification Report (Selected Breeds V2)

**Generated**: 2026-10-06 18:21:25  
**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Model Cohort**: Selected 8-Breed High-Accuracy Version 2  

---

## 1. Split Inventory

| Partition | Real Images | Synthetic Images | Total Images | Partition Purity | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Training** | 153 | 800 | **953** | 153 Real + 800 Controlled Synthetic | **VERIFIED** |
| **Validation** | 34 | 0 | **34** | **100% Real Photographs Only** | **STRICTLY ISOLATED** |
| **Locked Test** | 33 | 0 | **33** | **100% Real Photographs Only** | **FROZEN & IMMUTABLE** |
| **Total Master** | **220** | **800** | **1,020** | — | — |

---

## 2. Leakage & Overlap Audit

| Check Description | Overlap Count | Result |
| :--- | :---: | :--- |
| **Train / Locked Test Cryptographic SHA-256 Overlap** | **0 / 33** | **ZERO LEAKAGE (PASSED)** |
| **Validation / Locked Test SHA-256 Overlap** | **0 / 33** | **ZERO LEAKAGE (PASSED)** |
| **Train / Validation SHA-256 Overlap** | **0 / 34** | **ZERO LEAKAGE (PASSED)** |
| **Synthetic / Locked Test Overlap** | **0 / 33** | **ZERO CONTAMINATION (PASSED)** |
| **Synthetic / Validation Overlap** | **0 / 34** | **ZERO CONTAMINATION (PASSED)** |
| **Locked Test vs Test Partition Discrepancy** | **0** | **IDENTICAL & UNCHANGED** |

---

## 3. Training Partition Composition by Breed

| Breed | Species | Real Train Samples | Synthetic Train Samples (New) | Total Training Samples |
| :--- | :--- | :---: | :---: | :---: |
| **Dangi** | Cattle | 28 | 100 | **128** |
| **Gaolao** | Cattle | 23 | 100 | **123** |
| **Gir** | Cattle | 20 | 100 | **120** |
| **Ongole** | Cattle | 21 | 100 | **121** |
| **Sahiwal** | Cattle | 16 | 100 | **116** |
| **Manda** | Buffalo | 21 | 100 | **121** |
| **Surti** | Buffalo | 14 | 100 | **114** |
| **Toda** | Buffalo | 10 | 100 | **110** |

**Total Training Pool**: **953** images  
*Note: The 32 legacy SYNTH_ files remain excluded from training to guarantee exact 100-sample synthetic class balance.*
