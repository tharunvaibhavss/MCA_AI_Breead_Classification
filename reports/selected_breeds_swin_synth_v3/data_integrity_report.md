# Data Integrity & Leakage Verification Report

**Generated**: 2026-10-09 00:03:44  
**Auditor**: SHA-256 Cryptographic Hash Verification Engine  

## 1. Cryptographic Isolation Audit

To guarantee scientific validity, all partitions were audited using SHA-256 hash checks to prevent train-test contamination, validation leakage, and synthetic leakage into test sets.

| Check Description | Expected | Measured | Status |
| :--- | :---: | :---: | :---: |
| **Train / Test SHA-256 Overlap** | 0 | **0** | **PASSED** |
| **Validation / Test SHA-256 Overlap** | 0 | **0** | **PASSED** |
| **Train / Validation SHA-256 Overlap** | 0 | **0** | **PASSED** |
| **Synthetic / Test SHA-256 Overlap** | 0 | **0** | **PASSED** |
| **Synthetic / Validation SHA-256 Overlap** | 0 | **0** | **PASSED** |
| **Synthetic / Real Train SHA-256 Overlap** | 0 | **0** | **PASSED** |
| **Locked Test Set Modifications** | 0 | **0** | **PASSED** |
| **Corrupted / Unreadable Images** | 0 | **0** | **PASSED** |

---

## 2. Real vs Synthetic Integrity Notes

- Every real image in train, validation, and locked test is distinct and unique.
- The 4,160 synthetic images on disk contain 832 distinct base renders generated with multiple copies per render; all 4,160 images were incorporated into the training manifest without altering the dataset.
- Real validation (34 images) and locked test (33 images) contain **zero synthetic images**.
