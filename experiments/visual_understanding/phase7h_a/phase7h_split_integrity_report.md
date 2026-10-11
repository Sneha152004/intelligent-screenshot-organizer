# Phase 7H-A: Group-Aware Dataset Split Integrity Report

**Date**: 2026-10-02  
**Target Split Artifact**: `experiments/visual_understanding/phase7h_a/vsue_dataset_split_phase7h.json`  
**Total Verified Screenshots**: 833 Images  
**Random Seed**: 42  
**Split Ratios**: 60% Train / 20% Calibration / 20% Validation

---

## 1. Executive Summary & Partition Summary

This report documents the verification of the group-aware dataset split created for Phase 7H-A. The split ensures strict partition isolation, zero data leakage (by Screenshot ID or MD5 hash), and atomic assignment of all binary duplicate groups to single partitions.

```
                         Group-Aware Dataset Partition Split
                         +-----------------------------------+
                         |  833 Verified Screenshots         |
                         +-----------------+-----------------+
                                           |
         +---------------------------------+---------------------------------+
         |                                 |                                 |
+--------v-----------------------+ +-------v-----------------------+ +-------v-----------------------+
|  Train Partition               | |  Calibration Partition        | |  Validation Partition         |
|  502 Images (60.3%)            | |  162 Images (19.4%)            | |  169 Images (20.3%)            |
|  23 Duplicate Groups (46 imgs) | |  7 Duplicate Groups (14 imgs)  | |  9 Duplicate Groups (18 imgs)  |
|  456 Singletons                | |  148 Singletons               | |  151 Singletons               |
+--------------------------------+ +-------------------------------+ +-------------------------------+
```

---

## 2. Per-Partition Class Distribution Matrix

| Screen Type Label | Total Verified | Train Partition (60.3%) | Calibration Partition (19.4%) | Validation Partition (20.3%) | Stratification Status |
| :--- | ----: | ----: | ----: | ----: | :--- |
| **`document_article_view`** | 602 | 363 (60.3%) | 118 (19.6%) | 121 (20.1%) | **STRATIFIED** |
| **`product_detail`** | 137 | 82 (59.9%) | 26 (19.0%) | 29 (21.2%) | **STRATIFIED** |
| **`code_editor_terminal`** | 48 | 30 (62.5%) | 8 (16.7%) | 10 (20.8%) | **STRATIFIED** |
| **`dashboard_settings`** | 27 | 16 (59.3%) | 6 (22.2%) | 5 (18.5%) | **STRATIFIED** |
| **`cart_checkout_payment`** | 19 | 11 (57.9%) | 4 (21.1%) | 4 (21.1%) | **STRATIFIED** |
| *8 Remaining Classes* | 0 | 0 | 0 | 0 | Unsupported |
| **Total Verified Split** | **833** | **502 (60.3%)** | **162 (19.4%)** | **169 (20.3%)** | **100% ISOLATED** |

---

## 3. Data Leakage & Group Isolation Verification

| Audit Check | Test Description | Audit Outcome | Verdict |
| :--- | :--- | :---: | :---: |
| **SID Isolation** | Intersection of Train, Cal, and Val Screenshot IDs | **0 IDs** | **PASSED** |
| **MD5 Hash Isolation** | Intersection of Train, Cal, and Val MD5 Checksums | **0 Hashes** | **PASSED** |
| **Duplicate Group Atomic Assignment** | Verification that duplicate group members belong to a single partition | **0 Groups Split** | **PASSED** |
| **Unverified / Corrupted Exclusion** | Confirmation that 0-byte corrupted files (`ss_441`, `ss_443`, `ss_560`) are excluded | **3 Excluded** | **PASSED** |

---

## 4. Documented Limitations & Multi-Split Constraints

1. **Classes with Zero Data**: The 8 unsupported classes cannot be evaluated in the dataset split until team collection provides candidate screenshots.
2. **Low-Support Class Splitting**: `cart_checkout_payment` (19 total samples) produces small validation partitions (4 samples). While group-aware isolation is maintained, future training experiments should expand this class to support robust multi-fold validation.
