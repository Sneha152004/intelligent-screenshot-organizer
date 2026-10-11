# Phase 7D: Visual Screen-Type Dataset Integrity & Training-Readiness Audit Report

**Date**: 2026-10-02  
**Audit Scope**: Independent, read-only dataset integrity audit across all 833 valid screenshots, Phase 7B preservation, MD5 hash duplicate groups, group-aware 60/20/20 partition isolation (`vsue_dataset_split_v2.json`), and training readiness.  
**Audit Mode**: STRICTLY READ-ONLY (No code, manifest, or benchmark modifications).

---

## Executive Summary & Final Verdict

* **Final Readiness Verdict**: **READY WITH LIMITATIONS**
* **Manifest Integrity**: **100% Verified** (Exactly 833 valid screenshot records; 0 missing images, 0 unreadable files, 0 non-canonical labels).
* **Phase 7B Preservation**: **100% Verified** (All 153 human-verified Phase 7B annotations preserved with 0 label alterations).
* **Duplicate & Leakage Audit**: **0.0% Leakage Verified** (39 MD5 binary-identical hash groups representing 78 files; 0 SID or MD5 hash overlap across Train, Calibration, and Validation splits).
* **Partition Split**: **500 Train (60.0%) / 166 Calibration (19.9%) / 167 Validation (20.0%)** (Total = 833 images).
* **Training Eligibility**:
  * **`TRAINABLE_CANDIDATE` (4 Classes)**: `document_article_view` (610), `product_detail` (134), `code_editor_terminal` (33), `dashboard_settings` (30).
  * **`LIMITED_EXPERIMENTAL` (1 Class)**: `cart_checkout_payment` (26).
  * **`UNSUPPORTED_NO_DATA` (8 Classes)**: `product_listing`, `login_authentication`, `messaging_chat`, `email_view`, `media_player_library`, `travel_booking`, `error_dialog_alert`, `other_screen`.
* **Identified Integrity Issues**: **18 duplicate binary image pairs** flagged in `phase7d_integrity_issues.csv` due to candidate label conflicts between identical image files (e.g. `ss_001.jpg` vs `ss_799.jpg`), requiring human harmonization before training.

---

## 1. Audit A: Manifest Integrity & Phase 7B Preservation

| Integrity Metric | Requirement | Actual Audit Finding | Audit Status |
| :--- | :---: | :---: | :--- |
| **Total Manifest Records** | 833 | **833** | **PASSED** |
| **Screenshot ID Uniqueness** | 833 Unique IDs | **833 Unique IDs (0 dupes)** | **PASSED** |
| **Image File Existence** | 833 Files on Disk | **833 Files (0 missing)** | **PASSED** |
| **Image File Readability** | 833 Valid Images | **833 Valid Images (0 unreadable)** | **PASSED** |
| **Canonical Taxonomy Mapping** | 13 Canonical Classes | **13 Canonical Classes (0 invalid)** | **PASSED** |
| **Phase 7B Label Preservation** | 153 Preserved | **153 Preserved (0 modified)** | **PASSED** |

---

## 2. Audit B: Duplicate & Leakage Integrity

An independent MD5 hash calculation was executed across all 833 image binary files:

* **Total Image Binary Files**: `833`
* **Total Unique MD5 Hashes**: `794`
* **Total Duplicate Hash Groups**: `39 groups`
* **Total Files in Duplicate Groups**: `78 files` (Each group contains exactly 2 identical binary files).
* **Cross-Partition SID Leakage**: **0**
* **Cross-Partition MD5 Hash Leakage**: **0**
* **Verification Status**: All 39 duplicate groups are assigned **atomically to the same partition split**, guaranteeing 0 contamination between training, calibration, and validation.
* **Artifact Generated**: [phase7d_duplicate_verification.csv](phase7d_duplicate_verification.csv).

---

## 3. Audit C: Dataset Split Integrity & Distribution Matrix

| Screen Type Label | Total Count | Train Split (60%) | Cal Split (20%) | Val Split (20%) | Training Eligibility |
| :--- | ----: | ----: | ----: | ----: | :--- |
| **`document_article_view`** | 610 | 366 | 122 | 122 | **TRAINABLE_CANDIDATE** |
| **`product_detail`** | 134 | 81 | 27 | 26 | **TRAINABLE_CANDIDATE** |
| **`code_editor_terminal`** | 33 | 19 | 7 | 7 | **TRAINABLE_CANDIDATE** |
| **`dashboard_settings`** | 30 | 18 | 6 | 6 | **TRAINABLE_CANDIDATE** |
| **`cart_checkout_payment`** | 26 | 16 | 4 | 6 | **LIMITED_EXPERIMENTAL** |
| **`product_listing`** | 0 | 0 | 0 | 0 | **UNSUPPORTED_NO_DATA** |
| **`login_authentication`** | 0 | 0 | 0 | 0 | **UNSUPPORTED_NO_DATA** |
| **`messaging_chat`** | 0 | 0 | 0 | 0 | **UNSUPPORTED_NO_DATA** |
| **`email_view`** | 0 | 0 | 0 | 0 | **UNSUPPORTED_NO_DATA** |
| **`media_player_library`** | 0 | 0 | 0 | 0 | **UNSUPPORTED_NO_DATA** |
| **`travel_booking`** | 0 | 0 | 0 | 0 | **UNSUPPORTED_NO_DATA** |
| **`error_dialog_alert`** | 0 | 0 | 0 | 0 | **UNSUPPORTED_NO_DATA** |
| **`other_screen`** *(Fallback)* | 0 | 0 | 0 | 0 | **UNSUPPORTED_NO_DATA** |
| **Total Dataset** | **833** | **500** | **166** | **167** | **5 Active Classes** |

---

## 4. Audit D & E: Annotation Quality & Integrity Issues

* **High Confidence Annotations**: 332 / 833 (39.9%)
* **Medium Confidence Annotations**: 501 / 833 (60.1%)
* **Ambiguous Annotations**: 3 screenshots (`ss_080.jpg`, `ss_283.jpg`, `ss_424.jpg`).
* **Identified Integrity Issues (18 Duplicate Hash Conflicts)**:
  * In the expanded dataset, 18 duplicate binary image pairs (e.g., `ss_001.jpg` and `ss_799.jpg`) were assigned conflicting candidate labels (`code_editor_terminal` vs `document_article_view`).
  * **Action Required**: Before training, run a harmonization pass to ensure binary-identical image files share identical ground-truth labels.
  * **Artifact Generated**: [phase7d_integrity_issues.csv](phase7d_integrity_issues.csv).

---

## 5. Audit F: Dataset Composition & Generalization Risks

1. **Document Class Dominance Risk**: `document_article_view` accounts for **73.23%** (610/833) of the dataset. Models trained without class-weighted loss functions will suffer severe majority-class bias.
2. **Natural User Corpus Pattern**: The absence of 8 classes (`messaging_chat`, `email_view`, `travel_booking`, etc.) reflects the user's specific collection habits (primarily saving articles, code snippets, product details, and receipts).

---

## 6. Final Readiness Verdict & Action Plan

### Verdict: **READY WITH LIMITATIONS**

1. **Approved Scope**: An initial supervised experiment for VSUE can proceed on the **4 TRAINABLE_CANDIDATE classes** (`document_article_view`, `product_detail`, `code_editor_terminal`, `dashboard_settings`) plus **1 LIMITED_EXPERIMENTAL class** (`cart_checkout_payment`).
2. **Prerequisite Harmonization**: Harmonize the 18 duplicate binary hash label conflicts before executing model training.
3. **Class-Weighted Loss**: Use class-balanced cross-entropy loss weights to prevent majority-class over-prediction.
