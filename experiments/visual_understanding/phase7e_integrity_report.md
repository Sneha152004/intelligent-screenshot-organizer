# Phase 7E: Dataset Integrity & Finalization Report

**Date**: 2026-10-02  
**Final Completion Verdict**: **READY WITH DOCUMENTED LIMITATIONS**  
**Primary Dataset Artifact**: `experiments/visual_understanding/screen_type_manifest_final.csv` (833 records)  
**Partition Split Artifact**: `experiments/visual_understanding/vsue_dataset_split_final.json`

---

## 1. Before vs. After Harmonization Audit

| Metric | Phase 7D Baseline | Phase 7E Finalized Dataset | Status |
| :--- | :---: | :---: | :--- |
| **Total Screenshot Records** | 833 | **833** | **100% Preserved** |
| **Unique Image Hashes** | 794 | **794** | **100% Verified** |
| **Duplicate Binary Groups** | 39 | **39** | **100% Accounted** |
| **Duplicate Group Primary Label Conflicts** | **18** | **0** | **100% Resolved** |
| **Harmonized Screenshot Records** | 0 | **36** | **All 18 Pairs Unified** |
| **Cross-Partition SID / Hash Leakage** | **0** | **0** | **100% Isolated** |

---

## 2. Final Per-Partition Class Distribution Matrix

| Screen Type Label | Total Count | Train Split (60.3%) | Cal Split (19.4%) | Val Split (20.3%) | Training Eligibility Status |
| :--- | ----: | ----: | ----: | ----: | :--- |
| **`document_article_view`** | 602 | 363 | 118 | 121 | **`TRAINABLE_CANDIDATE`** |
| **`product_detail`** | 137 | 82 | 26 | 29 | **`TRAINABLE_CANDIDATE`** |
| **`code_editor_terminal`** | 48 | 30 | 8 | 10 | **`TRAINABLE_CANDIDATE`** |
| **`dashboard_settings`** | 27 | 16 | 6 | 5 | **`LIMITED_EXPERIMENTAL`** |
| **`cart_checkout_payment`** | 19 | 11 | 4 | 4 | **`LIMITED_EXPERIMENTAL`** |
| **`product_listing`** | 0 | 0 | 0 | 0 | **`UNSUPPORTED_NO_DATA`** |
| **`login_authentication`** | 0 | 0 | 0 | 0 | **`UNSUPPORTED_NO_DATA`** |
| **`messaging_chat`** | 0 | 0 | 0 | 0 | **`UNSUPPORTED_NO_DATA`** |
| **`email_view`** | 0 | 0 | 0 | 0 | **`UNSUPPORTED_NO_DATA`** |
| **`media_player_library`** | 0 | 0 | 0 | 0 | **`UNSUPPORTED_NO_DATA`** |
| **`travel_booking`** | 0 | 0 | 0 | 0 | **`UNSUPPORTED_NO_DATA`** |
| **`error_dialog_alert`** | 0 | 0 | 0 | 0 | **`UNSUPPORTED_NO_DATA`** |
| **`other_screen`** *(Fallback)* | 0 | 0 | 0 | 0 | **`UNSUPPORTED_NO_DATA`** |
| **Total Finalized Dataset** | **833** | **502** | **162** | **169** | **5 Active Classes** |

---

## 3. Data Leakage & Partition Isolation Verification

* **Train Partition**: **502 screenshots** (60.3%)
* **Calibration Partition**: **162 screenshots** (19.4%)
* **Validation Partition**: **169 screenshots** (20.3%)
* **SID Leakage Count**: **0**
* **MD5 Hash Leakage Count**: **0**
* **Verification**: All 39 duplicate binary groups are assigned **atomically to the same partition split**, guaranteeing 0 contamination between training, calibration, and validation.

---

## 4. Final Training Eligibility Assessment

1. **`TRAINABLE_CANDIDATE` (3 Classes)**: `document_article_view` (602), `product_detail` (137), `code_editor_terminal` (48). These classes have sufficient training, calibration, and validation support to train and evaluate an initial supervised model.
2. **`LIMITED_EXPERIMENTAL` (2 Classes)**: `dashboard_settings` (27) and `cart_checkout_payment` (19). These classes have smaller sample counts and should be evaluated as experimental heads.
3. **`UNSUPPORTED_NO_DATA` (8 Classes)**: Naturally absent from this user's dataset corpus. They remain in the taxonomy but will be omitted from loss evaluation in initial training.
