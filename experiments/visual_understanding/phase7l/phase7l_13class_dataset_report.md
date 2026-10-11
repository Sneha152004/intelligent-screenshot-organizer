# Phase 7L: 13-Class Visual Dataset Feasibility & Split Proposal Report

## Executive Summary

This report delivers the quantitative dataset analysis and feasibility evaluation for expanding visual screen-type supervised classification to the full 13-class taxonomy.

---

## 1. Class Availability & Feasibility Status

| Visual Screen Type | Verified Screenshots | Unique Canonical Groups | Feasibility Status | Feasibility Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **`travel_booking`** | 47 | **6** | **READY** | \(\ge 5\) unique groups; supports leakage-free train/calib/val splits. |
| **`code_editor_terminal`** | 31 | **7** | **READY** | \(\ge 5\) unique groups; supports leakage-free train/calib/val splits. |
| **`other_screen`** | 215 | **8** | **READY** | \(\ge 5\) unique groups; general fallback representation. |
| **`product_listing`** | 31 | 2 | **LIMITED** | 2 unique groups; insufficient for full 3-way split. |
| **`document_article_view`** | 21 | 2 | **LIMITED** | 2 unique groups; insufficient for full 3-way split. |
| **`messaging_chat`** | 2 | 2 | **LIMITED** | 2 unique groups (only 2 total samples in dataset). |
| **`product_detail`** | 27 | 1 | **INSUFFICIENT** | Single duplicate group; severe train/val template leakage. |
| **`login_authentication`** | 15 | 1 | **INSUFFICIENT** | Single duplicate group; severe train/val template leakage. |
| **`cart_checkout_payment`** | 10 | 4 | **INSUFFICIENT** | 4 unique groups; insufficient balance for 3-way split. |
| **`error_dialog_alert`** | 8 | 1 | **INSUFFICIENT** | Single duplicate group; severe train/val template leakage. |
| **`dashboard_settings`** | 8 | 1 | **INSUFFICIENT** | Single duplicate group; severe train/val template leakage. |
| **`media_player_library`** | 28 | 1 | **INSUFFICIENT** | Single duplicate group; severe train/val template leakage. |
| **`email_view`** | 0 | 0 | **INSUFFICIENT (ZERO)** | 0 examples in entire dataset. |

---

## 2. 5-Class Validated Baseline vs 13-Class Feasibility Comparison

- **Current Validated Model (`YOLOv8n-cls-5class`)**: Uses the frozen 5-class primary benchmark (`code_editor_terminal`, `travel_booking`, `document_article_view`, `dashboard_settings`, `media_player_library`). Validated in Phase 7K with **+20.00% multimodal accuracy gain**.
- **13-Class Training Recommendation**: **DO NOT RETRAIN YET.** Only 3 of the 13 classes meet the strict READY threshold (\(\ge 5\) unique duplicate groups). Retraining a 13-class model at this stage would cause catastrophic overfitting on single-group classes.

---

## 3. Recommended Roadmap for Future Visual Expansion (Phase 7M)

1. Collect and annotate targeted screenshot batches for underrepresented classes (`email_view`, `messaging_chat`, `cart_checkout_payment`, `login_authentication`, `error_dialog_alert`).
2. Require a minimum of **5 unique canonical duplicate groups per class** before authorizing a 13-class supervised vision model training.
3. Preserve `YOLOv8n-cls-5class` as the active validated visual model until a 13-class model passes independent benchmarking.
