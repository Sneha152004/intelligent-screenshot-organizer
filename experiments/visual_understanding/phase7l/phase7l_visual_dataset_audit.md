# Phase 7L: Visual Dataset Audit Report (13-Class Taxonomy)

## Executive Summary

This report documents an exhaustive audit of the 836 dataset screenshots across all 13 visual screen-type classes based on the authoritative Phase 7H master human review ledger (`phase7he_human_review_ledger.csv`) and Phase 7H-F/7H-G ground truth reconciliation data.

---

## 1. Ground Truth Master Ledger Summary (836 Screenshots)

- **Total Dataset Screenshots**: 836
- **Valid Screenshots**: 833
- **Corrupted Screenshots**: 3 (`ss_441.png`, `ss_443.png`, `ss_560.png` — strictly excluded)
- **Human-Verified Screenshots**: **443** (280 `correct` ledger corrections, 163 `approve` ledger approvals)
- **Pending Human Verification**: 338
- **Rejected Valid Screenshots**: 51
- **Ambiguous Screenshots**: 1

---

## 2. Verified Class Distribution & Canonical Group Breakdown

| Visual Screen Type | Verified Screenshots | Unique Duplicate Groups | Benchmark Split History | 13-Class Training Feasibility |
| :--- | :---: | :---: | :--- | :---: |
| `document_article_view` | 21 | 2 | Primary Benchmark (Class 1) | **LIMITED** |
| `travel_booking` | 47 | 6 | Primary Benchmark (Class 2) | **READY** |
| `dashboard_settings` | 8 | 1 | Primary Benchmark (Class 3) | **LIMITED** |
| `media_player_library` | 28 | 1 | Primary Benchmark (Class 4) | **LIMITED** |
| `code_editor_terminal` | 31 | 7 | Primary Benchmark (Class 5) | **READY** |
| `product_listing` | 31 | 2 | Zero-Shot Probe | **LIMITED** |
| `product_detail` | 27 | 1 | Unsupervised / Single Group | **INSUFFICIENT** |
| `cart_checkout_payment` | 10 | 4 | Extended Benchmark | **INSUFFICIENT** |
| `login_authentication` | 15 | 1 | Single Duplicate Group | **INSUFFICIENT** |
| `messaging_chat` | 2 | 2 | Extremely Low Count | **INSUFFICIENT** |
| `email_view` | 0 | 0 | Empty Class | **INSUFFICIENT (ZERO)** |
| `error_dialog_alert` | 8 | 1 | Single Duplicate Group | **INSUFFICIENT** |
| `other_screen` | 215 | 8 | General Fallback | **LIMITED** |
| **Total** | **443** | **36** | — | — |

---

## 3. Key Dataset Audit Findings

1. **Severe Group Imbalance**: 8 of the 13 classes (`product_detail`, `login_authentication`, `error_dialog_alert`, `media_player_library`, `dashboard_settings`, `cart_checkout_payment`, `messaging_chat`, `email_view`) have 4 or fewer unique canonical duplicate groups.
2. **Leakage Risks**: Splitting classes with 1 or 2 duplicate groups across Train, Calibration, and Validation causes severe data leakage (identical visual templates in train and val).
3. **Audit Verdict for Track B**: **`VISUAL EXPANSION NOT YET READY`**. Training a 13-class supervised vision model at this stage would introduce severe overfitting and group leakage. The 5-class YOLOv8n-cls model remains the validated visual model.
