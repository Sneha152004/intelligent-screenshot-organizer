# Phase 7H-G: Benchmark Readiness & Leakage Audit Report

**Date**: 2026-10-03 22:32:36  
**Random Seed**: 42  
**Audit Verdict**: **`READY FOR MODEL BENCHMARKING`**

---

## 1. Executive Summary

The Phase 7H-G benchmark construction has produced two deterministic, leakage-safe, group-aware benchmark manifests:
1. **Primary 5-Class Benchmark**: 135 screenshots across 128 unique canonical duplicate groups (83 Train / 27 Calibration / 25 Validation).
2. **Extended 8-Class Benchmark**: 147 screenshots across 139 unique canonical duplicate groups (89 Train / 30 Calibration / 28 Validation).

Both manifests passed 100% of automated leakage audit checks (zero group overlap, zero MD5 hash overlap, zero screenshot ID overlap, zero filename overlap, zero missing files, zero corrupted files, zero invalid taxonomy classes).

---

## 2. Partition Summary Table

| Benchmark | Total Screenshots | Unique Groups | Train Split | Calib Split | Val Split | Split Proportions | Leakage Verdict |
| :--- | ----: | ----: | ----: | ----: | ----: | :---: | :---: |
| **Primary 5-Class** | **135** | **128** | 83 (61.5%) | 27 (20.0%) | 25 (18.5%) | **61.5% / 20.0% / 18.5%** | **PASSED (0 Leakage)** |
| **Extended 8-Class** | **147** | **139** | 89 (60.5%) | 30 (20.4%) | 28 (19.0%) | **60.5% / 20.4% / 19.0%** | **PASSED (0 Leakage)** |
| **Zero-Shot Probe** | **31** | **31** | — | — | 31 (Probe) | **100% Isolated Probe** | **PASSED (Isolated)** |

---

## 3. Per-Class Partition Distribution

### Primary 5-Class Benchmark:
* **`document_article_view`**: Train 13, Calib 4, Val 4 (Total: 21 items / 21 groups)
* **`travel_booking`**: Train 29, Calib 9, Val 9 (Total: 47 items / 46 groups)
* **`dashboard_settings`**: Train 5, Calib 2, Val 1 (Total: 8 items / 8 groups)
* **`media_player_library`**: Train 17, Calib 6, Val 5 (Total: 28 items / 28 groups)
* **`code_editor_terminal`**: Train 19, Calib 6, Val 6 (Total: 31 items / 25 groups)

### Extended 8-Class Benchmark Additional Classes:
* **`cart_checkout_payment`**: Train 5, Calib 3, Val 2 (Total: 10 items / 9 groups)
* **`messaging_chat`**: Train 1, Calib 0, Val 1 (Total: 2 items / 2 groups)
* **`email_view`**: 0 items (Unrepresented in eligible ground truth)

---

## 4. Class Representation & Split Limitations

1. **`email_view` Limitation**: Possesses 0 eligible human-verified ground-truth items in current dataset; excluded from supervised training.
2. **`messaging_chat` Limitation**: Possesses only 2 eligible items (2 groups); assigned 1 to Train and 1 to Validation (Calibration has 0).
3. **`product_listing` Zero-Shot Probe**: All 31 screenshots are isolated into `phase7hg_zero_shot_probe.csv` for zero-shot evaluation.
