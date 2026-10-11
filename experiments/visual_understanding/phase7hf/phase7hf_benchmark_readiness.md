# Phase 7H-F: Visual Screen-Type Benchmark Readiness Report

**Date**: 2026-10-03 22:22:48  
**Corpus Evaluated**: 443 Human-Verified Ground-Truth Screenshots  
**Audit Verdict**: **`READY WITH CLASS RESTRICTIONS`**

---

## 1. Executive Summary & Verdict Rationale

The audit confirms that the **443 human-verified ground-truth screenshots** across **8 populated visual screen types** are **READY WITH CLASS RESTRICTIONS** for supervised visual model benchmarking.

### Verdict Justification:
1. **Sufficient Support for Core Head Classes**: 5 major screen types (`document_article_view`, `travel_booking`, `dashboard_settings`, `media_player_library`, `code_editor_terminal`) possess $\ge 45$ independent duplicate groups each (totaling 396 screenshots), enabling high-confidence group-aware train/calibration/validation splitting.
2. **Controlled Evaluation for Secondary Classes**: 3 secondary classes (`email_view`, `cart_checkout_payment`, `messaging_chat`) possess between 8 and 24 independent groups (totaling 45 screenshots), suitable for stratified evaluation.
3. **Class Restrictions Required**: `product_listing` has only 2 independent groups (2 screenshots) and must be excluded from split validation to prevent zero-train or zero-test splits. 4 taxonomy classes have zero human-verified examples in the current 443 subset (`product_detail`, `login_authentication`, `error_dialog_alert`, `other_screen`).
4. **Duplicate Group Harmonization**: 6 duplicate groups contain conflicting human decisions across items within the same duplicate group (e.g. singleton IDs assigned `singleton_<id>` vs shared hash group). Group-aware splitting must group by MD5 image hash / canonical duplicate group ID to prevent train/test leakage.

---

## 2. Benchmark Experiment Configuration Recommendations

### Recommended Configuration: 5-Head / 8-Populated Staged Benchmark

* **Primary 5-Head Benchmark** (Recommended for Initial Baseline):
  * Classes: `document_article_view`, `travel_booking`, `dashboard_settings`, `media_player_library`, `code_editor_terminal`.
  * Total Examples: 396 screenshots across 394 independent duplicate groups.
  * Split Ratio: Group-aware 60% Train (237) / 20% Calib (79) / 20% Val (80).
* **Extended 8-Class Benchmark** (Secondary Evaluation):
  * Classes: Primary 5 + `email_view` (24), `cart_checkout_payment` (13), `messaging_chat` (8).
  * Total Examples: 441 screenshots across 437 independent duplicate groups.
  * Group-aware splitting enforced with minimum 1 item per fold.
* **Excluded Classes**:
  * `product_listing` (2 items / 2 groups): Reserved as zero-shot/unseen evaluation probe.
  * 4 empty taxonomy classes: Documented as unrepresented in current human-verified subset.

---

## 3. Group-Aware Leakage Safeguards

* **Duplicate Group Splitting Policy**: GroupKFold / Group-aware train-test-split MUST be used to ensure no duplicate group is split across training, calibration, or validation sets.
* **Conflict Resolution**: Any duplicate group with multiple items must be kept intact in the same dataset split.
