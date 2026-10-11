# Additional Data Requirements Report

**Date**: 2026-10-02  
**Purpose**: Outline sampling requirements to address class imbalance in the visual screen-type dataset.

---

## 1. Current Representation Analysis (153-Screenshot Manifest)

| Screen Type Label | Sampled Count | Percentage | Representation Status |
| :--- | :---: | :---: | :--- |
| `document_article_view` | 106 | 69.28% | **Sufficient** |
| `product_detail` | 17 | 11.11% | **Sufficient** |
| `cart_checkout_payment` | 17 | 11.11% | **Sufficient** |
| `dashboard_settings` | 9 | 5.88% | **Low** (Needs +15 samples) |
| `code_editor_terminal` | 4 | 2.61% | **Low** (Needs +20 samples) |
| `product_listing` | 0 | 0.00% | **Critical Sparse** (Needs +25 samples) |
| `login_authentication` | 0 | 0.00% | **Critical Sparse** (Needs +15 samples) |
| `messaging_chat` | 0 | 0.00% | **Critical Sparse** (Needs +25 samples) |
| `email_view` | 0 | 0.00% | **Critical Sparse** (Needs +20 samples) |
| `media_player_library` | 0 | 0.00% | **Critical Sparse** (Needs +20 samples) |
| `travel_booking` | 0 | 0.00% | **Critical Sparse** (Needs +20 samples) |
| `error_dialog_alert` | 0 | 0.00% | **Critical Sparse** (Needs +15 samples) |

---

## 2. Recommended Expansion Plan

1. **Expand Manifest Sampling**: The current 153-image manifest represents ~18% of our 833 total screenshots. We should expand manifest annotations across the remaining **680 screenshots** in `data/screenshots/`.
2. **Targeted Sampling**: Filter screenshots matching tags `chat`, `email`, `login`, `video`, `flight`, `grid`, and `settings` to populate under-represented classes prior to training.
