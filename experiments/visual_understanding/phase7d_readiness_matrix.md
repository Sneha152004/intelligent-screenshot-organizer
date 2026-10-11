# Phase 7D: Training Readiness Matrix

**Date**: 2026-10-02  
**Purpose**: Per-class evidence-based readiness classification for initial VSUE supervised model training.

---

## Readiness Matrix Across Canonical 13 Screen Types

| Screen Type | Training Eligibility | Total Count | Train / Cal / Val | Strengths & Limitations | Recommended Next Action |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **`document_article_view`** | **TRAINABLE_CANDIDATE** | 610 | 366 / 122 / 122 | High sample count; diverse layouts (slides, text, articles). | Include in training with downweighted class loss. |
| **`product_detail`** | **TRAINABLE_CANDIDATE** | 134 | 81 / 27 / 26 | Strong visual components (hero image, price badge, buy CTA). | Include in training. |
| **`code_editor_terminal`** | **TRAINABLE_CANDIDATE** | 33 | 19 / 7 / 7 | Distinct visual features (line numbers, syntax highlighting). | Include in training with loss weighting. |
| **`dashboard_settings`** | **TRAINABLE_CANDIDATE** | 30 | 18 / 6 / 6 | Distinct toggles and list components. | Include in training with loss weighting. |
| **`cart_checkout_payment`** | **LIMITED_EXPERIMENTAL** | 26 | 16 / 4 / 6 | Small validation count (6 val screens). | Include as experimental head; flag metrics. |
| **`product_listing`** | **UNSUPPORTED_NO_DATA** | 0 | 0 / 0 / 0 | No examples present in corpus. | Retain class in taxonomy; omit from loss. |
| **`login_authentication`** | **UNSUPPORTED_NO_DATA** | 0 | 0 / 0 / 0 | No examples present in corpus. | Retain class in taxonomy; omit from loss. |
| **`messaging_chat`** | **UNSUPPORTED_NO_DATA** | 0 | 0 / 0 / 0 | No examples present in corpus. | Retain class in taxonomy; omit from loss. |
| **`email_view`** | **UNSUPPORTED_NO_DATA** | 0 | 0 / 0 / 0 | No examples present in corpus. | Retain class in taxonomy; omit from loss. |
| **`media_player_library`** | **UNSUPPORTED_NO_DATA** | 0 | 0 / 0 / 0 | No examples present in corpus. | Retain class in taxonomy; omit from loss. |
| **`travel_booking`** | **UNSUPPORTED_NO_DATA** | 0 | 0 / 0 / 0 | No examples present in corpus. | Retain class in taxonomy; omit from loss. |
| **`error_dialog_alert`** | **UNSUPPORTED_NO_DATA** | 0 | 0 / 0 / 0 | No examples present in corpus. | Retain class in taxonomy; omit from loss. |
| **`other_screen`** *(Fallback)*| **UNSUPPORTED_NO_DATA** | 0 | 0 / 0 / 0 | No unclassified fallback screens remaining. | Retain class in taxonomy; omit from loss. |
