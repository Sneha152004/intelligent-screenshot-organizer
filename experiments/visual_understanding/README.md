# Phase 7A: Visual Screen Understanding Engine (VSUE) Design & Dataset Preparation

This directory contains the dataset audit, consolidated screen-type taxonomy, annotation guidelines, candidate manifest, visual architecture comparison, and integration design for Phase 7A.

---

## Phase 7A Deliverables

1. [dataset_audit.md](dataset_audit.md): Complete audit of the 833 valid screenshots, dimensions, orientations, and OCR text densities.
2. [screen_type_taxonomy.md](screen_type_taxonomy.md): Consolidated 12-class Screen-Type Taxonomy specification.
3. [annotation_guidelines.md](annotation_guidelines.md): Standardized annotation guidelines and schema rules.
4. [screen_type_manifest.csv](screen_type_manifest.csv): 153-screenshot stratified candidate annotation manifest.
5. [visual_architecture_comparison.md](visual_architecture_comparison.md): Performance and size comparison of candidate visual backbones.
6. [integration_design.md](integration_design.md): Proposed visual output contract and non-overriding fusion architecture.

---

## Phase 7A Final Recommendations

1. **Dataset Sufficiency**: The current dataset of 833 screenshots is **sufficient** to train a lightweight screen-type recognizer across 12 consolidated interface categories.
2. **Retained Screen Types**: Consolidate the 18 initial candidates into **12 distinct screen types** (`product_listing`, `product_detail`, `cart_checkout_payment`, `login_authentication`, `messaging_chat`, `email_view`, `media_player_library`, `code_editor_terminal`, `document_article_view`, `travel_booking`, `dashboard_settings`, `error_dialog_alert`).
3. **Recommended Visual Encoder**: **MobileNetV3-Small** (with dynamic INT8 quantization) or a **Hybrid MobileNetV3 + OpenCV Layout Feature Extractor**.
4. **Expected Model Storage**: **1.05 MB (INT8 Quantized)** to **4.11 MB (FP32)**.
5. **Application Footprint Budget**: The complete application stays comfortably within **286–300 MB RAM**, strictly respecting the **100–300 MB** requirement.
