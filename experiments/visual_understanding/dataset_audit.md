# Phase 7A: Visual Screen Understanding Engine - Dataset Audit Report

**Date**: 2026-10-02  
**Target Scope**: Read-only dataset inspection across 836 metadata records and 833 valid screenshot images (`data/screenshots/`).  
**Objective**: Evaluate structural visual diversity, image dimensions, layout patterns, and OCR text density to determine dataset feasibility for training a specialized **Visual Screen Understanding Engine (VSUE)**.

---

## 1. Dataset Scale & Integrity

* **Total Metadata Records**: 836 (`data/metadata_v2.csv`)
* **Valid Readable Image Files**: **833 screenshots**
* **Excluded Zero-Byte Files**: 3 files (`ss_000.jpg`, `ss_347.png`, `ss_614.png`)
* **Metadata Alignment**: 100% of valid image files match metadata records.

---

## 2. Image Dimensions & Aspect Ratio Analysis

| Metric | Minimum | Maximum | Mean / Typical |
| :--- | :---: | :---: | :---: |
| **Width (px)** | 69 px | 1,600 px | 850 px |
| **Height (px)** | 13 px | 3,120 px | 1,650 px |
| **Aspect Ratio (W/H)** | 0.446 | 5.308 | **0.520 (Mobile Portrait)** |

### Orientation Breakdown:
* **Portrait ($H > W$)**: **783 screenshots (94.0%)** – Mobile app UI screens, vertical scroll views, chat logs, product listings.
* **Landscape ($W > H$)**: **50 screenshots (6.0%)** – Desktop IDE windows, web browser views, video players, landscape slides.

---

## 3. Structural Visual Component Analysis

Through OpenCV edge density and contour profile analysis, screenshots were categorized into visual layout archetypes:

1. **Document / Text-Heavy Screens (~69.6% | 580 screens)**:
   * Dominated by horizontal text lines, paragraphs, articles, code lines, or PDF slides.
   * High edge density in text regions; low visual component segmentation.
2. **UI Component-Heavy Screens (~30.4% | 253 screens)**:
   * Characterized by prominent structural UI elements: product cards, search/filter bars, checkout buttons, input text fields, toggle switches, chat bubbles, and modal dialogs.
   * Strong bounding contour groupings and visual grid structures.

---

## 4. OCR Text Density Distribution

OCR text density was measured using raw character and word counts from EasyOCR extractions:

| Density Tier | Definition | Screenshot Count | Percentage | Primary Feature Driver |
| :--- | :--- | :---: | :---: | :--- |
| **Sparse Text** | $< 20$ chars OR $< 5$ words | **37** | **4.4%** | Pure visual icons, blank UIs, toggles, graphics |
| **Medium Text** | $20 - 250$ chars | **730** | **87.6%** | Standard app screens, product cards, chat logs |
| **Dense Text** | $> 250$ chars | **66** | **7.9%** | Long code files, dense articles, documentation |

---

## 5. Decoupling Screen Types from Canonical Categories

> [!IMPORTANT]
> Existing canonical categories (**Work & Career, Shopping, Education, Travel**, etc.) reflect **semantic topic intent** and MUST NOT be used as screen-type ground truth.

### Demonstrative Cross-Category Mapping:

| Existing Category | Observed Screen Types in Category |
| :--- | :--- |
| **Work & Career** | `code_editor_terminal`, `document_article_view`, `dashboard_settings`, `email_view`, `messaging_chat` |
| **Shopping** | `product_listing`, `product_detail`, `cart_checkout_payment`, `dashboard_settings` |
| **Payment** | `cart_checkout_payment`, `login_authentication`, `dashboard_settings` |
| **Connectivity** | `dashboard_settings`, `error_dialog_alert`, `document_article_view` |

---

## 6. Dataset Feasibility Conclusion

* **Feasibility Status**: **SUFFICIENT FOR EXPERIMENTAL PHASE**
* **Conclusion**: The dataset contains 833 screenshots with rich structural diversity across 12 distinct interface layouts.
* **Recommendation**: Proceed with screen-type annotation on the representative 153-screenshot sample using the consolidated taxonomy.
