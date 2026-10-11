# Phase 7G: Comprehensive Error Analysis & Failure Mode Taxonomy

**Date**: 2026-10-02  
**Target Model**: MobileNetV3-Small Baseline (Phase 7F Checkpoint)  
**Validation Partition**: 169 Held-Out Validation Screenshots  
**Total Errors Audited**: Exactly 21 Instances (87.57% Top-1 Accuracy)

---

## 1. Executive Summary

An in-depth qualitative and quantitative error analysis was conducted on all 21 misclassified validation screenshots. The objective was to categorize the failure modes, assess the impact of class imbalance, identify visual ambiguity patterns, and provide actionable recommendations for downstream OCR + Visual multimodal fusion.

### Summary of Errors by Ground-Truth Screen Type

| Ground-Truth Screen Type | Validation Support | Correct Predictions | Errors Count | Per-Class Recall | Dominant Error Destination |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`document_article_view`** | 121 | 115 | 6 | **95.04%** | `code_editor_terminal` (3), `product_detail` (3) |
| **`product_detail`** | 29 | 21 | 8 | **72.41%** | `document_article_view` (8) |
| **`code_editor_terminal`** | 10 | 7 | 3 | **70.00%** | `document_article_view` (3) |
| **`dashboard_settings`** | 5 | 5 | 0 | **100.00%** | None (Zero errors) |
| **`cart_checkout_payment`** | 4 | 0 | 4 | **0.00%** | `document_article_view` (3), `product_detail` (1) |
| **Total Validation Split** | **169** | **148** | **21** | **87.57%** | `document_article_view` (14 / 21 errors) |

---

## 2. Failure Mode Taxonomy & Qualitative Breakdown

All 21 error instances fall into three primary failure categories:

```
                  +----------------------------------------------+
                  |  21 Validation Error Instances (100.0%)       |
                  +-----------------------+----------------------+
                                          |
         +--------------------------------+--------------------------------+
         |                                |                                |
+--------v----------------------+ +-------v----------------------+ +-------v----------------------+
|  Dominant Class Overbias      | |  OCR-Required Disambiguation | |  Domain Boundary Overlap     |
|  14 Errors (66.7%)            | |  6 Errors (28.6%)            | |  1 Error (4.8%)              |
+-------------------------------+ +------------------------------+ +------------------------------+
```

### Category 1: Dominant Class Overbias (14 Errors / 66.7%)

* **Description**: The visual model predicts the majority class (`document_article_view`) when the true screen type is a minority class (`product_detail`, `code_editor_terminal`, or `cart_checkout_payment`).
* **Underlying Cause**: Despite applying inverse class weighting during training, the overwhelming frequency of `document_article_view` in the dataset (~70% of all images) creates a strong structural bias toward text-heavy layouts.
* **Affected Screenshots**:
  * `product_detail` $\rightarrow$ `document_article_view` (8 instances): `ss_020.jpg`, `ss_022.jpg`, `ss_164.jpg`, `ss_182.png`, `ss_369.png`, `ss_450.png`, `ss_818.jpg`, `ss_820.jpg`.
  * `code_editor_terminal` $\rightarrow$ `document_article_view` (3 instances): `ss_015.jpg`, `ss_070.jpg`, `ss_813.jpg`.
  * `cart_checkout_payment` $\rightarrow$ `document_article_view` (3 instances): `ss_155.jpg`, `ss_353.png`, `ss_357.png`.

### Category 2: OCR-Required Disambiguation & Visual Layout Ambiguity (6 Errors / 28.6%)

* **Description**: Screenshots where purely visual features (fonts, margins, lines, colors) are insufficient to distinguish document text from code editor views or product listings.
* **Underlying Cause**: Documents containing monospaced code snippets, structured tables, or inline product image thumbnails visually mimic code editors or e-commerce detail pages.
* **Affected Screenshots**:
  * `document_article_view` $\rightarrow$ `code_editor_terminal` (3 instances): `ss_033.jpg`, `ss_058.jpg`, `ss_065.jpg` (Text formatting with dark themes or monospaced font headers).
  * `document_article_view` $\rightarrow$ `product_detail` (3 instances): `ss_105.jpg`, `ss_173.jpg`, `ss_388.jpg` (Articles displaying prominent top hero images or product review callouts).

### Category 3: Domain Boundary Overlap (1 Error / 4.8%)

* **Description**: Misclassification between visually adjacent UI domains.
* **Underlying Cause**: Cart and checkout screens display item thumbnails, item titles, prices, and call-to-action buttons that share identical visual layouts with product detail pages.
* **Affected Screenshot**:
  * `ss_218.png` (`cart_checkout_payment` $\rightarrow$ `product_detail`). Ground truth is a shopping cart screen with item lists, but the visual model classifies it as `product_detail` (Top-1 confidence: 0.8626).

---

## 3. Complete List of 21 Misclassified Validation Screenshots

| Screenshot ID | Ground Truth | Predicted Class | Confidence | Top-2 Class | Top-2 Margin | Error Category |
| :--- | :--- | :--- | :---: | :--- | :---: | :--- |
| `ss_015.jpg` | `code_editor_terminal` | `document_article_view` | 0.9905 | `product_detail` | 0.9817 | Dominant Class Overbias |
| `ss_020.jpg` | `product_detail` | `document_article_view` | 0.5568 | `code_editor_terminal` | 0.1158 | Dominant Class Overbias |
| `ss_022.jpg` | `product_detail` | `document_article_view` | 0.7803 | `product_detail` | 0.5619 | Dominant Class Overbias |
| `ss_033.jpg` | `document_article_view` | `code_editor_terminal` | 0.9900 | `document_article_view` | 0.9799 | OCR-Required Disambiguation |
| `ss_058.jpg` | `document_article_view` | `code_editor_terminal` | 0.6959 | `document_article_view` | 0.3958 | OCR-Required Disambiguation |
| `ss_065.jpg` | `document_article_view` | `code_editor_terminal` | 0.5059 | `document_article_view` | 0.1572 | OCR-Required Disambiguation |
| `ss_070.jpg` | `code_editor_terminal` | `document_article_view` | 0.9488 | `dashboard_settings` | 0.9001 | Dominant Class Overbias |
| `ss_105.jpg` | `document_article_view` | `product_detail` | 0.9525 | `document_article_view` | 0.9051 | OCR-Required Disambiguation |
| `ss_155.jpg` | `cart_checkout_payment` | `document_article_view` | 0.6052 | `code_editor_terminal` | 0.2106 | Dominant Class Overbias |
| `ss_164.jpg` | `product_detail` | `document_article_view` | 0.9961 | `product_detail` | 0.9922 | Dominant Class Overbias |
| `ss_173.jpg` | `document_article_view` | `product_detail` | 0.8166 | `document_article_view` | 0.6542 | OCR-Required Disambiguation |
| `ss_182.png` | `product_detail` | `document_article_view` | 0.9968 | `product_detail` | 0.9947 | Dominant Class Overbias |
| `ss_218.png` | `cart_checkout_payment` | `product_detail` | 0.8626 | `document_article_view` | 0.7257 | Domain Boundary Overlap |
| `ss_353.png` | `cart_checkout_payment` | `document_article_view` | 0.9074 | `product_detail` | 0.8272 | Dominant Class Overbias |
| `ss_357.png` | `cart_checkout_payment` | `document_article_view` | 0.8797 | `product_detail` | 0.7698 | Dominant Class Overbias |
| `ss_369.png` | `product_detail` | `document_article_view` | 0.9181 | `product_detail` | 0.8362 | Dominant Class Overbias |
| `ss_388.jpg` | `document_article_view` | `product_detail` | 0.5174 | `document_article_view` | 0.0439 | OCR-Required Disambiguation |
| `ss_450.png` | `product_detail` | `document_article_view` | 0.9981 | `code_editor_terminal` | 0.9962 | Dominant Class Overbias |
| `ss_813.jpg` | `code_editor_terminal` | `document_article_view` | 0.9905 | `product_detail` | 0.9817 | Dominant Class Overbias |
| `ss_818.jpg` | `product_detail` | `document_article_view` | 0.5568 | `code_editor_terminal` | 0.1158 | Dominant Class Overbias |
| `ss_820.jpg` | `product_detail` | `document_article_view` | 0.7803 | `product_detail` | 0.5619 | Dominant Class Overbias |

---

## 4. Key Takeaways for Multimodal Fusion Architecture

1. **Standalone Visual Limitations**: Pure image-only classification cannot reliably detect low-sample screen types like `cart_checkout_payment` (0/4 recall) due to extreme sample scarcity and visual similarity to product pages and documents.
2. **Complementary Role**: Visual screen classification excels at identifying high-level layout structure (`document_article_view` 95.04% recall, `dashboard_settings` 100% recall), but MUST be combined with EasyOCR semantic text features to resolve ambiguous UI boundaries.
3. **Preserve `FinalRefinedClassifier` Primacy**: These audit findings validate our core architectural decision: **`FinalRefinedClassifier` must remain the primary decision maker**, while VSUE provides supporting visual evidence.
