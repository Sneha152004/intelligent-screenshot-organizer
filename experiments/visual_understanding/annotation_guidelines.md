# Screen-Type Annotation Guidelines

**Date**: 2026-10-02  
**Purpose**: Standardized rules for annotating screenshots for the Visual Screen Understanding Engine dataset manifest.

---

## 1. Core Annotation Principles

1. **Annotate Visual Structure, Not Semantic Topic**:
   * Label screens based on **what interface components are visually present** (e.g. chat bubbles, code editor, product grid), NOT the topic of the text (e.g. an email about shopping is an `email_view`, not a `product_detail`).
2. **Primary vs. Secondary Screen Types**:
   * Assign the single dominant layout as `screen_type`.
   * If a second prominent layout component is embedded (e.g. a code snippet in a document), assign it to `secondary_screen_types`.
3. **Disambiguation & Edge Case Rules**:
   * **IDE Code vs. Document Code**: If line numbers and editor chrome are visible, label `code_editor_terminal`. If code is inside a blog post or slide, label `document_article_view`.
   * **Cart Overlay on Product Page**: If the cart drawer occupies >50% of the screen, label `cart_checkout_payment`; otherwise label `product_detail`.
   * **Modal Alert over App**: Label `error_dialog_alert` as primary if the modal dialog is centered and active.

---

## 2. Manifest Schema Definition

* `screenshot_id`: Filename identifier (e.g., `ss_001.jpg`)
* `filename`: Filename identifier
* `image_path`: Relative path to image file
* `existing_category`: Canonical topic category (reference only)
* `orientation`: `portrait`, `landscape`, or `square`
* `ocr_density`: `sparse`, `medium`, or `dense`
* `screen_type`: Primary consolidated screen type (1 of 12)
* `secondary_screen_types`: Optional secondary screen type
* `visual_elements`: Pipe-separated list of observed UI components (e.g. `chat_bubbles|avatar_icons|input_box`)
* `layout_description`: Concise structural description of the interface layout
* `annotation_confidence`: `high`, `medium`, `low`, or `unassigned`
* `annotation_reason`: Rationale for assigned label
* `review_status`: `verified`, `needs_review`, or `unreviewed`
