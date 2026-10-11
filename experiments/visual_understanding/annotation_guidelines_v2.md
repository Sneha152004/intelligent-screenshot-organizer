# Phase 7B: Enhanced Annotation Guidelines (v2)

**Date**: 2026-10-02  
**Purpose**: Standardized rules for human verification of screenshot visual structures, eliminating text/metadata bias.

---

## 1. Disambiguation Rules for Visually Similar Classes

1. **`product_listing` vs. `product_detail`**:
   * Label `product_listing` if $\ge 2$ product cards are visible in a grid/list.
   * Label `product_detail` if 1 dominant item is showcased with a large hero image and buy action CTA.
2. **`messaging_chat` vs. `email_view`**:
   * Label `messaging_chat` if left/right aligned speech bubbles or message pills are present.
   * Label `email_view` if structured sender headers ("From:", "Subject:") and continuous email thread text are present.
3. **`document_article_view` vs. `code_editor_terminal`**:
   * Label `code_editor_terminal` if vertical line numbers, syntax highlighting, or IDE window chrome are visible.
   * Label `document_article_view` if text is formatted as prose, paragraphs, or slides.

---

## 2. Review Status Definitions

* **`verified`**: Visual components cleanly match a single primary screen type with high confidence.
* **`ambiguous`**: Interface displays dual prominent layouts (e.g. settings modal over a document view). Secondary screen type must be populated.
* **`insufficient_visual_evidence`**: Screen is blank, severely corrupted, or lacks observable structural components.
* **`pending`**: Unreviewed record awaiting human inspection.

---

## 3. Confidence Level Protocols

* **`high`**: Key structural components (e.g. line numbers, chat bubbles, payment CTA, hero image) are clearly visible.
* **`medium`**: Visual structure is partially obscured or shared across boundary categories.
* **`low`**: Minimal visual features; primary label assigned tentatively.
