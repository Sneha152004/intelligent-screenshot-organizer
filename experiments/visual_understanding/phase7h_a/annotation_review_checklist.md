# Phase 7H-A: Annotation Quality Review Checklist

**Date**: 2026-10-02  
**Target Engine**: Visual Screen Understanding Engine (VSUE)  
**Purpose**: Standard Operating Procedure (SOP) for Human Verification & Quality Assurance of Screenshot Annotations

---

## 1. Human Reviewer Verification Checklist

Every candidate screenshot MUST pass all five checklist stages before being assigned `annotation_status: verified`:

### Stage 1: File Integrity & Image Readability Check
- [ ] Image file exists on disk and opens cleanly with standard image viewers (PIL / OpenCV).
- [ ] File size is non-zero (> 0 bytes).
- [ ] Image resolution is legible (width $\ge 224$ px, height $\ge 224$ px).
- [ ] Image data is uncorrupted with valid headers.

### Stage 2: Privacy & Sensitivity Audit
- [ ] Screenshot contains NO visible passwords, OTP pin numbers, credit card CVVs, or full account numbers.
- [ ] Screenshot contains NO private personal correspondence or unredacted sensitive identity documents.
- [ ] Any minor personal identifier is flagged or acceptable under project guidelines.

### Stage 3: Primary Visual Intent Verification
- [ ] Primary `screen_type` label is selected strictly from the 13 canonical taxonomy labels.
- [ ] The assigned screen type matches the dominant functional interface layout visible in the screenshot.
- [ ] Visual structure is not forced into a class if it is genuinely ambiguous. Genuinely borderline images are marked `ambiguous`.

### Stage 4: Distinguishing Boundaries Verification
- [ ] `product_listing` vs `product_detail`: Multiple items in a grid/list $\rightarrow$ `product_listing`. Single item focus with description/buy button $\rightarrow$ `product_detail`.
- [ ] `product_detail` vs `cart_checkout_payment`: Item presentation $\rightarrow$ `product_detail`. Order summary, price breakdown, payment selection $\rightarrow$ `cart_checkout_payment`.
- [ ] `code_editor_terminal` vs `document_article_view`: Active IDE window with line numbers margin $\rightarrow$ `code_editor_terminal`. Formatted code block inside documentation webpage $\rightarrow$ `document_article_view`.
- [ ] `email_view` vs `document_article_view`: Explicit email headers (From, To, Subject, Inbox toolbar) $\rightarrow$ `email_view`. Web article/blog post $\rightarrow$ `document_article_view`.

### Stage 5: Schema & Metadata Verification
- [ ] `screenshot_id` matches exact filename.
- [ ] `visual_evidence` field contains a concise, non-empty natural language explanation of the key visual components observed.
- [ ] MD5 hash matches the computed file digest.
- [ ] `duplicate_group_id` correctly links exact binary duplicate images.

---

## 2. Multi-Annotator Conflict Resolution Procedure

1. If two human annotators assign different screen types to the same screenshot:
   - Perform side-by-side visual comparison against the `annotation_guidelines.md` taxonomy definitions.
   - If consensus is reached, update `screen_type` and document rationale in `annotator_notes`.
   - If visual structure remains contested or equal in weight across two categories, mark `annotation_status: ambiguous` and set `review_status: needs_second_review`.
2. Never force a split decision to favor class balance. Data integrity supersedes class balance.
