# External-Domain Accuracy Analysis

**Sprint Workstream**: Accuracy Improvement Sprint — Final External Investigation  
**Date**: October 2026  
**Status**: **COMPLETED - DIAGNOSTIC REPORT GENERATED**  
**Analysis Data Source**: `experiments/accuracy_improvement/external_domain_analysis.json`

---

## 1. Executive Summary & Diagnostic Overview

To investigate why `RefinedSemanticCandidate` achieves **99.20% Top-1** accuracy on the Internal Holdout Test set but **61.67% Top-1** (37/60 correct, 23 errors) on the External Benchmark, an empirical domain-shift audit was performed across dataset distributions, OCR text characteristics, category support, and failure modes.

### Key Finding: Severe Domain & OCR Distribution Shift
The external benchmark exhibits a **7x increase in mean OCR length** (508.2 chars vs 73.5 chars) compared to internal validation data. The external benchmark consists predominantly of **full web pages, social media feeds (YouTube, Facebook, Instagram), e-commerce product pages (Amazon, Cafe listings), and scanned/handwritten academic materials**.

These long web pages introduce noisy navigation UI text, social metadata, and false keyword triggers ("Reel", "1 lakh views", "marvel", "Just now", "Dear Student") that hijack the rule-based classifier's keyword matching logic.

---

## 2. Dataset & OCR Distribution Comparison

| Dimension | Internal Validation Set | External Benchmark Set | Ratio / Shift Observed |
| :--- | :--- | :--- | :--- |
| **Total Screenshot Count** | 125 screenshots | 60 screenshots | Standard benchmark size |
| **Mean OCR Text Length** | **73.5 characters** | **508.2 characters** | **6.9x longer OCR text** |
| **Median OCR Text Length** | 48 characters | 325 characters | 6.8x longer median |
| **Zero OCR Screenshots** | 2 (1.6%) | 1 (1.7%) | Comparable zero-text rate |
| **Sparse OCR (<100 chars)**| 98 (78.4%) | 4 (6.7%) | Severe sparsity shift |
| **Dominant Content Type** | Native Mobile App UI | Full Web Pages, Feeds, Scans | **Major Domain Shift** |

---

## 3. Category Distribution & Per-Class Error Breakdown

The 60 external benchmark screenshots are heavily weighted toward `Education` (23 items) and `Shopping` (11 items):

| Ground-Truth Category | External Support | Correct Predictions | Error Count | Accuracy / Recall | Dominant Misclassifications |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Education** | 23 | 14 | 9 | 60.87% | 4 to `Communication`, 2 to `General`, 1 `Shopping`, 1 `Entertainment`, 1 `Work & Career` |
| **Shopping** | 11 | 7 | 4 | 63.64% | 2 to `General`, 1 `Communication`, 1 `Entertainment` |
| **Entertainment** | 8 | 2 | 6 | 25.00% | 5 to `General`, 1 to `Shopping` |
| **Work & Career** | 6 | 5 | 1 | 83.33% | 1 to `Education` |
| **Travel** | 5 | 5 | 0 | **100.00%** | None (Perfect recall) |
| **Payment** | 3 | 3 | 0 | **100.00%** | None (Perfect recall) |
| **Communication** | 2 | 1 | 1 | 50.00% | 0 (Low support) |
| **General** | 2 | 0 | 2 | 0.00% | 1 to `Communication`, 1 to `Connectivity` |
| **Total** | **60** | **37** | **23** | **61.67%** | — |

---

## 4. Taxonomy of the 23 External Errors

The 23 misclassifications fall into 5 distinct error categories:

```
                                  23 EXTERNAL ERRORS
                                          │
       ┌──────────────────┬───────────────┼───────────────┬────────────────┐
       ▼                  ▼               ▼               ▼                ▼
[1] Web UI Noise   [2] Garbled OCR  [3] False Keyword [4] General     [5] Annotation
     & Feed Metadata   & Handwriting    Triggers        Fallback        Ambiguity
     (7 errors)        (4 errors)       (4 errors)      (6 errors)      (2 errors)
```

1. **Web UI Boilerplate & Feed Metadata (7 errors / 30.4%)**:
   - Web pages and social feeds contain words like "Reel", "Just now", "Dear Student", "View all comments" that override true domain semantics and trigger `Communication` or `Entertainment`.
   - Examples: `ext_001` (Idiom list $\rightarrow$ `Entertainment`), `ext_043` (Reasoning question $\rightarrow$ `Communication`), `ext_055` (KIIT scholarship email $\rightarrow$ `Communication`), `ext_059` (Vocabulary lesson $\rightarrow$ `Communication`).

2. **Garbled OCR & Handwriting (4 errors / 17.4%)**:
   - Scanned handwritten notes or image-heavy layouts yield zero or noisy OCR tokens (`ext_002`, `ext_009`, `ext_010`, `ext_050`).
   - Example: `ext_009` (Handwritten math notes $\rightarrow$ OCR `#au sdhwan' Page Ho_ YOUVA Date...` $\rightarrow$ predicted `General`).

3. **False Keyword Triggers (4 errors / 17.4%)**:
   - Out-of-context matching of generic words inside web listings:
   - `ext_017`: Amazon pocket watch listing containing "marvel" $\rightarrow$ predicted `Entertainment`.
   - `ext_051`: YouTube feed containing "1 lakh views" $\rightarrow$ matched currency pattern $\rightarrow$ predicted `Shopping`.
   - `ext_013`: CS memory partition notes containing "fixed scheme" $\rightarrow$ predicted `Shopping`.
   - `ext_019`: Restaurant cafe listing $\rightarrow$ predicted `Communication`.

4. **General Fallback on Sparse/Complex Layouts (6 errors / 26.1%)**:
   - Social media posts, word puzzles, and music players where text is sparse or unmapped (`ext_022`, `ext_023`, `ext_026`, `ext_027`, `ext_033`, `ext_054`).

5. **Category Boundary Ambiguity (2 errors / 8.7%)**:
   - `ext_025` (Placement schedule: ground truth `Work & Career`, predicted `Education`).
   - `ext_039` (Android settings with Wi-Fi: ground truth `General`, predicted `Connectivity`).

---

## 5. Summary & Actionable Recommendations

- **Root Cause**: The primary driver of external performance loss is **unfiltered web UI boilerplate** and **false keyword triggers** on long web page text (mean 508.2 chars).
- **Potential Interventions**:
  1. **OCR Preprocessing & Web Boilerplate Filtering**: Clean social feed metadata ("View comments", "Just now", "Reels", "1 lakh views") and web navbar text before keyword scoring.
  2. **Selective Visual Override Expansion**: Utilize visual evidence for low-confidence `General` predictions on sparse or image-heavy screenshots (`ext_002`, `ext_026`, `ext_050`).
