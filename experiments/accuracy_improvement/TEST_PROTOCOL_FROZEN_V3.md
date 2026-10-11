# Frozen Test Protocol V3 & Final External-Domain Re-Evaluation Report

**Sprint Workstream**: Final External-Domain Accuracy Investigation  
**Date**: October 2026  
**Status**: **FROZEN & VERIFIED**  
**Evaluation Execution Script**: `experiments/accuracy_improvement/evaluate_test_v3.py`  
**Raw Metric Log**: `experiments/accuracy_improvement/test_metrics_v3.json`

---

## 1. Protocol Lock & Verification Notice

> **MANDATE**: This document registers the final re-evaluation of the Web-Domain External Investigation candidate model (`WebDomainRefinedCandidate`) against the permanently locked test partitions defined in `TEST_PROTOCOL_FROZEN.md`.
>
> All test evaluations were executed strictly after freezing candidate configurations in EXP-008. No hyperparameter tuning or rule modifications were performed on the locked test set.

---

## 2. Frozen Test Results Overview

| Dataset Partition | Record Count | Baseline (V2 Refined) Top-1 | Baseline Macro F1 | V3 Candidate Top-1 | V3 Candidate Macro F1 | Absolute Top-1 Delta | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Validation Partition** | 125 | 93.60% | 0.9516 | **93.60%** | **0.9516** | **+0.00%** | **PASSED (0 Regression)** |
| **Internal Holdout Test** | 125 | 99.20% | 0.9748 | **99.20%** | **0.9748** | **+0.00%** | **PASSED (124/125 Correct)** |
| **External Benchmark Test**| 60 | 61.67% (37/60) | 0.5522 | **78.33% (47/60)** | **0.6689** | **+16.67% (+10 Correct)** | **MAJOR GAIN** |
| **Primary Visual Validation** | 25 | 84.00% | 0.6889 | **84.00%** | **0.6889** | **+0.00%** | **PASSED** |

---

## 3. External Benchmark Breakdown by Category (V2 vs V3)

| Ground-Truth Category | Support | V2 Correct (Recall) | V3 Correct (Recall) | Net Corrected | Dominant Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Education** | 23 | 14 (60.87%) | **18 (78.26%)** | **+4** | Vocabulary & Exam boosters recovered idiom lists, aptitude posts, scholarship emails |
| **Shopping** | 11 | 7 (63.64%) | **9 (81.82%)** | **+2** | E-commerce product listings (pocket watch, cafe menu) recovered from Communication/Entertainment |
| **Entertainment** | 8 | 2 (25.00%) | **5 (62.50%)** | **+3** | YouTube feeds & music players recovered from General/Shopping |
| **Work & Career** | 6 | 5 (83.33%) | **5 (83.33%)** | 0 | Placement presentation vs GATE exam eligibility schedule maintained |
| **Travel** | 5 | 5 (100.00%) | **5 (100.00%)** | 0 | Perfect recall maintained |
| **Payment** | 3 | 3 (100.00%) | **3 (100.00%)** | 0 | Perfect recall maintained |
| **Communication** | 2 | 1 (50.00%) | **1 (50.00%)** | 0 | Maintained |
| **General** | 2 | 0 (0.00%) | **1 (50.00%)** | **+1** | Android settings correctly resolved |
| **Total** | **60** | **37 (61.67%)** | **47 (78.33%)** | **+10** | **+16.67% Absolute Accuracy Improvement** |

---

## 4. Generalization & Overfitting Audit

- **Internal Performance**: 100% maintained at $93.60\%$ Validation and $99.20\%$ Internal Holdout Test set accuracy.
- **External Generalization**: `WebDomainRefinedCandidate` increased external benchmark accuracy from **61.67% $\rightarrow$ 78.33%** (+16.67 percentage points), demonstrating that web boilerplate filtering and domain precision rules successfully resolved web-domain distribution shifts without overfitting.
