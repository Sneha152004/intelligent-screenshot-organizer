# Diagnostic Top-K Candidate Coverage Analysis

**Workstream Directory**: `experiments/final_external_v2_validation/`  
**Date**: October 2026  
**Dataset**: 244 External V2 Screenshots (`ext2_001.jpg` – `ext2_244.jpg`)  
**Data Source**: `topk_analysis.json`

---

## 1. Executive Summary & Diagnostic Findings

An architectural analysis of the production semantic classifier's prediction distribution on External V2 was conducted to evaluate the potential for evidence-based candidate reranking.

### Key Finding: 45.1% of Top-1 Errors are Recoverable in Top-3
Out of 71 total Top-1 classification errors on External V2:
- **23 screenshots (32.4% of errors)** have the correct ground-truth category sitting at **Top-2**.
- **9 screenshots (12.7% of errors)** have the correct ground-truth category sitting at **Top-3**.
- **Cumulative Top-3 Coverage**: **84.02% (205 / 244 screenshots)**.

This proves that **the correct semantic category is already present in the model's Top-3 candidate list for 84.02% of all screenshots**. An evidence-based reranker that selects among existing Top-3 candidates can achieve up to **84.02% Top-1 accuracy** without needing new classifier architectures or retraining.

---

## 2. Top-K Coverage Breakdown (n = 244)

| Candidate Rank | Screenshot Count | Percentage of Benchmark | Cumulative Coverage | Recoverable Errors |
| :--- | :--- | :--- | :--- | :--- |
| **Top-1 Correct** | 173 | 70.90% | 70.90% | — |
| **Top-2 Only (Correct at Rank 2)** | 23 | 9.43% | **80.33%** | +23 screenshots |
| **Top-3 Only (Correct at Rank 3)** | 9 | 3.69% | **84.02%** | +9 screenshots |
| **Outside Top-3 (Ranks 4–9)** | 39 | 15.98% | 100.00% | 39 screenshots |
| **Total Benchmark** | **244** | **100.00%** | — | **32 Recoverable Errors** |

---

## 3. Per-Category Top-1, Top-2, and Top-3 Coverage

| Category | Support | Top-1 Correct (Recall) | Top-2 Coverage (Recall) | Top-3 Coverage (Recall) | Top-3 Opportunity Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Education** | 36 | 21 (58.33%) | **27 (75.00%)** | **31 (86.11%)** | **+10 screenshots** |
| **Shopping** | 40 | 29 (72.50%) | **34 (85.00%)** | **35 (87.50%)** | **+6 screenshots** |
| **Payment** | 13 | 11 (84.62%) | **13 (100.00%)** | **13 (100.00%)** | **+2 screenshots** |
| **Travel** | 5 | 3 (60.00%) | **4 (80.00%)** | **4 (80.00%)** | **+1 screenshot** |
| **Communication** | 21 | 14 (66.67%) | **16 (76.19%)** | **17 (80.95%)** | **+3 screenshots** |
| **Entertainment** | 5 | 5 (100.00%) | **5 (100.00%)** | **5 (100.00%)** | Perfect recall |
| **Work & Career** | 11 | 2 (18.18%) | **7 (63.64%)** | **8 (72.73%)** | **+6 screenshots** |
| **Connectivity** | 5 | 4 (80.00%) | **4 (80.00%)** | **4 (80.00%)** | Maintained |
| **General** | 108 | 84 (77.78%) | **86 (79.63%)** | **88 (81.48%)** | **+4 screenshots** |
| **Total / Overall** | **244** | **173 (70.90%)** | **196 (80.33%)** | **205 (84.02%)** | **+32 screenshots** |

---

## 4. Primary Recoverable Error Pairs

The diagnostic analysis revealed specific error pairs where the true label sits at Top-2 or Top-3:

1. **`Education` $\rightarrow$ Predicted `Travel` / `Shopping` (Top-2: `Education`)**:
   - Course slides and tutorial pages containing travel/transportation examples or pricing tables where `Education` is the Top-2 candidate.
2. **`Work & Career` $\rightarrow$ Predicted `Travel` / `General` (Top-2: `Work & Career`)**:
   - Technical job postings, onboarding checklists, and developer dashboards where `Work & Career` sits at Rank 2.
3. **`Shopping` $\rightarrow$ Predicted `Travel` (Top-2: `Shopping`)**:
   - E-commerce product voucher pages where `Shopping` is at Rank 2 behind `Travel`.

---

## 5. Architectural Implications

Instead of forcing a single early Top-1 classification decision, preserving the Top-3 candidate hypotheses (`top1_category`, `top2_category`, `top3_category`) provides rich candidate context downstream:
* **Downstream Tag Generation**: Can generate domain tags for both Top-1 and Top-2 categories when candidates are close in confidence.
* **Downstream Intent & Chatbot**: Can retrieve screenshots based on alternative candidate hypotheses when user query matches Top-2 or Top-3 domains.
