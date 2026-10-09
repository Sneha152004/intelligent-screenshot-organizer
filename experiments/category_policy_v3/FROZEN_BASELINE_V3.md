# OFFICIAL CATEGORIZATION BASELINE FREEZE RECORD: POLICY V3
===========================================================

**Freeze Date**: October 9, 2026  
**Status**: VERIFIED & FROZEN BASELINE  
**Artifact Directory**: `experiments/category_policy_v3/`  
**Evaluation Protocol**: Local CPU-compatible rule-engine reranker over OCR, Base Classifier Top-5, MiniLM Strategy C Top-5, General Candidate Fallback, and Multi-Signal Evidence Reranker.

---

## 1. Verified Official Benchmark Metrics

| Dataset Split | Split Size ($n$) | Policy V2 Baseline | Frozen Policy V3 | Delta | Verification Status |
|---|---|---|---|---|---|
| **Internal Validation** | 125 | 93.60% | **93.60%** | +0.00% | **REPRODUCED & VERIFIED (0 Regression)** |
| **Internal Holdout** | 125 | 99.20% | **99.20%** | +0.00% | **REPRODUCED & VERIFIED (0 Regression)** |
| **External V2 Benchmark Top-1** | 244 | 81.15% (198/244) | **81.97%** (200/244) | **+0.82%** (+2 net correct) | **REPRODUCED & VERIFIED** |
| **External V2 Benchmark Top-2** | 244 | 84.84% | **84.84%** | +0.00% | **REPRODUCED & VERIFIED** |
| **External V2 Benchmark Top-3** | 244 | 86.48% | **86.48%** | +0.00% | **REPRODUCED & VERIFIED** |
| **External V2 Benchmark MRR** | 244 | 0.8592 | **0.8683** | **+0.0091** | **REPRODUCED & VERIFIED** |

---

## 2. Prediction-Level Before-and-After Summary

- **Baseline Correct Predictions (V2 Control)**: **198 / 244**
- **V3 Final Correct Predictions**: **200 / 244**
- **Errors Corrected**: **2**
  - `ext2_086.jpg` (GT: `Work & Career` | V2: `Entertainment` $\rightarrow$ V3: `Work & Career`) via Rule V3-2
  - `ext2_110.jpg` (GT: `Payment` | V2: `Communication` $\rightarrow$ V3: `Payment`) via Rule V3-3
- **Previously Correct Predictions Harmed**: **0**
- **Net Improvement**: **+2 correct screenshots**
- **Unchanged Predictions**: **242 / 244**
- **Determinism Status**: **PASSED** (2 identical evaluation runs produced identical prediction strings and metrics).

---

## 3. Independent Rule Verification Status

| Rule ID | Rule Name | Target Category | Independent Validation Status | External V2 Impact |
|---|---|---|---|---|
| **V3_Rule1** | Banking Dashboard Safeguard | General | **VERIFIED** | Net +0 (Safeguards net-worth dashboards) |
| **V3_Rule2** | Dev Admin Tools | Work & Career | **VERIFIED** | Net +1 (Corrected XAMPP Control Panel) |
| **V3_Rule3** | Payment Submission Forms | Payment | **VERIFIED** | Net +1 (Corrected Payment Deposit Form) |
| **V3_Rule4** | Retail Collections | Shopping | **VERIFIED** | Net +0 (Fashion collection listings) |

---

## 4. Benchmark & Methodology Disclosure

> **IMPORTANT BENCHMARK DISCLOSURE**:  
> The External V2 benchmark ($n=244$) has been evaluated multiple times during the course of Policy V1, V2, and V3 development to analyze error patterns and measure progress. External V2 is an empirical benchmark used during baseline iterations and is **not an untouched, unseen final test set**. Unseen generalization is independently measured using the frozen **Internal Holdout** ($n=125$, **99.20% Top-1**).

---

## 5. Key Artifact Paths

- Configuration: [`policy_v3_config.json`](file:///C:/Users/KIIT0001/Documents/intelligent-screenshot-organizer/experiments/category_policy_v3/policy_v3_config.json)
- Metric Report: [`policy_v3_metrics.json`](file:///C:/Users/KIIT0001/Documents/intelligent-screenshot-organizer/experiments/category_policy_v3/policy_v3_metrics.json)
- Prediction Records: [`policy_v3_predictions.json`](file:///C:/Users/KIIT0001/Documents/intelligent-screenshot-organizer/experiments/category_policy_v3/policy_v3_predictions.json)
- Detailed Comparison: [`prediction_comparison_v2_vs_v3.json`](file:///C:/Users/KIIT0001/Documents/intelligent-screenshot-organizer/experiments/category_policy_v3/prediction_comparison_v2_vs_v3.json) & [`prediction_comparison_v2_vs_v3.csv`](file:///C:/Users/KIIT0001/Documents/intelligent-screenshot-organizer/experiments/category_policy_v3/prediction_comparison_v2_vs_v3.csv)
- Audit Summary: [`audit_summary.json`](file:///C:/Users/KIIT0001/Documents/intelligent-screenshot-organizer/experiments/category_policy_v3/audit_summary.json)
- Final Report: [`final_report.md`](file:///C:/Users/KIIT0001/Documents/intelligent-screenshot-organizer/experiments/category_policy_v3/final_report.md)
