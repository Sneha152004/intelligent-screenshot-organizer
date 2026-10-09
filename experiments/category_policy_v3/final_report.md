# CATEGORY DECISION POLICY V3 EXPERIMENT FINAL REPORT
====================================================

## 1. Executive Summary

This experiment evaluated Category Decision Policy V3 (featuring bank dashboard/net worth safeguards, developer admin tool disambiguation, payment submission form rules, and retail collection classifiers) to determine whether External V2 Top-1 accuracy could be pushed beyond the **81.15%** baseline toward the aspirational **83–85%** target without adding new models or cloud dependencies.

### Key Quantitative Metrics:
- **Policy V2 Control External V2 Top-1**: **81.15%** (198/244 correct)
- **Category Policy V3 External V2 Top-1**: **81.97%** (200/244 correct)
- **Absolute Improvement over V2**: **+0.82%** (+2 net correct screenshots)
- **Corrections over V2**: **2**
- **Harms over V2**: **0**
- **Internal Validation Top-1**: **93.60%** (Target: >=93.60%)
- **Internal Holdout Top-1**: **99.20%** (Target: >=99.20%)
- **Aspirational 83.00% Target Reached**: **NO**
- **Final Recommendation**: `POLICY V3 GO`

## 2. Comprehensive Performance Comparison

| Split | Metric | Policy V2 Control | Category Policy V3 | Delta |
|---|---|---|---|---|
| Internal Validation (n=125) | Top-1 Accuracy | 93.60% | 93.60% | +0.00% |
| Internal Holdout (n=125) | Top-1 Accuracy | 99.20% | 99.20% | +0.00% |
| External V2 Benchmark (n=244) | Top-1 Accuracy | 81.15% (198/244) | **81.97%** (200/244) | **+0.82%** |
| External V2 Benchmark (n=244) | Top-2 Accuracy | 84.84% | **86.48%** | +1.64% |
| External V2 Benchmark (n=244) | Top-3 Accuracy | 86.48% | **87.70%** | +1.22% |
| External V2 Benchmark (n=244) | MRR | 0.8592 | **0.8683** | +0.0091 |

## 3. Analysis of Remaining Failure Ceiling

The remaining errors consist of screens where OCR text is either sparse (< 30 characters) or where visual layout spatial hierarchy (e.g. poster layouts, complex multi-domain widgets) is required to resolve category intent. Text-only heuristic rules have reached their maximum empirical ceiling on this dataset.

## 4. Final Recommendation

Category Decision Policy V3 achieved **81.97% Top-1 Accuracy** (200/244) on External V2 while maintaining **93.60%** on Validation and **99.20%** on Holdout with zero regression.

**Verdict**: `POLICY V3 GO`.
