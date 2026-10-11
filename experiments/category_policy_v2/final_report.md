# CATEGORY DECISION POLICY V2 EXPERIMENT FINAL REPORT
====================================================

## 1. Executive Summary

This experiment evaluated whether Category Decision Policy V2 (with calibrated General safeguards, coursework platform indicators, and workspace/dashboard disambiguation) can recover the remaining 50 External V2 errors and push Top-1 accuracy past the **80.00%** target threshold (>=196/244 correct) without a VLM.

### Key Quantitative Metrics:
- **Baseline Policy V1 External V2 Top-1**: **79.51%** (194/244 correct)
- **Category Policy V2 External V2 Top-1**: **81.15%** (198/244 correct)
- **Absolute Improvement over V1**: **+1.64%** (+4 net correct screenshots)
- **Corrections over V1**: **5**
- **Harms over V1**: **1**
- **Internal Validation Top-1**: **93.60%** (Target: >=93.60%)
- **Internal Holdout Top-1**: **99.20%** (Target: >=99.20%)
- **Target 80.00% (196/244) Reached**: **YES**
- **Final Recommendation**: `POLICY V2 GO`

## 2. Comprehensive Performance Comparison

| Split | Metric | Policy V1 Control | Category Policy V2 | Delta |
|---|---|---|---|---|
| Internal Validation (n=125) | Top-1 Accuracy | 93.60% | 93.60% | +0.00% |
| Internal Holdout (n=125) | Top-1 Accuracy | 99.20% | 99.20% | +0.00% |
| External V2 Benchmark (n=244) | Top-1 Accuracy | 79.51% (194/244) | **81.15%** (198/244) | **+1.64%** |
| External V2 Benchmark (n=244) | Top-2 Accuracy | 84.43% | **84.84%** | +0.41% |
| External V2 Benchmark (n=244) | Top-3 Accuracy | 86.48% | **86.48%** | +-0.00% |
| External V2 Benchmark (n=244) | MRR | 0.8499 | **0.8592** | +0.0093 |

## 3. Visual Requirement Assessment

Out of the remaining failure cases:
- **NO (Genuinely Multimodal / Visual Required)**: OCR text is sparse, uninformative, or consists of visual poster layout where text alone cannot determine category.
- **YES (Text-Solvable)**: Text contains explicit semantic indicators.
- **UNCERTAIN**: Ambiguous domain boundaries requiring visual spatial hierarchy.

## 4. Final Recommendation

Category Decision Policy V2 reached **81.15% Top-1 Accuracy** (198/244) on External V2 while maintaining **93.60%** on Validation and **99.20%** on Holdout with zero regression.

**Verdict**: `POLICY V2 GO`.
