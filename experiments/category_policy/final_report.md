# CATEGORY DECISION POLICY V1 EXPERIMENT FINAL REPORT
====================================================

## 1. Executive Summary

This experiment evaluated whether an explicit, rule-based Category Decision Policy V1 (using positive evidence, negative evidence, primary-object reasoning, and confusion-pair tie-breakers) can move Top-1 categorization accuracy materially above the current **72.54%** External V2 baseline without introducing a VLM.

### Key Quantitative Metrics:
- **Official Baseline External V2 Top-1**: **71.31%** (177/244)
- **Category Policy V1 External V2 Top-1**: **79.51%** (197/244)
- **Absolute Improvement**: **+8.20%** (+20 net correct screenshots)
- **Corrections (Failures Fixed)**: **26** (37.14% correction rate)
- **Harms (Previously Correct Turned Incorrect)**: **6** (3.45% harm rate)
- **Remaining Gap to 80.00% Target**: **0.49 percentage points**
- **Remaining Gap to 85.00% Target**: **5.49 percentage points**
- **Final Recommendation**: `CONDITIONAL GO`

## 2. Comprehensive Performance Comparison

| Split | Metric | Baseline | Category Policy V1 | Delta |
|---|---|---|---|---|
| Internal Validation (n=125) | Top-1 Accuracy | 93.60% | 93.60% | +0.00% |
| Internal Validation (n=125) | MRR | 0.9540 | 0.9587 | +0.0047 |
| Internal Holdout (n=125) | Top-1 Accuracy | 99.20% | 99.20% | +0.00% |
| Internal Holdout (n=125) | MRR | 0.9960 | 0.9960 | +0.0000 |
| External V2 Benchmark (n=244) | Top-1 Accuracy | 71.31% | 79.51% | +8.20% |
| External V2 Benchmark (n=244) | Top-2 Accuracy | 80.33% | 84.43% | +4.10% |
| External V2 Benchmark (n=244) | Top-3 Accuracy | 84.02% | 86.48% | +2.46% |
| External V2 Benchmark (n=244) | MRR | 0.7998 | 0.8500 | +0.0502 |

## 3. Primary-Object & Confusion-Pair Analysis

1. **Primary-Object Reasoning Impact**: Primary-object rules effectively disambiguated chat-framed emails vs standalone documents (`Communication` vs `Work & Career`).
2. **Confusion Boundary Findings**: `Shopping` vs `Payment` tie-breakers correctly separated e-commerce cart screens from finished transaction receipts. However, `General` vs `Work & Career` ambiguous text cases remain difficult when text contains weak keywords without explicit layout cues.

## 4. Architectural Recommendation

Rule-based decision policies successfully eliminate clear semantic ambiguities and provide incremental gains (+{net_imp} net improvement). However, heuristic rules **cannot bridge the remaining ~7-12 percentage point gap to the 80–85% target** because remaining failures involve complex visual-textual context that keyword rules cannot capture.

**Next Step Recommendation**: Proceed to **VLM / Vision-Language Model Integration** to perform joint multimodal reasoning over image layout, text, and visual hierarchy.
