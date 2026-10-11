# Frozen External V2 Categorization Recovery Evaluation Report

**Evaluation Target**: 244 External V2 Screenshots  
**Date**: October 2026  
**Status**: **FROZEN EVALUATION COMPLETE**

## Benchmark Overview (n = 244)

| Metric | Baseline Classifier | Recovery Engine V2 | Delta |
| :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | 71.31% (174/244) | **71.31% (174/244)** | **0.00 pp (Stable)** |
| **Top-2 Accuracy** | 80.33% (196/244) | **81.15% (198/244)** | **+0.82 pp** |
| **Top-3 Accuracy** | 84.02% (205/244) | **85.25% (208/244)** | **+1.23 pp** |
| **Macro F1 Score** | 0.6513 | **0.6540** | **+0.0027** |
| **Balanced Accuracy** | 0.7403 | **0.7442** | **+0.0039** |
| **MRR** | 0.7912 | **0.7950** | **+0.0038** |
| **Candidate Pool Coverage** | 82.79% (Top-3) | **88.93% (217/244)** | **+6.14 pp** |

## Failure Taxonomy Analysis (External V2)
- **Group A2 (Positive Control)**: 174 screenshots (71.31% correct Top-1)
- **Group A1 (Reranking Failures)**: 43 screenshots (17.62% in-pool but wrong Top-1)
- **Group B1 (Candidate Gen Failures)**: 27 screenshots (11.07% out of candidate pool)
