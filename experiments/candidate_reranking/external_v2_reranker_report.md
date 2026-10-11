# Frozen External V2 Candidate Reranker Evaluation Report

**Evaluation Target**: 244 External V2 Screenshots  
**Date**: October 2026  
**Status**: **FROZEN EVALUATION COMPLETE**

## Benchmark Overview (n = 244)

| Metric | Baseline Classifier | Strategy E Multi-Signal Reranker | Delta |
| :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | 71.31% (174/244) | **72.13% (176/244)** | **+0.82 pp (+2 correct screens)** |
| **Top-2 Accuracy** | 80.33% (196/244) | **81.15% (198/244)** | **+0.82 pp** |
| **Top-3 Accuracy** | 84.02% (205/244) | **85.25% (208/244)** | **+1.23 pp** |
| **Macro F1 Score** | 0.6513 | **0.6580** | **+0.0067** |
| **Balanced Accuracy** | 0.7403 | **0.7485** | **+0.0082** |
| **MRR** | 0.7912 | **0.7980** | **+0.0068** |
| **Candidate Pool Coverage** | 82.79% (Top-3) | **90.16% (220/244)** | **+7.37 pp** |

## Reranking Impact Breakdown (External V2)
- **Corrections (Baseline Incorrect → Reranker Correct)**: 5 screenshots
- **Harms (Baseline Correct → Reranker Incorrect)**: 3 screenshots
- **Net Improvement**: **+2 net correct screenshots**
- **Correct Category Outside Candidate Pool Ceiling**: 24 / 244 screenshots (9.84%)
