# Reproducibility Verification Report

**Evaluation Target**: 244 External V2 Screenshots  
**Date**: October 2026  
**Status**: **PASS**

## Run Comparison (Two-Stage Evidence Reranked Production Classifier)

| Metric | Run 1 Value | Run 2 Value | Match Status |
| :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | 71.3115% | 71.3115% | **IDENTICAL** |
| **Top-2 Accuracy** | 80.3279% | 80.3279% | **IDENTICAL** |
| **Top-3 Accuracy** | 84.0164% | 84.0164% | **IDENTICAL** |
| **Macro Precision** | 0.635933 | 0.635933 | **IDENTICAL** |
| **Macro Recall** | 0.740319 | 0.740319 | **IDENTICAL** |
| **Macro F1 Score** | 0.651274 | 0.651274 | **IDENTICAL** |
| **Balanced Accuracy** | 0.740319 | 0.740319 | **IDENTICAL** |
| **MRR** | 0.791193 | 0.791193 | **IDENTICAL** |
| **ECE** | 0.105533 | 0.105533 | **IDENTICAL** |

**Conclusion**: The evaluation pipeline is 100% deterministic and reproducible across runs.
