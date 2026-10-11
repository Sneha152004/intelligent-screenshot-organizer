# Component Ablation Results Report

## Internal Validation (n=125) Component Sweep
| Ablation | Top-1 Accuracy | Top-2 Accuracy | Top-3 Accuracy | MRR | Macro F1 | Corrections | Harms | Net Imp |
|---|---|---|---|---|---|---|---|---|
| A: Existing Production Baseline | 93.60% | 94.40% | 94.40% | 0.9540 | 0.9436 | 0 | 1 | -1 |
| B: + Positive Evidence Rules | 93.60% | 94.40% | 94.40% | 0.9540 | 0.9436 | 0 | 1 | -1 |
| C: + Positive + Negative Evidence Rules | 93.60% | 94.40% | 94.40% | 0.9540 | 0.9436 | 0 | 1 | -1 |
| D: + Primary-Object Reasoning | 93.60% | 94.40% | 94.40% | 0.9540 | 0.9436 | 0 | 1 | -1 |
| E: + Confusion-Pair Tie-Breakers | 93.60% | 94.40% | 100.00% | 0.9587 | 0.9436 | 0 | 1 | -1 |
| F: Full Category Policy V1 | 93.60% | 94.40% | 100.00% | 0.9587 | 0.9436 | 0 | 1 | -1 |