# External V2 Benchmark Empirical Evaluation Summary

> [!NOTE]
> **Empirical Evaluation Results (244 Screenshots)**:
> For the 244-screenshot External V2 benchmark dataset (`ext2_001.jpg` – `ext2_244.jpg`), the classification pipeline achieved the following empirical results without assuming unverified performance ranges:

---

## Empirical Performance Summary

- **Dataset Size**: 244 Screenshots (`ext2_001.jpg` – `ext2_244.jpg`)
- **Top-1 Accuracy**: **66.39%** (162 / 244)
- **Top-2 Recall**: **77.87%** (190 / 244)
- **Top-3 Recall**: **81.97%** (200 / 244)
- **Macro F1-Score**: **0.6082**
- **Balanced Accuracy**: **0.7054**
- **Mean Reciprocal Rank (MRR)**: **0.7350**
- **Core Domain Category Recalls**:
  - **Connectivity**: **80.00%** (4 / 5)
  - **Entertainment**: **80.00%** (4 / 5)
  - **Travel**: **80.00%** (4 / 5)
  - **Payment**: **76.92%** (10 / 13)
  - **Communication**: **76.19%** (16 / 21)
  - **Shopping**: **75.00%** (30 / 40)
  - **Education**: **58.33%** (21 / 36)
  - **Work & Career**: **45.45%** (5 / 11)
  - **General**: **62.96%** (68 / 108)

---

- **Benchmark Location**: `experiments/intelligence/external_v2/`
- **Prediction Audit Log**: `experiments/intelligence/external_v2/v2_prediction_audit.csv`
- **Status**: Recorded strictly with exact empirical metrics.
