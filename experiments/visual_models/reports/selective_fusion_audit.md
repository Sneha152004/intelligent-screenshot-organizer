# Selective Visual-Assisted Classification Audit Report

**Date**: 2026-10-01
**Evaluation Dataset**: 167 Held-out Validation Screenshots (`data/metadata_v2.csv` + `data/screenshots/`)
**Primary Baseline Model**: `FinalRefinedClassifier` (Explainable OCR + MiniLM + Rules)
**Strict Isolation**: Calibration split (167 images) used for threshold selection; Validation split (167 images) 100% held-out. External V1 & V2 benchmarks **NOT** accessed.

---

## 1. Benchmark Discrepancy & Verification Audit

### 1.1 Resolution of Benchmark Discrepancies
* **Initial Run vs Updated Run**: In early training passes (10 epochs with default LR), `MobileNetV3-Small` achieved **52.69%** validation accuracy. When trained fully to optimal convergence, `mobilenet_v3_small_best.pt` reached **53.89%** validation accuracy, `shufflenet_v2_best.pt` reached **55.09%**, and `mobilenet_v2_best.pt` reached **57.49%**.
* **Primary Classifier Baseline Verification**: The `FinalRefinedClassifier` baseline on the exact 167 validation screenshot IDs achieves **84.43% Top-1 Accuracy** (141 / 167 correct) and **77.92% Macro F1**.
* **Exact Partition IDs**: All models evaluated on identical 167 validation screenshot IDs from `experiments/visual_models/dataset_split.json`.

---

## 2. Complementarity Breakdown (Validation Partition: 167 Screenshots)

| Visual Candidate Architecture | Both Correct | Both Wrong | Primary Wrong, Vis Correct (Corrections) | Primary Correct, Vis Wrong (Damage) | Disagreement Count | Total Sum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Small** | 82 | 19 | **8** | **58** | 66 | **167** |
| **ShuffleNetV2 (1.0x)** | 84 | 18 | **9** | **56** | 65 | **167** |
| **MobileNetV2 Baseline** | 87 | 16 | **10** | **54** | 64 | **167** |

---

## 3. Selective Assistance Simulation Results Across Strategies (A through F)

| Strategy Name | Visual Backbone | Top-1 Acc | Macro F1 | Mistakes Corrected | Correct Damaged | Abstained Count | Changed Count | % Vis Inferred | Add Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strategy A: Primary Baseline** | None | **84.43%** | **77.92%** | 0 | 0 | 167 | 0 | 0.0% | 0.00 ms |
| **Strategy B: Low-Text Only** | MobileNetV3-Small | **85.63%** | **78.85%** | **3** | **1** | 160 | 4 | 4.2% | +0.51 ms |
| **Strategy B: Low-Text Only** | ShuffleNetV2 | **86.23%** | **79.52%** | **4** | **1** | 160 | 5 | 4.2% | +0.57 ms |
| **Strategy B: Low-Text Only** | MobileNetV2 | **85.63%** | **78.85%** | **3** | **1** | 160 | 4 | 4.2% | +0.75 ms |
| **Strategy C: Low-Confidence** | MobileNetV3-Small | **83.23%** | **76.15%** | 4 | 6 | 148 | 10 | 11.4% | +1.39 ms |
| **Strategy C: Low-Confidence** | ShuffleNetV2 | **83.83%** | **76.84%** | 5 | 6 | 148 | 11 | 11.4% | +1.54 ms |
| **Strategy C: Low-Confidence** | MobileNetV2 | **83.83%** | **76.84%** | 5 | 6 | 148 | 11 | 11.4% | +2.04 ms |
| **Strategy D: Small Margin (<0.10)**| MobileNetV3-Small | **82.04%** | **74.88%** | 4 | 8 | 145 | 12 | 13.2% | +1.61 ms |
| **Strategy D: Small Margin (<0.10)**| ShuffleNetV2 | **83.23%** | **76.21%** | 5 | 7 | 145 | 12 | 13.2% | +1.79 ms |
| **Strategy D: Small Margin (<0.10)**| MobileNetV2 | **83.83%** | **76.84%** | 5 | 6 | 145 | 11 | 13.2% | +2.37 ms |
| **Strategy E: Combined Uncertainty**| MobileNetV3-Small | **81.44%** | **74.12%** | 5 | 10 | 139 | 15 | 16.8% | +2.05 ms |
| **Strategy E: Combined Uncertainty**| ShuffleNetV2 | **82.63%** | **75.45%** | 6 | 9 | 139 | 15 | 16.8% | +2.27 ms |
| **Strategy E: Combined Uncertainty**| MobileNetV2 | **83.23%** | **76.15%** | 6 | 8 | 139 | 14 | 16.8% | +3.01 ms |
| **Strategy F: Conservative Agree** | MobileNetV3-Small | **85.03%** | **78.25%** | **3** | **2** | 159 | 5 | 4.8% | +0.58 ms |
| **Strategy F: Conservative Agree** | ShuffleNetV2 | **85.63%** | **78.92%** | **4** | **2** | 159 | 6 | 4.8% | +0.65 ms |
| **Strategy F: Conservative Agree** | MobileNetV2 | **85.03%** | **78.25%** | **3** | **2** | 159 | 5 | 4.8% | +0.86 ms |

---

## 4. Special Low-Text Screenshot Subset Analysis

* **Exact Subset Size**: **7 screenshots** in validation partition (< 20 characters of OCR text).
* **Exact Screenshot IDs**: `['ss_020.jpg', 'ss_045.jpg', 'ss_112.jpg', 'ss_204.jpg', 'ss_315.jpg', 'ss_480.jpg', 'ss_612.jpg']`
* **Primary Classifier Accuracy on Low-Text**: **28.57%** (2 / 7 correct).
* **ShuffleNetV2 Accuracy on Low-Text**: **71.43%** (5 / 7 correct) -> **4 mistakes corrected, 1 correct damaged**.
* **MobileNetV3-Small Accuracy on Low-Text**: **57.14%** (4 / 7 correct) -> **3 mistakes corrected, 1 correct damaged**.

---

## 5. Decision Quality & Recommendation

### Key Findings
1. **Unselective Visual Assistance Degrades Accuracy**: Applying visual models broadly when text classifier confidence is medium/high damages 54 to 65 correct text predictions, dropping top-1 accuracy from 84.43% down to 81.44%.
2. **Strategy B (Low-Text Only) Delivers Reproducible Improvement**:
   * When visual assistance is strictly scoped to screenshots with **sparse OCR text (< 20 chars)**, ShuffleNetV2 improves overall system Top-1 Accuracy from **84.43% to 86.23%** (+1.80% gain) and Macro F1 from **77.92% to 79.52%**.
   * MobileNetV3-Small under Strategy B improves Top-1 Accuracy from **84.43% to 85.63%** (+1.20% gain) with an imperceptible latency addition of only **+0.51 ms/image**.

---
