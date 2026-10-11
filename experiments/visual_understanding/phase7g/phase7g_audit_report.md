# Phase 7G: Independent Audit of the Visual Screen-Type Baseline Model

**Date**: 2026-10-02  
**Target Model**: MobileNetV3-Small Visual Screen Understanding Engine (VSUE) Baseline  
**Auditor**: Antigravity Independent Audit Engine (Phase 7G)  
**Audit Verdict**: **AUDIT PASSED** *(With Documented Minority-Class Limitations)*

---

## 1. Executive Summary & Final Verdict

This document presents the complete results of the independent, read-only audit of the Phase 7F Visual Screen-Type Baseline Model. The audit evaluated model serialization, dataset partition isolation, metric reproducibility, error taxonomy, confidence calibration, FP32 vs INT8 dynamic quantization equivalence, and single-threaded CPU runtime efficiency.

### Audit Verdict Summary

> [!NOTE]
> **AUDIT VERDICT: PASSED**  
> * **Metric Reproduction**: 100% exact match across all reported Top-1/2/3 accuracy, Macro F1, Weighted F1, and Balanced Accuracy metrics.
> * **Quantization Loss**: Zero accuracy loss under INT8 dynamic quantization (100% Top-1 agreement, 0 prediction flips).
> * **Runtime Budget**: Warm median CPU latency of **22.03 ms/image** and **4.24 MB** INT8 footprint easily satisfy the 100–300 MB deployment budget.
> * **Documented Limitation**: Extreme class imbalance severely limits image-only performance on minority classes (`cart_checkout_payment` 0.00% recall), confirming that visual classification MUST operate as a supporting signal alongside primary OCR semantic classification.

---

## 2. Audited Model Architecture & Configuration

| Parameter | Specification | Audit Verification |
| :--- | :--- | :--- |
| **Backbone Architecture** | MobileNetV3-Small | ImageNet1K pretrained weights (`MobileNet_V3_Small_Weights.DEFAULT`) |
| **Classifier Head** | Linear Layer (`1024` $\rightarrow$ `5`) | Replaced 1000-class head with 5-class linear output |
| **Input Shape & Preprocessing** | `(3, 224, 224)` | ImageNet normalization with letterbox padding & bilinear resize |
| **Training Dataset Split** | 502 screenshots | Group-aware split from 833 final manifest records |
| **Calibration Dataset Split** | 162 screenshots | Group-aware split from 833 final manifest records |
| **Validation Dataset Split** | 169 screenshots | Group-aware held-out split (5 supported classes) |
| **Loss Function** | CrossEntropyLoss | Weighted by inverse class frequency vector |
| **Optimizer & Schedule** | AdamW (`lr=0.001`, `weight_decay=1e-4`) | StepLR decay at epoch 8 & 16 |
| **Best Epoch** | Epoch 13 | Confirmed calibration Macro F1 peak (`0.7437`) |

---

## 3. Metric Reproduction & Baseline Verification

All evaluation metrics were independently re-computed over the 169 held-out validation screenshots using `visual_screen_type_model.pth`.

| Metric Name | Phase 7F Reported | Phase 7G Reproduced | Verification Status |
| :--- | :---: | :---: | :---: |
| **Validation Sample Count** | 169 | 169 | **MATCH** |
| **Correct Predictions** | 148 / 169 | 148 / 169 | **MATCH** |
| **Error Instances** | 21 / 169 | 21 / 169 | **MATCH** |
| **Top-1 Accuracy** | **87.57%** | **87.57%** | **EXACT MATCH** |
| **Top-2 Accuracy** | **94.08%** | **94.08%** | **EXACT MATCH** |
| **Top-3 Accuracy** | **98.82%** | **98.82%** | **EXACT MATCH** |
| **Macro F1-Score** | **67.96%** | **67.96%** | **EXACT MATCH** |
| **Weighted F1-Score** | **86.32%** | **86.32%** | **EXACT MATCH** |
| **Balanced Accuracy** | **67.49%** | **67.49%** | **EXACT MATCH** |

---

## 4. Per-Class Performance Breakdown & Confusion Matrix

### Per-Class Performance Metrics (169 Held-Out Validation Screenshots)

| Screen Type Label | Support | Precision | Recall | F1-Score | Representation Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`document_article_view`** | 121 | 89.15% | 95.04% | 0.9200 | SUPPORTED |
| **`product_detail`** | 29 | 84.00% | 72.41% | 0.7778 | SUPPORTED |
| **`code_editor_terminal`** | 10 | 70.00% | 70.00% | 0.7000 | SUPPORTED |
| **`dashboard_settings`** | 5 | 100.00% | 100.00% | 1.0000 | SUPPORTED |
| **`cart_checkout_payment`** | 4 | 0.00% | 0.00% | 0.0000 | SUPPORTED |
| *`product_listing`* | 0 | 0.00% | 0.00% | 0.0000 | UNSUPPORTED_NO_DATA |
| *`login_authentication`* | 0 | 0.00% | 0.00% | 0.0000 | UNSUPPORTED_NO_DATA |
| *`messaging_chat`* | 0 | 0.00% | 0.00% | 0.0000 | UNSUPPORTED_NO_DATA |
| *`email_view`* | 0 | 0.00% | 0.00% | 0.0000 | UNSUPPORTED_NO_DATA |
| *`media_player_library`* | 0 | 0.00% | 0.00% | 0.0000 | UNSUPPORTED_NO_DATA |
| *`travel_booking`* | 0 | 0.00% | 0.00% | 0.0000 | UNSUPPORTED_NO_DATA |
| *`error_dialog_alert`* | 0 | 0.00% | 0.00% | 0.0000 | UNSUPPORTED_NO_DATA |
| *`other_screen`* | 0 | 0.00% | 0.00% | 0.0000 | UNSUPPORTED_NO_DATA |

### 5x5 Confusion Matrix

```
                      Pred: doc  product  code  dash  cart
GT: document_article     115       3       3     0     0
GT: product_detail         8      21       0     0     0
GT: code_editor            3       0       7     0     0
GT: dashboard_settings     0       0       0     5     0
GT: cart_checkout          3       1       0     0     0
```

---

## 5. Confidence Calibration & Reliability Audit

The model's output probability distribution was evaluated for calibration across 10 equal bins.

```
                    Confidence Distribution & Reliability Data
                    +----------------------------------------+
                    | Overall Mean Confidence:   0.9512      |
                    | Mean Conf (Correct):       0.9697      |
                    | Mean Conf (Incorrect):     0.8212      |
                    | ECE (10 Bins):             0.0825      |
                    | Brier Score:               0.2106      |
                    +----------------------------------------+
```

* **Expected Calibration Error (ECE)**: **0.0825 (8.25%)**. Indicates reasonable calibration for a un-calibrated baseline.
* **Separability**: Correct predictions have a high mean confidence of **0.9697**, while incorrect predictions drop to **0.8212**, providing a clear threshold region for confidence gating.

---

## 6. FP32 vs INT8 Quantization Audit

The PyTorch INT8 dynamic quantized checkpoint (`visual_screen_type_model_int8.pth`) was compared directly against FP32 predictions across all 169 validation samples.

| Parameter / Metric | FP32 Model | INT8 Dynamic Quantized | Delta / Equivalence |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | **87.57%** | **87.57%** | **0.00% Loss** |
| **Top-1 Agreement** | 100.00% | 100.00% | 169 / 169 Match |
| **Prediction Flips** | — | 0 | **Zero Flips** |
| **Mean Max Logit Diff** | — | 0.030917 | Negligible numerical shift |
| **Checkpoint File Size** | 5.93 MB | **4.24 MB** | **28.5% Compression** |

---

## 7. Latency & Resource Footprint Audit

Single-threaded CPU runtime benchmarks were executed on 224x224 input tensors.

| Metric | Benchmark Result | Target Budget | Compliance |
| :--- | :---: | :---: | :---: |
| **FP32 Checkpoint Size** | 5.93 MB | < 30 MB | **PASS** |
| **INT8 Checkpoint Size** | 4.24 MB | < 10 MB | **PASS** |
| **Model Load Time** | 74.93 ms | < 500 ms | **PASS** |
| **Cold CPU Forward Pass** | 20.14 ms | < 100 ms | **PASS** |
| **Warm CPU Latency (Mean)** | 23.40 ms | < 50 ms | **PASS** |
| **Warm CPU Latency (Median)** | **22.03 ms** | < 50 ms | **PASS** |
| **Peak RAM Overhead** | **15.93 MB** | 100–300 MB | **PASS** |

---

## 8. Summary of Phase 7G Audit Deliverables

All 11 Phase 7G audit artifacts have been generated and saved under `experiments/visual_understanding/phase7g/`:

1. `audit_phase7g.py`: Comprehensive audit script.
2. `phase7g_audit_report.md`: This primary audit document.
3. `phase7g_reproduced_metrics.json`: Exact reproduced metrics JSON.
4. `phase7g_per_class_metrics.csv`: Per-class precision/recall/F1 CSV.
5. `phase7g_confusion_matrix.csv`: 5x5 confusion matrix CSV.
6. `phase7g_error_analysis.csv`: Detailed 21 error instances CSV.
7. `phase7g_error_analysis.md`: Qualitative failure mode breakdown.
8. `phase7g_fp32_int8_comparison.csv`: Sample-level FP32 vs INT8 comparison CSV.
9. `phase7g_confidence_analysis.json`: Confidence stats, ECE, and Brier score.
10. `phase7g_runtime_memory_report.json`: CPU latency & RAM benchmark JSON.
11. `phase7g_artifact_consistency_report.md`: Artifact integrity report.
12. `phase7g_test_results.md`: Unit & regression test execution log.

---

## 9. Conclusion & Transition Readiness

Phase 7G confirms that the Phase 7F MobileNetV3-Small model is a robust, highly efficient, and 100% reproducible baseline. With **87.57% Top-1 accuracy**, **22.03 ms warm CPU latency**, **4.24 MB INT8 footprint**, and **0.0825 ECE**, it is fully validated and ready to serve as a supporting visual signal for future multimodal fusion phases.
