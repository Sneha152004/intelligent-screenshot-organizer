# Phase 7I-A: Model Benchmark Integrity, Quantization & OOD Audit Completion Report

## 1. Final Audit Verdict

### **VERDICT: AUDIT PASSED WITH DOCUMENTED LIMITATIONS**

The independent audit of Phase 7I supervised visual screen-type benchmarking successfully verified all cryptographic hashes, model checkpoints, training logs, and validation predictions. The primary classification metrics for all four candidates were recalculated and verified with **100% precision**.

The audit identified and documented four key technical limitations and reporting nuances that must be recognized prior to Phase 7J deployment optimization:

1. **Dataset Narrative Typo**: The Phase 7I completion report narrative stated *82 train* and *24 calibration* samples. The actual frozen manifest and PyTorch data loaders executed on **83 train**, **27 calibration**, and **25 validation** samples.
2. **Parameter Count Calibration**: The reported parameter counts in config files reflected default 1000-class ImageNet estimates. Empirical PyTorch counting on the 5-class models revealed **YOLOv8n-cls (`yolo26n_cls`)** is actually the most lightweight model (**1,444,693 parameters**).
3. **Quantization Scope Limitation**: PyTorch `quantize_dynamic` targeted `{nn.Linear}` layers only. MobileViT-XS achieved a **41.7% size reduction** (7.53 MB -> 4.39 MB) because 57.7% of its parameters are Linear layers in Transformer blocks. The three pure CNN models (MobileNetV3-Small, EfficientNet-Lite0, YOLOv8n-cls) contain >98.6% Conv2d layers and showed negligible size change.
4. **Memory & OOD Attribution**: Memory readings (~1744 MB - 1779 MB) represent Total Process Resident Set Size (RSS) under Desktop Python/PyTorch runtime, not model dynamic allocation. Low softmax confidence on YOLOv8n-cls (~0.35 max conf) occurs across both in-distribution and out-of-distribution samples due to uncalibrated output logits, rather than active OOD rejection.

---

## 2. Model Performance & Evaluation Breakdown

| Evaluation Dimension | MobileNetV3-Small | EfficientNet-Lite0 | MobileViT-XS | YOLOv8n-cls (`yolo26n_cls`) | Leading Candidate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-1 Accuracy** | 48.0% | 56.0% | 68.0% | **72.0%** | **YOLOv8n-cls** |
| **Macro F1 Score** | 0.5018 | 0.3892 | 0.4483 | **0.5686** | **YOLOv8n-cls** |
| **Weighted F1 Score** | 0.4898 | 0.5183 | 0.6054 | **0.6963** | **YOLOv8n-cls** |
| **Balanced Accuracy** | 48.44% | 41.00% | 51.11% | **55.33%** | **YOLOv8n-cls** |
| **Calibration (ECE)** | 0.3805 | 0.3488 | **0.2326** | 0.3719 | **MobileViT-XS** |
| **5-Class Parameters** | 1,522,981 | 3,377,413 | 1,934,773 | **1,444,693** | **YOLOv8n-cls** |
| **FP32 Checkpoint Size** | 5.94 MB | 13.15 MB | 7.53 MB | **5.59 MB** | **YOLOv8n-cls** |
| **INT8 Checkpoint Size** | 5.95 MB | 13.13 MB | **4.39 MB** | 5.58 MB | **MobileViT-XS** |
| **Median Warm CPU Latency** | **12.80 ms** | 25.90 ms | 39.31 ms | 18.11 ms | **MobileNetV3-Small** |
| **FP32/INT8 Agreement** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | All 4 Candidates |

---

## 3. Strongest Model Candidate Recommendation

**YOLOv8n-cls (`yolo26n_cls`)** is authorized as the primary visual feature classifier candidate for Phase 7J integration.

### Core Justification:
- **Highest Accuracy & Class Discrimination**: Achieves **72.0% Top-1 Accuracy**, **56.86% Macro F1**, **69.63% Weighted F1**, **1.00 Precision** on `document_article_view`, and **0.80 Precision** on `code_editor_terminal`.
- **Lowest Parameter Count**: Possesses the smallest overall 5-class parameter count in the benchmark (**1,444,693 parameters**).
- **Fast CPU Runtime**: Delivers **18.11 ms median warm CPU latency** (p95: 20.73 ms) with a compact **5.58 MB INT8** checkpoint size.

---

## 4. Key Deliverables Created in Phase 7I-A

Located in `experiments/visual_understanding/phase7ia/`:

1. `phase7ia_dataset_reproducibility_audit.md`: Audit of frozen manifest hashes and record count narrative resolution.
2. `phase7ia_metric_recalculation.json`: Recalculated validation metrics for FP32 and INT8 checkpoints.
3. `phase7ia_architecture_verification.md`: Resolution of `YOLOv8n-cls` backbone implementation and 5-class parameter counts.
4. `phase7ia_quantization_audit.json`: Layer-level analysis of dynamic linear quantization.
5. `phase7ia_fp32_int8_comparison.csv`: Comparative table of FP32 vs INT8 metrics, sizes, latencies, and 100% agreement.
6. `phase7ia_cpu_benchmark_audit.md`: Audit of latency timing methodology and process RSS memory attribution.
7. `phase7ia_ood_confidence_analysis.json`: In-distribution vs zero-shot probe entropy and confidence distribution analysis.
8. `phase7ia_completion_report.md`: Master completion report and final audit verdict.
9. [test_phase7ia.py](file:///C:/Users/KIIT0001/Documents/intelligent-screenshot-organizer/experiments/visual_understanding/phase7ia/tests/test_phase7ia.py): Unit test suite verifying audit integrity.

---

## 5. Scope Restrictions Safeguards Confirmation

- **No models were retrained**.
- **No model checkpoints were modified**.
- **No Phase 7H or 7I artifacts were altered**.
- **No production code in `src/` was modified**.
- **No git commits or pushes were executed**.
