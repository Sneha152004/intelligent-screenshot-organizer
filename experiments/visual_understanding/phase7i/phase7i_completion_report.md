# Phase 7I: Supervised Visual Screen-Type Model Benchmarking Completion Report

## 1. Executive Summary

Phase 7I has successfully completed the supervised lightweight visual model benchmarking across four vision architecture candidates (**MobileNetV3-Small**, **EfficientNet-Lite0**, **MobileViT-XS**, and **YOLO26n-cls**). The experiment strictly followed all Phase 7H-G frozen primary benchmark specifications and validation isolation protocols.

### Key Highlights:
* **Preflight Verification**: Verified exact frozen MD5 checksums for all input benchmark manifests (`phase7hg_primary_manifest.csv`: `0e49c374a467c72e9ee2298897cd0898`, `phase7hg_extended_manifest.csv`: `fe670d7e43f73e43112736edd77bdea5`, `phase7hg_zero_shot_probe.csv`: `2ae93b1fa44938b5505ea12bf2e90225`).
* **Validation Isolation**: Strict separation was maintained. The 25-sample validation set was used exclusively for final metrics evaluation; early stopping and checkpoint selection were strictly guided by the 24-sample calibration split.
* **Aspect-Ratio-Preserving Preprocessing**: Implemented letterbox padding to \(224 \times 224\) with RGBA alpha compositing and ImageNet normalization. No content-destroying crop preprocessing was performed.
* **Winner Model Candidate**: **YOLO26n-cls (YOLOv8n-cls)** achieved the highest performance across all primary metrics (**72.0% Top-1 Accuracy**, **56.86% Macro F1**, **69.63% Weighted F1**, **55.33% Balanced Accuracy**) with fast CPU inference (**18.11 ms median warm latency**) and compact model size (**5.58 MB INT8**).
* **Test Verification**: All 6 unit tests in `experiments/visual_understanding/phase7i/tests/test_phase7i.py` passed cleanly.

---

## 2. Frozen Input Manifest Integrity Audit

Prior to training, preflight hash verification confirmed that all benchmark inputs strictly match the frozen Phase 7H-G manifests:

```
Verified phase7hg_primary_manifest.csv: 0e49c374a467c72e9ee2298897cd0898 (OK)
Verified phase7hg_extended_manifest.csv: fe670d7e43f73e43112736edd77bdea5 (OK)
Verified phase7hg_zero_shot_probe.csv: 2ae93b1fa44938b5505ea12bf2e90225 (OK)
```

Dataset splits utilized:
* **Primary Train Partition**: 82 samples across 5 visual screen types (`document_article_view`, `travel_booking`, `dashboard_settings`, `media_player_library`, `code_editor_terminal`).
* **Primary Calibration Partition**: 24 samples (strictly for early stopping & loss monitoring).
* **Primary Validation Partition**: 25 samples (strictly for final evaluation).
* **Zero-Shot Probe**: 31 out-of-taxonomy `product_listing` samples.

---

## 3. Model Training & Quantization Summary

All models were trained for up to 30 epochs on CPU using AdamW optimizer (learning rate \(10^{-3}\), weight decay \(10^{-4}\)), CosineAnnealingLR scheduler, and inverse class frequency CrossEntropyLoss. FP32 checkpoints were saved alongside PyTorch dynamic INT8 quantized checkpoints (`quantize_dynamic`).

### Training Overview:
| Candidate | FP32 Parameters | FP32 Size | INT8 Size | Best Calibration Epoch | Best Calibration Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Small** | 2,542,856 | 5.94 MB | 5.95 MB | Epoch 18 | 0.8143 |
| **EfficientNet-Lite0** | 4,652,008 | 13.15 MB | 13.13 MB | Epoch 26 | 0.9412 |
| **MobileViT-XS** | 2,317,848 | 7.53 MB | 4.39 MB | Epoch 21 | 0.9984 |
| **YOLO26n-cls** | 2,719,288 | 5.59 MB | 5.58 MB | Epoch 24 | 0.7289 |

---

## 4. Benchmark Validation Evaluation Metrics

Evaluation performed on the isolated 25-sample primary validation partition:

| Metric | MobileNetV3-Small | EfficientNet-Lite0 | MobileViT-XS | YOLO26n-cls | Winner |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-1 Accuracy** | 48.0% | 56.0% | 68.0% | **72.0%** | **YOLO26n-cls** |
| **Top-2 Accuracy** | 76.0% | **84.0%** | 68.0% | 80.0% | **EfficientNet-Lite0** |
| **Top-3 Accuracy** | 88.0% | **100.0%** | 80.0% | 88.0% | **EfficientNet-Lite0** |
| **Macro Precision** | 0.6267 | 0.3833 | 0.4166 | **0.6386** | **YOLO26n-cls** |
| **Macro Recall** | 0.4844 | 0.4100 | 0.5111 | **0.5533** | **YOLO26n-cls** |
| **Macro F1** | 0.5018 | 0.3892 | 0.4483 | **0.5686** | **YOLO26n-cls** |
| **Weighted F1** | 0.4898 | 0.5183 | 0.6054 | **0.6963** | **YOLO26n-cls** |
| **Balanced Accuracy** | 48.44% | 41.00% | 51.11% | **55.33%** | **YOLO26n-cls** |
| **ECE (Calibration)** | 0.3805 | 0.3488 | **0.2326** | 0.3719 | **MobileViT-XS** |

---

## 5. CPU Edge Deployment Benchmarks

Latencies measured on x86 CPU single-sample inference (Batch Size = 1):

| Deployment Metric | MobileNetV3-Small | EfficientNet-Lite0 | MobileViT-XS | YOLO26n-cls |
| :--- | :---: | :---: | :---: | :---: |
| **Cold-Start Latency** | **21.77 ms** | 40.14 ms | 56.23 ms | 24.93 ms |
| **Median Warm Latency** | **12.80 ms** | 25.90 ms | 39.31 ms | 18.11 ms |
| **p95 Warm Latency** | **14.43 ms** | 28.75 ms | 46.67 ms | 20.73 ms |
| **Peak Memory Footprint** | **1744.75 MB** | 1779.84 MB | 1766.22 MB | 1761.25 MB |

---

## 6. Zero-Shot Out-of-Distribution Probe Metrics

Evaluated on 31 unseen `product_listing` screenshots:
* **MobileNetV3-Small**: Mean confidence 0.8526 (Over-confident misclassifications).
* **EfficientNet-Lite0**: Mean confidence 0.7568.
* **MobileViT-XS**: Mean confidence 0.6185.
* **YOLO26n-cls**: Mean confidence **0.3488** (Properly uncertain on unseen OOD classes).

---

## 7. Artifact Integrity & Safeguards

### Created Artifacts:
* `experiments/visual_understanding/phase7i/model_configs.json`
* `experiments/visual_understanding/phase7i/dataset_preprocessing.py`
* `experiments/visual_understanding/phase7i/train_models.py`
* `experiments/visual_understanding/phase7i/evaluate_models.py`
* `experiments/visual_understanding/phase7i/tests/test_phase7i.py`
* `experiments/visual_understanding/phase7i/model_checkpoints/` (8 checkpoint files: FP32 and INT8 for all 4 models)
* `experiments/visual_understanding/phase7i/metrics/` (9 JSON and CSV metric files)
* `experiments/visual_understanding/phase7i/phase7i_model_comparison.md`
* `experiments/visual_understanding/phase7i/phase7i_completion_report.md`

### Integrity Guarantees:
* No changes made to production codebase (`src/`).
* No changes made to rules, EasyOCR, MiniLM, or `FinalRefinedClassifier`.
* No training performed on validation split or zero-shot probe data.
* No modifications to master human review ledger or frozen Phase 7H-G manifests.

---

## 8. Final Verdict & Next Phase Hand-Off

* **Phase Status**: **COMPLETED & VERIFIED CLEANLY**
* **Recommended Production Model**: **YOLO26n-cls (YOLOv8n-cls)**
* **Next Step (Phase 7J)**: Integrate `visual_screen_type_yolo26n_cls_int8.pth` into the multi-signal hybrid screenshot classification architecture.
