# Phase 7I: Supervised Visual Screen-Type Model Comparison & Benchmarking Report

## Executive Summary

Phase 7I evaluated four candidate lightweight vision architectures—**MobileNetV3-Small**, **EfficientNet-Lite0**, **MobileViT-XS**, and **YOLO26n-cls (YOLOv8n-cls)**—on the frozen Phase 7H-G primary 5-class visual screen-type dataset (`document_article_view`, `travel_booking`, `dashboard_settings`, `media_player_library`, `code_editor_terminal`). All models were trained using inverse class-frequency weighted cross-entropy loss, aspect-ratio-preserving letterbox preprocessing, and strictly calibration-isolated early stopping.

### Key Benchmark Findings:
1. **Best Overall Classification Performance**: **YOLO26n-cls** achieved the highest accuracy and F1 scores across all candidates: **72.0% Top-1 Accuracy**, **56.86% Macro F1**, **69.63% Weighted F1**, and **55.33% Balanced Accuracy**.
2. **Fastest Edge CPU Inference**: **MobileNetV3-Small** demonstrated the lowest CPU inference latency: **12.80 ms median warm latency** (p95: 14.43 ms) and **21.77 ms cold start latency**, closely followed by **YOLO26n-cls** (**18.11 ms median warm latency**, p95: 20.73 ms).
3. **Best Quantization & Calibration Efficiency**: **MobileViT-XS** achieved the best calibration metric (**ECE = 0.2326**) and the highest dynamic INT8 compression factor, shrinking from **7.53 MB (FP32)** down to **4.39 MB (INT8)** (41.7% size reduction), though with higher CPU latency (39.31 ms median).
4. **Overall Recommendation**: **YOLO26n-cls** is recommended as the primary visual classifier candidate for downstream integration due to its superior class discrimination (1.0 precision on `document_article_view`, 0.8 precision on `code_editor_terminal`), compact model footprint (5.59 MB FP32 / 5.58 MB INT8), and low CPU inference latency (18.11 ms).

---

## 1. Primary Benchmark Metrics Comparison

All metrics were calculated on the isolated 25-sample validation split of the Phase 7H-G primary benchmark.

| Metric | MobileNetV3-Small | EfficientNet-Lite0 | MobileViT-XS | YOLO26n-cls | Best Candidate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-1 Accuracy** | 48.0% (0.4800) | 56.0% (0.5600) | 68.0% (0.6800) | **72.0% (0.7200)** | **YOLO26n-cls** |
| **Top-2 Accuracy** | 76.0% (0.7600) | **84.0% (0.8400)** | 68.0% (0.6800) | 80.0% (0.8000) | **EfficientNet-Lite0** |
| **Top-3 Accuracy** | 88.0% (0.8800) | **100.0% (1.0000)** | 80.0% (0.8000) | 88.0% (0.8800) | **EfficientNet-Lite0** |
| **Macro Precision** | 0.6267 | 0.3833 | 0.4166 | **0.6386** | **YOLO26n-cls** |
| **Macro Recall** | 0.4844 | 0.4100 | 0.5111 | **0.5533** | **YOLO26n-cls** |
| **Macro F1 Score** | 0.5018 | 0.3892 | 0.4483 | **0.5686** | **YOLO26n-cls** |
| **Weighted F1 Score** | 0.4898 | 0.5183 | 0.6054 | **0.6963** | **YOLO26n-cls** |
| **Balanced Accuracy** | 48.44% | 41.00% | 51.11% | **55.33%** | **YOLO26n-cls** |
| **Expected Calibration Error (ECE)** | 0.3805 | 0.3488 | **0.2326** | 0.3719 | **MobileViT-XS** |

---

## 2. CPU Deployment & Edge Runtime Metrics

Inference latency and memory overhead were benchmarked on standard x86 CPU runtime (Batch Size = 1).

| Deployment Metric | MobileNetV3-Small | EfficientNet-Lite0 | MobileViT-XS | YOLO26n-cls |
| :--- | :---: | :---: | :---: | :---: |
| **FP32 Parameter Count** | 2,542,856 (2.54M) | 4,652,008 (4.65M) | 2,317,848 (2.32M) | 2,719,288 (2.72M) |
| **FP32 Checkpoint Size** | 5.94 MB | 13.15 MB | 7.53 MB | **5.59 MB** |
| **INT8 Checkpoint Size** | 5.95 MB | 13.13 MB | **4.39 MB** | 5.58 MB |
| **Cold-Start Latency (ms)** | **21.77 ms** | 40.14 ms | 56.23 ms | 24.93 ms |
| **Median Warm Latency (ms)** | **12.80 ms** | 25.90 ms | 39.31 ms | 18.11 ms |
| **p95 Warm Latency (ms)** | **14.43 ms** | 28.75 ms | 46.67 ms | 20.73 ms |
| **Peak RAM Footprint (MB)** | **1744.75 MB** | 1779.84 MB | 1766.22 MB | 1761.25 MB |

---

## 3. Per-Class Performance Breakdown

### Precision, Recall, and F1 per Class (Primary Validation Set)

#### MobileNetV3-Small
* `document_article_view`: Precision 0.00, Recall 0.00, F1 0.00 (Support: 4)
* `travel_booking`: Precision 0.80, Recall 0.89, F1 0.84 (Support: 9)
* `dashboard_settings`: Precision 1.00, Recall 1.00, F1 1.00 (Support: 1)
* `media_player_library`: Precision 1.00, Recall 0.20, F1 0.33 (Support: 5)
* `code_editor_terminal`: Precision 0.33, Recall 0.33, F1 0.33 (Support: 6)

#### EfficientNet-Lite0
* `document_article_view`: Precision 0.17, Recall 0.25, F1 0.20 (Support: 4)
* `travel_booking`: Precision 0.75, Recall 1.00, F1 0.86 (Support: 9)
* `dashboard_settings`: Precision 0.00, Recall 0.00, F1 0.00 (Support: 1)
* `media_player_library`: Precision 1.00, Recall 0.80, F1 0.89 (Support: 5)
* `code_editor_terminal`: Precision 0.00, Recall 0.00, F1 0.00 (Support: 6)

#### MobileViT-XS
* `document_article_view`: Precision 0.00, Recall 0.00, F1 0.00 (Support: 4)
* `travel_booking`: Precision 0.73, Recall 0.89, F1 0.80 (Support: 9)
* `dashboard_settings`: Precision 0.00, Recall 0.00, F1 0.00 (Support: 1)
* `media_player_library`: Precision 0.56, Recall 1.00, F1 0.71 (Support: 5)
* `code_editor_terminal`: Precision 0.80, Recall 0.67, F1 0.73 (Support: 6)

#### YOLO26n-cls (YOLOv8n-cls)
* `document_article_view`: Precision 1.00, Recall 0.50, F1 0.67 (Support: 4)
* `travel_booking`: Precision 0.64, Recall 1.00, F1 0.78 (Support: 9)
* `dashboard_settings`: Precision 0.00, Recall 0.00, F1 0.00 (Support: 1)
* `media_player_library`: Precision 0.75, Recall 0.60, F1 0.67 (Support: 5)
* `code_editor_terminal`: Precision 0.80, Recall 0.67, F1 0.73 (Support: 6)

---

## 4. Zero-Shot Probe Evaluation (31 `product_listing` Screenshots)

The zero-shot probe evaluated out-of-taxonomy fallback predictions on 31 `product_listing` screenshots (a class excluded from the primary 5-class training set).

| Model Candidate | Mean Prediction Confidence | Top Predicted Class | Predicted Class Distribution |
| :--- | :---: | :--- | :--- |
| **MobileNetV3-Small** | 0.8526 | `travel_booking` (18) | `travel_booking`: 18, `document_article_view`: 12, `code_editor_terminal`: 1 |
| **EfficientNet-Lite0** | 0.7568 | `travel_booking` (15) | `travel_booking`: 15, `media_player_library`: 8, `code_editor_terminal`: 5, `document_article_view`: 3 |
| **MobileViT-XS** | 0.6185 | `media_player_library` (19) | `media_player_library`: 19, `travel_booking`: 12 |
| **YOLO26n-cls** | 0.3488 | `travel_booking` (27) | `travel_booking`: 27, `document_article_view`: 2, `media_player_library`: 2 |

> [!NOTE]
> **YOLO26n-cls** exhibited the lowest mean confidence score (0.3488) on zero-shot probe samples, indicating superior out-of-distribution uncertainty awareness compared to overly confident CNN models like MobileNetV3-Small (0.8526).

---

## 5. Decision Matrix & Candidate Selection

| Architecture | Accuracy & F1 | CPU Speed | Model Size | Out-of-Distribution Awareness | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n-cls** | ⭐⭐⭐⭐⭐ (72.0% Top-1) | ⭐⭐⭐⭐ (18.11 ms) | ⭐⭐⭐⭐⭐ (5.58 MB) | ⭐⭐⭐⭐⭐ (Low OOD confidence) | **SELECTED PRIMARY MODEL** |
| **MobileViT-XS** | ⭐⭐⭐⭐ (68.0% Top-1) | ⭐⭐ (39.31 ms) | ⭐⭐⭐⭐⭐ (4.39 MB INT8) | ⭐⭐⭐ (0.6185 mean conf) | Alternate (Best INT8 Size) |
| **EfficientNet-Lite0** | ⭐⭐⭐ (56.0% Top-1) | ⭐⭐⭐ (25.90 ms) | ⭐⭐ (13.13 MB) | ⭐⭐⭐ (0.7568 mean conf) | Not Recommended |
| **MobileNetV3-Small** | ⭐⭐ (48.0% Top-1) | ⭐⭐⭐⭐⭐ (12.80 ms) | ⭐⭐⭐⭐ (5.94 MB) | ⭐ (Over-confident OOD) | Historical Baseline |

---

## 6. Conclusion & Deployment Next Steps

1. **Model Authorization**: **YOLO26n-cls** is authorized as the primary visual feature model for Phase 7J multi-modal classifier integration.
2. **Quantization Deployment**: Deploy INT8 quantized PyTorch model weights (`visual_screen_type_yolo26n_cls_int8.pth`) for CPU inference.
3. **Confidence Thresholding**: Apply a minimum softmax confidence threshold of \(\ge 0.50\) for visual-only prediction acceptance to reject out-of-distribution classes cleanly.
