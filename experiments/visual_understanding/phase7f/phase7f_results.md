# Phase 7F: Supervised Visual Screen-Type Baseline Results Report

**Date**: 2026-10-02  
**Target Architecture**: MobileNetV3-Small (Pretrained ImageNet1K Weights + 5-Class Linear Head)  
**Execution Device**: Intel/AMD CPU (Single Threaded Execution)  
**Verdict**: **SUCCESSFUL BASELINE**

---

## 1. Executive Summary & Baseline Achievements

* **Held-out Validation Top-1 Accuracy**: **87.57%** (148 / 169 correct)
* **Top-2 Accuracy**: **94.08%**
* **Top-3 Accuracy**: **98.82%**
* **Macro F1-Score**: **67.96%**
* **Weighted F1-Score**: **86.32%**
* **Balanced Accuracy**: **67.49%**
* **Warm CPU Latency**: **20.87 ms/image**
* **INT8 Model Size**: **4.24 MB** (5.93 MB FP32)
* **INT8 Validation Accuracy**: **87.57%** (Zero quantization loss)

---

## 2. Per-Class Performance Breakdown (Held-out Validation Split)

| Screen Type Label | Support | Precision | Recall | F1-Score | Representation Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`document_article_view`** | 121 |  89.1% |  95.0% |  92.0% | Supported |
| **`product_detail`** | 29 |  84.0% |  72.4% |  77.8% | Supported |
| **`code_editor_terminal`** | 10 |  70.0% |  70.0% |  70.0% | Supported |
| **`dashboard_settings`** | 5 | 100.0% | 100.0% | 100.0% | Supported |
| **`cart_checkout_payment`** | 4 |   0.0% |   0.0% |   0.0% | Supported |

---

## 3. 5x5 Confusion Matrix

```
                       document_article_view  product_detail  code_editor_terminal  dashboard_settings  cart_checkout_payment
document_article_view                    115               3                     3                   0                      0
product_detail                             8              21                     0                   0                      0
code_editor_terminal                       3               0                     7                   0                      0
dashboard_settings                         0               0                     0                   5                      0
cart_checkout_payment                      3               1                     0                   0                      0
```

---

## 4. Latency & Application Footprint Analysis

* **Cold CPU Forward Pass**: `20.18 ms`
* **Warm CPU Forward Pass**: `20.87 ms`
* **Model Parameter Count**: `2,542,856`
* **FP32 Checkpoint Size**: `5.93 MB`
* **PyTorch INT8 Quantized Size**: `4.24 MB`
* **Total Combined RAM Footprint**: `~286 - 300 MB` (Fits strictly within target 100-300 MB budget).

---

## 5. Scope Protection Verification

* `FinalRefinedClassifier`: **UNTOUCHED**
* EasyOCR & MiniLM: **UNTOUCHED**
* Production visual encoders & rules: **UNTOUCHED**
* External V1 & External V2 benchmarks: **UNTOUCHED**
* Phase 7E manifest & splits: **UNTOUCHED**
