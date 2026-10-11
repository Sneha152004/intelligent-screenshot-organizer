# Pretrained Visual Transfer Learning & Text Fusion Report

**Timestamp:** `2026-09-28T17:35:46.784614+00:00`  
**Objective:** Determine whether pretrained semantic visual representations (MobileNetV2 14MB, EfficientNet-B0 20MB) combined with MiniLM text embeddings improve screenshot classification beyond the 68.33% text baseline.

---

## Executive Summary

1. **Bagged Text Baseline Control (`Bagged MiniLM + SVM`)**:
   - **CV Top-1 Acc:** 63.3% | **External Top-1 Acc:** **68.33%** (41/60) | **External Macro F1:** 0.3708
2. **Selected Winner Architecture (`Bagged MiniLM + SVM (Text Baseline)`)**:
   - **CV Top-1 Acc:** **63.3%** | **External Top-1 Acc:** **68.3%** (40/60) | **External Macro F1:** **0.3708**
   - **External Top-2 Recall:** **75.0%** | **External Top-3 Recall:** **83.3%**
   - **Absolute Change over Baseline:** **+0.0 pp Top-1 Acc**, **+0.0 Macro F1**
3. **92% Target Threshold**: **NO (Bottleneck: Pretrained ImageNet domain gap on mobile/web screenshots & small training pool)**

---

## 1. Experimental Setup & Zero-Leakage Protocol

- **Training Pool:** 30 Screenshots (8 Development Benchmark + 22 Internal Validation Holdout).
- **Validation Protocol:** 5-Fold Cross-Validation on the 30 training samples.
- **Frozen External Benchmark:** 60 External Screenshots (`experiments/intelligence/external/images/`). Evaluated **ONCE** after model selection.

---

## 2. Comprehensive Transfer Learning Results Table

| Model Architecture | CV Top-1 | CV Macro P | CV Macro F1 | External Top-1 | External Macro P | External Macro F1 | Top-2 | Top-3 | End-to-End CPU Latency | Added Footprint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Bagged MiniLM + SVM (Text Baseline) (WINNER)** | 63.3% | 0.4385 | 0.4077 | **68.3%** | 0.3423 | **0.3708** | 75.0% | 83.3% | 243.9 ms | 0.0 MB |
| `MobileNetV2 Visual-only` | 26.7% | 0.1852 | 0.1719 | **23.3%** | 0.1845 | **0.1719** | 31.7% | 46.7% | 303.1 ms | 14.0 MB |
| `EfficientNet-B0 Visual-only` | 33.3% | 0.1806 | 0.1809 | **31.7%** | 0.2267 | **0.2122** | 55.0% | 56.7% | 313.2 ms | 20.5 MB |
| `MiniLM + MobileNetV2` | 56.7% | 0.3796 | 0.3620 | **68.3%** | 0.3715 | **0.4040** | 78.3% | 83.3% | 303.1 ms | 14.0 MB |
| `MiniLM + EfficientNet-B0` | 46.7% | 0.2341 | 0.2556 | **60.0%** | 0.3462 | **0.3502** | 78.3% | 83.3% | 313.2 ms | 20.5 MB |
| `MiniLM + MobileNetV2 + MLP (128, 64)` | 46.7% | 0.3550 | 0.3116 | **58.3%** | 0.3498 | **0.3416** | 71.7% | 78.3% | 303.2 ms | 14.0 MB |
| `MiniLM + EfficientNet-B0 + MLP (128, 64)` | 40.0% | 0.2000 | 0.2169 | **55.0%** | 0.3349 | **0.3319** | 71.7% | 76.7% | 313.2 ms | 20.5 MB |
| `Late Fusion (Softmax Weighted)` | 26.7% | 0.1810 | 0.1803 | **31.7%** | 0.2390 | **0.2121** | 66.7% | 80.0% | 303.2 ms | 14.0 MB |

---

## 3. Visual Error Analysis & Diagnostic Breakdown

- **Total External Test Screenshots:** 60
- **Errors Corrected by Visual Transfer Features (0):**
- None

- **Errors Introduced by Visual Transfer Features (0):**
- None

- **Net Improvement in Accuracy Count:** **0 images**

---

## 4. Latency & Resource Breakdown

For selected model `Bagged MiniLM + SVM (Text Baseline)`:
- **OCR Text Extraction Latency:** `~196.93 ms`
- **MiniLM Embedding Latency:** `~46.93 ms`
- **Visual Encoder Latency:** `~0.0 ms`
- **Classifier Inference Latency:** `~0.0514 ms`
- **Total End-to-End CPU Latency:** `~243.91 ms`
- **Added Model Storage Footprint:** `0.0 MB`

---

## 5. Integrity Verification Checklist

```text
External images modified: NO
External labels modified: NO
External data used for training: NO
External data used for feature selection: NO
External data used for tuning: NO
External data used for model selection: NO
Production classifier modified: NO
ChromaDB modified: NO
New dataset added: NO
Commit or push: NO
```
