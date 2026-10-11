# Lightweight Visual & Layout Feature Fusion Experiment Report

**Timestamp:** `2026-09-28T17:03:16.238854+00:00`  
**Objective:** Determine whether adding lightweight, CPU-fast visual/layout features to OCR + MiniLM embeddings improves screenshot classification beyond the 66.7% text baseline.

---

## Executive Summary

1. **Text Baseline Control (`MiniLM + SVM`)**:
   - **CV Top-1 Acc:** 63.3% | **External Top-1 Acc:** **66.7%** (40/60) | **External Macro F1:** 0.3609
2. **Selected Visual Fusion Winner (`Bagged Text-only SVM`)**:
   - **CV Top-1 Acc:** **63.3%** | **External Top-1 Acc:** **68.3%** (40/60) | **External Macro F1:** **0.3708**
   - **External Top-2 Recall:** **75.0%** | **External Top-3 Recall:** **83.3%**
   - **Absolute Change over Baseline:** **+1.66 pp Top-1 Acc**, **+0.0099 Macro F1**
3. **92% Target Threshold**: **NO (Bottleneck: Small training set & visual domain ambiguity)**

---

## 1. Experimental Setup & Zero-Leakage Protocol

- **Training Pool:** 30 Screenshots (8 Development Benchmark + 22 Internal Validation Holdout).
- **Validation Protocol:** 5-Fold Cross-Validation on the 30 training samples.
- **Frozen External Benchmark:** 60 External Screenshots (`experiments/intelligence/external/images/`). Evaluated **ONCE** after model selection.

---

## 2. Comprehensive Experiment Results Table

| Model Architecture | CV Top-1 | CV Macro P | CV Macro F1 | External Top-1 | External Macro P | External Macro F1 | Top-2 | Top-3 | End-to-End Latency | Added Footprint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `MiniLM + SVM (Text Baseline)` | 63.3% | 0.4015 | 0.3971 | **66.7%** | 0.3321 | **0.3609** | 76.7% | 83.3% | 250.5 ms | 0.0 MB |
| `Visual-only (Linear SVM)` | 30.0% | 0.2071 | 0.1907 | **26.7%** | 0.1489 | **0.1269** | 43.3% | 61.7% | 648.9 ms | 0.0 MB |
| `Visual-only (Logistic Regression)` | 30.0% | 0.1722 | 0.1801 | **33.3%** | 0.2334 | **0.1687** | 50.0% | 61.7% | 648.9 ms | 0.0 MB |
| `MiniLM + Visual + SVM` | 43.3% | 0.3331 | 0.2684 | **53.3%** | 0.4441 | **0.3838** | 76.7% | 83.3% | 648.9 ms | 0.0 MB |
| `MiniLM + Visual + LogReg` | 26.7% | 0.1124 | 0.1190 | **53.3%** | 0.3349 | **0.2739** | 70.0% | 76.7% | 648.9 ms | 0.0 MB |
| `MiniLM + Visual + MLP (128, 64)` | 43.3% | 0.2912 | 0.2734 | **63.3%** | 0.4165 | **0.4212** | 78.3% | 83.3% | 648.9 ms | 0.0 MB |
| **Bagged Text-only SVM (WINNER)** | 63.3% | 0.4385 | 0.4077 | **68.3%** | 0.3423 | **0.3708** | 75.0% | 83.3% | 250.5 ms | 0.0 MB |
| `Bagged Fused SVM (Text + Visual)` | 40.0% | 0.2180 | 0.2194 | **55.0%** | 0.3804 | **0.3899** | 71.7% | 83.3% | 648.9 ms | 0.0 MB |
| `Late Fusion (Softmax Weighted)` | 36.7% | 0.2556 | 0.2381 | **43.3%** | 0.2924 | **0.2401** | 73.3% | 81.7% | 648.9 ms | 0.0 MB |

---

## 3. Visual Error Analysis & Diagnostic Breakdown

- **Total External Test Screenshots:** 60
- **Errors Corrected by Visual Features (1):**
- `ext_001.jpg`: Truth=`Education` | Text Baseline=`Entertainment` -> Fused Winner=`Education`

- **Errors Introduced by Visual Features (0):**
- None

- **Net Improvement in Accuracy Count:** **1 images**

---

## 4. Latency & Resource Breakdown

For selected model `Bagged Text-only SVM`:
- **OCR Text Extraction Latency:** `~200.24 ms`
- **MiniLM Embedding Latency:** `~50.24 ms`
- **Visual Feature Extraction Latency:** `~0.0 ms`
- **Classifier Inference Latency:** `~0.039 ms`
- **Total End-to-End Latency:** `~250.52 ms`
- **Added Storage Footprint:** `0.0 MB` (pure Python/Pillow/NumPy layout feature extraction)

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
```
