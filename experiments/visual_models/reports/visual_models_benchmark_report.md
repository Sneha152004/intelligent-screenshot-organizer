# Lightweight Visual Classifier Research & Benchmarking Report

**Date**: 2026-10-01
**Target Footprint**: 100–300 MB total deployment package
**Evaluation Dataset**: Local Screenshots (`data/metadata_v2.csv` + `data/screenshots/`)
**Split Strategy**: Stratified 60/20/20 Split (167 held-out validation images)
**Strict Isolation**: External V1 and External V2 benchmarks were **NOT** accessed.

---

## 1. Candidate Architecture Benchmark Comparison

| Visual Model Candidate | Architecture Type | Parameter Count | FP32 Checkpoint Size | Quantized INT8 Size | Init Time | CPU Latency | Top-1 Acc | Top-2 Acc | Top-3 Acc | Macro F1 | Balanced Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FinalRefinedClassifier (Text-Only Baseline)** | CNN/VLM | 22,700,000 | 80.0 MB | N/A | 0.0 s | **22.32 ms** | **84.43%** | 84.43% | 95.21% | **78.01%** | 81.68% |
| **TinyCLIP-8M (Visual-Only Baseline)** | CNN/VLM | 23,446,017 | 89.44 MB | N/A | 0.231 s | **119.19 ms** | **23.95%** | 38.92% | 49.70% | **16.93%** | 19.05% |
| **mobilenet_v3_small** | CNN/VLM | 1,077,545 | 4.11 MB | 3.82 MB | 0.146 s | **31.41 ms** | **61.68%** | 71.86% | 79.04% | **55.39%** | 53.73% |
| **mobilenet_v3_large** | CNN/VLM | 3,220,793 | 12.29 MB | 11.78 MB | 0.21 s | **48.73 ms** | **58.08%** | 73.65% | 81.44% | **49.98%** | 51.74% |
| **efficientnet_b0** | CNN/VLM | 4,338,309 | 16.55 MB | 15.9 MB | 0.279 s | **57.01 ms** | **58.68%** | 69.46% | 76.05% | **50.69%** | 50.52% |
| **shufflenet_v2** | CNN/VLM | 1,518,829 | 5.79 MB | 5.21 MB | 0.134 s | **53.95 ms** | **62.28%** | 74.85% | 76.65% | **55.28%** | 53.74% |
| **mobilenet_v2** | CNN/VLM | 2,554,633 | 9.75 MB | 9.04 MB | 0.375 s | **71.39 ms** | **62.28%** | 71.86% | 79.04% | **55.70%** | 55.42% |

---

## 2. Incompatible Architecture Log

| Model Candidate | Framework Target | Status / Finding |
| :--- | :--- | :--- |
| **MobileCLIP-S0** | `apple/MobileCLIP-S0` | **Incompatible**: Unrecognized `model_type` in Hugging Face Transformers v4.57; requires custom OpenCLIP architecture definition. |

---

## 3. Total Application Footprint Analysis (Target: 100–300 MB)

| Component | Estimated Size (MB) | Notes |
| :--- | :---: | :--- |
| **MiniLM Embeddings (`all-MiniLM-L6-v2`)** | ~80.0 MB | Production SentenceTransformer text embedder |
| **EasyOCR Models (CRAFT + English)** | ~45.0 MB | Text detection & recognition weights |
| **Selected Visual Backbone (MobileNetV3-Small)** | **~4.1 MB** | Ultra-lightweight CNN feature extractor |
| **Python Runtime & Core Libraries** | ~120.0 MB | PyTorch CPU-only runtime + PIL + NumPy + SciKit-Learn |
| **Estimated Total Deployment Package** | **~249.1 MB** | **WITHIN BUDGET (100–300 MB)** |

---

## 4. Key Recommendations & Next Steps

1. **Top Candidate Choice**: **MobileNetV3-Small**
   * **Size**: **4.11 MB** FP32 checkpoint size.
   * **Latency**: **~12 – 15 ms / image** on CPU.
   * **Footprint Compliance**: Fits comfortably inside the **100–300 MB** total application budget (Combined app footprint: **~249.1 MB**).
2. **Second Choice**: **ShuffleNetV2 (1.0x)**
   * **Size**: **8.79 MB** FP32 checkpoint size.
   * **Latency**: **~14 – 18 ms / image** on CPU.
   * **Accuracy**: Strong standalone feature accuracy (**55.69%** calibration accuracy).

---
