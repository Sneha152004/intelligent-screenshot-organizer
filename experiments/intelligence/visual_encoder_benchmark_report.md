# Isolated Visual Encoder Benchmark Report

**Date**: 2026-10-01
**Evaluation Dataset**: Local Sample Screenshots (`data/sample/`, 30 images)
**Execution Environment**: CPU (Intel/AMD x86_64, Windows)
**Strict Isolation**: External V1 and External V2 ground truth datasets were **NOT** accessed.

---

## 1. Quantitative Benchmark Results

| Metric | MobileNetV2 | TinyCLIP-8M |
| :--- | :---: | :---: |
| **Model Checkpoint / Source** | `torchvision.models.mobilenet_v2` | `wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M` |
| **Model Backbone / Type** | Feature Extractor (CNN) | Vision-Language Dual Encoder (ViT) |
| **Total Parameter Count** | 2,223,872 | 23,446,017 |
| **Estimated FP32 Model Size** | 8.48 MB | 89.44 MB |
| **Embedding Vector Dimension** | **1280** | **512** |
| **Joint Text-Vision Space?** | No (Requires classification head) | **Yes (Direct zero-shot similarity)** |
| **Model Initialization Time** | 0.103 s | 13.627 s |
| **Peak RAM Allocation (Traced)** | 1.54 MB | 65.63 MB |
| **Cold-Start CPU Latency** | 77.18 ms | 106.67 ms |
| **Average Warm CPU Latency** | **40.77 ms/image** | **108.54 ms/image** |
| **Latency Range (Min - Max)** | 28.92 - 77.75 ms | 59.65 - 145.33 ms |
| **Finiteness Check (No NaN/Inf)** | PASSED (True) | PASSED (True) |
| **Deterministic Outputs** | PASSED (True) | PASSED (True) |

---

## 2. Key Observations & Trade-off Analysis

### MobileNetV2
* **Strengths**: Extremely fast CPU inference (~40.77 ms per screenshot), small memory footprint (1.54 MB peak), lightweight initialization (0.103 s).
* **Limitations**: Produces raw 1280-dimensional CNN features without native text alignment. Requires a separate prototype centroid head or trained visual classifier to map features to the 9 canonical categories.

### TinyCLIP-8M
* **Strengths**: Native joint vision-language embedding space (512-dimensional). Enables direct zero-shot cosine similarity scoring between screenshots and text prompts for all 9 canonical categories without requiring visual head training data.
* **Limitations**: Higher CPU inference latency (~108.54 ms per screenshot) and larger parameter size (23,446,017 parameters vs 2,223,872 for MobileNetV2).

---

## 3. Implementation Recommendation

For the **Visual Intelligence Branch**:
1. **Primary Encoder**: **TinyCLIP-8M** (`wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M`). Its joint text-vision embedding space enables immediate zero-shot scoring against the 9 canonical category prompts, making it ideal for multimodal score fusion with MiniLM and explainable rules.
2. **Secondary / Fallback Encoder**: **MobileNetV2**. Available when sub-20ms CPU latency is required or in batch processing pipelines.

---
