# Experimental Multimodal Classifier Evaluation Report

**Date**: 2026-10-01
**Dataset**: Local Screenshots (`data/metadata_v2.csv` + `data/screenshots/`)
**Split Strategy**: Stratified 60/20/20 (Train / Calibration / Held-out Validation)
**Held-out Validation Size**: 168 images (100% identical IDs for all 3 configs)
**Calibration Set Size**: 167 images (Used to select w_text = 0.70, w_visual = 0.30)
**Strict Isolation**: External V1 and External V2 benchmarks were **NOT** accessed.

---

## 1. Executive Metrics Comparison

| Metric | Config A: Text-Only | Config B: Visual-Only | Config C: Multimodal Fusion |
| :--- | :---: | :---: | :---: |
| **Primary Classifier Model** | `FinalRefinedClassifier` | `VisualCategoryEvidenceEngine` | `MultimodalScreenshotClassifier` |
| **Top-1 Accuracy** | **84.52%** | 23.81% | **82.74%** |
| **Top-2 Recall** | **84.52%** | 38.10% | **85.71%** |
| **Top-3 Recall** | **95.24%** | 49.40% | **94.05%** |
| **Balanced Accuracy** | **81.68%** | 18.89% | **80.26%** |
| **Macro F1 Score** | **78.02%** | 16.80% | **73.22%** |
| **Average Latency per Image** | **24.64 ms** | 119.05 ms | **144.09 ms** |
| **Additional Visual Latency** | Baseline (0 ms) | N/A | **+119.45 ms** |
| **Visual Fallback Frequency** | N/A | N/A | **0 / 168 (0.0%)** |

---

## 2. Per-Category Performance Breakdown

### Config A: Text-Only (`FinalRefinedClassifier`)
| Category | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Education** | 18 | 88.89% | 88.89% | 88.89% |
| **Shopping** | 27 | 93.10% | 100.00% | 96.43% |
| **Payment** | 5 | 83.33% | 100.00% | 90.91% |
| **Travel** | 24 | 92.31% | 100.00% | 96.00% |
| **Communication** | 15 | 93.75% | 100.00% | 96.77% |
| **Entertainment** | 26 | 96.30% | 100.00% | 98.11% |
| **Work & Career** | 28 | 100.00% | 85.71% | 92.31% |
| **Connectivity** | 6 | 17.65% | 50.00% | 26.09% |
| **General** | 19 | 40.00% | 10.53% | 16.67% |

### Config C: Multimodal Fusion (`MultimodalScreenshotClassifier`)
| Category | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Education** | 18 | 88.89% | 88.89% | 88.89% |
| **Shopping** | 27 | 89.66% | 96.30% | 92.86% |
| **Payment** | 5 | 41.67% | 100.00% | 58.82% |
| **Travel** | 24 | 92.31% | 100.00% | 96.00% |
| **Communication** | 15 | 83.33% | 100.00% | 90.91% |
| **Entertainment** | 26 | 92.59% | 96.15% | 94.34% |
| **Work & Career** | 28 | 96.00% | 85.71% | 90.57% |
| **Connectivity** | 6 | 30.00% | 50.00% | 37.50% |
| **General** | 19 | 33.33% | 5.26% | 9.09% |

---

## 3. 9x9 Confusion Matrices

### Config A (Text-Only)
| Ground Truth \ Predicted | Education | Shopping | Payment | Travel | Communication | Entertainment | Work & Career | Connectivity | General |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Education** | 16 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| **Shopping** | 0 | 27 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Payment** | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Travel** | 0 | 0 | 0 | 24 | 0 | 0 | 0 | 0 | 0 |
| **Communication** | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 0 | 0 |
| **Entertainment** | 0 | 0 | 0 | 0 | 0 | 26 | 0 | 0 | 0 |
| **Work & Career** | 1 | 1 | 1 | 0 | 0 | 0 | 24 | 0 | 1 |
| **Connectivity** | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 3 | 2 |
| **General** | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 14 | 2 |

### Config C (Multimodal)
| Ground Truth \ Predicted | Education | Shopping | Payment | Travel | Communication | Entertainment | Work & Career | Connectivity | General |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Education** | 16 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Shopping** | 0 | 26 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| **Payment** | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Travel** | 0 | 0 | 0 | 24 | 0 | 0 | 0 | 0 | 0 |
| **Communication** | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 0 | 0 |
| **Entertainment** | 0 | 0 | 0 | 0 | 1 | 25 | 0 | 0 | 0 |
| **Work & Career** | 1 | 1 | 1 | 0 | 0 | 0 | 24 | 0 | 1 |
| **Connectivity** | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 3 | 1 |
| **General** | 1 | 0 | 6 | 1 | 1 | 1 | 1 | 7 | 1 |

---

## 4. Key Comparative Findings & Design Analysis

1. **Text Classifier Superiority**:
   * The original text classifier (`FinalRefinedClassifier`) achieves **84.52% Top-1 Accuracy** and **78.02% Macro F1** on local validation screenshots.
   * High-precision domain keywords, regular expressions, and MiniLM text embeddings provide strong, explicit signals for UI screenshot text.

2. **Multimodal Fusion Impact**:
   * Multimodal fusion achieves **82.74% Top-1 Accuracy** and **73.22% Macro F1**.
   * Confidence-aware dynamic weighting preserves the text classifier's high accuracy when strong text indicators are present, while integrating zero-shot visual prompt matching for explainability.

3. **Latency Trade-Off**:
   * Text-only classification runs in **24.64 ms** per image on CPU.
   * Adding the TinyCLIP visual branch introduces an additional **+119.45 ms** per image, bringing total multimodal latency to **144.09 ms**.

4. **Preservation of Original Engine**:
   * The original text classifier was completely preserved and remains independently usable without any changes.
---
