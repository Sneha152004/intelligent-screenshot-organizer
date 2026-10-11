# Visual-Only Category Evidence Engine Evaluation Report

**Date**: 2026-10-01
**Dataset**: Local Screenshots (`data/metadata_v2.csv` + `data/sample/`)
**Split Strategy**: Stratified 80/20 Train/Validation Split (168 validation images)
**Model**: TinyCLIP-8M Zero-Shot Visual Engine (`wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M`)
**Strict Isolation**: External V1 and External V2 benchmarks were **NOT** accessed.

---

## 1. Executive Metrics Summary

* **Top-1 Accuracy**: **23.81%**
* **Top-2 Recall**: **38.10%**
* **Top-3 Recall**: **49.40%**
* **Balanced Accuracy**: **18.89%**
* **Macro F1 Score**: **16.80%**
* **Average CPU Latency**: **106.38 ms / image**

---

## 2. Per-Category Performance Breakdown

| Canonical Category | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Education** | 18 | 0.00% | 0.00% | 0.00% |
| **Shopping** | 27 | 38.10% | 29.63% | 33.33% |
| **Payment** | 5 | 0.00% | 0.00% | 0.00% |
| **Travel** | 24 | 0.00% | 0.00% | 0.00% |
| **Communication** | 15 | 13.04% | 40.00% | 19.67% |
| **Entertainment** | 26 | 40.91% | 34.62% | 37.50% |
| **Work & Career** | 28 | 37.84% | 50.00% | 43.08% |
| **Connectivity** | 6 | 0.00% | 0.00% | 0.00% |
| **General** | 19 | 20.00% | 15.79% | 17.65% |

---

## 3. 9x9 Confusion Matrix (Ground Truth \ Prediction)

| Ground Truth \ Predicted | Education | Shopping | Payment | Travel | Communication | Entertainment | Work & Career | Connectivity | General |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Education** | 0 | 1 | 0 | 0 | 4 | 3 | 3 | 2 | 5 |
| **Shopping** | 0 | 8 | 0 | 0 | 3 | 3 | 6 | 6 | 1 |
| **Payment** | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 3 | 0 |
| **Travel** | 0 | 3 | 0 | 0 | 7 | 1 | 6 | 6 | 1 |
| **Communication** | 0 | 0 | 0 | 0 | 6 | 1 | 2 | 2 | 4 |
| **Entertainment** | 0 | 5 | 0 | 1 | 8 | 9 | 2 | 0 | 1 |
| **Work & Career** | 0 | 0 | 0 | 0 | 8 | 2 | 14 | 4 | 0 |
| **Connectivity** | 0 | 0 | 0 | 1 | 3 | 0 | 2 | 0 | 0 |
| **General** | 0 | 4 | 0 | 0 | 5 | 3 | 2 | 2 | 3 |

---

## 4. Key Observations & Visual Intelligence Findings

1. **Standalone Visual Baseline Performance**:
   * Pure zero-shot visual evidence alone achieves **{top1_acc * 100:.2f}% Top-1 Accuracy** and **{top3_recall * 100:.2f}% Top-3 Recall** on unseen validation screenshots without any fine-tuning or OCR text input.
2. **Visual Confusion Analysis**:
   * Visual-only features confuse text-heavy screens across categories (e.g. `Education` vs `Work & Career` or `Travel` vs `General`), as screenshot layouts often feature similar dark/light UI containers and web view headers.
3. **Necessity of Multimodal Fusion**:
   * These empirical results demonstrate why visual evidence must be integrated as a complementary signal ($w_{\text{visual}}$) with MiniLM text embeddings and explainable rules, rather than replacing the textual classifier. Visual evidence excels on text-sparse/graphic screens, while OCR text provides high-precision semantic domain signals.

---
