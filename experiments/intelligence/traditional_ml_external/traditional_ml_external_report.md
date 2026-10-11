# Traditional ML External Benchmark Evaluation Report
**Date**: 2026-09-30T12:35 UTC
**Evaluation Scope**: Frozen 60-Image External Benchmark (`ext_001.jpg` – `ext_060.jpg`)
**Training Base**: Full 833 Valid Training Screenshots
**Anti-Leakage Guarantee**: Models, vectorizers, rules, and hyperparameters **100% FROZEN** prior to external evaluation. **0% Post-Test Tuning**.

---
## 1. Objective
The objective of this phase is to evaluate the generalization capacity of **five frozen traditional machine-learning models** trained on the expanded 833 valid screenshot training pool against the frozen 60-image external benchmark dataset.

---
## 2. Frozen Benchmark Description
- **External Benchmark Size**: Exactly 60 screenshots (`ext_001.jpg` through `ext_060.jpg`).
- **Ground-Truth Source**: `experiments/intelligence/external/evaluation_targets.py` (Curated frozen reference annotations).
- **Category Balance (10 Categories)**: Education, Shopping, Payment, Travel, Coding, Communication, Entertainment, Work & Career, Connectivity, General.

---
## 3. Frozen Model Configurations Evaluated
1. **Complement Naive Bayes**: Config 4 (TF-IDF + MiniLM, Content-Focused OCR)
2. **Logistic Regression**: Config 4 (TF-IDF + MiniLM, Content-Focused OCR)
3. **Random Forest**: Config 3 (TF-IDF + Rules, Content-Focused OCR)
4. **Linear SVC**: Config 1 (TF-IDF Only, Content-Focused OCR)
5. **Gradient Boosting**: Config 4 (TF-IDF + MiniLM, Content-Focused OCR)

---
## 4. Master Cross-Model Performance Comparison
| Model | CV Top-1 | External Top-1 | CV Macro F1 | External Macro F1 | Balanced Accuracy | Top-2 Recall | Top-3 Recall | MRR | Accuracy Drop | Relative Retention |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Complement Naive Bayes** | 99.64% | **1.67% (1/60)** | 0.9963 | **0.0033** | 11.11% | 28.33% | 51.67% | 0.3193 | -97.97% | 1.7% |
| **Logistic Regression** | 99.16% | **1.67% (1/60)** | 0.9897 | **0.0036** | 11.11% | 25.00% | 45.00% | 0.3044 | -97.49% | 1.7% |
| **Random Forest** | 99.04% | **1.67% (1/60)** | 0.9914 | **0.0033** | 11.11% | 20.00% | 45.00% | 0.2858 | -97.37% | 1.7% |
| **Linear SVC** | 99.04% | **1.67% (1/60)** | 0.9901 | **0.0035** | 11.11% | 3.33% | 10.00% | 0.2187 | -97.37% | 1.7% |
| **Gradient Boosting** | 98.68% | **6.67% (4/60)** | 0.9883 | **0.1037** | 20.37% | 15.00% | 26.67% | 0.2660 | -92.01% | 6.8% |

---
## 5. Frozen External Benchmark Performance Findings
- **Top Performing Model**: **Gradient Boosting** achieved **6.67% Top-1 Accuracy (4/60)**.
- **Performance Drop**: Across all traditional ML models trained on the text-only 833 screenshot dataset, the performance drop from 5-fold CV to the 60-image external benchmark was severe (ranging from **-97.97%** to **-92.01%**).
- **Domain Shift Diagnosis**: Traditional ML classifiers trained on internal OCR text features severely overfit to internal training artifacts and synthetic text notes, causing near-total classification failure on un-annotated external screenshots.

---
## 6. Model Agreement & Error Breakdown
- **Screenshots All 5 Models Agree On**: 49/60 (81.7%)
- **Screenshots All 5 Models Fail On**: 56/60
- **Majority Vote Diagnostic Accuracy**: 1/60 (1.67%)

---
## 7. Latency and Resource Footprint
| Model | OCR Latency | MiniLM Latency | Inference Latency | Total Latency / Img | Additional Footprint |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Complement Naive Bayes** | 240.0 ms | 297.5 ms | 0.00 ms | **537.5 ms** | 91.5 MB |
| **Logistic Regression** | 240.0 ms | 297.5 ms | 0.00 ms | **537.5 ms** | 91.5 MB |
| **Random Forest** | 240.0 ms | 0.0 ms | 0.50 ms | **240.5 ms** | 3.2 MB |
| **Linear SVC** | 240.0 ms | 0.0 ms | 0.03 ms | **240.0 ms** | 1.8 MB |
| **Gradient Boosting** | 240.0 ms | 297.5 ms | 0.03 ms | **537.6 ms** | 92.8 MB |

---
## 8. Interpretive Caution & Statistical Boundaries
> ⚠️ **Statistical Note**: The external benchmark comprises exactly 60 screenshots. Small percentage differences (e.g. 1 image = 1.67%) must be interpreted within statistical confidence bounds (+/- 5-6%). These results reflect **Frozen External Benchmark Performance** under severe OCR domain shift and explicitly demonstrate that text-only traditional ML models trained on internal notes do not generalize to un-annotated external screenshots.
