# Traditional ML OCR Screenshot Classification Experiment Report

**Date**: 2026-09-30 11:57 UTC
**Scope**: 833 Valid Training Screenshots (5-Fold Stratified CV)
**Anti-Leakage Guarantee**: External Benchmark (60 Images) **0% ACCESSED / 100% UNTOUCHED**.

## 1. Objective

This experiment evaluated **5 traditional machine-learning classifiers** across **5 feature configurations** and **3 OCR representations** strictly using 5-Fold Stratified Cross-Validation on the 833 valid screenshot training pool to select the best traditional ML baseline prior to external benchmark evaluation.

---

## 2. Dataset

- **Total Metadata Records**: 836
- **Valid Training Screenshots**: 833
- **Excluded Zero-Byte Files**: `ss_441.png`, `ss_443.png`, `ss_560.png`
- **Canonical Categories (10)**: `Coding`, `Communication`, `Connectivity`, `Education`, `Entertainment`, `General`, `Payment`, `Shopping`, `Travel`, `Work & Career`

---

## 3. Experimental Protocol

- **Cross-Validation**: Stratified 5-Fold Cross-Validation (`dataset_loader.py`).
- **Fold Isolation**: All TF-IDF vectorization, feature scaling (StandardScaler/MinMaxScaler), and model fitting performed strictly inside each fold's training split.
- **Zero External Benchmark Leakage**: `ext_001.jpg` through `ext_060.jpg` were not loaded, scored, or accessed.

---

## 4. OCR Representations

- **Content-Focused OCR (Primary)**: Strips system status bar indicators and generic UI buttons.
- **Normalized OCR**: Unicode normalization, lowercase conversion, character deduplication, noise filtering.
- **Raw OCR**: Unmodified EasyOCR output string.

---

## 5. Feature Engineering

- **Feature Group A (TF-IDF Text)**: Word (1, 2) n-grams, min_df=2, sublinear_tf=True, max_features=10,000.
- **Feature Group B (Rules/Evidence)**: 10 category keyword scores + max score + margin + conflict flag (13 features).
- **Feature Group C (MiniLM Semantic)**: 384d `all-MiniLM-L6-v2` embedding + 10 category prototype similarities (394 features).
- **Feature Group D (Structured OCR)**: 11 numeric text layout and entity count features.

---

## 6. Five Traditional ML Models

1. **Logistic Regression** (`LogisticRegression`, C=1.0)
2. **Linear SVM** (`LinearSVC`, C=1.0, balanced weights)
3. **Complement Naive Bayes** (`ComplementNB`, alpha=1.0)
4. **Random Forest** (`RandomForestClassifier`, 100 trees, max_depth=15)
5. **Gradient Boosting** (`GradientBoostingClassifier`, 100 estimators, lr=0.1)

---

## 7. Cross-Validation Results Summary Matrix

| Model & Feature Configuration | Top-1 Accuracy (Mean ± Std) | Macro F1 (Mean ± Std) | Balanced Accuracy (Mean ± Std) | Weighted F1 (Mean) |
| :--- | :---: | :---: | :---: | :---: |
| **LogisticRegression | Config 1 (TF-IDF Only)** | 98.80% (±0.66%) | **0.9874** (±0.0067) | 98.86% (±0.59%) | 0.9883 |
| **LinearSVC | Config 1 (TF-IDF Only)** | 99.04% (±0.48%) | **0.9901** (±0.0052) | 99.07% (±0.51%) | 0.9905 |
| **ComplementNB | Config 1 (TF-IDF Only)** | 99.40% (±0.38%) | **0.9940** (±0.0042) | 99.42% (±0.48%) | 0.9939 |
| **RandomForest | Config 1 (TF-IDF Only)** | 98.80% (±0.65%) | **0.9895** (±0.0046) | 98.98% (±0.44%) | 0.9880 |
| **GradientBoosting | Config 1 (TF-IDF Only)** | 98.80% (±0.66%) | **0.9882** (±0.0092) | 98.71% (±1.16%) | 0.9879 |
| **LogisticRegression | Config 2 (TF-IDF + Structured)** | 98.20% (±1.25%) | **0.9747** (±0.0155) | 96.85% (±2.02%) | 0.9817 |
| **LinearSVC | Config 2 (TF-IDF + Structured)** | 98.92% (±0.59%) | **0.9890** (±0.0066) | 98.93% (±0.63%) | 0.9893 |
| **ComplementNB | Config 2 (TF-IDF + Structured)** | 99.28% (±0.59%) | **0.9929** (±0.0070) | 99.28% (±0.74%) | 0.9928 |
| **RandomForest | Config 2 (TF-IDF + Structured)** | 98.92% (±0.70%) | **0.9900** (±0.0061) | 99.01% (±0.59%) | 0.9892 |
| **GradientBoosting | Config 2 (TF-IDF + Structured)** | 98.80% (±0.66%) | **0.9882** (±0.0092) | 98.71% (±1.16%) | 0.9879 |
| **LogisticRegression | Config 3 (TF-IDF + Rules)** | 97.60% (±0.76%) | **0.9710** (±0.0104) | 97.05% (±1.09%) | 0.9754 |
| **LinearSVC | Config 3 (TF-IDF + Rules)** | 98.56% (±0.81%) | **0.9832** (±0.0099) | 98.39% (±0.96%) | 0.9856 |
| **ComplementNB | Config 3 (TF-IDF + Rules)** | 98.92% (±0.70%) | **0.9894** (±0.0069) | 98.87% (±0.75%) | 0.9892 |
| **RandomForest | Config 3 (TF-IDF + Rules)** | 99.04% (±0.48%) | **0.9914** (±0.0046) | 99.13% (±0.46%) | 0.9905 |
| **GradientBoosting | Config 3 (TF-IDF + Rules)** | 98.56% (±0.61%) | **0.9858** (±0.0084) | 98.48% (±1.05%) | 0.9855 |
| **LogisticRegression | Config 4 (TF-IDF + MiniLM)** | 99.16% (±0.81%) | **0.9897** (±0.0109) | 98.91% (±1.18%) | 0.9917 |
| **LinearSVC | Config 4 (TF-IDF + MiniLM)** | 98.56% (±0.61%) | **0.9793** (±0.0115) | 97.59% (±1.37%) | 0.9855 |
| **ComplementNB | Config 4 (TF-IDF + MiniLM) ⭐** | 99.64% (±0.29%) | **0.9963** (±0.0031) | 99.65% (±0.31%) | 0.9964 |
| **RandomForest | Config 4 (TF-IDF + MiniLM)** | 98.32% (±0.70%) | **0.9801** (±0.0088) | 98.43% (±0.80%) | 0.9836 |
| **GradientBoosting | Config 4 (TF-IDF + MiniLM)** | 98.68% (±0.70%) | **0.9883** (±0.0065) | 98.82% (±0.69%) | 0.9868 |
| **LogisticRegression | Config 5 (Full: TF-IDF + Rules + MiniLM + Structured)** | 99.16% (±0.81%) | **0.9897** (±0.0109) | 98.91% (±1.18%) | 0.9917 |
| **LinearSVC | Config 5 (Full: TF-IDF + Rules + MiniLM + Structured)** | 98.56% (±0.61%) | **0.9793** (±0.0115) | 97.59% (±1.37%) | 0.9855 |
| **ComplementNB | Config 5 (Full: TF-IDF + Rules + MiniLM + Structured)** | 99.52% (±0.24%) | **0.9958** (±0.0024) | 99.57% (±0.26%) | 0.9952 |
| **RandomForest | Config 5 (Full: TF-IDF + Rules + MiniLM + Structured)** | 97.96% (±1.05%) | **0.9729** (±0.0192) | 97.24% (±2.22%) | 0.9800 |
| **GradientBoosting | Config 5 (Full: TF-IDF + Rules + MiniLM + Structured)** | 98.68% (±0.70%) | **0.9884** (±0.0065) | 98.82% (±0.69%) | 0.9868 |

---

## 8. Feature Ablation Analysis

Ablation evaluation for winning model architecture `ComplementNB` across feature configurations:

| Ablation Configuration | Top-1 Accuracy | Macro F1 | Balanced Accuracy | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: |
| **A. TF-IDF Only** | 99.40% | **0.9941** | 99.43% | 0.9940 |
| **B. TF-IDF + Rules** | 98.92% | **0.9893** | 98.86% | 0.9892 |
| **C. TF-IDF + MiniLM** | 99.64% | **0.9963** | 99.65% | 0.9964 |
| **D. TF-IDF + Rules + MiniLM** | 99.28% | **0.9935** | 99.25% | 0.9928 |
| **E. Full (TF-IDF + Rules + MiniLM + Structured)** | 99.52% | **0.9958** | 99.56% | 0.9952 |

---

## 9. OCR Representation Comparison

| Representation | Top-1 Accuracy | Macro F1 | Balanced Accuracy | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Content-Focused OCR** | 99.64% | **0.9963** | 99.65% | 0.9964 |
| **Normalized OCR** | 99.64% | **0.9963** | 99.65% | 0.9964 |
| **Raw OCR** | 99.64% | **0.9963** | 99.65% | 0.9964 |

---

## 10. Model Selection for Next Phase

- **Highest CV Top-1 Accuracy**: `ComplementNB | Config 4 (TF-IDF + MiniLM)` (99.64%)
- **Highest CV Macro F1**: `ComplementNB | Config 4 (TF-IDF + MiniLM)` (0.9963)
- **Highest Balanced Accuracy**: `ComplementNB | Config 4 (TF-IDF + MiniLM)` (99.65%)

> 📌 **FROZEN MODEL RECOMMENDATION FOR NEXT PHASE**: `ComplementNB` trained on `Config 4 (TF-IDF + MiniLM)` using `Content-Focused OCR`.

---

## 11. Leakage Verification Checklist

- [x] Exactly 0 external benchmark images (`ext_001.jpg` – `ext_060.jpg`) loaded or scored.
- [x] 833 valid screenshots evaluated strictly in 5-Fold Stratified CV.
- [x] 3 zero-byte files excluded.
- [x] All TF-IDF vectorization and feature scaling fit strictly inside each fold split.
- [x] MiniLM encoder frozen without fine-tuning.
- [x] No production code or database schema modified.
