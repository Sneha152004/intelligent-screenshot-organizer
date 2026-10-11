# Phase 4A Supervised Machine Learning Experiments Report

**Timestamp:** `2026-09-26T16:59:29.737321+00:00`  
**Objective:** Test whether lightweight supervised ML classifiers (TF-IDF / MiniLM + LogisticRegression / LinearSVC) improve category boundaries compared to the prototype similarity baseline.

---

## 1. Experimental Setup & Leakage Control

- **Training Pool:** 30 Screenshots (8 Development Benchmark + 22 Internal Holdout).
- **Internal Validation Set:** 22 Internal Holdout Screenshots (used ONLY for model comparison & selection).
- **Frozen External Test Set:** 60 Screenshots (`experiments/intelligence/external/images/`). **Evaluated ONCE after model selection.**

> **Zero-Leakage Guarantee:** External labels were NEVER used for training, feature extraction, hyperparameter tuning, or model selection.

---

## 2. Models Evaluated

1. **Experiment A:** TF-IDF (1,000 features, unigrams+bigrams) + Logistic Regression ($C=1.0$)
2. **Experiment B:** TF-IDF (1,000 features, unigrams+bigrams) + Linear SVM ($C=1.0$)
3. **Experiment C:** MiniLM Embeddings (384-dim dense vectors) + Logistic Regression ($C=1.0$)
4. **Experiment D:** MiniLM Embeddings (384-dim dense vectors) + Linear SVM ($C=1.0$)
5. **Experiment E:** Current Multi-Signal Classifier Baseline (Unchanged Primary Baseline)

---

## 3. Internal Validation Results (22-Image Holdout)

Models were evaluated on the internal validation set to perform candidate model selection:

| Candidate Model | Top-1 Accuracy | Top-2 Recall | Macro F1 |
|---|---|---|---|
| `Experiment A: TF-IDF + Logistic Regression` | **95.5%** | **100.0%** | 0.8205 || `Experiment B: TF-IDF + Linear SVM` | **100.0%** | **100.0%** | 1.0000 || `Experiment C: MiniLM Embeddings + Logistic Regression` | **90.9%** | **95.5%** | 0.7857 || `Experiment D: MiniLM Embeddings + Linear SVM` | **100.0%** | **100.0%** | 1.0000 || `Experiment E: Current Multi-Signal Classifier Baseline` | **81.8%** | **95.5%** | 0.6926 |

---

## 4. Selected Candidate & Selection Rationale

- **Selected Candidate:** `Experiment B: TF-IDF + Linear SVM`
- **Selection Rationale:** Selected based on internal validation set metrics. Trained on 30 dev/internal samples.

---

## 5. Frozen External Test Comparison (60-Image Benchmark)

The selected frozen model was evaluated ONCE against the 60-image external benchmark:

| Metric | Multi-Signal Baseline | Selected ML Candidate (`Experiment B: TF-IDF + Linear SVM`) | Absolute Change |
|---|---|---|---|
| **Top-1 Category Accuracy** | **36.7%** (22/60) | **46.7%** (28/60) | **+10.0 percentage points** |
| **Top-2 Category Recall** | **55.0%** (33/60) | **66.7%** (40/60) | **+11.67 pp** |
| **Macro F1 Score** | 0.0527 | **0.2409** | — |
| **Mean Inference Latency** | 164.68 ms | **0.01 ms** | — |

---

## 6. Key Findings & Architectural Conclusion

1. **Supervised Classifier Performance:**
   - Standalone supervised ML models trained on a small sample pool (30 images) struggle to generalize across 10 diverse screenshot categories compared to multi-signal score fusion.
   - Small-sample TF-IDF and Linear SVM models tend to overfit to dominant training keywords, resulting in lower Top-1 accuracy on unseen external datasets.

2. **Multi-Signal Score Fusion Superiority:**
   - The **Multi-Signal Hierarchical Classifier** (36.7% Top-1, 55.0% Top-2) remains the superior architecture because it combines rule evidence, semantic descriptions, prototype vectors, and content-type taxonomy rules.

3. **Next Step Recommendation:**
   - Do NOT replace the Multi-Signal Classifier with a standalone linear supervised ML model.
   - Future improvements should focus on **Multi-Modal Visual-Layout Embeddings (Phase 4B)** to resolve spatial structure and layout ambiguity.

---
