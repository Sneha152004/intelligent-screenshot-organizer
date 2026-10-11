# Explainable OCR Evidence + Semantic Classification Report (Refined Coding & General Layer)
**Date**: 2026-09-30T13:22 UTC
**Scope**: 833 Valid Training Screenshots (Stratified Split: 70% Dev-Train, 15% Validation, 15% Holdout)
**Anti-Leakage Guarantee**: External Benchmark (`ext_001.jpg` – `ext_060.jpg`) **0% ACCESSED / 100% UNTOUCHED**.

---
## 1. Objective
This experiment refined the explainable multi-signal screenshot categorization pipeline by implementing targeted contextual evidence rules for **Coding** (VS Code, terminal, git commit, traceback) and a true fallback mechanism for **General**.

---
## 2. Dataset and Stratified Development Split
- **Total Valid Samples**: 833
- **Development Training (70%)**: 583 screenshots
- **Validation Split (15%)**: 125 screenshots
- **Internal Holdout Split (15%)**: 125 screenshots

---
## 3. Audits of Baseline Rules
### Coding Rule Audit
- **Isolated Keywords**: `python`, `code`, `import`, `git` carried low weight (+1.2), causing screenshots with code snippets to be outweighed by Education semantic similarity.
- **Disambiguation**: `python` + `tutorial` / `course` -> Education vs `python` + `traceback` / `vs code` / `def` -> Coding.
### General Fallback Audit
- **True Fallback Definition**: Assigned ONLY when evidence for the other 9 categories is 0/insufficient AND max non-General semantic similarity is weak (<0.40).

---
## 4. Task 3: Controlled Variant Testing (Validation Split - 125 Screenshots)
| Variant Name | Val Top-1 | Val Macro F1 | Val Bal Acc | Coding F1 | General F1 | Ambiguous Count | Latency / Img |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant A (Current Baseline)** | 85.60% | **0.7576** | 82.22% | 0.3636 | 0.0000 | 18 | 15.4 ms |
| **Variant B (Improved Coding Only)** | 85.60% | **0.7576** | 82.22% | 0.3636 | 0.0000 | 19 | 15.9 ms |
| **Variant C (Improved General Fallback Only)** | 85.60% | **0.7576** | 82.22% | 0.3636 | 0.0000 | 18 | 14.2 ms |
| **Variant D (Improved Coding + General Together)** | 85.60% | **0.7576** | 82.22% | 0.3636 | 0.0000 | 19 | 14.3 ms |

> 📌 **Selected Variant**: **Variant A (Current Baseline)** (Validation Macro F1 = **0.7576**).

---
## 5. Task 4: Final Frozen Holdout Evaluation (Internal Holdout Split - 125 Screenshots)
- **Top-1 Accuracy**: **84.80% (106/125)**
- **Top-2 Recall**: **89.60%**
- **Top-3 Recall**: **97.60%**
- **Macro F1 Score**: **0.7694**
- **Balanced Accuracy**: **82.73%**
- **Mean Reciprocal Rank (MRR)**: **0.9020**

### Per-Category Performance (Internal Holdout Split)
| Category | Support | Precision | Recall | F1 Score | Correct Predictions |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Coding** | 10 | 1.0000 | 0.3000 | **0.4615** | 3 / 10 |
| **Communication** | 11 | 0.8462 | 1.0000 | **0.9167** | 11 / 11 |
| **Connectivity** | 4 | 1.0000 | 1.0000 | **1.0000** | 4 / 4 |
| **Education** | 11 | 0.9167 | 1.0000 | **0.9565** | 11 / 11 |
| **Entertainment** | 19 | 1.0000 | 1.0000 | **1.0000** | 19 / 19 |
| **General** | 11 | 0.3333 | 0.0909 | **0.1429** | 1 / 11 |
| **Payment** | 3 | 0.2143 | 1.0000 | **0.3529** | 3 / 3 |
| **Shopping** | 20 | 0.9524 | 1.0000 | **0.9756** | 20 / 20 |
| **Travel** | 19 | 0.9048 | 1.0000 | **0.9500** | 19 / 19 |
| **Work & Career** | 17 | 1.0000 | 0.8824 | **0.9375** | 15 / 17 |

---
## 6. Task 5: Detailed Error Analysis by Failure Mode
- **ss_177.jpg**: Ground Truth = `General`, Predicted = `Payment` | Failure Mode: `genuinely_ambiguous`
- **ss_023.jpg**: Ground Truth = `Coding`, Predicted = `Education` | Failure Mode: `programming_education_vs_coding_confusion`
- **ss_021.jpg**: Ground Truth = `Coding`, Predicted = `Communication` | Failure Mode: `weak_contextual_evidence`
- **ss_302.png**: Ground Truth = `Work & Career`, Predicted = `General` | Failure Mode: `contextual_ambiguity`
- **ss_094.png**: Ground Truth = `General`, Predicted = `Payment` | Failure Mode: `genuinely_ambiguous`
- **ss_819.jpg**: Ground Truth = `General`, Predicted = `Payment` | Failure Mode: `genuinely_ambiguous`
- **ss_043.jpg**: Ground Truth = `Coding`, Predicted = `Communication` | Failure Mode: `weak_contextual_evidence`
- **ss_294.png**: Ground Truth = `Work & Career`, Predicted = `General` | Failure Mode: `contextual_ambiguity`
- **ss_011.jpg**: Ground Truth = `Coding`, Predicted = `Travel` | Failure Mode: `weak_contextual_evidence`
- **ss_019.jpg**: Ground Truth = `Coding`, Predicted = `Travel` | Failure Mode: `programming_education_vs_coding_confusion`

---
## 7. Task 6: Explainability Traces (Representative Samples for Viva / Presentation)
- **ss_292.png** -> Pred: `Work & Career` (GT: `Work & Career`) | Correct: `True` | Conf: `High` | Score: `0.610` | Evidence: ["Med KW: 'job' (x2)", "Med KW: 'career' (x2)"]
- **ss_177.jpg** -> Pred: `Payment` (GT: `General`) | Correct: `False` | Conf: `Low` | Score: `0.495` | Evidence: ["Med KW: 'account' (x2)"]
- **ss_754.png** -> Pred: `Communication` (GT: `Communication`) | Correct: `True` | Conf: `High` | Score: `0.660` | Evidence: ["Med KW: 'email' (x2)", "Med KW: 'mail' (x2)"]
- **ss_458.png** -> Pred: `Shopping` (GT: `Shopping`) | Correct: `True` | Conf: `High` | Score: `0.585` | Evidence: ["Med KW: 'product' (x2)", "Med KW: 'order' (x2)"]
- **ss_798.png** -> Pred: `Entertainment` (GT: `Entertainment`) | Correct: `True` | Conf: `High` | Score: `0.614` | Evidence: ["Med KW: 'music' (x2)", "Med KW: 'video' (x2)"]
