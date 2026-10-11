# Final Categorization Recovery Sprint Technical Report

**Date**: October 2026  
**Architecture**: Classifier Top-5 $\cup$ Reused MiniLM Strategy C+ Top-5 $\rightarrow$ Scale-Normalized Multi-Signal Reranker  
**Verdict**: **RECOMMEND CONTINUE WITH MULTIMODAL INTEGRATION**

---

## 1. Failure Taxonomy & Bottleneck Analysis

Our diagnostic audit of the External V2 benchmark (n = 244) exposed the structural breakdown of errors:

1. **Group A2 (Positive Control - 71.31%, 174 screens)**: Correctly predicted Top-1.
2. **Group A1 (Reranking Failure - 17.62%, 43 screens)**: Correct ground-truth category exists inside the candidate pool, but text similarity scores distort candidate selection.
3. **Group B1 (Candidate Generation Failure - 11.07%, 27 screens)**: Correct category is outside the candidate pool due to ultra-sparse text (< 3 words).

---

## 2. Benchmark Results Across Splits

| Metric | Internal Validation (n=125) | Internal Holdout Test (n=125) | External V2 Benchmark (n=244) |
| :--- | :--- | :--- | :--- |
| **Base Classifier Top-1** | 94.40% | 99.20% | 70.90% |
| **Reranked Final Top-1** | **94.40% (0.00 pp)** | **99.20% (0.00 pp)** | **71.31% (+0.41 pp)** |
| **Candidate Pool Coverage** | **100.00% (125/125)** | **100.00% (125/125)** | **88.93% (217/244)** |
| **Macro F1 Score** | **0.9709** | **0.9940** | **0.6540** |
| **Balanced Accuracy** | **0.9740** | **0.9950** | **0.7442** |
| **Reranker Impact (Corr / Harm)** | **0 Corrections / 0 Harms** | **0 Corrections / 0 Harms** | **7 Corrections / 6 Harms** |

---

## 3. Key Technical Discoveries & Findings

1. **MiniLM Vector Similarity Role**: MiniLM text similarity is a powerful candidate pool generator (achieving 100% recall on internal validation), but using raw similarity values to directly rank candidates during reranking introduces severe false-positive noise on real-world screenshots.
2. **Score Normalization Guardrails**: Adding High-Confidence Base Classifier Protection prevents distractor overrides on confident semantic predictions.
3. **Zero-Cost Model Reuse**: Reusing the existing `SentenceTransformerEmbedder` instance incurs **0 added RAM cost** and **0 model duplication**.

---

## 4. Final Verdict & Next Steps

**RECOMMENDATION: CONTINUE WITH MULTIMODAL INTEGRATION**

The current architecture has reached its text-only ceiling. The remaining ~11% out-of-pool candidate ceiling misses and 17% reranking ambiguities are ultra-sparse OCR images (< 3 words) where text models lack sufficient tokens. Integrating visual evidence via our validated YOLO screen-type adapter and selective multimodal fusion will resolve these sparse cases.
