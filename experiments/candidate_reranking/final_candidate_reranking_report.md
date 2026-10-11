# Final Candidate Pool & Evidence Reranking Technical Report

**Date**: October 2026  
**Architecture**: Base Semantic Classifier Top-3 $\cup$ Reused MiniLM Strategy C Top-5 $\rightarrow$ Multi-Signal Evidence Reranker  
**Verdict**: **RECOMMEND GO FOR PRODUCTION INTEGRATION**

---

## 1. Executive Summary

We conducted a complete experimental investigation into expanding semantic category candidate generation by uniting our primary semantic classifier's Top-3 predictions with MiniLM Strategy C's Top-5 text similarity candidates, followed by multi-signal evidence reranking.

### Key Benchmark Results Across Splits

| Metric | Internal Validation (n=125) | Internal Holdout Test (n=125) | External V2 Benchmark (n=244) |
| :--- | :--- | :--- | :--- |
| **Base Classifier Top-1** | 94.40% | 99.20% | 71.31% |
| **Reranked Final Top-1** | **95.20% (+0.80 pp)** | **99.20% (0.00 pp)** | **72.13% (+0.82 pp)** |
| **Expanded Pool Recall@5** | **100.00% (125/125)** | **100.00% (125/125)** | **90.16% (220/244)** |
| **Macro F1 Score** | **0.9752** | **0.9940** | **0.6580** |
| **Balanced Accuracy** | **0.9780** | **0.9950** | **0.7485** |
| **Reranker Impact (Corr/Harm)** | **1 Correction / 0 Harms** | **0 Corrections / 0 Harms** | **5 Corrections / 3 Harms** |

---

## 2. Resource & Model Reuse Audit

- **Model Duplication**: **0** (Existing `SentenceTransformerEmbedder` instance reused directly).
- **Added Memory Footprint**: **0 MB** (Shared PyTorch model RAM = 132.28 MB).
- **Added Disk Space**: **0 MB**.
- **Inference Latency**: **~14.21 ms / screenshot**.

---

## 3. Preserved Metadata for Downstream Layers

Every screenshot record preserves rich Top-K metadata:
- `top1_category`, `top1_score`
- `top2_category`, `top2_score`
- `top3_category`, `top3_score`
- `candidate_pool` (categories, scores, provenance sources: `["classifier", "minilm"]`)
- `final_category`, `confidence_str`
- `reranked`, `reranking_reason`

This preserves Top-K candidate hypotheses for Tag Generation, Intent Classification, Chatbot Prompting, and Vector Retrieval.

---

## 4. Final Recommendation

**RECOMMEND GO FOR PRODUCTION INTEGRATION**

The candidate pool expansion and multi-signal evidence reranker succeed across all 8 success criteria without model duplication, zero memory increase, zero regression on holdout data, and improved Final Top-1 accuracy on External V2.
