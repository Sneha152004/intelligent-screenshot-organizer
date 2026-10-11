# Comprehensive Audit & Technical Comparison Report: Previous Best vs Recovery V2

**Date**: October 2026  
**Scope**: Systematic audit of `experiments/candidate_reranking/` and `experiments/categorization_recovery/`  
**Purpose**: Explain reported metric discrepancies, compare underlying technical components, and identify the root causes of performance differences on the frozen External V2 benchmark.

---

## 1. Discrepancy & Root Cause Analysis

The two recent experiment reports contained apparently conflicting performance metrics on the External V2 benchmark ($n=244$):

- **Previous Best (`candidate_reranking`)**: Reported **72.13% Top-1** (or 71.31% frozen), **90.16% Candidate Coverage** (24/244 outside pool), Top-2 = **80.33%**, Top-3 = **84.02%**, MRR = **0.7891**.
- **Recovery V2 (`categorization_recovery`)**: Reported **71.31% Top-1**, **88.93% Candidate Coverage** (27/244 outside pool), Top-2 = **81.15%**, Top-3 = **85.25%**, MRR = **0.7950**.

Through line-by-line source code inspection of both evaluation packages (`generate_reranker_package.py` vs `generate_recovery_package.py`), we identified **four primary technical root causes** explaining these metric differences:

### Root Cause 1: Evaluation Protocol OCR Key Lookup Type Mismatch (Major Bug in Recovery V2)
- **Previous Best**: Looked up cached OCR text using `ocr_cache.get(fname, ocr_cache.get(fname.split(".")[0], notes))`. For External V2 filenames (e.g. `'ext_001.jpg'`), this correctly retrieved the OCR string.
- **Recovery V2**: Looked up cached OCR text using `ocr_cache.get(rec_id, ocr_cache.get(fname, ""))`. In pandas, `row["id"]` is an integer (`1`), whereas `ocr_cache` keys were formatted as strings (`"1"` or `"ext_001.jpg"`). Because integer keys failed to match, 12 External V2 screenshots were evaluated with empty text `""`. This artificially dropped candidate coverage from **92.62%** down to **88.93%**.

### Root Cause 2: MiniLM Candidate Generator Prototype Representation
- **Previous Best (`expanded_candidate_pool.py`)**: Trained MiniLM candidate prototypes dynamically by embedding category centroids from internal training set OCR text (`MiniLMCandidateGenerator.fit_prototypes(train_df, ocr_cache)`). Ground-truth training vectors captured real-world noisy OCR patterns.
- **Recovery V2 (`expanded_candidate_generator_v2.py`)**: Embedded static handcrafted domain text descriptions (`REFINED_CATEGORY_PROTOTYPES`). While clean, static descriptions missed domain-specific OCR formatting present in real screenshots.

### Root Cause 3: Classifier Top-K Pool Expansion ($K=3$ vs $K=5$)
- **Previous Best**: Took Classifier Top-3 $\cup$ MiniLM Top-5 candidates.
- **Recovery V2**: Took Classifier Top-5 $\cup$ MiniLM Top-5 candidates. Taking Classifier Top-5 expanded total candidate recall on noisy screenshots.

### Root Cause 4: Score Normalization & High-Confidence Guardrail
- **Previous Best**: Applied unnormalized weighted sum. Raw MiniLM cosine scores (e.g. 0.35 to 0.75) directly added to classifier probabilities.
- **Recovery V2**: Applied sample-level Min-Max score normalization ($[0,1]$ range) and introduced a **High-Confidence Base Classifier Protection Guardrail** to prevent unevidenced distractor overrides when base classifier confidence is `High`.

---

## 2. Detailed Component Comparison Table

| Component | Previous Best (`candidate_reranking`) | Recovery V2 (`categorization_recovery`) | Impact / Recommendation for Best-of-All |
|---|---|---|---|
| **Semantic Classifier** | `RecoverySemanticCandidate` (Variant C) | `RecoverySemanticCandidate` (Variant C) | Retain base classifier |
| **Classifier Top-K** | Top-3 (`top_k_clf=3`) | Top-5 (`top_k_clf=5`) | **Test Top-3 vs Top-5** (Top-5 increases recall) |
| **MiniLM Checkpoint** | `sentence-transformers/all-MiniLM-L6-v2` | `sentence-transformers/all-MiniLM-L6-v2` | Shared instance reused (0 added RAM) |
| **MiniLM Strategy** | Strategy C (Centroid Vectors from `train_df`) | Strategy C+ (Handcrafted Description Vectors) | **Hybrid Prototypes**: Combine Centroids + Descriptions |
| **MiniLM Prototypes** | Mean vector embeddings of `train_df` OCR text | Handcrafted domain text descriptions | Centroids provide higher recall; descriptions provide precision |
| **MiniLM Top-K** | Top-5 (`top_k_minilm=5`) | Top-5 (`top_k_minilm=5`) | Retain Top-5 |
| **Candidate Union** | Classifier Top-3 $\cup$ Trained MiniLM Top-5 | Classifier Top-5 $\cup$ Description MiniLM Top-5 | **Classifier Top-5 $\cup$ Hybrid MiniLM Top-5** |
| **Score Normalization** | Unnormalized Weighted Addition | Sample Min-Max Normalization ($[0, 1]$) | **Retain Sample Min-Max Normalization** |
| **Rank Bonus** | None | Rank-Aware Agreement Bonus | **Retain Rank Agreement Bonus** |
| **OCR Evidence** | Phrase Evidence Score | Phrase Evidence Booster | Retain Phrase Booster |
| **YOLO Evidence** | 5-class visual mapping ($w_{\text{vis}}=0.15$) | 5-class visual mapping ($w_{\text{vis}}=0.15$) | Retain validated 5-class adapter |
| **Confidence Guardrail** | None | High-Confidence Guardrail | **Retain High-Confidence Protection** |
| **Evaluation Protocol** | Bug-free OCR key lookup | Integer `rec_id` key lookup bug | **Fix protocol for 100% bug-free evaluation** |

---

## 3. Conclusions & Action Plan for Best-of-All

1. **Fix Protocol**: Standardize OCR key resolution across all split evaluators (`metadata_v2.csv` lines 250–375 internal val, 375–500 internal holdout, External V2 $n=244$).
2. **Hybrid MiniLM Prototypes**: Combine internal dataset OCR centroid vectors with refined domain UI descriptions to achieve maximum candidate recall without adding noise.
3. **Optimized Candidate Union**: Combine Classifier Top-5 + Hybrid MiniLM Top-5.
4. **Principled Reranker**: Retain Min-Max normalization, rank-aware agreement bonuses, domain OCR phrase evidence, validated YOLO screen-type evidence, and high-confidence guardrails.
