# Comprehensive Repository Rules & Heuristics Audit Report

## 1. Overview & Purpose

This audit catalogues all existing category-specific rules, heuristics, keyword maps, phrase boosters, penalties, thresholds, and guardrails across the repository prior to constructing **Category Decision Policy V1**.

---

## 2. Inventory of Existing Rules

### Rule 1: Base Semantic Classifier Prediction (TF-IDF / Keyword Matching Variant C)
- **ID**: `RULE_001_TFIDF_SEMANTIC_CLASSIFIER`
- **Affected Categories**: All 10 Canonical Categories (`Coding`, `Communication`, `Connectivity`, `Education`, `Entertainment`, `General`, `Payment`, `Shopping`, `Travel`, `Work & Career`).
- **Positive Evidence**: Learned TF-IDF vocabulary weights fitted on internal training split ($n=250$).
- **Negative Evidence**: Implicit; zero-weight terms yield zero probability.
- **Thresholds**: High confidence ($>0.80$), Medium ($0.50-0.80$), Low ($<0.50$).
- **Implementation Location**: [exp_recovery_candidate.py](file:///C:/Users/KIIT0001/Documents/intelligent-screenshot-organizer/experiments/categorization_recovery/exp_recovery_candidate.py)
- **Status**: Production-Active in Candidate Generation (Top-3).
- **Validation Evidence**: Achieves 94.40% on Internal Validation and 99.20% on Internal Holdout.

---

### Rule 2: MiniLM Prototype Embedding Cosine Similarity
- **ID**: `RULE_002_MINILM_EMBEDDING_PROTOTYPES`
- **Affected Categories**: All 10 Canonical Categories.
- **Positive Evidence**: 384-dimensional `sentence-transformers/all-MiniLM-L6-v2` cosine similarity between screenshot OCR embeddings and category prototype mean vectors.
- **Negative Evidence**: Low cosine similarity ($<0.20$).
- **Thresholds**: Generates Top-5 candidate ranking descending by cosine similarity.
- **Implementation Location**: [expanded_candidate_pool.py](file:///C:/Users/KIIT0001/Documents/intelligent-screenshot-organizer/experiments/candidate_reranking/expanded_candidate_pool.py)
- **Status**: Production-Active in Candidate Pool Generation.
- **Validation Evidence**: Increased candidate pool coverage from 88.93% to 92.62% on External V2.

---

### Rule 3: Mandatory General Candidate Fallback
- **ID**: `RULE_003_MANDATORY_GENERAL_FALLBACK`
- **Affected Category**: `General`.
- **Positive Evidence**: Always appends `General` to candidate pool if absent from Classifier Top-5 and MiniLM Top-5.
- **Negative Evidence**: None.
- **Thresholds**: Unconditional fallback append rule.
- **Implementation Location**: [run_final_baseline_audit.py](file:///C:/Users/KIIT0001/Documents/intelligent-screenshot-organizer/experiments/final_categorization_baseline/run_final_baseline_audit.py)
- **Status**: Production-Active Candidate Pool Rule.
- **Validation Evidence**: Expanded candidate pool coverage to **100.00% (244/244)** on External V2 with zero harms.

---

### Rule 4: Domain OCR Phrase Evidence Booster (`CATEGORY_EVIDENCE_MAP`)
- **ID**: `RULE_004_DOMAIN_OCR_PHRASE_BOOSTER`
- **Affected Categories**: `Education`, `Shopping`, `Payment`, `Travel`, `Coding`, `Communication`, `Entertainment`, `Work & Career`, `Connectivity`.
- **Positive Evidence**: Exact string match of category-specific domain phrases in raw OCR text (e.g., `add to cart`, `boarding pass`, `payment successful`, `vs code`, `online course`).
- **Negative Evidence**: None per category.
- **Thresholds**: $+0.20$ score boost for 1 phrase match, $+0.35$ score boost for $\ge 2$ phrase matches.
- **Implementation Location**: [best_of_all_reranker.py](file:///C:/Users/KIIT0001/Documents/intelligent-screenshot-organizer/experiments/best_of_all_categorization/best_of_all_reranker.py)
- **Status**: Production-Active Reranking Rule.

---

### Rule 5: General Evidence Penalty
- **ID**: `RULE_005_GENERAL_EVIDENCE_PENALTY`
- **Affected Category**: `General`.
- **Positive Evidence**: None.
- **Negative Evidence**: Presence of $\ge 1$ domain phrase match for any competing non-General candidate in pool.
- **Thresholds**: Multiplies `General` composite score by `0.50` (or `0.40`).
- **Implementation Location**: [best_of_all_reranker.py](file:///C:/Users/KIIT0001/Documents/intelligent-screenshot-organizer/experiments/best_of_all_categorization/best_of_all_reranker.py)
- **Status**: Production-Active Reranking Rule.

---

### Rule 6: High-Confidence Protection Guardrail
- **ID**: `RULE_006_HIGH_CONF_GUARDRAIL`
- **Affected Categories**: All Categories.
- **Positive Evidence**: Base classifier confidence == `High`.
- **Negative Evidence**: Reranked candidate has $< 2$ domain OCR phrase matches and low visual confidence.
- **Thresholds**: Protects base Top-1 prediction from unevidenced overrides.
- **Implementation Location**: [best_of_all_reranker.py](file:///C:/Users/KIIT0001/Documents/intelligent-screenshot-organizer/experiments/best_of_all_categorization/best_of_all_reranker.py)
- **Status**: Production-Active Reranking Rule.

---

### Rule 7: YOLOv8n-cls Visual Screen Type Evidence Mapping
- **ID**: `RULE_007_YOLO_VISUAL_SCREEN_TYPE_MAP`
- **Affected Categories**: `Coding`, `Travel`, `Entertainment`, `Education`, `Connectivity`.
- **Positive Evidence**: YOLO prediction matches `VIS_5_TO_SEMANTIC_MAP` (`code_editor_terminal` $\rightarrow$ `Coding`, `travel_booking` $\rightarrow$ `Travel`, `media_player_library` $\rightarrow$ `Entertainment`, `document_article_view` $\rightarrow$ `Education`, `dashboard_settings` $\rightarrow$ `Connectivity`).
- **Negative Evidence**: Unmapped screen type (`other_screen`) or low confidence ($< 0.40$).
- **Thresholds**: `vis_conf >= 0.40`; adds `min(1.0, vis_conf) * w_vis` to composite score.
- **Implementation Location**: [visual_screen_type_adapter.py](file:///C:/Users/KIIT0001/Documents/intelligent-screenshot-organizer/experiments/visual_understanding/phase7j/visual_screen_type_adapter.py)
- **Status**: Production-Active Visual Evidence Rule.

---

### Rule 8: Contextual Combination Disambiguation Rules
- **ID**: `RULE_008_CONTEXTUAL_COMBINATION_RULES`
- **Affected Categories**: `Education`, `Coding`, `Shopping`, `Payment`, `Work & Career`, `Entertainment`.
- **Positive Evidence**: Multi-word contextual pairs (e.g., `python` + `tutorial` $\rightarrow$ `Education`, `python` + `traceback` $\rightarrow$ `Coding`, `₹` + `add to cart` $\rightarrow$ `Shopping`, `₹` + `payment successful` $\rightarrow$ `Payment`, `linkedin` + `job` $\rightarrow$ `Work & Career`).
- **Negative Evidence**: None.
- **Thresholds**: Adds weights ranging from $4.0$ to $5.0$.
- **Implementation Location**: [ocr_evidence_semantic.py](file:///C:/Users/KIIT0001/Documents/intelligent-screenshot-organizer/experiments/intelligence/ocr_evidence_semantic/ocr_evidence_semantic.py)
- **Status**: Research-Active in Evidence Module.
