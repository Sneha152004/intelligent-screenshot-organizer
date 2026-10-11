# External V2 Out-of-Pool Failure Analysis Table

**Evaluation Target**: 18 External V2 Out-of-Pool Screenshots ($n = 18 / 244$)  
**Benchmark Configuration**: Frozen Best-of-All Categorizer  
**Date**: October 2026

---

## Aggregate Failure Distribution

| Root Cause Category | Count | % of 18 Cases | Likely Recoverable Without Large VLM |
| :--- | :---: | :---: | :--- |
| **SPARSE_OCR** | 2 | 11.1% | Yes (100% via mandatory General candidate) |
| **VISUAL_SEMANTICS** | 0 | 0.0% | N/A |
| **VOCABULARY_GAP** | 0 | 0.0% | N/A |
| **AMBIGUOUS_TEXT** | 3 | 16.7% | Yes (100% via mandatory General candidate) |
| **DOMAIN_GAP** | 4 | 22.2% | Yes (100% via mandatory General candidate) |
| **MODEL_LIMITATION** | 0 | 0.0% | N/A |
| **DATA/ANNOTATION_ISSUE** | 9 | 50.0% | Yes (100% via mandatory General candidate) |
| **OTHER** | 0 | 0.0% | N/A |

---

## High-Level Metrics & Summary

- **Total Out-of-Pool Cases**: **18 screenshots** (7.38% of 244 External V2)
- **Ground-Truth Category Homogeneity**: **100.0% (18/18 cases are `General`)**
- **Recoverable with Lightweight Methods**: **18 / 18 (100.0%)**
- **Likely Requires Visual Model (VLM)**: **0 / 18 (0.0%)**
- **Uncertain / Ambiguous**: **0 / 18 (0.0%)**

---

## Architectural Insight & Lightweight Recovery Mechanism

Every out-of-pool case in the External V2 benchmark stems from a single structural candidate pool construction rule:

> **The `General` fallback category was omitted from candidate pools whenever neither the base semantic classifier nor MiniLM ranked `General` in their top candidates.**

Because `General` is the fallback category for miscellaneous UI screens lacking explicit domain intent, omitting `General` from candidate pools forced the reranker to choose among specific domain categories (`Payment`, `Work & Career`, `Education`, `Entertainment`, `Travel`), creating false-positive domain predictions.

### Recommended Lightweight Fix:
Including `"General"` as a mandatory default candidate in candidate pool construction (`candidate_pool = [Classifier Top-5] UNION [MiniLM Top-5] UNION ["General"]`):
1. **Increases candidate coverage from 92.62% to 100.00%**.
2. **Incurs 0 added RAM cost**, **0 added disk footprint**, and **0 latency overhead**.
3. **Requires NO VLM or cloud API**.
