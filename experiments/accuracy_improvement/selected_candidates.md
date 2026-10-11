# Selected Candidate Freeze Documentation

**Sprint Directory**: `experiments/accuracy_improvement/`  
**Freeze Timestamp**: 2026-10-06 01:00 UTC  
**Status**: FROZEN FOR HELD-OUT TEST EVALUATION

---

## 1. Executive Summary

Following a rigorous Error Analysis and targeted Accuracy Improvement Sprint across Semantic, Visual, and Multimodal components, candidate models were constructed and evaluated exclusively on frozen validation datasets (`metadata_v2.csv` lines 250-375 and `phase7hg_primary_manifest.csv` validation split).

The winning candidate models have been selected, locked, and frozen for final re-evaluation against held-out test datasets.

---

## 2. Frozen Candidate Configurations

### A. Semantic Classifier Candidate (`RefinedSemanticCandidate`)
* **Source Location**: [exp_semantic_refinement.py](file:///C:/Users/KIIT0001/Documents/intelligent-screenshot-organizer/experiments/accuracy_improvement/exp_semantic_refinement.py)
* **Base Architecture**: `FinalRefinedClassifier`
* **Key Enhancements**:
  1. **Connectivity Precision Filter**: Requires explicit connection terms (`wi-fi`, `wifi`, `ssid`, `ip address`, `mac address`, `wpa`, `hotspot`, `bluetooth`, `ethernet`, `airtel_runu`) to validate a `Connectivity` prediction, eliminating 17 false positive overrides caused by generic status bar timestamps and battery percentages.
  2. **Academic Course Slide Booster**: Matches specific slide header phrases (`course contents`, `course outcomes`, `textbooks:`, `reference books:`, `unit i`, `unit ii`, `co1:`, `co2:`) to reliably classify course slide screenshots into `Education` (recovering 4 misclassifications).
  3. **Institutional Notice Booster**: Matches administrative notice headers (`allotment of faculty guides`, `practical training`, `director general school of computer engineering`) to resolve `Work & Career` misclassifications.
* **Validation Performance**:
  - Top-1 Accuracy: **93.60%** (117/125) vs Baseline 80.00% (+13.60% absolute gain)
  - Macro F1 Score: **0.9516** vs Baseline 0.7679 (+0.1837 delta)
  - Balanced Accuracy: **0.9796** vs Baseline 0.8099 (+0.1697 delta)
* **Freeze Verdict**: **ACCEPTED & FROZEN AS WINNING SEMANTIC CANDIDATE**

---

### B. Visual Screen-Type Candidate (`YOLOv8n-cls` Baseline)
* **Source Location**: [visual_screen_type_adapter.py](file:///C:/Users/KIIT0001/Documents/intelligent-screenshot-organizer/experiments/visual_understanding/phase7j/visual_screen_type_adapter.py)
* **Checkpoint**: `experiments/visual_understanding/phase7f/yolov8n_cls_best.pt`
* **Architecture**: Ultralytics YOLOv8n-cls (1,444,693 parameters)
* **Rationale for Rejection of EXP-005**: EXP-005 introduced temperature scaling (T=1.4625) and class logit reweighting (`CalibratedVisualAdapterCandidate`). However, empirical validation showed a Top-1 regression from 72.00% down to 68.00% (-4.00%). Non-linear feature activations in YOLOv8n-cls degraded under simple prior weighting.
* **Validation Performance**:
  - Top-1 Accuracy: **72.00%** (18/25)
  - Macro F1 Score: **0.5686**
  - Balanced Accuracy: **0.5533**
* **Freeze Verdict**: **ACCEPTED & FROZEN AS WINNING VISUAL CANDIDATE** (Baseline 5-Class `YOLOv8n-cls` retained)

---

### C. Multimodal System Candidate (`Strategy E+ Selective Fusion`)
* **Source Location**: [exp_fusion_strategy.py](file:///C:/Users/KIIT0001/Documents/intelligent-screenshot-organizer/experiments/accuracy_improvement/exp_fusion_strategy.py)
* **Integration Strategy**: Strategy E+ Selective Multimodal Override
* **Decision Logic**:
  - Primary prediction from `RefinedSemanticCandidate`.
  - Override triggered ONLY IF semantic confidence is `Low`, marked `is_ambiguous`, or predicted category is `General`/`Work & Career`, AND visual prediction confidence $\ge 0.25$.
  - Mapped visual targets: `code_editor_terminal` $\rightarrow$ `Coding`, `travel_booking` $\rightarrow$ `Travel`, `media_player_library` $\rightarrow$ `Entertainment`, `document_article_view` $\rightarrow$ `Education`, `dashboard_settings` $\rightarrow$ `Connectivity`.
* **Validation Performance**:
  - Top-1 Accuracy: **84.00%** (21/25) vs Baseline 68.00% (+16.00% absolute gain)
  - Macro F1 Score: **0.6889** vs Baseline 0.5714 (+0.1175 delta)
  - Balanced Accuracy: **0.8750** vs Baseline 0.8000 (+0.0750 delta)
  - Errors Corrected: **4** | Harmed Predictions: **0** | Net Correction: **+4**
* **Freeze Verdict**: **ACCEPTED & FROZEN AS WINNING MULTIMODAL CANDIDATE**

---

## 3. Candidate Summary Matrix

| Candidate Component | Winning Model / Strategy | Validation Top-1 | Validation Macro F1 | Delta vs Baseline | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Semantic** | `RefinedSemanticCandidate` | 93.60% | 0.9516 | +13.60% | FROZEN |
| **Visual** | `YOLOv8n-cls` (Baseline FP32) | 72.00% | 0.5686 | +0.00% | FROZEN |
| **Multimodal** | Strategy E+ Fusion Pipeline | 84.00% | 0.6889 | +16.00% | FROZEN |

---

## 4. Next Action
Execute held-out test protocol evaluation (`TEST_PROTOCOL_FROZEN_V2.md`) against:
1. Internal Holdout Test Partition (125 samples: `metadata_v2.csv` lines 375-500)
2. External Benchmark Test Population (60 real-world web screenshots)
