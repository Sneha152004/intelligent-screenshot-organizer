# Phase 7K: Visual Evidence Value Analysis Report

## Executive Summary

This analysis evaluates when and why visual screen-type evidence from **Ultralytics YOLOv8n-cls** improves screenshot classification when combined with the frozen semantic classifier (**FinalRefinedClassifier**).

---

## 1. Interaction Categorization Matrix (Validation Set)

| Interaction Category | Screenshot Count | Description | Impact on Final Accuracy |
| :--- | :---: | :--- | :---: |
| **Category 1: Semantic Correct, Visual Agrees** | **17 (68.0%)** | Both branches independently predict matching ground truth. | **Preserved (Correct)** |
| **Category 2: Semantic Correct, Visual Disagrees** | **0 (0.0%)** | Semantic branch is correct; visual predicts different layout. | **0 Harmed (Preserved)** |
| **Category 3: Semantic Wrong, Visual Corrects** | **5 (20.0%)** | Text OCR is sparse (`General` Low conf); visual identifies UI layout (`code_editor_terminal`). | **+5 Corrected (+20.0%)** |
| **Category 4: Both Branches Wrong** | **3 (12.0%)** | Text OCR is uninformative and visual screen-type belongs to unsupported/ambiguous class. | **Unchanged (Error)** |
| **Category 5: Unsupported Visual Coverage** | **0 (0.0%)** | Visual model encounters non-5-class UI; semantic fallback preserved. | **Safe Handling** |

---

## 2. Deep Dive: The 5 Corrected Validation Screenshots

In 5 validation samples (`ss_028.jpg`, `ss_037.jpg`, `ss_041.jpg`, `ss_045.jpg`, `ss_062.jpg`), the text-based semantic classifier defaulted to `General` with `confidence = Low` and `is_ambiguous = True` due to sparse OCR text keywords.

### How Visual Evidence Resolved the Error:
1. **Visual Feature Cue**: YOLOv8n-cls extracted structural visual features (dark background theme, editor line-number gutter, syntax color blocks, code indentation).
2. **Visual Model Output**: Predicted `code_editor_terminal` with high relative probability margin over alternative visual classes.
3. **Selective Overriding Rule (Strategy E)**: When `semantic_category == 'General'` AND `confidence == 'Low'`, the harness mapped `code_editor_terminal` \(\rightarrow\) `Coding`.
4. **Outcome**: All 5 misclassified screenshots were correctly reclassified as `Coding`, raising Top-1 Accuracy from **68.00% to 88.00%** with **0 harmed samples**.

---

## 3. Why Selective Overriding Succeeds (And Score Fusion Fails)

- **Uncalibrated Logits**: Raw YOLOv8n-cls softmax probabilities are compressed near ~0.35 across classes. Simple score addition/multiplication (Strategy D) is overwhelmed by high semantic scores, yielding no improvement (68.00% \(\rightarrow\) 68.00%).
- **Selective Deference (Strategy C & E)**: By deferring to visual evidence ONLY when the semantic classifier is uncertain (`confidence == 'Low'` or `is_ambiguous == True`), the multimodal system achieves maximum precision without corrupting high-confidence semantic predictions.
