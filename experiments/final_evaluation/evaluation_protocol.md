# Final Model Evaluation Protocol

**Workstream**: Final Model Evaluation & Frozen Test Validation  
**Date**: October 2026  
**Status**: **FROZEN EVALUATION PROTOCOL**  

---

## 1. Objective & Protocol Overview

This document specifies the exact, standardized evaluation protocol for the final model evaluation milestone of the Intelligent Screenshot Organizer.

The protocol establishes a locked, reproducible benchmark procedure for:
1. **Semantic Classifier**: `FinalRefinedClassifier`
2. **Visual Screen-Type Model**: Production 5-Class `Ultralytics YOLOv8n-cls`
3. **Multimodal System**: Integrated `OCRPipeline` with Selective Fusion (Strategy E)

---

## 2. Dataset & Split Definitions

### 2.1 Standardized Evaluation Partitions

```text
TRAINING PARTITION (83 Visual / 250 Semantic Screenshots)
        ↓
CALIBRATION & VALIDATION PARTITION (27 Calib + 25 Val Visual / 125 Val Semantic)
        ↓
FROZEN TEST PARTITIONS (125 Internal Holdout + 60 External Benchmark)
```

1. **Training Partition**:
   - Visual: 83 screenshots across 78 canonical duplicate groups from Phase 7H-G (`phase7hg_primary_manifest.csv`).
   - Semantic: 250 screenshots (`ss_001.jpg` to `ss_250.jpg`) from `metadata_v2.csv`.

2. **Validation / Calibration Partition**:
   - Visual Calibration: 27 screenshots (26 groups).
   - Visual Validation: 25 screenshots (24 groups).
   - Semantic Validation: 125 screenshots (`ss_251.jpg` to `ss_375.jpg`).

3. **Frozen Final Test Partitions**:
   - **Internal Holdout Test Set**: 125 screenshots (`ss_376.jpg` to `ss_500.jpg`) from `metadata_v2.csv`.
   - **External Benchmark Test Set**: 60 screenshots (`ext_001.jpg` to `ext_060.jpg`) from `external_manifest_v2.csv`.

### 2.2 Corrupted File Handling
- The 3 known corrupted/empty zero-byte screenshot files (`ss_441.png`, `ss_443.png`, `ss_560.png`) are excluded from training and evaluation sets.
- If a missing or corrupted file is encountered during inference, the pipeline MUST catch the exception gracefully, log a warning, and default to semantic fallback without crashing.

---

## 3. Production Model & Component Specifications

### 3.1 Semantic Classifier
- **Class**: `FinalRefinedClassifier` (`experiments/intelligence/final_refinement/final_refinement_experiment.py`)
- **Input**: Raw text OCR string and optional notes string.
- **Preprocessing**: `normalize_ocr_text()` lowercase stripping and `content_focused_ocr_text()` keyword extraction.
- **Output**: Category prediction string, confidence rating (`High`, `Medium`, `Low`), and `is_ambiguous` flag.

### 3.2 Visual Screen-Type Model
- **Model**: `Ultralytics YOLOv8n-cls` adapted to 5 classes (`yolov8n-cls.pt`)
- **Input**: Screenshot image path.
- **Preprocessing**: 224x224 RGB letterbox resize maintaining aspect ratio with pad color `(114, 114, 114)`.
- **Output**: 5-class probability vector (`code_editor_terminal`, `travel_booking`, `media_player_library`, `document_article_view`, `dashboard_settings`).

### 3.3 Multimodal Selective Fusion (Strategy E)
- **Module**: `src/pipeline.py` (`OCRPipeline`)
- **Override Condition**:
  ```python
  if (sem_confidence == "Low" or is_ambiguous or sem_category == "General") and vis_conf >= 0.40:
      if vis_st in VIS_5_TO_SEMANTIC_MAP:
          final_category = VIS_5_TO_SEMANTIC_MAP[vis_st]
          fusion_applied = True
  ```
- **Category Map**:
  - `code_editor_terminal` \(\rightarrow\) `Coding`
  - `travel_booking` \(\rightarrow\) `Travel`
  - `media_player_library` \(\rightarrow\) `Entertainment`
  - `document_article_view` \(\rightarrow\) `Education`
  - `dashboard_settings` \(\rightarrow\) `Connectivity`

---

## 4. Evaluation Metrics & Performance Benchmarks

### 4.1 Classification Metrics
For all partitions (Train, Validation, Test), report:
- **Top-1 Accuracy**
- **Top-2 / Top-3 Accuracy** (where multi-class probabilities are available)
- **Macro Precision, Recall, F1-Score**
- **Weighted Precision, Recall, F1-Score**
- **Balanced Accuracy**
- **Per-Class Support, Precision, Recall, F1-Score**
- **Confusion Matrix** (Saved as 300 DPI PNG figures)

### 4.2 Multimodal Fusion Transition Metrics
- **Semantic Predictions Changed Count**
- **Errors Corrected Count** (Semantic Incorrect \(\rightarrow\) Multimodal Correct)
- **Errors Harmed Count** (Semantic Correct \(\rightarrow\) Multimodal Incorrect)
- **Net Correction** (Corrected minus Harmed)

### 4.3 Runtime Latency & Computational Footprint
- **Semantic Inference Latency** (warm median, p95, cold start)
- **Visual Inference Latency** (warm median, p95, cold start)
- **Combined Pipeline Latency** (warm median, p95, cold start)
- **Two-Stage Retrieval Search Latency** (ChromaDB vector lookup + SQLite enrichment)

---

## 5. Environment & Reproducibility Constraints

- **Random Seed**: `42` for all split verifications and numpy/torch operations.
- **Python Version**: Python 3.9.13
- **Dependencies**: PyTorch 2.x, Ultralytics 8.x, scikit-learn 1.x, EasyOCR, ChromaDB.
- **Execution Rule**: The test evaluation scripts MUST be executed twice to verify 100% deterministic prediction agreement.
