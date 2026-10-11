# Pipeline Integration & Architecture Report

**Phase**: Final Visual Integration & Product Pipeline  
**Date**: October 2026  
**Status**: **INTEGRATED & CONTRACT VALIDATED**  

---

## 1. System Architecture Overview

The Intelligent Screenshot Organizer product features a modular 3-Layer Storage & Multimodal Processing Architecture coordinated by `src/pipeline.py` (`OCRPipeline`).

```text
                               ┌─────────────────────────┐
                               │   RAW SCREENSHOT IMAGE  │
                               └────────────┬────────────┘
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      ▼                                           ▼
          ┌──────────────────────┐                    ┌──────────────────────┐
          │  EasyOCR / Mock OCR  │                    │ LocalFileStore (L1)  │
          └───────────┬──────────┘                    └───────────┬──────────┘
                      │                                           │
                      ▼                                           ▼
          ┌──────────────────────┐                    ┌──────────────────────┐
          │FinalRefinedClassifier│                    │ YOLOv8n-cls (Vision) │
          └───────────┬──────────┘                    └───────────┬──────────┘
                      │                                           │
                      ▼                                           ▼
          Semantic Category & Score                     Visual Layout Type & Score
                      │                                           │
                      └─────────────────────┬─────────────────────┘
                                            ▼
                               ┌─────────────────────────┐
                               │ Selective Fusion Layer  │
                               └────────────┬────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │   ScreenshotMetadata    │
                               └────────────┬────────────┘
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      ▼                                           ▼
          ┌──────────────────────┐                    ┌──────────────────────┐
          │SQLiteMetadataStore(L2│                    │VectorStoreManager(L3)│
          └──────────────────────┘                    └──────────────────────┘
```

---

## 2. Component Contracts & Interfaces

### 2.1 Layer 1: Local File Store (`src/storage/file_store.py`)
- **Class**: `LocalFileStore`
- **Contract**: Accepts raw image source paths, verifies file integrity, copies file to local screenshot repository `data/screenshots/`, and returns relative URI path.

### 2.2 OCR Engine (`src/easy_ocr.py`, `src/mock_ocr.py`)
- **Interface**: `BaseOCREngine`
- **Contract**: Accepts image file path, executes optical character recognition, and returns cleaned string output.

### 2.3 Semantic Classification Branch (`experiments/intelligence/final_refinement/`)
- **Class**: `FinalRefinedClassifier`
- **Contract**: Accepts `raw_text` and optional `notes`, applies multi-signal taxonomy rules, and outputs prediction dictionary:
  - `prediction`: Semantic category (e.g. `Coding`, `Travel`, `Shopping`, `General`)
  - `confidence`: Confidence rating (`High`, `Medium`, `Low`)
  - `is_ambiguous`: Boolean flag indicating classification ambiguity.

### 2.4 Visual Layout Branch (`experiments/visual_understanding/phase7j/`)
- **Class**: `VisualScreenTypeAdapter`
- **Contract**: Preprocesses screenshot image to 224x224 letterbox tensor, executes `YOLOv8n-cls` forward pass, and outputs:
  - `screen_type`: Visual layout class (`code_editor_terminal`, `travel_booking`, `media_player_library`, `document_article_view`, `dashboard_settings`)
  - `confidence`: Softmax probability score.

### 2.5 Multimodal Selective Fusion (Strategy E)
- **Rules**:
  ```python
  if (sem_confidence == "Low" or is_ambiguous or sem_category == "General") and vis_conf >= 0.40:
      if vis_st in VIS_5_TO_SEMANTIC_MAP:
          final_category = VIS_5_TO_SEMANTIC_MAP[vis_st]
          fusion_applied = True
  ```
- **Mapping**:
  - `code_editor_terminal` \(\rightarrow\) `Coding`
  - `travel_booking` \(\rightarrow\) `Travel`
  - `media_player_library` \(\rightarrow\) `Entertainment`
  - `document_article_view` \(\rightarrow\) `Education`
  - `dashboard_settings` \(\rightarrow\) `Connectivity`

---

## 3. Metadata Serialization Schema (Layer 2)

Rich metadata is structured in `ScreenshotMetadata` (`src/models/metadata.py`) and saved to SQLite (`SQLiteMetadataStore`):

```json
{
  "screenshot_id": "ss_2026_10_001",
  "filename": "screenshot_001.png",
  "image_uri": "data/screenshots/ss_2026_10_001.png",
  "category": "Coding",
  "tags": ["python", "vscode"],
  "ocr_available": true,
  "summary": "VS Code editor window",
  "evidence": {
    "text_snippet": "def train_model(): import torch",
    "visual_evidence": {
      "screen_type": "code_editor_terminal",
      "confidence": 0.94,
      "model_name": "YOLOv8n-cls",
      "model_version": "1.0.0-phase7j",
      "fusion_applied": true,
      "fusion_reason": "Visual layout 'code_editor_terminal' (conf 0.94) overrode low-confidence semantic fallback 'General' -> 'Coding'",
      "original_semantic_category": "General"
    },
    "fusion_applied": true,
    "fusion_reason": "Visual layout 'code_editor_terminal' overrode low-confidence semantic fallback",
    "original_semantic_category": "General"
  }
}
```

---

## 4. Storage Backward Compatibility

- **SQLite Schema Preservation**: The SQLite table `screenshot_metadata` continues to use standard columns (`screenshot_id`, `filename`, `image_uri`, `category`, `tags`, `summary`, `evidence`, `created_at`, `updated_at`).
- **JSON Column Storage**: Visual evidence dictionary is JSON-serialized inside the existing `evidence` column without requiring SQL schema migrations or DDL modifications.
- **Legacy Record Compatibility**: Legacy records lacking `visual_evidence` deserialize cleanly without KeyError exceptions.

---

## 5. Integration Verification Summary

All component boundaries were tested via unit tests in `experiments/visual_understanding/final_visual_integration/tests/test_final_integration.py` and `tests/test_end_to_end_pipeline.py`. Contract validation confirmed zero breaking API changes across the entire product pipeline.
