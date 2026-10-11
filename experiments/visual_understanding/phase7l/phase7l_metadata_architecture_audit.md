# Phase 7L: Metadata Architecture Audit Report (Track A)

## Executive Summary

This architecture audit evaluates the project's metadata structure, pipeline ingestion paths, storage layer, and model execution flow to establish a backward-compatible integration boundary for multimodal visual evidence.

---

## 1. Existing System Component & Metadata Path Audit

| Subsystem Component | Module Location | Schema / Contract | Integration Entry Point | Preservation Rule |
| :--- | :--- | :--- | :--- | :---: |
| **Screenshot Model** | `src/models/screenshot.py` | `Screenshot(screenshot_id, filename, image_uri)` | Ingestion pipeline | **UNTOUCHED** |
| **Metadata Model** | `src/models/metadata.py` | `ScreenshotMetadata` dataclass | Application-level metadata model | **EXTENDED COMPATIBLY** |
| **Metadata Storage** | `src/storage/metadata_store.py` | `SQLiteMetadataStore` (data/metadata.sqlite) | Layer 2 SQLite persistence | **EXTENDED COMPATIBLY** |
| **Semantic Classifier** | `experiments/intelligence/final_refinement/` | `FinalRefinedClassifier` | Multi-signal semantic category | **UNTOUCHED (FROZEN)** |
| **Visual Adapter** | `experiments/visual_understanding/phase7j/` | `VisualScreenTypeAdapter` | Visual screen-type evidence | **UNTOUCHED (FROZEN)** |
| **Multimodal Harness** | `experiments/visual_understanding/phase7k/` | `MultimodalFusionHarness` | Selective evidence overriding | **ISOLATED REFERENCE** |

---

## 2. Preservation of the Three Distinct Concepts

> [!IMPORTANT]
> **Architectural Principle**:
> The system maintains three distinct non-overlapping concepts:
> 1. **Semantic Category** (Owned by `FinalRefinedClassifier`): e.g. `Coding`, `Travel`, `Shopping`, `Education`.
> 2. **Visual Screen Type** (Owned by `YOLOv8n-cls`): e.g. `code_editor_terminal`, `travel_booking`, `document_article_view`.
> 3. **Multimodal Fusion Decision** (Owned by `MultimodalFusionHarness`): Selective overriding when semantic prediction is ambiguous or low-confidence.

These three concepts MUST NOT be collapsed into a single string attribute.

---

## 3. Compatibility & Schema Extensibility

- **`ScreenshotMetadata` Dataclass**: `evidence` (line 77) and `classification_evidence` (line 95) are flexible `Dict[str, Any]` fields.
- **SQLite Table (`screenshot_metadata`)**: `evidence` (line 73) and `classification_evidence` (line 89) are stored as JSON-serialized text strings (`TEXT`).
- **Backward Compatibility**:
  - Legacy metadata records containing `{}` deserialize with `visual_evidence = None`.
  - New records populate `evidence["visual_evidence"]` with structured visual evidence (`screen_type`, `confidence`, `model_name`, `fusion_applied`, `fusion_reason`, `original_semantic_category`).
  - No new SQL columns or breaking DDL migrations are required.
