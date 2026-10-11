# Product Integration Audit & Architecture Document

## 1. Executive Summary & Audit Purpose

This document completes the pre-implementation audit of the Intelligent Screenshot Organizer repository. The goal is to audit all existing production and experimental modules, establish the canonical end-to-end data flow, and define the integration blueprint for connecting OCR, semantic classification, visual screen-type analysis, multimodal fusion, metadata persistence, embedding generation, and ChromaDB vector search.

---

## 2. Component-by-Component Audit

### 2.1 Entry Point & Data Loading (`run_pipeline.py`, `src/loader.py`)
- **Current Entry Point**: `run_pipeline.py` at the project root coordinates the 3-layer architecture (`LocalFileStore`, `SQLiteMetadataStore`, `VectorStoreManager`, `SentenceTransformerEmbedder`, `OCRPipeline`).
- **Data Loading**: `src/loader.py` (`load_dataset`) reads metadata CSV files (`data/metadata.csv` or `data/metadata_v2.csv`), validates image file existence in `data/screenshots/`, and returns structured document representations.

### 2.2 OCR Text Extraction (`src/easy_ocr.py`, `src/mock_ocr.py`)
- **OCR Engine**: `EasyOCREngine` (in `src/easy_ocr.py`) uses EasyOCR for GPU/CPU text extraction. `MockOCREngine` serves as a lightweight fallback for fast deterministic testing.
- **Contract**: `extract_text(image_path: str) -> str`.

### 2.3 Text Chunking (`src/chunker.py`)
- **Text Chunker**: `ParagraphChunker` (in `src/chunker.py`) splits extracted OCR text into paragraph-level chunks (`chunk(text: str) -> List[str]`).

### 2.4 Semantic Text Classification (`FinalRefinedClassifier`)
- **Primary Semantic Classifier**: `FinalRefinedClassifier` (in `experiments/intelligence/final_refinement/final_refinement_experiment.py`).
- **Role**: Determines the primary semantic category (`Coding`, `Shopping`, `Travel`, `Education`, `Communication`, `Entertainment`, `Connectivity`, `Work & Career`, `Payment`, `General`).
- **Contract**: `predict_final(raw_text: str, notes: str = "", variant: str = "C") -> Dict[str, Any]`. Returns `prediction`, `confidence` (`High`/`Low`), `is_ambiguous` (`bool`), `margin`, and `category_scores`.
- **Status**: Authoritative. Must NOT be retrained or modified.

### 2.5 Visual Screen-Type Analysis (`YOLOv8n-cls` 5-Class Model)
- **Production Visual Model**: `YOLOv8n-cls` (5-Class Validated Model) in `experiments/visual_understanding/final_visual_integration/pipeline_integration.py`.
- **Supported Visual Types**: `document_article_view`, `travel_booking`, `dashboard_settings`, `media_player_library`, `code_editor_terminal`.
- **Preprocessing**: 224x224 RGB letterbox resize with padding and ImageNet normalization.
- **Status**: Production-selected visual model. Experimental 13-class model evaluated but retained as secondary experiment.

### 2.6 Multimodal Selective Fusion (Strategy E)
- **Rule Engine**: Merges semantic and visual signals conservatively:
  - If semantic prediction is confident (`High` confidence, unambiguous, non-`General`), **preserve semantic prediction**.
  - If semantic prediction is ambiguous / low confidence / fallback `General`, and visual model detects validated supporting layout evidence (`code_editor_terminal` -> `Coding`, `travel_booking` -> `Travel`, `media_player_library` -> `Entertainment`, `document_article_view` -> `Education`, `dashboard_settings` -> `Connectivity`), **apply visual override**.
- **Original Prediction Preservation**: Preserves `semantic_category`, `visual_evidence`, `fusion_applied`, and `fusion_reason` inside metadata for full explainability.

### 2.7 Application Metadata & Storage (`src/models/metadata.py`, `src/storage/metadata_store.py`)
- **Metadata Model**: `ScreenshotMetadata` dataclass supporting identity (`screenshot_id`, `filename`, `image_uri`), category, tags, summary, and `evidence["visual_evidence"]`.
- **SQLite Persistence**: `SQLiteMetadataStore` persists `ScreenshotMetadata` records in SQLite table `screenshot_metadata`. The `evidence` JSON dictionary column natively stores `visual_evidence` without requiring SQL DDL schema migration.

### 2.8 Embeddings & ChromaDB Indexing (`src/embedder.py`, `src/vector_store.py`)
- **Dense Embeddings**: `SentenceTransformerEmbedder` (MiniLM-L6-v2, 384 dimensions).
- **Vector Store**: `VectorStoreManager` indexes OCR text chunks in ChromaDB with metadata (`screenshot_id`, `chunk_index`, `category`).

### 2.9 Two-Stage Retrieval (`src/pipeline.py`)
- **Search Flow**: `search_and_enrich(query_text)`:
  1. Queries ChromaDB (Layer 3) to get top-K chunk matches.
  2. Resolves `screenshot_id` against `SQLiteMetadataStore` (Layer 2) and `LocalFileStore` (Layer 1) to return enriched search results with full metadata and image URIs.

---

## 3. Canonical End-to-End Architecture

```text
                         SCREENSHOT
                             │
                             ▼
                    Image Validation
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
             OCR                       Visual Model
        (EasyOCR Engine)             (YOLOv8n-cls 5-Class)
              │                             │
              ▼                             ▼
          Text Chunks                  screen_type
      (ParagraphChunker)                    │
              │                             │
              ▼                             │
     FinalRefinedClassifier                 │
              │                             │
              ▼                             │
       semantic category                    │
              │                             │
              └──────────────┬──────────────┘
                             ▼
                     Multimodal Fusion
               (Selective Overriding Rule)
                             │
                             ▼
                     ScreenshotMetadata
             (evidence.visual_evidence)
                       │             │
                       │             ▼
                       │    SQLiteMetadataStore
                       │     (Layer 2 Storage)
                       │
                       ▼
             SentenceTransformerEmbedder
               (MiniLM 384d Vectors)
                       │
                       ▼
               VectorStoreManager
             (Layer 3 ChromaDB Index)
                       │
                       ▼
             2-Stage Semantic Retrieval
```

---

## 4. Safe Fallback Matrix

| Scenario | Component Impact | Fallback Behavior |
| :--- | :--- | :--- |
| **Normal Processing** | All components active | Full multimodal enrichment, SQLite persistence & ChromaDB indexing |
| **Visual Model Unavailable** | Checkpoint file missing | Pipeline logs warning, sets `visual_evidence = None`, proceeds with semantic-only classification |
| **Visual Inference Exception** | Runtime exception during inference | Exception caught gracefully, logged, fallback to semantic-only classification |
| **Unsupported Visual Class** | Visual layout not mapped | Visual evidence recorded, semantic prediction remains authoritative |
| **OCR Text Extraction Failure** | OCR returns empty string | OCR falls back to filename/notes; visual model provides primary structural evidence if confident |
| **Corrupted Image File** | File unreadable by PIL/OpenCV | Screenshot rejected gracefully with warning; no ingestion pipeline crash |

---

## 5. Integration Implementation Plan

1. **Production Pipeline Upgrade (`src/pipeline.py`)**: Integrate `FinalRefinedClassifier` and `IntegratedMultimodalPipeline` visual fusion into `OCRPipeline.process()`.
2. **End-to-End Validation Script**: Build `experiments/visual_understanding/final_visual_integration/end_to_end_validation.py` to validate end-to-end execution across agreement, ambiguity resolution, unsupported visual classes, sparse OCR, and corrupted inputs.
3. **End-to-End Product Test Suite (`tests/test_end_to_end_pipeline.py`)**: Create comprehensive product-level unit & integration tests covering all 15 test scenarios.
4. **Reports**: Generate `end_to_end_validation_report.md`, `pipeline_integration_report.md`, `performance_report.md`, and `product_integration_completion_report.md`.
