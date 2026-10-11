# Phase 7J: System Architecture Audit & Visual Integration Boundary

## Executive Summary

This architecture audit defines the structural boundary and integration entry points for incorporating the visual screen-type classifier (**Ultralytics YOLOv8n-cls**) into the Intelligent Screenshot Organizer repository without modifying production behavior or altering existing contracts.

---

## 1. Existing System Component Survey

| System Domain | Authoritative Module | Role / Contract | Preservation Policy |
| :--- | :--- | :--- | :--- |
| **Screenshot Ingestion** | `src/loader.py`, `src/models/screenshot.py` | Physical image file loading and `Screenshot` entity creation | **UNTOUCHED** |
| **Semantic Classification** | `FinalRefinedClassifier` (`experiments/intelligence/final_refinement/`) | Multi-signal semantic categorization into 10 canonical categories | **UNTOUCHED** |
| **OCR Pipeline** | `EasyOCR` (`src/easy_ocr.py`) | Text extraction and bounding box detection | **UNTOUCHED** |
| **Embedding Engine** | `MiniLM` (`src/embedder.py`) | 384-dimensional text semantic embeddings | **UNTOUCHED** |
| **Metadata Representation** | `ScreenshotMetadata` (`src/models/metadata.py`) | Standardized schema storing category, evidence, URLs, entities | **UNTOUCHED** |
| **Visual Evidence Branch** | `experiments/visual_understanding/phase7j/` | Visual screen-type recognition providing supporting evidence | **PHASE 7J ISOLATED** |

---

## 2. Distinction Between Semantic Category & Visual Screen Type

> [!IMPORTANT]
> **Semantic Category \(\neq\) Visual Screen Type**
>
> - **Semantic Category** (Owned by `FinalRefinedClassifier`): Represents the high-level functional or domain context of the screenshot. Examples: `Shopping`, `Coding`, `Travel`, `Payment`, `Education`.
> - **Visual Screen Type** (Owned by `YOLOv8n-cls`): Represents the visual layout composition and UI page structure. Examples: `product_detail`, `code_editor_terminal`, `travel_booking`, `document_article_view`, `dashboard_settings`.

### Examples of Concept Disambiguation:
1. A screenshot of an Amazon item page has **Semantic Category** = `Shopping` and **Visual Screen Type** = `product_detail`.
2. A screenshot of VS Code showing Python code has **Semantic Category** = `Coding` and **Visual Screen Type** = `code_editor_terminal`.
3. A screenshot of a flight confirmation has **Semantic Category** = `Travel` and **Visual Screen Type** = `travel_booking`.

---

## 3. Theoretical Integration Boundary

The visual branch connects in parallel with the semantic branch at a non-production orchestration layer:

```text
                     Screenshot Image
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
    [Semantic Branch]                 [Visual Branch]
  FinalRefinedClassifier            YOLOv8n-cls Adapter
            │                                 │
     semantic category                   screen_type
  ("Shopping", "Coding")          ("product_detail", etc.)
            │                                 │
            └────────────────┬────────────────┘
                             ▼
                 [Integration Harness]
              Combined Screenshot Result
```

- **Semantic Category**: Remains authoritative for categorization, directory organization, and intent tagging.
- **Visual Screen Type**: Serves strictly as **supporting visual evidence** attached under visual metadata.

---

## 4. Preservation & Scope Restrictions

1. **No Production Modifications**: `src/` modules (`loader.py`, `pipeline.py`, `multimodal_classifier.py`, `metadata.py`, `easy_ocr.py`, `embedder.py`) remain completely untouched.
2. **No Model Retraining**: Checkpoint `visual_screen_type_yolo26n_cls.pth` and `_int8.pth` remain frozen and unmodified.
3. **No Score Fusion**: No score averaging, multiplication, or joint probability fusion between semantic confidence and visual confidence is performed.
4. **No Git Modifications**: Git repository state remains clean and uncommitted.
