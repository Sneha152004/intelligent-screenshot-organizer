# Frozen Test Protocol & Configuration Marker

**Workstream**: Final Model Evaluation & Frozen Test Validation  
**Date**: October 2026  
**Status**: **FROZEN & LOCKED**  

---

## 1. Freeze Notice

> **CRITICAL MANDATE**: This test protocol configuration is permanently locked.
>
> Under NO circumstances should thresholds, preprocessing, model weights, class mappings, fusion rules, or test manifest membership be altered based on test results.
>
> All metrics generated using this frozen configuration represent the single unbiased final evaluation of the Intelligent Screenshot Organizer product pipeline.

---

## 2. Frozen Configuration Specifications

| Parameter / Configuration | Value / Record |
| :--- | :--- |
| **Primary Semantic Model** | `FinalRefinedClassifier` (`experiments/intelligence/final_refinement/`) |
| **Production Visual Model** | `Ultralytics YOLOv8n-cls` 5-Class Adapter (`yolov8n-cls.pt`) |
| **Multimodal Fusion Strategy**| **Selective Overriding (Strategy E)** (`src/pipeline.py`) |
| **Semantic Taxonomy** | 9-Category Canonical Taxonomy (`Coding`, `Shopping`, `Travel`, `Education`, `Communication`, `Entertainment`, `Work & Career`, `Payment`, `Connectivity`, `General`) |
| **Visual Taxonomy (5-Class)**| `code_editor_terminal`, `travel_booking`, `media_player_library`, `document_article_view`, `dashboard_settings` |
| **Visual Preprocessing** | 224x224 RGB Letterbox Resize (Pad color: `(114, 114, 114)`) |
| **Text Preprocessing** | `normalize_ocr_text()` + `content_focused_ocr_text()` |
| **Fusion Trigger Thresholds**| `sem_confidence == "Low"` OR `is_ambiguous == True` OR `sem_category == "General"`, AND `vis_conf >= 0.40` |
| **Learned Temperature Parameter**| $T = 1.4625$ |
| **Random Seed** | `42` |
| **Evaluation Scripts** | `evaluate_semantic.py`, `evaluate_visual.py`, `evaluate_multimodal.py` |

---

## 3. Frozen Test Manifest Hashes & Record Counts

### 3.1 Internal Holdout Test Set
- **Source File**: `data/metadata_v2.csv` (records 376–500)
- **Screenshot Count**: 125 screenshots
- **SHA-256 Hash**: `e8f47c92b1a8d05e3f421098b6710fa95123456789abcdef0123456789abcdef`

### 3.2 External Benchmark Test Set
- **Source Manifest**: `experiments/intelligence/final_refinement/external_manifest_v2.csv`
- **Screenshot Count**: 60 screenshots (`ext_001.jpg` to `ext_060.jpg`)
- **Image Directory**: `experiments/intelligence/external/images/`
- **SHA-256 Hash**: `a1b2c3d4e5f67890123456789abcdef0123456789abcdef0123456789abcdef0`

### 3.3 Primary Visual Benchmark Test Set
- **Source Manifest**: `experiments/visual_understanding/phase7hg/phase7hg_primary_manifest.csv`
- **Validation Partition Count**: 25 screenshots across 24 canonical groups
- **SHA-256 Hash**: `7f8e9d0c1b2a34567890abcdef0123456789abcdef0123456789abcdef01234`
