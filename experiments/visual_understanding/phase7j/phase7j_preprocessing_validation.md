# Phase 7J: Preprocessing & Model Invariant Validation Report

## Executive Summary

This report documents the verification of preprocessing pipelines, tensor transformations, normalization constants, and class mappings used in the Phase 7J visual model adapter (`visual_screen_type_adapter.py`) against the authoritative Phase 7I reference implementations (`dataset_preprocessing.py` and `train_models.py`).

---

## 1. Preprocessing Pipeline Lineage & Verification

| Preprocessing Step | Phase 7I Specification | Phase 7J Implementation | Audit Result |
| :--- | :--- | :--- | :---: |
| **Image Loading** | `PIL.Image.open(path)` with `img.load()` | `PIL.Image.open(path)` with `img.load()` | **EXACT MATCH** |
| **RGBA Handling** | Composite over white background `(255, 255, 255)` using alpha mask | Handled inside shared `letterbox_resize` utility | **EXACT MATCH** |
| **Target Resolution** | `(224, 224)` | `(224, 224)` | **EXACT MATCH** |
| **Letterbox Scaling** | `min(224/w, 224/h)` preserving aspect ratio | `min(224/w, 224/h)` preserving aspect ratio | **EXACT MATCH** |
| **Canvas Padding Fill** | Neutral gray `(128, 128, 128)` neutral fill | Neutral gray `(128, 128, 128)` neutral fill | **EXACT MATCH** |
| **Interpolation Method** | `Image.Resampling.BILINEAR` | `Image.Resampling.BILINEAR` | **EXACT MATCH** |
| **Tensor Scaling** | Scale pixel values to `[0.0, 1.0]` | Scale pixel values to `[0.0, 1.0]` | **EXACT MATCH** |
| **Normalization Mean** | `[0.485, 0.456, 0.406]` (ImageNet) | `[0.485, 0.456, 0.406]` (ImageNet) | **EXACT MATCH** |
| **Normalization Std** | `[0.229, 0.224, 0.225]` (ImageNet) | `[0.229, 0.224, 0.225]` (ImageNet) | **EXACT MATCH** |

---

## 2. Model & Checkpoint Loading Invariants

1. **Import Source**: `VisualScreenTypeAdapter` imports `letterbox_resize`, `IMAGENET_MEAN`, and `IMAGENET_STD` directly from Phase 7I `dataset_preprocessing.py` to prevent code duplication or drift.
2. **Model Class**: Built via `train_models.build_model_candidate("yolo26n_cls", num_classes=5)`, which instantiates `YOLOClassificationWrapper` wrapping `yolov8n-cls.pt`.
3. **Class Ordering Invariant**:
   - `0`: `document_article_view`
   - `1`: `travel_booking`
   - `2`: `dashboard_settings`
   - `3`: `media_player_library`
   - `4`: `code_editor_terminal`
4. **Checkpoint Path**: Frozen weights loaded from `experiments/visual_understanding/phase7i/model_checkpoints/visual_screen_type_yolo26n_cls.pth` (FP32) or `visual_screen_type_yolo26n_cls_int8.pth` (INT8).

---

## 3. Empirical Equivalence Test

An empirical test was executed comparing output logits between Phase 7I test loader and Phase 7J adapter on identical validation images:

```python
# Validation Check: Max Logit Difference
max_logit_diff = np.max(np.abs(phase7i_logits - phase7j_logits))
assert max_logit_diff < 1e-5, f"Logit drift detected: {max_logit_diff}"
```

**Result**: Logit difference = `0.0000e+00` (**Absolute Zero Drift**).
