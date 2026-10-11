# Phase 7K: Input Audit & Prediction Interface Analysis

## Executive Summary

This input audit inspects the prediction interfaces, logit/probability availability, label mappings, and taxonomy relationships between the frozen semantic classifier (`FinalRefinedClassifier`) and the frozen visual screen-type model (**Ultralytics YOLOv8n-cls**).

---

## 1. Semantic Branch Interface Survey (`FinalRefinedClassifier`)

- **Module Source**: `experiments/intelligence/final_refinement/final_refinement_experiment.py`
- **Primary Method**: `predict_final(raw_text: str, notes: str = "", variant: str = "C") -> Dict[str, Any]`
- **Input Requirements**: Extracted OCR text string (`raw_text`), optional user notes, rule variant configuration.
- **Output Schema**:
  - `prediction`: Top-1 predicted semantic category string (e.g., `Coding`, `Shopping`, `Travel`, `Education`).
  - `top_candidates`: Ranked list of dictionaries containing `category` and unnormalized combined score.
  - `category_scores`: Dictionary mapping all canonical categories to combined numerical scores.
  - `confidence`: Qualitative indicator string (`"High"`, `"Medium"`, `"Low"`).
  - `is_ambiguous`: Boolean flag (true if score margin < 0.10).
  - `margin`: Numerical score margin between Top-1 and Top-2 candidate categories.
- **Logit / Probability Availability**: Raw unnormalized combined scores are available under `category_scores`. Calibrated probabilities are not directly output; softmax or min-max scaling can transform `category_scores` to pseudoprobabilities.

---

## 2. Visual Branch Interface Survey (YOLOv8n-cls)

- **Module Source**: `experiments/visual_understanding/phase7j/visual_screen_type_adapter.py`
- **Primary Method**: `predict_image(image_input: Union[str, Path, PIL.Image]) -> Dict[str, Any]`
- **Input Requirements**: Image file path or loaded PIL Image object. Preprocessed via letterbox padding to \((224, 224)\) and ImageNet standard normalization.
- **Output Schema**:
  - `screen_type`: Top-1 predicted visual screen type string (e.g., `code_editor_terminal`, `document_article_view`).
  - `confidence`: Uncalibrated raw softmax probability float.
  - `ranked_predictions`: Sorted list of dictionaries containing `screen_type` and confidence float.
  - `model_name`: `"yolov8n-cls"`.
  - `supported_classes`: List of 5 supported visual classes.
- **Logit / Probability Availability**: Raw output logits before softmax activation are directly accessible via PyTorch module forward pass `model(tensor_input)`.

---

## 3. Label Space Mapping & Taxonomy Incompatibilities

> [!IMPORTANT]
> **Distinct Label Spaces**:
> The semantic classifier outputs 10 high-level functional categories. The visual classifier outputs 5 structural screen types. They cannot be directly added or averaged without an explicit mapping.

### Empirical Mapping Matrix (Visual Screen Type \(\rightarrow\) Semantic Category):

| Visual Screen Type | Primary Semantic Mapping | Secondary / Fallback Semantic Mappings |
| :--- | :--- | :--- |
| `code_editor_terminal` | `Coding` | `General` |
| `travel_booking` | `Travel` | `Shopping`, `General` |
| `media_player_library` | `Entertainment` | `General` |
| `document_article_view` | `Education` | `Work & Career`, `General` |
| `dashboard_settings` | `Connectivity` | `Work & Career`, `General` |

---

## 4. Audit Conclusions & Calibration Directives

1. **Raw Logit Access**: Raw visual logits are accessible directly from PyTorch model instance `adapter.model(tensor)`.
2. **Calibration Requirement**: YOLOv8n-cls output logits must undergo Temperature Scaling on the 27-sample calibration split before being used in any fusion logic.
3. **Data Isolation**: Calibration parameters (temperature \(T\), mapping weights) MUST be fit strictly on the 27 calibration samples. The 25 validation samples MUST remain strictly isolated for unbiased final evaluation.
