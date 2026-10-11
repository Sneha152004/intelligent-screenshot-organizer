# Visual Screen-Type Data Contract Specification

## 1. Executive Overview

This specification establishes the formal data contract for visual screen-type evidence provided by the lightweight visual model (**Ultralytics YOLOv8n-cls**).

The contract defines input requirements, output schema, full vs supported taxonomy boundaries, and unknown class handling policies.

---

## 2. Input Requirements

| Field | Type | Constraint | Description |
| :--- | :--- | :--- | :--- |
| `image_input` | `Union[str, Path, PIL.Image]` | File path or loaded PIL Image object | Raw screenshot image entity |
| `target_resolution` | `Tuple[int, int]` | `(224, 224)` | Input resolution target for model tensor conversion |
| `preprocessing` | `str` | Aspect-ratio letterbox | Preserves original image ratio using `(128, 128, 128)` padding and RGBA composite |
| `normalization` | `str` | ImageNet standard | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` |

---

## 3. Output Schema Contract

```json
{
  "screen_type": "document_article_view",
  "confidence": 0.4521,
  "ranked_predictions": [
    {
      "screen_type": "document_article_view",
      "confidence": 0.4521
    },
    {
      "screen_type": "code_editor_terminal",
      "confidence": 0.2104
    },
    {
      "screen_type": "travel_booking",
      "confidence": 0.1532
    },
    {
      "screen_type": "dashboard_settings",
      "confidence": 0.1041
    },
    {
      "screen_type": "media_player_library",
      "confidence": 0.0802
    }
  ],
  "model_name": "yolov8n-cls",
  "model_version": "1.0.0-phase7i",
  "checkpoint_type": "FP32",
  "inference_latency_ms": 18.11,
  "supported_classes": [
    "document_article_view",
    "travel_booking",
    "dashboard_settings",
    "media_player_library",
    "code_editor_terminal"
  ],
  "is_supported_class": true,
  "evidence_status": "SUPPORTING_VISUAL_EVIDENCE"
}
```

### Schema Invariants:
1. **`screen_type`**: Non-empty string. Must belong to the 13-class taxonomy.
2. **`confidence`**: Float value strictly in range \([0.0, 1.0]\).
3. **`ranked_predictions`**: List sorted in strictly descending order of confidence.
4. **`model_name`**: Fixed string `"yolov8n-cls"`.
5. **`evidence_status`**: Declares that this prediction is **supporting visual evidence**, NOT the primary semantic category.

---

## 4. Taxonomy Boundaries: Full vs Supported Classes

> [!IMPORTANT]
> **Taxonomy Scope Differentiation**:
>
> The project maintains a 13-class visual screen-type taxonomy. However, the current supervised YOLOv8n-cls model was benchmarked on the frozen 5-class primary dataset.

| Full 13-Class Taxonomy | Supervised Model Support | Class Description |
| :--- | :---: | :--- |
| `document_article_view` | **SUPPORTED (Class 1)** | Reading text articles, documentation, or long-form posts |
| `travel_booking` | **SUPPORTED (Class 2)** | Flight/hotel booking forms, itineraries, or reservation views |
| `dashboard_settings` | **SUPPORTED (Class 3)** | System settings, admin dashboards, or analytics panels |
| `media_player_library` | **SUPPORTED (Class 4)** | Video players, audio playlists, or photo/media galleries |
| `code_editor_terminal` | **SUPPORTED (Class 5)** | IDEs, text editors, terminal windows, or error traces |
| `product_listing` | *UNSUPPORTED IN 5-CLASS MODEL* | E-commerce grid/list search results |
| `product_detail` | *UNSUPPORTED IN 5-CLASS MODEL* | Individual product specification page |
| `cart_checkout_payment` | *UNSUPPORTED IN 5-CLASS MODEL* | Shopping cart, checkout screen, or payment modal |
| `login_authentication` | *UNSUPPORTED IN 5-CLASS MODEL* | Sign-in forms or 2FA authentication screens |
| `messaging_chat` | *UNSUPPORTED IN 5-CLASS MODEL* | Instant messaging conversations or chat threads |
| `email_view` | *UNSUPPORTED IN 5-CLASS MODEL* | Email client inbox or message reader |
| `error_dialog_alert` | *UNSUPPORTED IN 5-CLASS MODEL* | System error popup or warning modal |
| `other_screen` | *UNSUPPORTED IN 5-CLASS MODEL* | General fallback UI screen |

---

## 5. Unknown / Unsupported Class Handling Policy

1. **Low Confidence \(\neq\) Out-of-Distribution (OOD)**:
   - Phase 7I-A demonstrated that YOLOv8n-cls produces mean max confidence of ~0.3481 on in-distribution validation data and ~0.3488 on out-of-distribution probe data.
   - Therefore, thresholding softmax confidence (e.g. `confidence < 0.50 => OOD`) **CANNOT** be used to reliably detect unsupported classes.
2. **Conservative Evidence Policy**:
   - The visual adapter maps predictions into the 5 supported classes without fabricating unsupported labels.
   - The visual signal is flagged with `evidence_status = "SUPPORTING_VISUAL_EVIDENCE"`.
   - The downstream `FinalRefinedClassifier` remains authoritative for semantic categorization.
