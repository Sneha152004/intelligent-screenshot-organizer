# Phase 7K: Production Integration Proposal (Metadata Schema Extension)

## Executive Summary

This proposal outlines the recommended schema extension for integrating visual screen-type evidence into the production `ScreenshotMetadata` model without breaking existing APIs or altering semantic classification ownership.

---

## 1. Proposed Schema Extension for `ScreenshotMetadata`

```python
# Proposed addition to src/models/metadata.py (For Future Implementation Phase)

@dataclass
class ScreenshotMetadata:
    # Existing fields remain untouched
    screenshot_id: str
    filename: str
    image_uri: str
    category: str = "General"  # Authoritative (Owned by FinalRefinedClassifier)
    content_type: Optional[str] = "unknown"

    # ... existing evidence and signal fields ...

    # Proposed Non-Authoritative Visual Evidence Structure
    visual_evidence: Dict[str, Any] = field(
        default_factory=lambda: {
            "screen_type": "unknown",  # Mapped visual screen type
            "confidence": 0.0,  # Calibrated visual confidence
            "model_name": "yolov8n-cls",  # Model identity
            "model_version": "1.0.0-phase7i",  # Checkpoint version
            "is_supported_class": True,  # 5-class vs 13-class flag
            "evidence_status": "SUPPORTING_VISUAL_EVIDENCE"  # Status flag
        }
    )
```

---

## 2. Integration Data Flow Architecture

```text
                        Screenshot Ingestion
                                 │
           ┌─────────────────────┴─────────────────────┐
           ▼                                           ▼
   [Semantic Branch]                           [Visual Branch]
 FinalRefinedClassifier                      VisualScreenTypeAdapter
(OCR + Text Embeddings)                        (YOLOv8n-cls Image)
           │                                           │
  Semantic Category                            Visual Screen Type
  ("Coding", "Travel")                       ("code_editor_terminal")
           │                                           │
           └─────────────────────┬─────────────────────┘
                                 ▼
                     Multimodal Fusion Harness
                    (Strategy E: Selective Override)
                                 │
                                 ▼
                    ScreenshotMetadata Object
         category: "Coding" (Final Fused Category)
         evidence.visual_screen_type: "code_editor_terminal"
```

---

## 3. Preserved File Safeguards Confirmation

- **`src/models/metadata.py`**: NOT modified by Phase 7K.
- **`src/storage/metadata_store.py`**: NOT modified by Phase 7K.
- **Pre-existing Git status**: All pre-existing git modifications from outside Phase 7J/7K were strictly preserved and left untouched.
