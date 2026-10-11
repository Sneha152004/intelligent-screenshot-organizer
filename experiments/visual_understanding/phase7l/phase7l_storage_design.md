# Phase 7L: Storage Design & Schema Migration Plan (Track A)

## Executive Summary

This report defines the storage design, serialization schema, and migration approach for persisting multimodal visual evidence in `ScreenshotMetadata` and `SQLiteMetadataStore`.

---

## 1. Storage Schema Design

Visual evidence is stored inside the `evidence` dictionary field of `ScreenshotMetadata`:

```json
{
  "screenshot_id": "ss_045.jpg",
  "filename": "ss_045.jpg",
  "image_uri": "data/screenshots/ss_045.jpg",
  "category": "Coding",
  "content_type": "code_editor",
  "evidence": {
    "visual_evidence": {
      "screen_type": "code_editor_terminal",
      "confidence": 0.3707,
      "model_name": "yolov8n-cls",
      "model_version": "1.0.0-phase7i",
      "checkpoint_type": "FP32",
      "is_supported_class": true,
      "fusion_applied": true,
      "fusion_strategy": "Strategy E (Selective Overriding)",
      "fusion_reason": "Visual layout 'code_editor_terminal' overrode low-confidence/ambiguous semantic fallback 'General'",
      "original_semantic_category": "General"
    }
  }
}
```

---

## 2. Serialization & Deserialization Flow

```text
[ScreenshotMetadata Object]
        │
        ├── category = "Coding" (Authoritative Semantic Category)
        └── evidence["visual_evidence"] = {...} (Supporting Visual Signal)
        │
        ▼
   json.dumps(metadata.evidence)
        │
        ▼
[SQLite screenshot_metadata Table]
   evidence TEXT Column
        │
        ▼
   json.loads(row["evidence"])
        │
        ▼
[Deserialized ScreenshotMetadata Object]
   visual_evidence preserved cleanly
```

---

## 3. Backward & Forward Compatibility Guarantee

1. **Legacy Records**: Old records without `visual_evidence` deserialize cleanly. `evidence.get("visual_evidence")` evaluates to `None`.
2. **Schema Stability**: No DDL table migrations (`ALTER TABLE`) or new columns are required.
3. **Roundtrip Integrity**: Test `test_metadata_roundtrip` verifies `save_metadata()` \(\rightarrow\) `get_metadata()` preserves all visual evidence attributes with zero data loss.
