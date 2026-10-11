# Visual Screen Understanding Engine (VSUE) - Integration Design

**Date**: 2026-10-02  
**Target Architecture**: Decoupled Visual Evidence Branch for `FinalRefinedClassifier`.

---

## 1. Proposed Visual Output Contract

The Visual Screen Understanding Engine (VSUE) emits a structured visual evidence payload:

```json
{
  "screen_type": "product_listing",
  "screen_type_confidence": 0.87,
  "secondary_screen_types": ["product_detail"],
  "visual_elements": [
    "product_grid_cards",
    "price_tags",
    "item_thumbnails",
    "filter_bar"
  ],
  "layout_features": {
    "orientation": "portrait",
    "has_repeated_cards": true,
    "has_navigation_bar": true,
    "has_dialog": false,
    "has_form_fields": false,
    "edge_density": 0.142
  },
  "visual_evidence": [
    "Grid layout detected with 6 repeating product card contours",
    "Prominent bottom navigation bar observed"
  ],
  "model_version": "vsue_experimental_v1"
}
```

---

## 2. Decoupled Evidence Fusion Architecture

```
                  Screenshot Image
                         |
        +----------------+----------------+
        |                                 |
        v                                 v
   [EasyOCR]                     [VSUE Visual Branch]
        |                                 |
 [Normalized Text]              [Visual Evidence Object]
        |                                 |
  [MiniLM + Rules]                        |
        |                                 |
 [Text Prediction & Margin]               |
        |                                 |
        +----------------+----------------+
                         |
                         v
              [Evidence Fusion Layer]
                         |
                         v
        Final Structured Screenshot Metadata
```

---

## 3. Non-Overriding Integrity Guarantee

1. **Text Primary Principle**: `FinalRefinedClassifier` remains the authoritative primary classifier.
2. **Selective Activation**: VSUE evidence is consulted **only** when text OCR is sparse (<20 chars) or text prediction confidence margin is ambiguous (<0.10).
3. **Inspectable Payload**: The VSUE output contract is logged alongside text predictions for full explainability.
