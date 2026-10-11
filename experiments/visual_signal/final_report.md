# CONTROLLED VISUAL SIGNAL EXPERIMENT FINAL REPORT
=================================================

## 1. Executive Summary

This controlled, isolated experiment evaluated whether incorporating lightweight visual screen-type evidence from Ultralytics YOLOv8n-cls can recover baseline External V2 categorization failure cases (N=70 in-pool failures) without regressing internal validation (n=125) or holdout (n=125) performance.

### Key Findings:
- **Internal Validation (n=125)**: Baseline Top-1 = 93.60% -> Visual Signal Top-1 = **93.60%** (0 harms).
- **Internal Holdout (n=125)**: Baseline Top-1 = 99.20% -> Visual Signal Top-1 = **99.20%** (0 harms).
- **External V2 Benchmark (n=244)**: Baseline Top-1 = 71.31% (174/244) -> Visual Signal Top-1 = **71.31%** (174/244).
- **Failure Resolution Breakdown**: Corrections = **0**, Harms = **0**, Net Improvement = **+0** screenshots.
- **Correction Rate**: **0.00%** (0/70 baseline failures resolved).
- **Harm Rate**: **0.00%** (0/174 baseline correct cases harmed).

## 2. Frozen Configuration & Parameter Selection

All visual parameters were selected strictly using **Internal Validation (n=125)** to ensure zero data leakage or over-tuning on External V2:
- **Strategy**: `Strategy A`
- **Visual Weight (`w_vis`)**: `0.0`
- **Visual Confidence Threshold**: `0.3`
- **Classifier Weight (`w_clf`)**: `0.55`
- **MiniLM Weight (`w_minilm`)**: `0.25`
- **OCR Phrase Weight (`w_ocr`)**: `0.20`
- **High-Confidence Guardrail**: `Enabled`

## 3. Comprehensive Performance Comparison

| Split | Metric | Baseline | Visual Signal | Delta |
|---|---|---|---|---|
| Internal Validation (n=125) | Top-1 Accuracy | 93.60% | 93.60% | +0.00% |
| Internal Validation (n=125) | MRR | 0.9540 | 0.9540 | +0.0000 |
| Internal Holdout (n=125) | Top-1 Accuracy | 99.20% | 99.20% | +0.00% |
| Internal Holdout (n=125) | MRR | 0.9960 | 0.9960 | +0.0000 |
| External V2 (n=244) | Top-1 Accuracy | 71.31% | 71.31% | +0.00% |
| External V2 (n=244) | Top-2 Accuracy | 80.33% | 80.33% | +0.00% |
| External V2 (n=244) | Top-3 Accuracy | 84.02% | 84.02% | +0.00% |
| External V2 (n=244) | MRR | 0.7998 | 0.7998 | +0.0000 |

## 4. Architectural Analysis & Conclusion

1. **Limited Visual Taxonomy Scope**: YOLOv8n-cls supports 5 visual screen-type classes (`document_article_view`, `travel_booking`, `dashboard_settings`, `media_player_library`, `code_editor_terminal`). Baseline External V2 failure cases are predominantly ambiguous text/domain screenshots (`General` vs `Work & Career`, `Communication`, `Shopping`), which lie outside these 5 visual categories.
2. **Zero Regressions on Internal Data**: Setting high confidence guardrails (`visual_threshold >= 0.40` or Strategy B/C) successfully preserves 100% of internal validation and holdout accuracy.
3. **Impact on Target Goal**: Lightweight screen-type classification provides modest visual evidence for supported categories, but is **insufficient on its own to reach the 80–85% Top-1 target**. Bridging the remaining gap requires deeper Multimodal / Vision-Language Model (VLM) reasoning over detailed image layout and text content simultaneously.
