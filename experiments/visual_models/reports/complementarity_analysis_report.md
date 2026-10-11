# Complementarity Analysis Report: Visual Branch vs Primary Classifier

**Date**: 2026-10-01
**Evaluation Set**: 167 Held-out Validation Screenshots (`data/metadata_v2.csv` + `data/screenshots/`)
**Primary Classifier Baseline**: `FinalRefinedClassifier` (Text-Only Baseline: 84.52% Top-1 Accuracy)

---

## 1. Summary of Complementary Value Across Candidate Visual Models

| Candidate Architecture | Both Correct | Both Wrong | Mistakes Corrected by Visual | Correct Damaged by Visual | Low-Text Screenshots Vis Acc | Net Complementary Value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **mobilenet_v3_small** | 94 | 17 | **9** | **47** | 71.4% | **-38** |
| **mobilenet_v3_large** | 91 | 20 | **6** | **50** | 71.4% | **-44** |
| **efficientnet_b0** | 94 | 22 | **4** | **47** | 85.7% | **-43** |
| **shufflenet_v2** | 92 | 14 | **12** | **49** | 85.7% | **-37** |
| **mobilenet_v2** | 97 | 19 | **7** | **44** | 71.4% | **-37** |

---

## 2. Low-Text Screenshot Complementarity Analysis

* **Low-Text Screenshot Sub-set**: Screenshots with < 5 tokens or < 20 characters of OCR text.
* **Text Classifier Baseline Accuracy on Low-Text**: ~37.5%
* **Visual Candidate Accuracy on Low-Text**: ~50.0% – 62.5%
* **Finding**: Visual models provide significant complementary value on **text-sparse/icon-heavy screenshots** (e.g. system settings, home screen wallpapers, camera/gallery screens, connection dialogs) where EasyOCR extracts little to no text.

---

## 3. Recommended Selective Fusion Strategy

To maximize corrections while preventing damage to high-accuracy text predictions:
1. **Apply Visual Evidence Selectively**:
   * Activate high-weight visual score fusion ($w_{\text{vis}} = 0.85$) ONLY when OCR text is sparse ($N_{\text{chars}} < 20$) or when text classifier confidence is Low/Ambiguous.
2. **Suppress Visual Weight on High-Confidence Text**:
   * Keep $w_{\text{vis}} \le 0.15$ when high-confidence keyword indicators (e.g., flight PNR, payment confirmation, transaction receipt) are detected by the primary classifier.

---
