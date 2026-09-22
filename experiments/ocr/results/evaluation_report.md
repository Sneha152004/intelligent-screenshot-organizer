# Phase 3B: Real OCR Evaluation Report

## Experiment Overview
Evaluated **EasyOCR** against manually verified reference OCR text across 8 representative screenshot domains.

## Quantitative Metrics Summary

| Screenshot | Category | Char Sim | CER | Word Sim | WER | Key Terms | Observations |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `ss_001.jpg` | Connectivity | 0.24 | 0.76 | 0.00 | 2.00 | 2/2 | High CER; OCR output differs substantially from the reference text. |
| `ss_002.jpg` | Payment | 0.75 | 0.25 | 0.64 | 0.36 | 2/2 | Good text extraction alignment |
| `ss_003.jpg` | Education | 0.53 | 0.47 | 0.45 | 0.55 | 2/2 | High CER; OCR output differs substantially from the reference text. |
| `ss_008.jpg` | Shopping | 0.16 | 0.84 | 0.03 | 0.97 | 2/2 | High CER; OCR output differs substantially from the reference text. |
| `ss_0011.jpg` | Travel | 0.52 | 0.48 | 0.53 | 0.47 | 2/2 | High CER; OCR output differs substantially from the reference text. |
| `ss_0021.jpg` | Communication | 0.70 | 0.30 | 0.77 | 0.23 | 2/2 | Good text extraction alignment |
| `ss_0028.jpg` | Coding | 0.04 | 0.96 | 0.00 | 1.43 | 1/2 | Missing key terms: ['Express']; High CER; OCR output differs substantially from the reference text. |
| `ss_0030.jpg` | Work & Career | 0.87 | 0.13 | 0.82 | 0.18 | 2/2 | Good text extraction alignment |

## Aggregate Observations
- **Average Character Similarity:** 0.4745 (Average CER: 0.5255)
- **Average Word Similarity:** 0.4047 (Average WER: 0.7741)
- **Strengths:** EasyOCR runs locally on CPU without external binary dependencies and successfully retains selected key terms across most evaluated screenshots.
- **Weaknesses:** Sensitive to special character punctuation (e.g., passwords like `Air@53054`) and multi-line UI layouts.
