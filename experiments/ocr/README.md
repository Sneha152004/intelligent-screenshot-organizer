# Phase 3B: Real OCR Evaluation Experiment

## Overview
This directory contains the experimental scaffolding used to evaluate real Optical Character Recognition (OCR) engines against the manually verified reference OCR text (`MockOCREngine` reference text) established during Phase 1 & Phase 2.

> [!NOTE]
> **Production Isolation:** This experiment operates entirely in isolation. It does NOT modify or replace `src/mock_ocr.py`, `src/pipeline.py`, or any production pipeline components.

---

## OCR Engine Trade-Off Analysis & Selection Rationale

| OCR Candidate | Windows Installation Complexity | Python Compatibility | Screenshot / UI Performance | Local Execution | Selection Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tesseract (`pytesseract`)** | **High:** Requires installing external C++ `tesseract.exe` binary on Windows and configuring OS PATH variables. | Good (`pip install pytesseract`). | Moderate on standard text; poor on small UI fonts, dark mode, and non-standard digital screenshot layouts. | Yes | **Rejected for Prototype:** High setup barrier for cross-platform developers; optimized for scanned 300 DPI documents rather than UI screenshots. |
| **EasyOCR (`easyocr`)** | **Zero:** Pure Python / PyTorch based (`pip install easyocr`). No C++ binary installations required. | Excellent (Python 3.9–3.11). | **High:** Uses deep learning detection (CRAFT) and recognition (ResNet+LSTM), handling varied UI fonts, multi-line blocks, and dark mode screens effectively. | Yes (Downloads ~45MB models on first run). | **RECOMMENDED & SELECTED:** Cleanest integration for Windows/CPU environments with zero OS-level binary dependencies. |
| **PaddleOCR (`paddleocr`)** | **High:** Requires `paddlepaddle`, C++ build tools, and specific protobuf/numpy version constraints. | Moderate (Can conflict with PyTorch/sentence-transformers). | Very High (SOTA recognition accuracy). | Yes | **Deferred:** Excellent performance but introduces heavy environment dependency conflicts on Windows. |

---

## Evaluation Methodology & Metrics

1. **Evaluated Subset:** 8 representative screenshots selected across 8 distinct functional categories:
   - `ss_001.jpg` (Connectivity - Airtel Wi-Fi password)
   - `ss_002.jpg` (Payment - UPI receipt ₹689)
   - `ss_003.jpg` (Education - KIIT Deep Learning curriculum)
   - `ss_008.jpg` (Shopping - MD Combo food menu prices)
   - `ss_0011.jpg` (Travel - IndiGo flight boarding pass)
   - `ss_0021.jpg` (Communication - WhatsApp/Telegram notifications)
   - `ss_0028.jpg` (Coding - VS Code Express.js controller code)
   - `ss_0030.jpg` (Work & Career - OpenAI AI Hackathon announcement)

2. **Metrics:**
   - **Character Error Rate (CER):** Normalized Levenshtein edit distance at the character level.
   - **Character Similarity:** $\max(0.0, 1.0 - \text{CER})$.
   - **Word Error Rate (WER):** Levenshtein edit distance at tokenized word level.
   - **Word Similarity:** $\max(0.0, 1.0 - \min(1.0, \text{WER}))$.
   - **Key Term Retention:** Verification of critical passwords, payment amounts, seat numbers, and codes.

---

## How to Run the Experiment

```bash
python experiments/ocr/test_ocr.py
```

Outputs will be saved automatically to:
- `experiments/ocr/results/raw_ocr_outputs.json`
- `experiments/ocr/results/evaluation_report.md`
