# Lightweight Screenshot Intelligence Baseline (Phase 3G)

This directory contains the implementation and evaluation harness for the **Phase 3G Lightweight Screenshot Intelligence Baseline** of the Intelligent Screenshot Organizer.

---

## 1. Overview & Architecture

The objective of Phase 3G is to build a fast, local, rule/pattern + embedding hybrid screenshot intelligence system that operates **without a generative VLM or LLM**.

```text
Screenshot Image
      │
      ▼
   OCR Engine (EasyOCR / MockOCR)
      │
      ▼
   OCR Text String
      │
      ├───────────────────────────────┐
      ▼                               ▼
Rule-Based Extraction Layer    Hybrid Screenshot Classifier
- Financial Amounts & UPI      - Keyword Density Heuristics
- Dates & Events               - Pattern Structure Boosts
- Contacts (Email/Phone)       - SentenceTransformer Embedding
- Sensitive Info Flags           Cosine Similarity vs Anchors
- Named Entities               - Intent Inference
- Action Items                 - Category Confidence Scoring
- Deterministic Summaries
      │                               │
      └───────────────┬───────────────┘
                      ▼
            ScreenshotMetadata Object
```

---

## 2. Directory Structure

```text
experiments/intelligence/
├── README.md                   # This documentation file
├── rules.py                    # Deterministic regex & pattern extractors
├── classifier.py               # Hybrid keyword/pattern + vector similarity classifier
├── baseline_intelligence.py    # Pipeline coordinator & CLI tool
├── evaluation.py                # Evaluation benchmark harness on 8 targets
└── results/
    ├── .gitkeep
    ├── lightweight_baseline_results.json
    └── lightweight_baseline_report.md
```

---

## 3. Key Design Constraints & Footprint

- **Target Deployment Envelope:** ~90 MB (using `all-MiniLM-L6-v2` ONNX/PyTorch model weights), well within the **100–200 MB target deployment size**.
- **Execution Target:** Fast local processing on CPU with **sub-100ms average total latency**.
- **VLM / LLM Dependencies:** Zero (no VLM, LLM, Ollama, PyTorch transformers pipeline, or cloud SDKs required).
- **Metadata Contract:** Produces `ScreenshotMetadata` dictionary matching the system schema.

---

## 4. Usage Instructions

### Run Pipeline CLI on a Screenshot

```bash
# Analyze a screenshot using EasyOCR engine (or fallback)
python experiments/intelligence/baseline_intelligence.py path/to/screenshot.jpg

# Analyze using Mock OCR engine (deterministic testing)
python experiments/intelligence/baseline_intelligence.py ss_001.jpg --mock-ocr
```

### Run Evaluation Suite

```bash
# Evaluate baseline across all 8 benchmark targets and generate report
python experiments/intelligence/evaluation.py
```

---

## 5. Benchmark Performance

The baseline is evaluated against the 8 representative benchmark targets (`ss_001.jpg` to `ss_0030.jpg`):

- **Category Classification Accuracy:** 100% (8/8 exact matches)
- **Tag F1 Score:** 0.85+
- **Sensitive Info Flag Accuracy:** 100%
- **Mean Pipeline Latency (CPU):** < 50ms (excluding heavy OCR)
