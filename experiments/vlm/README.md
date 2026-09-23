# Phase 3E: Model-Agnostic VLM Evaluation Harness

## Overview & Purpose

Phase 3E introduces a **Model-Agnostic Evaluation Harness** for benchmarking Visual Language Model (VLM) candidates for screenshot understanding.

> [!NOTE]
> **Prototype Evaluation Layer:** This directory provides an isolated benchmarking harness. It does **NOT** install real model weights, modify production pipelines, alter SQLite/ChromaDB storage, or depend on external cloud APIs. Candidate VLMs are evaluated via swappable adapter interfaces prior to production selection.

---

## Evaluation Benchmark Dataset

The harness evaluates VLM candidates across **8 representative screenshots** spanning distinct functional domains:

1. `ss_001.jpg` (Connectivity - Airtel Wi-Fi password dialog)
2. `ss_002.jpg` (Payment - UPI transaction receipt ₹689)
3. `ss_003.jpg` (Education - KIIT Deep Learning course syllabus)
4. `ss_008.jpg` (Shopping - MD Combo food menu prices)
5. `ss_0011.jpg` (Travel - IndiGo flight boarding pass)
6. `ss_0021.jpg` (Communication - WhatsApp/Telegram chat notifications)
7. `ss_0028.jpg` (Coding - VS Code Express.js controller code)
8. `ss_0030.jpg` (Work & Career - OpenAI AI Hackathon announcement)

> [!IMPORTANT]
> **Reference Annotation Notice:** The target annotations used by the evaluation harness are engineering reference annotations established during Phase 3D (`MockVLM` default outputs), serving as an initial benchmark target rather than independently human-validated ground truth.

---

## Evaluation Architecture

The harness separates model inference from evaluation logic using a swappable adapter pattern:

```text
               +-----------------------------+
               |      VLMEvaluator           |
               +--------------+--------------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
   +--------------------+          +--------------------+
   |  BaseVLMAdapter    |          |   BaseOCREngine    |
   +---------+----------+          +---------+----------+
             |                               |
             v                               v
    +-----------------+             +-----------------+
    | MockVLMAdapter  |             |  MockOCREngine  |
    | (or Real VLM)   |             |  (or EasyOCR)   |
    +-----------------+             +-----------------+
```

### Centralized Standardized System Prompt
All candidate VLMs receive a unified, model-agnostic system prompt instructing them to parse the screenshot image together with supporting OCR text, categorize the domain, generate tags, infer user intent, estimate importance (1–5), extract entities/dates/action items, detect sensitive information, and emit strictly compliant JSON without conversational wrapping.

---

## Evaluation Metrics

Rather than compressing model performance into a single arbitrary score, the framework evaluates independent metric dimensions:

1. **Schema Validity Rate:** Percentage of outputs adhering to JSON structure, required keys, correct data types, and valid importance boundaries (1–5).
2. **Category Accuracy:** Exact match rate for broad functional categories.
3. **Tag Precision & Recall:** Case-insensitive overlap between predicted and expected concept tags.
4. **Intent Exact Match:** Normalized match rate for inferred user intent.
5. **Entities Recall:** Overlap rate of extracted named entities.
6. **Dates Recall:** Overlap rate of extracted temporal dates.
7. **Action Items Recall:** Overlap rate of extracted action items.
8. **Sensitive Info Accuracy:** Multi-label boolean flag accuracy (`contains_password`, `contains_payment_info`, `contains_personal_contact`).

---

## Performance & Deployment Measurements

The framework measures:
- **Per-Image Latency:** Single-screenshot inference execution time (seconds).
- **Total Evaluation Time:** Wall-clock time across the benchmark suite.
- **Model Memory Footprint (MB):** Parameter/RAM footprint tracking for mobile and desktop deployment constraints.
- **CPU Compatibility:** The evaluation framework operates on CPU without requiring GPU acceleration.

---

## How to Run the Evaluation (Mock Demo Mode)

Run the evaluation harness using the built-in `MockVLMAdapter` (zero model download required):

```bash
python experiments/vlm/evaluate_vlm.py --engine mock
```

### CLI Arguments

- `--engine`: Adapter name (`mock` for demo mode). Default: `mock`.
- `--output`: Destination directory for output files. Default: `experiments/vlm/results`.
- `--limit`: Optional integer to limit evaluation to top N screenshots.

### Output Artifacts

Execution generates two result artifacts in `experiments/vlm/results/`:
1. `vlm_evaluation_results.json`: Machine-readable JSON containing per-screenshot results, latency, schema validation logs, and aggregate metrics.
2. `vlm_evaluation_report.md`: Human-readable Markdown summary report.

---

## Future Real VLM Adapter Integration

To benchmark a real local VLM candidate (e.g. `SmolVLM2`, `Qwen2.5-VL`, `LLaVA-Phi3`), implement a custom subclass of `BaseVLMAdapter` in `evaluate_vlm.py`:

```python
class RealVLMAdapter(BaseVLMAdapter):
    def __init__(self, model_path: str):
        # Load local model & processor
        ...

    @property
    def model_name(self) -> str:
        return "SmolVLM2-500M-Instruct"

    @property
    def model_footprint_mb(self) -> float:
        return 1000.0  # ~1GB model RAM footprint

    def analyze(self, image_path: str, ocr_text: str = "") -> dict:
        # Run inference using STANDARDIZED_VLM_PROMPT and return parsed JSON dict
        ...
```
