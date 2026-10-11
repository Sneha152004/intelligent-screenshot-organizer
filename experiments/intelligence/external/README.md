# External Screenshot Evaluation Dataset

This folder contains the **external / unseen evaluation dataset** for the
Intelligent Screenshot Organizer's lightweight screenshot-intelligence
pipeline.

The purpose of this dataset is to measure **generalization to screenshots
that were not used during development, rule design, prototype construction,
threshold tuning, or the original benchmark evaluation**.

> **Important:** The reference annotations in this folder are evaluation
> targets. They must be treated as frozen ground-truth-style annotations for
> the external evaluation and should not be modified after evaluation begins.

---

## 1. Dataset Contents

This external dataset contains **60 screenshots**.

Expected folder structure:

```text
experiments/
└── intelligence/
    └── external/
        ├── README.md
        ├── manifest.csv
        ├── evaluation_targets.py
        └── images/
            ├── ext_001.jpg
            ├── ext_002.jpg
            ├── ...
            └── ext_060.jpg
```

### Files

| File | Purpose |
|---|---|
| `README.md` | Documents the external evaluation dataset and evaluation protocol |
| `manifest.csv` | Tabular reference manifest containing annotations for every screenshot |
| `evaluation_targets.py` | Python representation of the frozen reference annotations used by evaluation code |
| `images/` | Contains the 60 external screenshots |

---

## 2. Naming Convention

External screenshots use the following naming convention:

```text
ext_001.jpg
ext_002.jpg
...
ext_060.jpg
```

The `ext_` prefix deliberately distinguishes these images from the
development/sample screenshot collection.

The filename is the primary identifier used to associate an image with its
reference annotation.

---

## 3. Annotation Schema

Each screenshot has a reference annotation containing:

```text
filename
category
tags
intent
summary
importance
entities
dates
action_items
sensitive_info
```

### Category

The functional category assigned to the screenshot.

The categories represented in this dataset are:

- **Coding**: 1
- **Communication**: 2
- **Education**: 22
- **Entertainment**: 8
- **General**: 2
- **Payment**: 3
- **Shopping**: 11
- **Travel**: 5
- **Work & Career**: 6

### Tags

A list of concise semantic labels describing important concepts present in
the screenshot.

### Intent

The inferred user-oriented purpose of retaining or using the screenshot.

### Summary

A short textual description of the screenshot's meaningful content.

### Importance

An integer importance value where applicable.

### Entities

Structured entities detected in the screenshot, represented with text and
entity type.

### Dates

Dates or date-related information visible in the screenshot.

### Action Items

Tasks or actions that can reasonably be extracted from the screenshot.

### Sensitive Information

Structured flags describing whether sensitive information such as passwords,
payment information, or personal contact information is present.

---

## 4. `manifest.csv`

`manifest.csv` is the tabular source of the reference annotations.

It can be inspected with:

```powershell
Get-Content experiments\intelligence\external\manifest.csv
```

or loaded programmatically:

```python
import pandas as pd

df = pd.read_csv(
    "experiments/intelligence/external/manifest.csv"
)

print(df.head())
print(df["category"].value_counts())
```

JSON-valued fields such as `entities`, `dates`, `action_items`, and
`sensitive_info` are stored in serialized form inside the CSV.

---

## 5. `evaluation_targets.py`

`evaluation_targets.py` provides the same reference annotations in a
Python-friendly format.

Typical usage:

```python
from experiments.intelligence.external.evaluation_targets import (
    get_evaluation_targets,
    get_target_by_filename,
)

targets = get_evaluation_targets()

target = get_target_by_filename("ext_001.jpg")
```

The module returns defensive copies so that evaluation code does not
accidentally mutate the stored reference annotations.

The expected filenames are:

```text
ext_001.jpg ... ext_060.jpg
```

---

## 6. External Evaluation Principle

This dataset exists specifically to answer a different question from the
development benchmark.

### Development benchmark

The original benchmark was used while developing and validating the
lightweight intelligence pipeline.

It helped establish:

- rule patterns
- category prototypes
- classifier behavior
- tag extraction behavior
- intent inference
- sensitive-information detection
- latency characteristics

### External evaluation

This dataset is intended to measure how the already-developed system behaves
on new screenshots.

The external evaluation should therefore follow this rule:

```text
DO NOT modify the intelligence system
after inspecting external evaluation results
and then report those same results as the
original external evaluation.
```

If rules, thresholds, category anchors, or other classifier logic are changed
after seeing the external results, the evaluation should be considered a new
experiment and rerun on a fresh held-out dataset.

---

## 7. Recommended Evaluation Flow

The intended evaluation flow is:

```text
                  External Images
                        │
                        ▼
                Local OCR Pipeline
                        │
                        ▼
              Rule / Pattern Extraction
                        │
                        ▼
             Hybrid Classification
                        │
                        ▼
               Intent Inference
                        │
                        ▼
             Structured Metadata
                        │
                        ▼
              ┌─────────────────┐
              │ Compare against │
              │ frozen targets  │
              └────────┬────────┘
                       │
                       ▼
              External Evaluation
                    Metrics
```

The evaluation should use the same production-like pipeline and metric
definitions used for the preceding intelligence evaluations wherever
possible.

This keeps comparisons meaningful and prevents changing the evaluation
protocol simply because the external dataset produced different results.

---

## 8. Metrics to Record

The external evaluation should report at least:

### Classification

- Overall category accuracy
- Per-category accuracy
- Confusion matrix, if available

### Tags

- Precision
- Recall
- F1 score

### Intent

- Intent accuracy or another explicitly defined matching metric

### Structured Extraction

- Entity precision / recall
- Date precision / recall
- Action-item precision / recall

### Sensitive Information

Per-flag metrics for:

- `contains_password`
- `contains_payment_info`
- `contains_personal_contact`

### Reliability

- Schema validity rate
- Number of failed predictions
- Failure categories

### Performance

- OCR latency
- Rule extraction latency
- Embedding latency
- Classification latency
- Total warm latency
- Cold-start latency, when relevant
- Memory / storage footprint

---

## 9. Data Leakage Precautions

The external dataset should remain separate from development data.

Do **not**:

- copy external images into the development benchmark
- use their annotations to create new classifier anchors
- tune thresholds specifically to these screenshots
- add their exact text patterns solely because they appear in this dataset
- modify expected labels to match model predictions
- use external results to claim performance of an already-retuned model

The purpose is to preserve a meaningful distinction between:

```text
Development data
       ↓
System design / tuning
       ↓
Frozen system
       ↓
External evaluation data
       ↓
Generalization measurement
```

---

## 10. Adding Future External Datasets

If another independent evaluation collection is created, it should preferably
use a separate directory rather than overwriting this dataset.

For example:

```text
experiments/
└── intelligence/
    ├── external/
    │   ├── images/
    │   ├── manifest.csv
    │   └── evaluation_targets.py
    │
    └── external_v2/
        ├── images/
        ├── manifest.csv
        └── evaluation_targets.py
```

This preserves experiment history and prevents accidental mixing of
evaluation sets.

---

## 11. Dataset Status

Current external dataset:

- **Images:** 60
- **Manifest entries:** 60
- **Reference targets:** 60
- **Filename range:** `ext_001.jpg` → `ext_060.jpg`
- **Purpose:** Generalization / external evaluation
- **Development tuning:** Must not be performed against this dataset

Before running the final experiment, verify that the image directory contains
all 60 expected files and that every image has exactly one corresponding
manifest/reference-target entry.

---

## 12. Verification Checklist

Before evaluation:

- [ ] `images/` contains 60 screenshots.
- [ ] Files are named `ext_001.jpg` through `ext_060.jpg`.
- [ ] `manifest.csv` contains 60 entries.
- [ ] `evaluation_targets.py` contains 60 targets.
- [ ] Every image has exactly one reference annotation.
- [ ] No external image was part of the development benchmark.
- [ ] No external annotation was used to modify the classifier before the
      evaluation.
- [ ] Evaluation configuration is frozen before running the experiment.
- [ ] Evaluation results are saved separately from development results.

---

## 13. Reproducibility

The external dataset should be considered a versioned evaluation artifact.

When results are reported, record:

- dataset version or commit
- model / classifier configuration
- OCR configuration
- embedding model
- evaluation script version
- Python/runtime environment
- CPU/GPU environment
- test count
- evaluation timestamp

This makes it possible to distinguish changes in the intelligence system from
changes in the evaluation dataset or runtime environment.

---

## 14. Phase 4A.1 Manual Dataset Quality Audit Protocol

During Phase 4A.1, a **ground-truth quality audit infrastructure** was established to audit the 60 external screenshots and their reference annotations prior to retrieval development.

> **Important Principles:**
> 1. **Frozen Dataset**: The 60 images and ground-truth reference annotations are frozen evaluation targets.
> 2. **Manual Review Separation**: Manual ground-truth review is strictly a **DATASET QUALITY AUDIT** to verify consistency, correct categories, content-types, tags, and intents. It is entirely separate from model accuracy evaluation.
> 3. **Human Review Required**: Any annotation corrections must be explicitly confirmed by a human reviewer. No automated script may alter reference annotations.
> 4. **No Model Tuning**: Results from this manual audit MUST NOT be used to tune rules, classifier thresholds, or prototype anchors.

### Audit & Review Infrastructure Tools

| Artifact / Tool | Path | Purpose |
|---|---|---|
| Audit & Verification Tool | `review_external.py` | Automated dataset integrity check, SHA-256 overlap check, taxonomy validation, and HTML/CSV generator |
| Offline HTML Reviewer | `manual_review.html` | Self-contained, offline visual review interface displaying screenshots alongside ground-truth metadata |
| CSV Manual Checklist | `manual_review_checklist.csv` | Human reviewer audit matrix with 14 criteria per image initialized to `UNREVIEWED` |

### Running the Audit Tool

To re-run dataset integrity checks and re-generate the review HTML/CSV:

```bash
python experiments/intelligence/external/review_external.py
```

