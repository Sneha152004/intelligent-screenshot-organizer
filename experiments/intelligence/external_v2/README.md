# External V2 Benchmark

## 1. Overview & Purpose
The **External V2 Benchmark** is an independent evaluation benchmark for the Intelligent Screenshot Organizer. It contains **244 real-world screenshot images** (`ext2_001.jpg` through `ext2_244.jpg`) reserved exclusively for evaluating generalizability and out-of-distribution performance.

- **Dataset Size**: 244 screenshots (`ext2_001.jpg` – `ext2_244.jpg`)
- **Manifest Location**: `experiments/intelligence/external_v2/manifest.csv`
- **Evaluation Targets Module**: `experiments/intelligence/external_v2/evaluation_targets.py`
- **Audit Report**: `experiments/intelligence/external_v2/label_audit_report.md`
- **Prediction Audit**: `experiments/intelligence/external_v2/v2_prediction_audit.csv`
- **Status**: **`READY TO FREEZE`**

---

## 2. Empirical Performance Metrics

- **Top-1 Accuracy**: **66.39%** (162 / 244)
- **Top-2 Recall**: **77.87%** (190 / 244)
- **Top-3 Recall**: **81.97%** (200 / 244)
- **Macro F1-Score**: **0.6082**
- **Balanced Accuracy**: **0.7054**
- **Mean Reciprocal Rank (MRR)**: **0.7350**

---

## 3. Canonical Category Taxonomy (9 Categories)

The top-level category taxonomy consists of exactly **9 canonical categories**:

1. **Education**: Course lectures, textbooks, tutorial slides, academic reading material.
2. **Shopping**: E-commerce product listings, cart checkouts, order confirmations.
3. **Payment**: Financial transactions, UPI transfers, bank account statements, payment receipts.
4. **Travel**: Flight bookings, navigation maps, hotel reservations, train tickets.
5. **Communication**: Chat apps, messaging, email threads, social messaging.
6. **Entertainment**: Video streaming, music players, social media feeds, gaming screens.
7. **Work & Career**: IDEs, professional developer tools, resume building, recruitment, corporate slides.
8. **Connectivity**: Wi-Fi settings, network speed tests, Bluetooth devices, system settings.
9. **General**: Generic, unsupported, or ambiguous screens lacking strong domain evidence.

---

## 4. Category vs. Tag Distinction

> [!WARNING]
> **Coding is NOT a category.** Coding is represented strictly as a tag/content attribute.

### Category and Tag Examples:
- **Python Tutorial Slide / Lecture**:
  - **Category**: `Education`
  - **Tags**: `Coding`, `Python`, `Programming`
- **Professional VS Code IDE Development**:
  - **Category**: `Work & Career`
  - **Tags**: `Coding`, `VS Code`, `Programming`, `Software Development`
- **Wi-Fi Settings Screen**:
  - **Category**: `Connectivity`
  - **Tags**: `Wi-Fi`, `Network`, `Settings`
- **Generic / Unsupported Screen**:
  - **Category**: `General`
  - **Tags**: *(empty or UI tags)*

---

## 5. Benchmark Governance Rules

To preserve scientific rigor and prevent data leakage:

1. **Never use External V2 for training**: Do not include V2 samples in training sets or cross-validation splits.
2. **Never use External V2 for rule discovery**: Do not inspect V2 images to design keyword heuristics or prompt templates.
3. **Never use External V2 for threshold tuning**: Do not optimize classification margins or similarity thresholds against V2.
4. **Never use External V2 for model selection**: Do not select model architectures or feature extractors based on V2 scores.
5. **Never modify labels based on classifier predictions**: Ground-truth labels must be assigned independently without looking at classifier predictions.
6. **Freeze labels before evaluation**: The manifest must be fully reviewed and locked before any evaluation run.
7. **One-Time Evaluation Constraint**: Once evaluated, classifier code must NOT be tuned using V2 failure cases.

---

## 6. Historical V1 Immutability Notice

The historical External V1 benchmark located at `experiments/intelligence/external/` is a separate historical benchmark evaluated during early project phases. It remains **completely untouched**.
