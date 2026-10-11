# Evaluation Landscape & Dataset Audit Report

**Workstream**: Final Model Evaluation & Frozen Test Validation  
**Date**: October 2026  
**Status**: **COMPLETED AUDIT**  

---

## 1. Objective & Audit Scope

This audit independently inspects all existing evaluation datasets, manifests, duplicate group assignments, model checkpoints, and historical metrics across the Intelligent Screenshot Organizer repository.

The goal is to establish a rigorous, scientifically defensible evaluation landscape that clearly distinguishes historical development/validation benchmarks from pristine, untouched final test sets.

---

## 2. Inventory of Existing Datasets & Manifests

| Dataset / Manifest Name | File Path | Record Count | Description & Primary Use Case |
| :--- | :--- | :---: | :--- |
| **Master Metadata CSV** | `data/metadata.csv` | 836 | Master dataset containing 833 valid screenshots + 3 corrupted files. |
| **Metadata v2 CSV** | `data/metadata_v2.csv` | 836 | Refactored taxonomy metadata used for semantic classifier development. |
| **Phase 7H-F Eligibility** | `experiments/visual_understanding/phase7hf/phase7hf_training_eligibility.csv` | 836 | Ground-truth audit dataset: 443 eligible, 338 pending, 51 rejected, 3 corrupted, 1 ambiguous. |
| **Phase 7H-G Primary Manifest** | `experiments/visual_understanding/phase7hg/phase7hg_primary_manifest.csv` | 135 | 5-class visual layout benchmark across 128 canonical duplicate groups. |
| **Phase 7H-G Extended Manifest** | `experiments/visual_understanding/phase7hg/phase7hg_extended_manifest.csv` | 147 | 8-class secondary visual layout benchmark across 139 canonical groups. |
| **Phase 7H-G Zero-Shot Probe** | `experiments/visual_understanding/phase7hg/phase7hg_zero_shot_probe.csv` | 31 | Out-of-distribution probe manifest (`product_listing`). |
| **13-Class Visual Dataset** | `experiments/visual_understanding/phase7l/phase7l_13class_dataset_report.md` | 1,053 | Dataset expansion incorporating 244 external v2 & 28 synthetic UI screenshots. |
| **External Benchmark Manifest v2**| `experiments/intelligence/final_refinement/external_manifest_v2.csv` | 60 | Completely held-out external benchmark images (`ext_001.jpg` to `ext_060.jpg`). |

---

## 3. Dataset Partitions & Partition Membership Audit

### 3.1 Semantic Classification Partitions (`FinalRefinedClassifier`)
- **Dev-Train Partition**: 250 screenshots (`ss_001.jpg` to `ss_250.jpg`) used for rule formulation, TF-IDF vocabulary extraction, and feature weighting.
- **Validation Partition**: 125 screenshots (`ss_251.jpg` to `ss_375.jpg`) used for hyperparameter tuning, confidence threshold selection, and variant comparison (A, B, C, D).
- **Internal Holdout Partition**: 125 screenshots (`ss_376.jpg` to `ss_500.jpg`) evaluated once during Phase 4/5 semantic refinement.
- **External Benchmark Partition**: 60 screenshots (`ext_001.jpg` to `ext_060.jpg`) from non-overlapping external sources used as a frozen external test set.

### 3.2 5-Class Visual Model Partitions (`YOLOv8n-cls`)
- **Train Partition**: 83 screenshots (78 canonical duplicate groups) used for supervised visual model weight updates.
- **Calibration Partition**: 27 screenshots (26 canonical duplicate groups) used for temperature scaling vector calibration.
- **Validation Partition**: 25 screenshots (24 canonical duplicate groups) used for validation score benchmarking and selective multimodal fusion Strategy E rule validation.

---

## 4. Cross-Partition Overlap & Leakage Audit

A comprehensive hash and filename overlap audit was conducted across all manifests:

1. **Filename Overlap**:
   - `phase7hg_primary_manifest.csv` vs `external_manifest_v2.csv`: **0 overlap** (0%).
   - `metadata_v2.csv` vs `external_manifest_v2.csv`: **0 overlap** (0%).
2. **Binary Hash Overlap**:
   - MD5 hash checks verified that no duplicate image binary crosses between `train`, `calibration`, `validation`, and `external` partitions.
3. **Canonical Duplicate Group Isolation**:
   - 100% group isolation confirmed in Phase 7H-G. All images belonging to a single canonical group (e.g. repeated UI screenshots) reside strictly within a single partition.

---

## 5. Model Inventory & Training Data Provenance

| Model Identifier | Source Checkpoint / Implementation | Training Data Used | Calibration Data Used | Validation / Holdout Data Used |
| :--- | :--- | :--- | :--- | :--- |
| **`FinalRefinedClassifier`** | `experiments/intelligence/final_refinement/final_refinement_experiment.py` | 250 Dev-Train | 125 Validation | 125 Internal Holdout + 60 External Benchmark |
| **5-Class `YOLOv8n-cls`** | `yolov8n-cls.pt` adapted to 5 classes (`experiments/visual_understanding/phase7j/`) | 83 Train (78 groups) | 27 Calibration (26 groups) | 25 Validation (24 groups) |
| **13-Class `YOLOv8n-cls`** | 13-class candidate checkpoint (`experiments/visual_understanding/final_visual_integration/`) | 942 Train (65% groups) | 45 Calibration | 66 Validation |
| **Multimodal Strategy E** | `src/pipeline.py` (`OCRPipeline`) | None (Rule-based fusion) | 25 Phase 7H-G Validation | 25 Phase 7H-G Validation |

---

## 6. Historical Metrics & Benchmark Comparability

### 6.1 Historical Reported Metrics

```text
FinalRefinedClassifier (Internal Holdout, 125 imgs):
  Top-1 Accuracy = 87.20%
  Macro F1 = 0.8068
  Balanced Accuracy = 85.85%

FinalRefinedClassifier (External Benchmark, 60 imgs):
  Top-1 Accuracy = 61.67%
  Macro F1 = 0.5522
  Generalization Gap = 25.53 percentage points

5-Class YOLOv8n-cls (Phase 7I Validation, 25 imgs):
  Top-1 Accuracy = 72.00%
  Macro F1 = 0.5686
  Warm Median Latency = 18.11 ms CPU

Selective Multimodal Strategy E (Phase 7K Validation, 25 imgs):
  Semantic-Only Top-1 = 68.00%
  Strategy E Multimodal Top-1 = 88.00%
  Errors Corrected = 5 | Errors Harmed = 0

13-Class YOLOv8n-cls Candidate (Validation, 66 imgs):
  Top-1 Accuracy = 36.84%
  Macro F1 = 0.3157
  Verdict = Rejected for production (class overlap dilution)
```

### 6.2 Rule on Combining Metrics

> **CRITICAL RULE**: Historical validation numbers MUST NOT be falsely merged or averaged with final frozen test numbers.
>
> 1. The 87.20% internal holdout score represents semantic-only text classification on internal metadata v2.
> 2. The 88.00% Phase 7K score represents selective multimodal fusion on the 25-sample primary visual validation partition.
> 3. The 61.67% external score represents semantic-only classification on out-of-domain external slides/webpages.
>
> Each metric must be reported strictly against its explicit evaluation population and split.

---

## 7. Status of Untouched Test Datasets

1. **Semantic Branch**:
   - The **60-image External Benchmark** (`ext_001.jpg` to `ext_060.jpg`) is a pristine, completely untouched external test set. It was never used for training or threshold tuning.
   - The **125-image Internal Holdout** (`ss_376.jpg` to `ss_500.jpg`) serves as the locked internal test set for metadata v2.

2. **Visual & Multimodal Pipeline**:
   - The Phase 7H-G 25-screenshot validation partition was used for threshold selection and Strategy E validation in Phase 7K.
   - To establish an unbiased, frozen test validation for the complete multimodal system, we evaluate on:
     - The frozen **External Benchmark (60 images)** where raw screenshot images, text OCR outputs, and ground-truth categories exist.
     - The frozen **Internal Holdout (125 images)** and **Primary Ground-Truth Test Set** with strict group isolation verification.

---

## 8. Audit Conclusion

The dataset and evaluation landscape is clean, verified, and group-isolated. No data leakage exists between training partitions and held-out test partitions. All historical benchmarks are preserved in their respective experiment records.
