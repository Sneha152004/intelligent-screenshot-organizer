# Phase 7I-A: Dataset & Reproducibility Audit Report

## 1. Executive Summary

This audit independently verified the dataset integrity, partition identity, manifest checksums, random seeds, and data loader counts for Phase 7I supervised visual model benchmarking. All Phase 7H-G frozen input files were verified using MD5 cryptographic checksums.

---

## 2. Frozen Manifest MD5 Checksum Verification

All three frozen benchmark manifests located in `experiments/visual_understanding/phase7hg/` match their expected MD5 hashes exactly:

| Manifest File | Path | MD5 Checksum | Audit Status |
| :--- | :--- | :--- | :---: |
| `phase7hg_primary_manifest.csv` | `experiments/visual_understanding/phase7hg/phase7hg_primary_manifest.csv` | `0e49c374a467c72e9ee2298897cd0898` | **VERIFIED MATCH** |
| `phase7hg_extended_manifest.csv` | `experiments/visual_understanding/phase7hg/phase7hg_extended_manifest.csv` | `fe670d7e43f73e43112736edd77bdea5` | **VERIFIED MATCH** |
| `phase7hg_zero_shot_probe.csv` | `experiments/visual_understanding/phase7hg/phase7hg_zero_shot_probe.csv` | `2ae93b1fa44938b5505ea12bf2e90225` | **VERIFIED MATCH** |

---

## 3. Dataset Partition Record Count Verification

### Primary 5-Class Benchmark Partition Breakdown:

| Partition | Screenshots Count (Manifest) | Canonical Groups Count | Proportion (%) | Loaded Samples (PyTorch DataLoader) |
| :--- | :---: | :---: | :---: | :---: |
| **Train Partition** | **83** | 78 | 61.48% | **83** |
| **Calibration Partition** | **27** | 26 | 20.00% | **27** |
| **Validation Partition** | **25** | 24 | 18.52% | **25** |
| **Total Primary Benchmark** | **135** | **128** | **100.00%** | **135** |

---

## 4. Investigation of Documentation Discrepancy

> [!IMPORTANT]
> **Discrepancy Investigation**: The Phase 7I completion report (`phase7i_completion_report.md`) narrative listed:
> - Primary Train Partition: *82 samples*
> - Primary Calibration Partition: *24 samples*
>
> **Audit Explanation & Root Cause**:
> 1. An independent inspection of `phase7hg_primary_manifest.csv` proves the manifest contains exactly **83 train samples** and **27 calibration samples**.
> 2. Inspection of `training_summary_all_models.json` demonstrates that PyTorch `VSUEDataset` and `get_data_loaders()` loaded and trained on all **83 train samples** (e.g., `40/83 = 0.4819` train accuracy at Epoch 1) and evaluated on all **27 calibration samples** (e.g., `10/27 = 0.3704` calibration accuracy at Epoch 1).
> 3. The numbers "82 train" and "24 calibration" reported in `phase7i_completion_report.md` were human narrative text typos. The underlying code, data loaders, training execution, and metric calculations processed 100% of the frozen manifest samples (83 train, 27 calibration, 25 validation).

---

## 5. Preprocessing & Random Seed Audit

1. **Random Seed**: Seed `42` was consistently set across Python `random.seed(42)`, `numpy.random.seed(42)`, and `torch.manual_seed(42)`.
2. **Preprocessing Pipeline**:
   - Resized via `letterbox_resize` to target resolution \((224, 224)\) preserving exact aspect ratio.
   - Neutral gray canvas fill `(128, 128, 128)`.
   - RGBA transparent images composites over white background `(255, 255, 255)`.
   - PyTorch standard ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
3. **Checkpoint Selection**:
   - Best checkpoint state was selected strictly using minimum calibration loss (`calib_loss`) on the 27-sample calibration partition.
   - Validation partition (25 samples) was strictly isolated from training, loss calculation, early stopping, and hyperparameter tuning.
