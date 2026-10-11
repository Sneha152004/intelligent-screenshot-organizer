# Phase 7F: First Supervised Visual Screen-Type Model Baseline

**Date**: 2026-10-02  
**Architecture**: MobileNetV3-Small (Pretrained ImageNet1K Weights + 5-Class Linear Head)  
**Execution Device**: Intel/AMD CPU (Single Thread Execution)  
**Verdict**: **SUCCESSFUL BASELINE**

---

## 1. Setup & Reproduction Guide

To run training, evaluation, and unit tests for Phase 7F:

```bash
# 1. Run isolated training pipeline (fits best epoch on Calibration set)
python experiments/visual_understanding/phase7f/train_vsue_baseline.py

# 2. Run independent evaluation on held-out Validation set
python experiments/visual_understanding/phase7f/evaluate_vsue_baseline.py

# 3. Run unit test suite
python -m unittest experiments/visual_understanding/phase7f/test_phase7f.py
```

---

## 2. Directory Artifacts Guide

* `train_vsue_baseline.py`: PyTorch training script with inverse class weighting and calibration-based early stopping.
* `evaluate_vsue_baseline.py`: Independent evaluation script computing Top-1/2/3 Acc, Macro F1, confusion matrix, CPU latency, and INT8 quantization metrics.
* `dataset_utils.py`: Letterbox padding, image loading, dataset loader, and inverse class frequency weight calculations.
* `vsue_model.py`: MobileNetV3-Small model architecture and VSUE JSON contract payload formatter.
* `test_phase7f.py`: Unit test suite covering image processing, preprocessing, class mapping, forward pass, checkpoint save/load, and CPU inference.
* `visual_screen_type_model.pth`: Saved best model PyTorch checkpoint selected via calibration Macro F1.
* `visual_screen_type_model_int8.pth`: Dynamic INT8 quantized model weights (1.05 MB).
* `model_config.json`: Hyperparameters, architecture settings, input dimensions, and normalization values.
* `validation_predictions.csv`: Per-image predictions and payloads on the 169 held-out validation screenshots.
* `per_class_metrics.csv`: Per-class precision, recall, F1-score, and support metrics.
* `confusion_matrix.csv`: 5x5 confusion matrix.
* `phase7f_results.md`: Comprehensive evaluation results report.
* `requirements_phase7f.txt`: Python package requirements.

---

## 3. Strict Scope Protection Verification

* `FinalRefinedClassifier`: **UNTOUCHED**
* EasyOCR & MiniLM: **UNTOUCHED**
* Production visual encoders & rules: **UNTOUCHED**
* External V1 & External V2 benchmarks: **UNTOUCHED**
* Phase 7E manifest & splits: **UNTOUCHED**
* Zero Git mutation executed.
