# Lightweight Visual Classifier Research & Benchmarking Suite

This directory contains the experimental research, training, benchmarking, and complementarity evaluation suite for lightweight visual models in the **Intelligent Screenshot Organizer**.

---

## 1. Directory Structure

```
experiments/visual_models/
├── dataset_split.py              # Stratified 60/20/20 train/cal/val split generator
├── dataset_split.json            # Reproducible partition image IDs (seed=42)
├── model_registry.py             # Model definitions for MobileNetV3, EfficientNet-B0, ShuffleNetV2, etc.
├── train_visual_classifier.py    # Training script for 9-class heads on frozen CNN backbones
├── evaluate_visual_models.py     # Comprehensive benchmarking script on 167 held-out validation images
├── evaluate_complementarity.py   # Complementarity analysis vs primary FinalRefinedClassifier
├── results/                      # Trained best model checkpoints (*_best.pt) & training logs
└── reports/                      # Generated benchmark & complementarity reports
    ├── visual_models_benchmark_report.md
    └── complementarity_analysis_report.md
```

---

## 2. Reproducing the Experiments

Run the following commands sequentially from the project root:

```bash
# 1. Generate stratified 60/20/20 dataset partitions (499 train / 167 cal / 167 val)
python experiments/visual_models/dataset_split.py

# 2. Train 9-class heads on candidate CNN backbones (MobileNetV3-Small/Large, EfficientNet-B0, ShuffleNetV2, MobileNetV2)
python experiments/visual_models/train_visual_classifier.py

# 3. Evaluate visual-only benchmarks against text-only and TinyCLIP baselines on held-out validation split
python experiments/visual_models/evaluate_visual_models.py

# 4. Perform complementarity analysis against primary FinalRefinedClassifier
python experiments/visual_models/evaluate_complementarity.py
```

---

## 3. Strict Non-Negotiable Rules Enforced

* Primary classifier (`FinalRefinedClassifier`), EasyOCR, MiniLM, rules, and prototypes remained **100% untouched**.
* External V1 and External V2 benchmarks remained **100% untouched**.
* All model training and early stopping relied exclusively on Train and Calibration splits. The 167 validation screenshot IDs were **strictly held out**.
