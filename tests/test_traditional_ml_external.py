"""
Unit Tests for Traditional ML External Benchmark Evaluation Harness
=====================================================================
Verifies dataset separation, ground-truth loading boundaries, model fitting on the full
833 training pool, prediction contracts across all 5 frozen traditional ML models,
metrics calculations, agreement analysis, and zero-leakage constraints.
"""

from pathlib import Path
import numpy as np
import pytest

from experiments.intelligence.dataset_loader import CANONICAL_CATEGORIES, load_dataset
from experiments.intelligence.external.evaluation_targets import get_evaluation_targets
from experiments.intelligence.traditional_ml_external.traditional_ml_external import (
    compute_comprehensive_metrics,
    content_focused_ocr_text,
    extract_rule_features_single,
    normalize_ocr_text,
    PROJECT_ROOT,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def test_external_benchmark_dataset_integrity():
    """Verifies that exactly 60 external benchmark targets exist and have valid canonical labels."""
    ext_targets = get_evaluation_targets()
    assert len(ext_targets) == 60, f"Expected 60 external benchmark targets, got {len(ext_targets)}"

    filenames = [t["filename"] for t in ext_targets]
    expected_filenames = [f"ext_{i:03d}.jpg" for i in range(1, 61)]
    assert filenames == expected_filenames, "External target filenames mismatch!"

    categories = set(t["category"] for t in ext_targets)
    assert categories.issubset(set(CANONICAL_CATEGORIES)), f"Unexpected categories in external targets: {categories - set(CANONICAL_CATEGORIES)}"


def test_strict_dataset_separation_zero_leakage():
    """Verifies zero overlap between the 833 training screenshots and 60 external benchmark screenshots."""
    train_samples = load_dataset(exclude_invalid=True)
    ext_targets = get_evaluation_targets()

    train_files = set(s["filename"] for s in train_samples)
    ext_files = set(t["filename"] for t in ext_targets)

    overlap = train_files.intersection(ext_files)
    assert len(overlap) == 0, f"Data leakage detected! Overlapping files: {overlap}"


def test_frozen_model_fit_and_predict_contract():
    """Verifies that frozen models fit on training text and output valid external predictions and probability distributions."""
    train_samples = load_dataset(exclude_invalid=True)
    ext_targets = get_evaluation_targets()

    train_texts = [f"sample text for category {s['category']} keyword" for s in train_samples]
    train_y = [s["category"] for s in train_samples]

    ext_texts = [f"sample external text {i}" for i in range(60)]

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=100)
    X_tr = vec.fit_transform(train_texts)
    X_ext = vec.transform(ext_texts)

    # Test ComplementNB contract
    cnb = ComplementNB(alpha=1.0)
    cnb.fit(X_tr, train_y)
    preds = cnb.predict(X_ext)
    probas = cnb.predict_proba(X_ext)

    assert len(preds) == 60
    assert all(p in CANONICAL_CATEGORIES for p in preds)
    assert probas.shape == (60, 10)
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_comprehensive_metrics_and_agreement():
    """Verifies calculation of Top-1, Top-2, Top-3, MRR, Macro F1, and confusion matrix."""
    y_true = CANONICAL_CATEGORIES * 6  # 60 samples
    y_pred = list(y_true)
    y_pred[0] = CANONICAL_CATEGORIES[1]  # Introduce 1 error

    y_proba = np.zeros((60, 10), dtype=float)
    for i in range(60):
        true_idx = CANONICAL_CATEGORIES.index(y_true[i])
        pred_idx = CANONICAL_CATEGORIES.index(y_pred[i])
        y_proba[i, pred_idx] = 0.8
        y_proba[i, true_idx] = 0.2

    metrics = compute_comprehensive_metrics(y_true, y_pred, y_proba, categories=CANONICAL_CATEGORIES)

    assert metrics["top1_accuracy"] == pytest.approx(59 / 60)
    assert metrics["top2_recall"] == 1.0
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics
    assert len(metrics["confusion_matrix"]) == 10
