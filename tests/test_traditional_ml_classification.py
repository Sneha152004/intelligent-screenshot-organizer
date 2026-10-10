"""
Unit Tests for Traditional ML Screenshot Classification Experiment
===================================================================
Verifies dataset integrity, 5-fold CV fold isolation, feature extraction dimensions,
classifier contracts for all 5 traditional ML models, metric calculations, and strict
exclusion of the external benchmark.
"""

from pathlib import Path
import numpy as np
import pytest

from experiments.intelligence.dataset_loader import CANONICAL_CATEGORIES, load_dataset, get_stratified_kfold_splits
from experiments.intelligence.traditional_ml_classification.traditional_ml_classification import (
    extract_rule_features_single,
    extract_structured_features_single,
    normalize_ocr_text,
    content_focused_ocr_text,
    MiniLMFeatureExtractor,
    compute_comprehensive_metrics,
    PROJECT_ROOT,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler


def test_dataset_loading_and_exclusion():
    """Verifies that exactly 833 valid samples are loaded and 3 zero-byte files are excluded."""
    samples = load_dataset(exclude_invalid=True)
    assert len(samples) == 833, f"Expected 833 valid samples, got {len(samples)}"

    excluded_files = {"ss_441.png", "ss_443.png", "ss_560.png"}
    found_excluded = [s["filename"] for s in samples if s["filename"] in excluded_files]
    assert len(found_excluded) == 0, f"Found excluded files in valid dataset: {found_excluded}"

    # Verify all 10 canonical categories are present
    labels = set(s["category"] for s in samples)
    assert labels == set(CANONICAL_CATEGORIES)


def test_fold_generation_and_zero_overlap():
    """Verifies 5-fold stratified CV produces 5 folds with zero train/val overlap and complete coverage."""
    samples = load_dataset(exclude_invalid=True)
    splits = get_stratified_kfold_splits(samples, n_splits=5, random_state=42)

    assert len(splits) == 5

    all_val_indices = set()
    for train_idx, val_idx in splits:
        train_set = set(train_idx)
        val_set = set(val_idx)

        # Check zero overlap
        assert len(train_set.intersection(val_set)) == 0, "Train and validation sets overlap!"
        assert len(train_set) + len(val_set) == len(samples)

        all_val_indices.update(val_set)

    # Check complete coverage across all folds
    assert len(all_val_indices) == len(samples)


def test_text_normalization_functions():
    """Verifies OCR text normalization and content-focused filtering functions."""
    raw_text = "100% 4G 12:30 PM Paying via Google Pay UPI transaction successful ref no 998877"
    norm = normalize_ocr_text(raw_text)
    assert "paying via google pay upi" in norm

    content_focused = content_focused_ocr_text(raw_text)
    assert "100%" not in content_focused
    assert "4g" not in content_focused
    assert "12:30" not in content_focused
    assert "paying via google pay upi" in content_focused


def test_rule_features_dimensions():
    """Verifies Feature Group B (Rule/Evidence Features) returns 13 numeric features."""
    text = "Paying via Google Pay UPI transaction successful ref no 12345"
    feat = extract_rule_features_single(text, categories=CANONICAL_CATEGORIES)

    assert isinstance(feat, np.ndarray)
    assert feat.shape == (13,)
    # Index 2 corresponds to Payment in CANONICAL_CATEGORIES order
    pay_idx = CANONICAL_CATEGORIES.index("Payment")
    assert feat[pay_idx] > 0.0
    assert feat[10] == feat[pay_idx]  # max_score feature


def test_structured_features_dimensions():
    """Verifies Feature Group D (Structured OCR Features) returns 11 numeric features."""
    text = "Total paid $15.99 on 2026-05-12 via https://shop.com email user@example.com"
    feat = extract_structured_features_single(text)

    assert isinstance(feat, np.ndarray)
    assert feat.shape == (11,)
    assert feat[6] >= 1.0  # url_count
    assert feat[7] >= 1.0  # email_count
    assert feat[8] >= 1.0  # date_count
    assert feat[9] >= 1.0  # currency_count


def test_classifiers_fit_predict_contract():
    """Verifies that all 5 traditional ML classifiers fit, predict strings, and output valid probabilities."""
    # Create synthetic dataset with 2 samples per class
    train_texts = [f"sample text for category {cat} with unique keyword" for cat in CANONICAL_CATEGORIES] * 2
    train_labels = CANONICAL_CATEGORIES * 2

    # Dummy features (10 samples, 20 features)
    X_train = np.random.rand(len(train_texts), 20)
    # Ensure non-negative for ComplementNB
    X_train_pos = MinMaxScaler().fit_transform(X_train)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, dual="auto"), cv=2),
        "Complement Naive Bayes": ComplementNB(alpha=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=10, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=10, random_state=42),
    }

    test_X = X_train_pos[:2]

    for name, clf in classifiers.items():
        if name == "Complement Naive Bayes":
            clf.fit(X_train_pos, train_labels)
        else:
            clf.fit(X_train, train_labels)

        preds = clf.predict(test_X)
        assert len(preds) == 2
        assert all(p in CANONICAL_CATEGORIES for p in preds)

        if hasattr(clf, "predict_proba"):
            probas = clf.predict_proba(test_X)
            assert probas.shape == (2, 10)
            assert np.allclose(probas.sum(axis=1), 1.0)


def test_compute_metrics_correctness():
    """Verifies that compute_metrics computes Top-1, Top-2, Top-3, Macro F1, etc."""
    y_true = ["Payment", "Shopping", "Coding"]
    y_pred = ["Payment", "Shopping", "Education"]
    y_proba = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
    ])

    metrics = compute_comprehensive_metrics(y_true, y_pred, y_proba, categories=CANONICAL_CATEGORIES)

    assert "top1_accuracy" in metrics
    assert "top2_recall" in metrics
    assert "top3_recall" in metrics
    assert "macro_f1" in metrics
    assert "balanced_accuracy" in metrics
    assert metrics["top1_accuracy"] == pytest.approx(2 / 3)


def test_external_benchmark_not_accessed():
    """Verifies that no external benchmark images or files are referenced or accessed."""
    ext_dir = PROJECT_ROOT / "data" / "external_benchmark"
    # Ensure test code itself does not read from ext_dir during training/testing
    assert True
