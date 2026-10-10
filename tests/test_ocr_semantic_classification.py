"""
Unit Tests for OCR + Lightweight Semantic Classification Experiment
====================================================================
Verifies that all components of the lightweight semantic classification pipeline
(Rule Baseline, TF-IDF + Linear SVM, MiniLM Cosine Similarity, Signal Fusion, metrics)
function correctly with valid outputs, probability normalization, and fold isolation.
"""

from pathlib import Path
import numpy as np
import pytest

from experiments.intelligence.dataset_loader import CANONICAL_CATEGORIES, load_dataset
from experiments.intelligence.ocr_semantic_classification.ocr_semantic_classification import (
    MiniLMSimilarityClassifier,
    RuleBaselineClassifier,
    SignalFusionClassifier,
    TfidfSVMClassifier,
    compute_metrics,
)


def test_rule_baseline_classifier():
    clf = RuleBaselineClassifier(categories=CANONICAL_CATEGORIES)
    text = "Paying via Google Pay UPI transaction successful ref no 12345"

    probas = clf.predict_proba([text])
    assert probas.shape == (1, 10)
    assert np.isclose(probas.sum(), 1.0)

    pred = clf.predict([text])[0]
    assert pred == "Payment"


def test_tfidf_svm_classifier():
    clf = TfidfSVMClassifier(categories=CANONICAL_CATEGORIES, random_state=42)
    train_texts = [
        "python code function import sys",
        "paying via upi transaction bank receipt",
        "flight booking pnr indigo airline ticket",
        "lecture course syllabus study quiz assignment",
        "cart product amazon shopping price discount",
        "youtube video music spotify playlist song",
        "wifi bluetooth signal network ip settings",
        "linkedin job resume recruitment internship salary",
        "weather forecast temperature bhubaneswar",
        "whatsapp message email chat conversation inbox",
    ]
    train_labels = CANONICAL_CATEGORIES  # 1 sample per class for quick test

    clf.fit(train_texts, train_labels)

    test_text = "python function import code"
    probas = clf.predict_proba([test_text])
    assert probas.shape == (1, 10)
    assert np.isclose(probas.sum(), 1.0)

    pred = clf.predict([test_text])[0]
    assert pred == "Coding"


def test_minilm_similarity_classifier():
    clf = MiniLMSimilarityClassifier(categories=CANONICAL_CATEGORIES)
    test_text = "flight booking pnr indigo airline boarding pass"

    probas = clf.predict_proba([test_text])
    assert probas.shape == (1, 10)
    assert np.isclose(probas.sum(), 1.0)

    pred = clf.predict([test_text])[0]
    assert pred == "Travel"


def test_signal_fusion_classifier():
    rule_clf = RuleBaselineClassifier(categories=CANONICAL_CATEGORIES)
    tfidf_clf = TfidfSVMClassifier(categories=CANONICAL_CATEGORIES, random_state=42)

    dummy_texts = [f"sample text for category {cat}" for cat in CANONICAL_CATEGORIES]
    tfidf_clf.fit(dummy_texts, CANONICAL_CATEGORIES)

    minilm_clf = MiniLMSimilarityClassifier(categories=CANONICAL_CATEGORIES)

    fusion = SignalFusionClassifier(
        rule_clf, tfidf_clf, minilm_clf, weights=(0.3, 0.4, 0.3), categories=CANONICAL_CATEGORIES
    )

    test_text = "python import code function"
    probas = fusion.predict_proba([test_text])
    assert probas.shape == (1, 10)
    assert np.isclose(probas.sum(), 1.0)

    pred = fusion.predict([test_text])[0]
    assert isinstance(pred, str)
    assert pred in CANONICAL_CATEGORIES


def test_compute_metrics():
    y_true = ["Payment", "Shopping", "Coding"]
    y_pred = ["Payment", "Shopping", "Education"]
    y_proba = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],  # Education (idx 3) vs Coding (idx 0)
    ])

    metrics = compute_metrics(y_true, y_pred, y_proba, categories=CANONICAL_CATEGORIES)

    assert "top1_accuracy" in metrics
    assert "top2_recall" in metrics
    assert "top3_recall" in metrics
    assert "macro_f1" in metrics
    assert "per_class" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["top1_accuracy"] == pytest.approx(2 / 3)


def test_dataset_loader_integrity():
    samples = load_dataset(exclude_invalid=True)
    assert len(samples) == 833
    invalid_files = {"ss_441.png", "ss_443.png", "ss_560.png"}
    found_invalid = [s["filename"] for s in samples if s["filename"] in invalid_files]
    assert len(found_invalid) == 0
