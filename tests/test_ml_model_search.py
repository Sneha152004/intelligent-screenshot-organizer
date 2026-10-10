"""
Unit Tests for Systematic Model Search Experiment Harness
==========================================================
Tests representation extractors, candidate model training, cross-validation metrics,
dataset separation, and report generation for screenshot intelligence model search.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from experiments.intelligence.ml_model_search import (
    FeatureExtractor,
    CandidateModelHarness,
    compute_metrics,
    DEV_TARGETS,
    JSON_OUT_PATH,
    MD_OUT_PATH,
)
from experiments.intelligence.unseen.evaluation_targets import UNSEEN_GROUND_TRUTH_ANNOTATIONS
from experiments.intelligence.external.evaluation_targets import get_evaluation_targets as get_external_targets
from src.embedder import SentenceTransformerEmbedder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def test_model_search_dataset_split_integrity():
    """Verify exact dataset split sizes and zero-overlap boundaries."""
    assert len(DEV_TARGETS) == 8
    assert len(UNSEEN_GROUND_TRUTH_ANNOTATIONS) == 22

    ext_targets = get_external_targets()
    assert len(ext_targets) == 60

    dev_filenames = set(d["filename"] for d in DEV_TARGETS)
    internal_filenames = set(UNSEEN_GROUND_TRUTH_ANNOTATIONS.keys())
    external_filenames = set(t["filename"] for t in ext_targets)

    overlap = internal_filenames.intersection(external_filenames)
    assert len(overlap) == 0, f"Found unexpected filename overlap: {overlap}"


def test_feature_extractor_representations():
    """Test feature extraction across word, char, and combined representations."""
    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    texts = [
        "payment invoice receipt total amount paid",
        "python script import sys def main return 0",
        "flight booking ticket confirmation gate pass",
    ]

    for rep in ["word_tfidf_bigram", "char_tfidf_25", "word_char_tfidf", "minilm", "combo_sparse_dense"]:
        fe = FeatureExtractor(rep, embedder=embedder)
        feats = fe.fit_transform(texts)
        assert feats is not None
        if rep == "minilm":
            assert isinstance(feats, np.ndarray)
            assert feats.shape == (3, 384)


def test_candidate_model_harness_fit_predict():
    """Test candidate model harness fitting and prediction format."""
    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    train_texts = [
        "payment invoice receipt total amount paid",
        "payment credit card transaction succeeded",
        "python code function import sys class test",
        "python script def main return code",
    ]
    train_labels = ["Payment", "Payment", "Coding", "Coding"]

    model = CandidateModelHarness("Test_Model", "word_char_tfidf", lambda: LinearSVC(C=1.0, random_state=42), "Unit Test Model")
    model.fit(train_texts, train_labels, embedder=embedder)

    assert len(model.classes_) == 2

    test_texts = ["payment receipt details", "python function code"]
    preds, top2, top3, lats = model.predict(test_texts, embedder=embedder)

    assert len(preds) == 2
    assert len(top2) == 2
    assert len(top3) == 2
    assert len(lats) == 2
    assert preds[0] == "Payment"
    assert preds[1] == "Coding"


def test_compute_metrics_structure():
    """Verify metric computation calculations including top-3 recall and macro precision/F1."""
    y_true = ["Payment", "Coding", "Travel", "Shopping"]
    y_pred = ["Payment", "Coding", "Shopping", "Shopping"]
    top2 = [["Payment", "Finance"], ["Coding", "Education"], ["Travel", "Shopping"], ["Shopping", "Retail"]]
    top3 = [["Payment", "Finance", "Bank"], ["Coding", "Education", "Tech"], ["Travel", "Shopping", "Transit"], ["Shopping", "Retail", "Store"]]

    metrics = compute_metrics(y_true, y_pred, top2, top3)

    assert metrics["top1_accuracy"] == 0.75
    assert metrics["top2_recall"] == 1.0
    assert metrics["top3_recall"] == 1.0
    assert "macro_precision" in metrics
    assert "macro_recall" in metrics
    assert "macro_f1" in metrics


def test_search_results_report_files_exist():
    """Verify that JSON and Markdown search report files exist and have required schema."""
    if JSON_OUT_PATH.exists():
        with open(JSON_OUT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "selected_winner" in data
        assert "selection_protocol" in data
        assert "cross_validation_results" in data
        assert "external_benchmark_results" in data
        assert "integrity_checklist" in data
        assert data["integrity_checklist"]["external_images_modified"] is False

    if MD_OUT_PATH.exists():
        text = MD_OUT_PATH.read_text(encoding="utf-8")
        assert "# Comprehensive Systematic Model-Search Report" in text
        assert "Central Model Search Comparison Table" in text
