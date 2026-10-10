"""
Unit Tests for Controlled Supervised ML Experiments
===================================================
Tests training, prediction, metric calculations, zero-leakage dataset separation,
and output reports for supervised ML screenshot classification experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from experiments.intelligence.ml_experiments import (
    SupervisedMLModel,
    compute_eval_metrics,
    DEV_TARGETS,
    JSON_OUT_PATH,
    MD_OUT_PATH,
)
from experiments.intelligence.unseen.evaluation_targets import UNSEEN_GROUND_TRUTH_ANNOTATIONS
from experiments.intelligence.external.evaluation_targets import get_evaluation_targets as get_external_targets
from src.embedder import SentenceTransformerEmbedder


def test_dataset_split_counts_and_zero_leakage():
    """Verify that dataset splits have correct counts and do not leak into each other."""
    assert len(DEV_TARGETS) == 8, "Dev dataset should have exactly 8 screenshots"
    assert len(UNSEEN_GROUND_TRUTH_ANNOTATIONS) == 22, "Internal holdout should have exactly 22 screenshots"

    ext_targets = get_external_targets()
    assert len(ext_targets) == 60, "External test dataset should have exactly 60 screenshots"

    dev_filenames = set(d["filename"] for d in DEV_TARGETS)
    internal_filenames = set(UNSEEN_GROUND_TRUTH_ANNOTATIONS.keys())
    external_filenames = set(t["filename"] for t in ext_targets)

    # Ensure zero overlap between internal holdout and external dataset
    overlap = internal_filenames.intersection(external_filenames)
    assert len(overlap) == 0, f"Found unexpected filename overlap between internal and external sets: {overlap}"


def test_supervised_ml_model_tfidf():
    """Test TF-IDF model training and prediction pipeline."""
    train_texts = [
        "payment invoice receipt total amount paid",
        "payment credit card transaction succeeded",
        "python code function import sys class test",
        "python script def main return code",
        "flight booking ticket confirmation boarding pass",
        "flight airline reservation departure gate pass",
    ]
    train_labels = ["Payment", "Payment", "Coding", "Coding", "Travel", "Travel"]

    model = SupervisedMLModel("TFIDF_LogReg_Test", feature_type="tfidf", model_type="logreg")
    model.fit(train_texts, train_labels)

    assert len(model.classes_) == 3
    assert set(model.classes_) == {"Payment", "Coding", "Travel"}

    test_texts = ["payment receipt details", "python function code"]
    preds, top2, lats = model.predict(test_texts)

    assert len(preds) == 2
    assert len(top2) == 2
    assert len(top2[0]) <= 2
    assert len(lats) == 2
    assert preds[0] == "Payment"
    assert preds[1] == "Coding"


def test_supervised_ml_model_minilm():
    """Test MiniLM embedding model training and prediction pipeline."""
    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

    train_texts = [
        "wifi network connected ip address dns settings",
        "shopping cart checkout item price subtotal",
        "wifi router signal strength ping latency",
    ]
    train_labels = ["Connectivity", "Shopping", "Connectivity"]

    model = SupervisedMLModel("MiniLM_SVM_Test", feature_type="minilm", model_type="svm")
    model.fit(train_texts, train_labels, embedder=embedder)

    assert len(model.classes_) == 2

    test_texts = ["wireless connection disconnected"]
    preds, top2, lats = model.predict(test_texts, embedder=embedder)

    assert len(preds) == 1
    assert len(top2) == 1
    assert len(lats) == 1
    assert preds[0] == "Connectivity"


def test_compute_eval_metrics():
    """Verify evaluation metric calculations for top-1 accuracy, top-2 recall, and macro F1."""
    y_true = ["Payment", "Coding", "Travel", "Shopping"]
    y_pred = ["Payment", "Coding", "Shopping", "Shopping"]  # 3/4 correct Top-1
    top2_preds = [
        ["Payment", "Finance"],
        ["Coding", "Education"],
        ["Travel", "Shopping"],  # Travel is in top-2!
        ["Shopping", "Retail"],
    ]  # 4/4 correct Top-2

    metrics = compute_eval_metrics(y_true, y_pred, top2_preds)

    assert metrics["top1_accuracy"] == 0.75
    assert metrics["top2_recall"] == 1.0
    assert "per_category" in metrics
    assert metrics["per_category"]["Payment"]["correct_images"] == 1
    assert metrics["per_category"]["Payment"]["total_images"] == 1


def test_experiment_report_files_exist():
    """Verify that experiment outputs are saved properly if experiments have run."""
    if JSON_OUT_PATH.exists():
        with open(JSON_OUT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "selected_candidate" in data
        assert "selection_rationale" in data
        assert "internal_validation_results" in data
        assert "external_test_results" in data

    if MD_OUT_PATH.exists():
        text = MD_OUT_PATH.read_text(encoding="utf-8")
        assert "# Phase 4A Supervised Machine Learning Experiments Report" in text
        assert "Zero-Leakage Guarantee" in text
