"""
Unit Tests for Visual Feature Fusion Experiment Harness
======================================================
Tests visual feature extraction, feature normalization, small-data fusion models,
bagged ensemble logic, dataset separation, and output report generation.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from experiments.intelligence.visual_model_experiment import (
    CompactVisualFeatureExtractor,
    StratifiedBaggedClassifier,
    VisionExperimentModel,
    compute_eval_metrics,
    DEV_TARGETS,
    JSON_OUT_PATH,
    MD_OUT_PATH,
)
from experiments.intelligence.unseen.evaluation_targets import UNSEEN_GROUND_TRUTH_ANNOTATIONS
from experiments.intelligence.external.evaluation_targets import get_evaluation_targets as get_external_targets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def test_visual_feature_extractor():
    """Verify visual feature extraction dimensionality, determinism, and fallback handling."""
    extractor = CompactVisualFeatureExtractor()

    # Create a temporary dummy image
    test_img = Image.new("RGB", (100, 200), color="blue")
    tmp_path = Path("scratch_test_img.png")
    test_img.save(tmp_path)

    try:
        feats = extractor.extract_features(str(tmp_path))
        assert isinstance(feats, np.ndarray)
        assert feats.shape == (54,)
        assert not np.isnan(feats).any()
        assert not np.isinf(feats).any()

        # Determinism check
        feats2 = extractor.extract_features(str(tmp_path))
        np.testing.assert_array_almost_equal(feats, feats2)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    # Fallback check for missing file
    fallback_feats = extractor.extract_features("non_existent_image.png")
    assert isinstance(fallback_feats, np.ndarray)
    assert fallback_feats.shape == (54,)
    assert (fallback_feats == 0.0).all()


def test_stratified_bagged_classifier():
    """Verify stratified bagged classifier training and prediction pipeline."""
    X = np.array([
        [1.0, 2.0], [1.1, 2.1], [1.2, 1.9],
        [5.0, 6.0], [5.1, 5.9], [4.9, 6.1],
    ])
    y = ["Payment", "Payment", "Payment", "Coding", "Coding", "Coding"]

    bagged = StratifiedBaggedClassifier(lambda: LinearSVC(C=1.0, random_state=42), n_estimators=5, random_state=42)
    bagged.fit(X, y)

    assert len(bagged.models) == 5
    assert bagged.classes_ == ["Coding", "Payment"]

    preds = bagged.predict(np.array([[1.05, 2.05], [5.05, 6.05]]))
    assert len(preds) == 2
    assert preds[0] == "Payment"
    assert preds[1] == "Coding"


def test_vision_experiment_model_modes():
    """Test VisionExperimentModel across text_only, vis_only, early_fusion, and late_fusion modes."""
    text_embs = np.random.randn(6, 384)
    vis_feats = np.random.randn(6, 54)
    labels = ["Payment", "Payment", "Coding", "Coding", "Travel", "Travel"]

    for mode in ["text_only", "vis_only", "early_fusion", "late_fusion"]:
        m = VisionExperimentModel(f"Test_{mode}", mode, lambda: LinearSVC(C=1.0, random_state=42), "Unit test model")
        m.fit(text_embs, vis_feats, labels)

        preds, top2, top3, lat = m.predict(text_embs[:2], vis_feats[:2])
        assert len(preds) == 2
        assert len(top2) == 2
        assert len(top3) == 2
        assert lat >= 0.0


def test_visual_dataset_separation_integrity():
    """Verify dataset split counts and zero overlap between internal and external targets."""
    assert len(DEV_TARGETS) == 8
    assert len(UNSEEN_GROUND_TRUTH_ANNOTATIONS) == 22

    ext_targets = get_external_targets()
    assert len(ext_targets) == 60

    internal_filenames = set(UNSEEN_GROUND_TRUTH_ANNOTATIONS.keys())
    external_filenames = set(t["filename"] for t in ext_targets)

    overlap = internal_filenames.intersection(external_filenames)
    assert len(overlap) == 0, f"Found unexpected filename overlap: {overlap}"


def test_visual_report_files_exist():
    """Verify that JSON and Markdown experiment output files exist and match expected schemas."""
    if JSON_OUT_PATH.exists():
        with open(JSON_OUT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "selected_winner" in data
        assert "selection_protocol" in data
        assert "cross_validation_results" in data
        assert "external_benchmark_results" in data
        assert "visual_error_analysis" in data
        assert "integrity_checklist" in data

    if MD_OUT_PATH.exists():
        text = MD_OUT_PATH.read_text(encoding="utf-8")
        assert "# Lightweight Visual & Layout Feature Fusion Experiment Report" in text
        assert "Comprehensive Experiment Results Table" in text
        assert "Visual Error Analysis" in text
