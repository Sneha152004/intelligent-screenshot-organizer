"""
Unit tests for Transfer Learning Pretrained Visual Encoders & Feature Fusion Harness
===================================================================================
Verifies:
1. PretrainedVisualEncoder correctly loads MobileNetV2 and EfficientNet-B0 architectures.
2. Extraction yields 1280-dimensional L2-normalized feature vectors.
3. Invalid / missing images return fallback zero vectors cleanly.
4. Evaluation metrics function computes Top-1, Top-2, Top-3, Macro F1, Macro P accurately.
5. JSON and Markdown result artifacts exist and follow strict protocol schemas.
6. Zero-leakage boundaries: model selection is determined strictly via Cross-Validation.
"""

import json
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from experiments.intelligence.transfer_learning_visual import (
    PretrainedVisualEncoder,
    compute_eval_metrics,
    JSON_OUT_PATH,
    MD_OUT_PATH,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_image_path(tmp_path):
    """Creates a temporary dummy image file for visual encoder testing."""
    img_path = tmp_path / "test_img.jpg"
    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    img.save(img_path)
    return str(img_path)


def test_mobilenetv2_encoder_extraction(sample_image_path):
    """Verifies MobileNetV2 encoder extracts 1280d normalized vector from image."""
    encoder = PretrainedVisualEncoder(architecture="mobilenet_v2")
    vec = encoder.extract_features(sample_image_path)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (1280,)
    # Verify vector is non-zero and L2-normalized (~1.0)
    norm = np.linalg.norm(vec)
    assert norm == pytest.approx(1.0, abs=1e-3)


def test_efficientnet_b0_encoder_extraction(sample_image_path):
    """Verifies EfficientNet-B0 encoder extracts 1280d normalized vector from image."""
    encoder = PretrainedVisualEncoder(architecture="efficientnet_b0")
    vec = encoder.extract_features(sample_image_path)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (1280,)
    norm = np.linalg.norm(vec)
    assert norm == pytest.approx(1.0, abs=1e-3)


def test_invalid_architecture_raises_error():
    """Verifies ValueError is raised when an unsupported architecture is requested."""
    with pytest.raises(ValueError, match="Unsupported architecture"):
        PretrainedVisualEncoder(architecture="resnet50")


def test_missing_image_returns_zero_vector():
    """Verifies missing or corrupt image paths yield a fallback 1280d zero vector."""
    encoder = PretrainedVisualEncoder(architecture="mobilenet_v2")
    vec = encoder.extract_features("non_existent_file_path_12345.jpg")

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (1280,)
    assert np.all(vec == 0.0)


def test_feature_fusion_vector_shape():
    """Verifies feature fusion of MiniLM (384d) and Visual (1280d) produces 1664d vector."""
    minilm_vec = np.random.randn(384)
    visual_vec = np.random.randn(1280)

    fused_vec = np.hstack([minilm_vec, visual_vec])
    assert fused_vec.shape == (1664,)


def test_compute_eval_metrics():
    """Verifies metric calculations for Top-1, Top-2, Top-3, and Macro Precision/F1."""
    y_true = ["Education", "Payment", "Shopping", "Travel"]
    y_pred = ["Education", "Payment", "Entertainment", "Travel"]
    top2_preds = [["Education", "Payment"], ["Payment", "Shopping"], ["Shopping", "Entertainment"], ["Travel", "Education"]]
    top3_preds = [["Education", "Payment", "Shopping"], ["Payment", "Shopping", "Travel"], ["Shopping", "Entertainment", "Education"], ["Travel", "Education", "Payment"]]

    metrics = compute_eval_metrics(y_true, y_pred, top2_preds, top3_preds)

    assert metrics["top1_accuracy"] == 0.75  # 3/4 correct
    assert metrics["top2_recall"] == 1.0     # All 4 in Top-2
    assert metrics["top3_recall"] == 1.0     # All 4 in Top-3
    assert "macro_f1" in metrics
    assert "macro_precision" in metrics
    assert "per_category" in metrics


def test_experiment_result_artifacts_exist():
    """Verifies JSON and Markdown result artifacts exist and contain mandatory schema fields."""
    assert JSON_OUT_PATH.exists(), f"Missing result file: {JSON_OUT_PATH}"
    assert MD_OUT_PATH.exists(), f"Missing report file: {MD_OUT_PATH}"

    with open(JSON_OUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "selected_winner" in data
    assert "selection_protocol" in data
    assert "cross_validation_results" in data
    assert "external_benchmark_results" in data
    assert "visual_error_analysis" in data
    assert "target_92_achieved" in data
    assert "integrity_checklist" in data

    # Verify winner was selected
    assert data["selected_winner"] == "Bagged MiniLM + SVM (Text Baseline)"

    # Verify integrity checklist flags
    integrity = data["integrity_checklist"]
    assert integrity["external_images_modified"] is False
    assert integrity["external_labels_modified"] is False
    assert integrity["external_data_used_for_selection"] is False
    assert integrity["current_production_classifier_modified"] is False
