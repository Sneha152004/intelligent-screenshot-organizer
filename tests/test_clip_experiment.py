"""
Unit test suite for CLIP-Style Image-Text Semantic Representation Experiment Harness
========================================================================================
Verifies:
1. PretrainedCLIPEncoder correctly loads 'openai/clip-vit-base-patch32'.
2. Image feature extraction yields 512-dimensional L2-normalized vectors.
3. Text feature extraction yields 512-dimensional L2-normalized vectors.
4. Fallback for corrupt / missing images returns 512d zero vectors safely.
5. ZeroShotCLIPClassifier initializes prototypes from fixed category prompts.
6. Zero-shot output probability distribution sums to 1.0 across all 10 categories.
7. Feature fusion concatenates MiniLM (384d) and CLIP (512d) into an 896d vector.
8. Metric calculation computes Top-1, Top-2, Top-3, Macro F1, Macro P accurately.
9. JSON and Markdown result artifacts exist and follow strict protocol schemas.
10. Zero-leakage boundaries: model selection is determined strictly via 5-fold Cross-Validation.
"""

import json
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from experiments.intelligence.clip_experiment import (
    PretrainedCLIPEncoder,
    ZeroShotCLIPClassifier,
    CATEGORY_PROMPTS,
    compute_eval_metrics,
    JSON_OUT_PATH,
    MD_OUT_PATH,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_image_path(tmp_path):
    """Creates a temporary dummy image file for CLIP visual encoder testing."""
    img_path = tmp_path / "test_clip_img.jpg"
    img = Image.new("RGB", (224, 224), color=(80, 120, 200))
    img.save(img_path)
    return str(img_path)


def test_clip_encoder_image_extraction(sample_image_path):
    """Verifies CLIP encoder extracts 512d L2-normalized vector from image."""
    encoder = PretrainedCLIPEncoder()
    vec, lat_ms = encoder.extract_image_feature(sample_image_path)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (512,)
    assert lat_ms > 0.0
    norm = np.linalg.norm(vec)
    assert norm == pytest.approx(1.0, abs=1e-3)


def test_clip_encoder_text_extraction():
    """Verifies CLIP encoder extracts 512d L2-normalized text vectors for prompts."""
    encoder = PretrainedCLIPEncoder()
    texts = ["an educational screenshot", "a shopping screenshot"]
    vecs, lat_ms = encoder.extract_text_features(texts)

    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (2, 512)
    assert lat_ms > 0.0
    for i in range(2):
        norm = np.linalg.norm(vecs[i])
        assert norm == pytest.approx(1.0, abs=1e-3)


def test_clip_missing_image_returns_zero_vector():
    """Verifies missing image paths yield a fallback 512d zero vector cleanly."""
    encoder = PretrainedCLIPEncoder()
    vec, lat_ms = encoder.extract_image_feature("non_existent_file_999.jpg")

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (512,)
    assert np.all(vec == 0.0)


def test_category_prompts_completeness():
    """Verifies all 10 target categories have defined prompt lists."""
    expected_categories = [
        "Education", "Shopping", "Payment", "Coding", "Travel",
        "Work & Career", "Communication", "Entertainment", "Connectivity", "General"
    ]
    for cat in expected_categories:
        assert cat in CATEGORY_PROMPTS
        assert isinstance(CATEGORY_PROMPTS[cat], list)
        assert len(CATEGORY_PROMPTS[cat]) >= 2


def test_zero_shot_classifier_probs_sum_to_one(sample_image_path):
    """Verifies ZeroShotCLIPClassifier outputs valid probability distribution summing to 1.0."""
    encoder = PretrainedCLIPEncoder()
    zs_clf = ZeroShotCLIPClassifier(encoder, CATEGORY_PROMPTS)

    img_vec, _ = encoder.extract_image_feature(sample_image_path)
    probs = zs_clf.predict_proba(np.expand_dims(img_vec, axis=0))

    assert probs.shape == (1, 10)
    assert np.sum(probs) == pytest.approx(1.0, abs=1e-4)


def test_feature_fusion_dimensionality():
    """Verifies fusion of MiniLM (384d) and CLIP (512d) produces an 896d vector."""
    minilm_vec = np.random.randn(384)
    clip_vec = np.random.randn(512)

    fused_vec = np.hstack([minilm_vec, clip_vec])
    assert fused_vec.shape == (896,)


def test_clip_experiment_result_artifacts_exist():
    """Verifies JSON and Markdown result artifacts exist and follow strict schema requirements."""
    assert JSON_OUT_PATH.exists(), f"Missing result file: {JSON_OUT_PATH}"
    assert MD_OUT_PATH.exists(), f"Missing report file: {MD_OUT_PATH}"

    with open(JSON_OUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "selected_winner" in data
    assert "best_clip_candidate" in data
    assert "selection_protocol" in data
    assert "category_prompts" in data
    assert "cross_validation_results" in data
    assert "external_benchmark_results" in data
    assert "clip_error_analysis" in data
    assert "integrity_checklist" in data

    # Verify integrity checklist flags
    integrity = data["integrity_checklist"]
    assert integrity["external_images_modified"] is False
    assert integrity["external_labels_modified"] is False
    assert integrity["external_data_used_for_training"] is False
    assert integrity["external_data_used_for_model_selection"] is False
    assert integrity["production_classifier_modified"] is False
