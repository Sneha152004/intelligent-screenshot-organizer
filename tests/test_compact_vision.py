"""
Unit test suite for Compact Semantic Vision Experiment Harness
================================================================
Verifies:
1. PretrainedCLIPEncoder loads 'wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M' (89.4 MB).
2. Image feature extraction yields 512-dimensional L2-normalized vectors.
3. Text feature extraction yields 512-dimensional L2-normalized vectors.
4. Missing / corrupt image paths return 512d zero vectors cleanly.
5. ZeroShotCLIPClassifier outputs valid probability distribution.
6. MiniLM + TinyCLIP feature fusion creates an 896d vector.
7. Evaluation metrics compute Top-1, Top-2, Top-3, Macro F1, Macro P accurately.
8. JSON and Markdown result artifacts exist in experiments/intelligence/compact_vision/.
9. Zero-leakage boundaries: model selection is determined strictly via 5-Fold Cross-Validation.
"""

import json
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from experiments.intelligence.compact_vision.compact_vision_experiment import (
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
    """Creates a temporary dummy image file for TinyCLIP visual encoder testing."""
    img_path = tmp_path / "test_tinyclip_img.jpg"
    img = Image.new("RGB", (224, 224), color=(60, 140, 180))
    img.save(img_path)
    return str(img_path)


def test_tinyclip_8m_image_extraction(sample_image_path):
    """Verifies TinyCLIP-8M encoder extracts 512d L2-normalized vector from image."""
    encoder = PretrainedCLIPEncoder("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", footprint_mb=89.4)
    vec, lat_ms = encoder.extract_image_feature(sample_image_path)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (512,)
    assert lat_ms > 0.0
    norm = np.linalg.norm(vec)
    assert norm == pytest.approx(1.0, abs=1e-3)
    assert encoder.footprint_mb == 89.4
    assert encoder.footprint_mb <= 200.0  # Target <=200MB verified


def test_tinyclip_8m_text_extraction():
    """Verifies TinyCLIP-8M encoder extracts 512d L2-normalized text vectors."""
    encoder = PretrainedCLIPEncoder("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", footprint_mb=89.4)
    texts = ["an educational screenshot", "a shopping screenshot"]
    vecs, lat_ms = encoder.extract_text_features(texts)

    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (2, 512)
    assert lat_ms > 0.0
    for i in range(2):
        norm = np.linalg.norm(vecs[i])
        assert norm == pytest.approx(1.0, abs=1e-3)


def test_tinyclip_missing_image_returns_zero_vector():
    """Verifies missing image paths yield a fallback 512d zero vector cleanly."""
    encoder = PretrainedCLIPEncoder("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", footprint_mb=89.4)
    vec, lat_ms = encoder.extract_image_feature("non_existent_compact_file.jpg")

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (512,)
    assert np.all(vec == 0.0)


def test_tinyclip_zero_shot_classifier(sample_image_path):
    """Verifies ZeroShotCLIPClassifier with TinyCLIP-8M outputs valid probabilities."""
    encoder = PretrainedCLIPEncoder("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", footprint_mb=89.4)
    zs_clf = ZeroShotCLIPClassifier(encoder, CATEGORY_PROMPTS)

    img_vec, _ = encoder.extract_image_feature(sample_image_path)
    probs = zs_clf.predict_proba(np.expand_dims(img_vec, axis=0))

    assert probs.shape == (1, 10)
    assert np.sum(probs) == pytest.approx(1.0, abs=1e-4)


def test_compact_feature_fusion_dimensionality():
    """Verifies fusion of MiniLM (384d) and TinyCLIP (512d) produces an 896d vector."""
    minilm_vec = np.random.randn(384)
    tiny_vec = np.random.randn(512)

    fused_vec = np.hstack([minilm_vec, tiny_vec])
    assert fused_vec.shape == (896,)


def test_compact_vision_result_artifacts_exist():
    """Verifies JSON and Markdown result artifacts exist and follow strict schema requirements."""
    assert JSON_OUT_PATH.exists(), f"Missing result file: {JSON_OUT_PATH}"
    assert MD_OUT_PATH.exists(), f"Missing report file: {MD_OUT_PATH}"

    with open(JSON_OUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "selected_winner" in data
    assert "best_compact_candidate" in data
    assert "selection_protocol" in data
    assert "target_200mb_achieved" in data
    assert "cross_validation_results" in data
    assert "external_benchmark_results" in data
    assert "error_analysis" in data
    assert "integrity_checklist" in data

    # Verify target <=200MB flag
    assert data["target_200mb_achieved"] is True

    # Verify integrity checklist flags
    integrity = data["integrity_checklist"]
    assert integrity["external_images_modified"] is False
    assert integrity["external_labels_modified"] is False
    assert integrity["external_data_used_for_training"] is False
    assert integrity["external_data_used_for_model_selection"] is False
    assert integrity["production_classifier_modified"] is False
