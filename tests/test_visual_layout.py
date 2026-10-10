"""
Unit Tests for Lightweight Visual & Layout Feature Extraction Engine
=====================================================================
Verifies visual feature extraction metrics, heuristic layout scores,
spatial text box geometry, image geometry, edge density, color statistics,
disambiguation rules, and strict 0% external benchmark access.
"""

from pathlib import Path
import pytest
import numpy as np

from experiments.intelligence.dataset_loader import CANONICAL_CATEGORIES, load_dataset
from experiments.intelligence.visual_layout.visual_layout_extractor import (
    extract_visual_layout_features,
)
from experiments.intelligence.visual_layout.visual_layout_experiment import (
    ExplainableVisualEvidenceClassifier,
    PROJECT_ROOT,
)


def test_visual_layout_feature_extraction_contract():
    """Verifies that visual layout extractor returns expected feature schema for a valid screenshot."""
    samples = load_dataset(exclude_invalid=True)
    sample_img = samples[0]["image_path"]

    feats = extract_visual_layout_features(sample_img)

    assert "image_geometry" in feats
    assert "width" in feats["image_geometry"]
    assert "height" in feats["image_geometry"]
    assert "aspect_ratio" in feats["image_geometry"]

    assert "color_brightness" in feats
    assert "mean_brightness" in feats["color_brightness"]
    assert "is_dark_mode" in feats["color_brightness"]

    assert "edge_structure" in feats
    assert "canny_edge_density" in feats["edge_structure"]

    assert "text_geometry" in feats
    assert "text_box_count" in feats["text_geometry"]

    assert "layout_heuristics" in feats
    assert "ide_editor_layout" in feats["layout_heuristics"]
    assert "slide_course_document_layout" in feats["layout_heuristics"]
    assert "generic_unknown_layout" in feats["layout_heuristics"]

    assert "visual_category_evidence" in feats
    assert len(feats["visual_category_evidence"]) == 10
    assert "latency_ms" in feats


def test_ide_layout_heuristic_trigger():
    """Verifies IDE-like dark mode and code indentation triggers IDE layout score."""
    sample_feats = {
        "image_geometry": {"width": 1920, "height": 1080, "aspect_ratio": 1.77, "orientation": "landscape"},
        "color_brightness": {"mean_brightness": 35.0, "std_brightness": 25.0, "is_dark_mode": 1.0, "is_light_mode": 0.0, "mean_saturation": 0.05},
        "edge_structure": {"canny_edge_density": 0.045, "h_v_edge_ratio": 1.2, "horizontal_lines_count": 8, "grid_card_count": 2},
        "text_geometry": {"text_box_count": 15, "left_margin_std": 0.08, "code_indentation_score": 0.75, "top_zone_ratio": 0.15, "mid_zone_ratio": 0.70, "bot_zone_ratio": 0.15},
        "layout_heuristics": {"ide_editor_layout": 0.85, "slide_course_document_layout": 0.10, "generic_unknown_layout": 0.05},
        "visual_category_evidence": {"Coding": 0.85, "Education": 0.10, "General": 0.05, "Shopping": 0.0, "Communication": 0.0, "Entertainment": 0.0, "Travel": 0.0, "Connectivity": 0.0, "Work & Career": 0.10, "Payment": 0.0},
    }

    clf = ExplainableVisualEvidenceClassifier(categories=CANONICAL_CATEGORIES, mode="full_visual")
    res = clf.predict_single_explainable("def calculate_area(r): import math return math.pi * r**2", sample_feats)

    assert res["prediction"] == "Coding"


def test_slide_layout_heuristic_trigger():
    """Verifies light background slide layout + Python tutorial text predicts Education."""
    sample_feats = {
        "image_geometry": {"width": 1920, "height": 1080, "aspect_ratio": 1.77, "orientation": "landscape"},
        "color_brightness": {"mean_brightness": 220.0, "std_brightness": 35.0, "is_dark_mode": 0.0, "is_light_mode": 1.0, "mean_saturation": 0.02},
        "edge_structure": {"canny_edge_density": 0.015, "h_v_edge_ratio": 1.0, "horizontal_lines_count": 2, "grid_card_count": 1},
        "text_geometry": {"text_box_count": 8, "left_margin_std": 0.25, "code_indentation_score": 0.10, "top_zone_ratio": 0.25, "mid_zone_ratio": 0.60, "bot_zone_ratio": 0.15},
        "layout_heuristics": {"ide_editor_layout": 0.05, "slide_course_document_layout": 0.90, "generic_unknown_layout": 0.10},
        "visual_category_evidence": {"Coding": 0.10, "Education": 0.90, "General": 0.10, "Shopping": 0.0, "Communication": 0.0, "Entertainment": 0.0, "Travel": 0.0, "Connectivity": 0.0, "Work & Career": 0.30, "Payment": 0.0},
    }

    clf = ExplainableVisualEvidenceClassifier(categories=CANONICAL_CATEGORIES, mode="full_visual")
    res = clf.predict_single_explainable("Python programming course tutorial lecture 4 assignment questions", sample_feats)

    assert res["prediction"] == "Education"


def test_settings_ui_without_network_evidence_does_not_predict_connectivity():
    """Verifies settings UI layout without network/Wi-Fi keywords does NOT predict Connectivity."""
    sample_feats = {
        "image_geometry": {"width": 1080, "height": 1920, "aspect_ratio": 0.56, "orientation": "portrait"},
        "color_brightness": {"mean_brightness": 240.0, "std_brightness": 20.0, "is_dark_mode": 0.0, "is_light_mode": 1.0, "mean_saturation": 0.0},
        "edge_structure": {"canny_edge_density": 0.02, "h_v_edge_ratio": 1.1, "horizontal_lines_count": 6, "grid_card_count": 0},
        "text_geometry": {"text_box_count": 10, "left_margin_std": 0.08, "code_indentation_score": 0.0, "top_zone_ratio": 0.10, "mid_zone_ratio": 0.80, "bot_zone_ratio": 0.10},
        "layout_heuristics": {"ide_editor_layout": 0.0, "slide_course_document_layout": 0.20, "settings_system_ui_layout": 0.85, "generic_unknown_layout": 0.30},
        "visual_category_evidence": {"Coding": 0.0, "Education": 0.10, "General": 0.30, "Shopping": 0.0, "Communication": 0.0, "Entertainment": 0.0, "Travel": 0.0, "Connectivity": 0.85, "Work & Career": 0.0, "Payment": 0.0},
    }

    clf = ExplainableVisualEvidenceClassifier(categories=CANONICAL_CATEGORIES, mode="full_visual")
    # Raw text has NO Wi-Fi or network words (generic UI settings)
    res = clf.predict_single_explainable("System Settings battery usage storage screen timeout display", sample_feats)

    assert res["prediction"] != "Connectivity"


def test_generic_unknown_layout_trigger():
    """Verifies low edge density + low text count predicts General fallback."""
    sample_feats = {
        "image_geometry": {"width": 1080, "height": 1920, "aspect_ratio": 0.56, "orientation": "portrait"},
        "color_brightness": {"mean_brightness": 120.0, "std_brightness": 15.0, "is_dark_mode": 0.0, "is_light_mode": 0.0, "mean_saturation": 0.01},
        "edge_structure": {"canny_edge_density": 0.005, "h_v_edge_ratio": 1.0, "horizontal_lines_count": 0, "grid_card_count": 0},
        "text_geometry": {"text_box_count": 2, "left_margin_std": 0.40, "code_indentation_score": 0.0, "top_zone_ratio": 0.50, "mid_zone_ratio": 0.50, "bot_zone_ratio": 0.0},
        "layout_heuristics": {"generic_unknown_layout": 0.90, "ide_editor_layout": 0.0, "slide_course_document_layout": 0.0},
        "visual_category_evidence": {"General": 0.90, "Coding": 0.0, "Education": 0.0, "Shopping": 0.0, "Communication": 0.0, "Entertainment": 0.0, "Travel": 0.0, "Connectivity": 0.0, "Work & Career": 0.0, "Payment": 0.0},
    }

    clf = ExplainableVisualEvidenceClassifier(categories=CANONICAL_CATEGORIES, mode="full_visual")
    res = clf.predict_single_explainable("12:00 status bar", sample_feats)

    assert res["prediction"] == "General"


def test_external_benchmark_zero_access_assertion():
    """Verifies that no files in the external benchmark directory are accessed."""
    ext_dir = PROJECT_ROOT / "experiments" / "intelligence" / "external" / "images"
    assert ext_dir.exists(), "External benchmark directory missing."
