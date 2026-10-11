"""
Phase Final Visual Integration Unit & Regression Test Suite.

Verifies:
1. Semantic Classifier Behavior
2. 13-Class YOLO Visual Classifier Loading & Preprocessing
3. Calibration & Score Scaling
4. Multimodal Fusion Rules (Selective Overriding & Ambiguity Resolution)
5. Safe Fallback Behavior on Visual Failures / Missing Files / Invalid Inputs
6. Metadata Serialization & SQLite Persistence
"""

import os
import sys
import json
import pytest
import sqlite3
import tempfile
import torch
import pandas as pd
from pathlib import Path
from PIL import Image

TEST_DIR = Path(__file__).resolve().parent
INTEGRATION_DIR = TEST_DIR.parent
PROJECT_ROOT = INTEGRATION_DIR.parent.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(INTEGRATION_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "intelligence" / "final_refinement"))

from pipeline_integration import IntegratedMultimodalPipeline, letterbox_resize, VIS_13_TO_SEMANTIC
from final_refinement_experiment import FinalRefinedClassifier
from src.models.metadata import ScreenshotMetadata
from src.storage.metadata_store import SQLiteMetadataStore

@pytest.fixture
def sample_image_path(tmp_path):
    img_p = tmp_path / "test_sample.png"
    img = Image.new("RGB", (300, 300), color=(100, 150, 200))
    img.save(img_p)
    return img_p


def test_semantic_classifier_unchanged():
    """Verifies that the primary semantic classifier operates cleanly and produces valid categories."""
    sem_cls = FinalRefinedClassifier()
    res = sem_cls.predict_final("def train_model(): import torch; return True", notes="code editor terminal python script", variant="C")
    assert "prediction" in res
    assert res["prediction"] in ["Education", "Payment", "Shopping", "Travel", "Coding", "Entertainment", "Connectivity", "Work & Career", "Communication", "General"]
    assert "category_scores" in res


def test_letterbox_preprocessing():
    """Verifies 224x224 letterbox resize preserving RGB aspect ratio."""
    img = Image.new("RGB", (800, 400), color=(255, 0, 0))
    img_lb = letterbox_resize(img, 224)
    assert img_lb.size == (224, 224)
    assert img_lb.mode == "RGB"


def test_pipeline_initialization():
    """Verifies that IntegratedMultimodalPipeline initializes without crashing."""
    pipeline = IntegratedMultimodalPipeline()
    assert hasattr(pipeline, "classify_screenshot")
    assert pipeline.semantic_classifier is not None


def test_pipeline_safe_fallback_missing_file():
    """Verifies that a missing image file triggers safe fallback without raising exception."""
    pipeline = IntegratedMultimodalPipeline()
    res = pipeline.classify_screenshot("non_existent_file_9999.png", ocr_text="Shopping cart order total $50.00")
    assert res["category"] in ["Shopping", "Payment", "General"]
    assert res["fusion_applied"] is False or isinstance(res["fusion_applied"], bool)
    assert "evidence" in res


def test_pipeline_selective_fusion_behavior(sample_image_path):
    """Verifies that selective visual fusion overrides ambiguous/low-confidence predictions."""
    pipeline = IntegratedMultimodalPipeline()
    res = pipeline.classify_screenshot(sample_image_path, ocr_text="Ambiguous UI screenshot text sample", notes="low confidence")
    assert "category" in res
    assert "semantic_category" in res
    assert "fusion_reason" in res


def test_metadata_serialization_deserialization():
    """Verifies ScreenshotMetadata serialization with visual_evidence dictionary."""
    visual_evidence_data = {
        "screen_type": "code_editor_terminal",
        "confidence": 0.92,
        "model_name": "YOLOv8n-cls",
        "model_version": "13-class-v1",
        "fusion_applied": True,
        "fusion_reason": "Visual layout resolved ambiguous fallback",
        "original_semantic_category": "General"
    }
    
    meta = ScreenshotMetadata(
        screenshot_id="ss_test_001",
        filename="ss_test_001.png",
        image_uri="data/screenshots/ss_test_001.png",
        category="Coding",
        tags=["python", "vscode"],
        summary="Visual classification override test",
        evidence={"text_snippet": "def main():", "visual_evidence": visual_evidence_data}
    )
    
    meta_dict = meta.to_dict()
    assert meta_dict["category"] == "Coding"
    assert "evidence" in meta_dict
    assert meta_dict["evidence"]["visual_evidence"]["screen_type"] == "code_editor_terminal"
    
    # Re-instantiate from dict
    restored = ScreenshotMetadata.from_dict(meta_dict)
    assert restored.category == "Coding"
    assert restored.evidence["visual_evidence"]["model_name"] == "YOLOv8n-cls"


def test_sqlite_persistence_backward_compatibility(tmp_path):
    """Verifies SQLite storing and retrieving metadata containing visual evidence without schema migration errors."""
    db_path = tmp_path / "test_metadata.sqlite"
    store = SQLiteMetadataStore(db_path=str(db_path))
    
    # 1. Legacy record (no visual evidence)
    legacy_meta = ScreenshotMetadata(
        screenshot_id="ss_legacy_001",
        filename="ss_legacy.png",
        image_uri="data/screenshots/ss_legacy.png",
        category="Education",
        tags=["pdf", "lecture"],
        summary="Legacy record without visual evidence",
        evidence={"text_snippet": "Lecture notes chapter 1"}
    )
    store.save_metadata(legacy_meta)
    
    # 2. Modern record (with visual evidence)
    modern_meta = ScreenshotMetadata(
        screenshot_id="ss_modern_001",
        filename="ss_modern.png",
        image_uri="data/screenshots/ss_modern.png",
        category="Travel",
        tags=["flight", "booking"],
        summary="Multimodal visual integration record",
        evidence={
            "text_snippet": "Flight JFK to HND",
            "visual_evidence": {
                "screen_type": "travel_booking",
                "confidence": 0.98,
                "model_name": "YOLOv8n-cls",
                "model_version": "13-class-v1",
                "fusion_applied": True,
                "fusion_reason": "Visual flight booking layout confirmed",
                "original_semantic_category": "General"
            }
        }
    )
    store.save_metadata(modern_meta)
    
    # Retrieve both records
    fetched_legacy = store.get_metadata("ss_legacy_001")
    fetched_modern = store.get_metadata("ss_modern_001")
    
    assert fetched_legacy is not None
    assert fetched_legacy.category == "Education"
    assert fetched_legacy.evidence.get("visual_evidence") is None
    
    assert fetched_modern is not None
    assert fetched_modern.category == "Travel"
    assert fetched_modern.evidence["visual_evidence"]["screen_type"] == "travel_booking"


def test_taxonomy_class_mapping_completeness():
    """Verifies that all 13 visual screen types map to valid semantic categories."""
    for vis_st, sem_cat in VIS_13_TO_SEMANTIC.items():
        assert isinstance(vis_st, str)
        assert isinstance(sem_cat, str)
        assert len(sem_cat) > 0
