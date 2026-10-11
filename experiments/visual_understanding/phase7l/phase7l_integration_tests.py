"""
Phase 7L: Multimodal Pipeline Schema Integration Helper & Tests.

Demonstrates non-destructive metadata serialization, SQLite persistence,
backward compatibility, and selective fusion metadata round-tripping.
"""

import os
import sys
import json
import sqlite3
import unittest
from pathlib import Path
from typing import Dict, Any, Optional

PHASE7L_DIR = Path(__file__).resolve().parent
VISUAL_UNDERSTANDING_DIR = PHASE7L_DIR.parent
PROJECT_ROOT = VISUAL_UNDERSTANDING_DIR.parent.parent
PHASE7I_DIR = VISUAL_UNDERSTANDING_DIR / "phase7i"
PHASE7J_DIR = VISUAL_UNDERSTANDING_DIR / "phase7j"
PHASE7K_DIR = VISUAL_UNDERSTANDING_DIR / "phase7k"

sys.path.insert(0, str(PHASE7I_DIR))
sys.path.insert(0, str(PHASE7J_DIR))
sys.path.insert(0, str(PHASE7K_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.metadata import ScreenshotMetadata
from storage.metadata_store import SQLiteMetadataStore
from phase7k_multimodal_harness import MultimodalFusionHarness


def create_sample_multimodal_metadata(
    screenshot_id: str = "test_001.jpg",
    filename: str = "test_001.jpg",
    category: str = "Coding",
    visual_screen_type: str = "code_editor_terminal",
    visual_confidence: float = 0.3707,
    fusion_applied: bool = True
) -> ScreenshotMetadata:
    """Helper creating ScreenshotMetadata with structured visual_evidence inside evidence dict."""
    visual_evidence_dict = {
        "screen_type": visual_screen_type,
        "confidence": round(float(visual_confidence), 4),
        "model_name": "yolov8n-cls",
        "model_version": "1.0.0-phase7i",
        "is_supported_class": True,
        "fusion_applied": fusion_applied,
        "fusion_strategy": "Strategy E (Selective Overriding)",
        "fusion_reason": "Visual layout 'code_editor_terminal' overrode low-confidence/ambiguous semantic fallback 'General'",
        "original_semantic_category": "General"
    }

    return ScreenshotMetadata(
        screenshot_id=screenshot_id,
        filename=filename,
        image_uri=f"data/screenshots/{filename}",
        category=category,
        content_type="code_editor",
        evidence={
            "visual_evidence": visual_evidence_dict,
            "extracted_text_snippet": "import os, sys"
        },
        confidence_scores={"category": 0.88, "visual": visual_confidence}
    )


class TestPhase7LIntegrationHelper(unittest.TestCase):
    def setUp(self):
        self.test_db = PHASE7L_DIR / "temp_test_metadata.sqlite"
        if self.test_db.exists():
            os.remove(self.test_db)
        self.store = SQLiteMetadataStore(db_path=str(self.test_db))

    def tearDown(self):
        if self.test_db.exists():
            os.remove(self.test_db)

    def test_metadata_roundtrip_with_visual_evidence(self):
        """1. Verify metadata roundtrip: Save -> Load -> Validate visual evidence fields."""
        meta = create_sample_multimodal_metadata("test_ss_001", "ss_045.jpg", "Coding", "code_editor_terminal", 0.3707, True)
        self.store.save_metadata(meta)

        loaded = self.store.get_metadata("test_ss_001")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.category, "Coding")
        self.assertIn("visual_evidence", loaded.evidence)

        vis = loaded.evidence["visual_evidence"]
        self.assertEqual(vis["screen_type"], "code_editor_terminal")
        self.assertEqual(vis["model_name"], "yolov8n-cls")
        self.assertTrue(vis["fusion_applied"])

    def test_backward_compatibility_old_metadata(self):
        """2. Verify legacy metadata without visual_evidence loads seamlessly."""
        legacy_meta = ScreenshotMetadata(
            screenshot_id="legacy_001",
            filename="legacy_001.jpg",
            image_uri="data/screenshots/legacy_001.jpg",
            category="Shopping",
            evidence={"legacy_field": "some_value"}
        )
        self.store.save_metadata(legacy_meta)

        loaded = self.store.get_metadata("legacy_001")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.category, "Shopping")
        self.assertNotIn("visual_evidence", loaded.evidence)


if __name__ == "__main__":
    unittest.main()
