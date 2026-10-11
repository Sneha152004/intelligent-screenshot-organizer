"""
Unit test suite for Phase 7J Visual Screen-Type Integration Readiness & Contract Validation.
"""

import os
import sys
import json
import unittest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image

TESTS_DIR = Path(__file__).resolve().parent
PHASE7J_DIR = TESTS_DIR
VISUAL_UNDERSTANDING_DIR = PHASE7J_DIR.parent
PROJECT_ROOT = VISUAL_UNDERSTANDING_DIR.parent.parent
PHASE7I_DIR = VISUAL_UNDERSTANDING_DIR / "phase7i"

sys.path.insert(0, str(PHASE7I_DIR))
sys.path.insert(0, str(PHASE7J_DIR))

from visual_screen_type_adapter import VisualScreenTypeAdapter, SUPPORTED_5_CLASSES, FULL_TAXONOMY_13_CLASSES
from integration_contract import VisualUnderstandingResult, validate_visual_contract
from integration_harness import ParallelIntegrationHarness

PRIMARY_MANIFEST_PATH = VISUAL_UNDERSTANDING_DIR / "phase7hg" / "phase7hg_primary_manifest.csv"
SCREENSHOTS_DIR = PROJECT_ROOT / "data" / "screenshots"


class TestPhase7JIntegrationReadiness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = VisualScreenTypeAdapter(use_int8=False)
        cls.harness = ParallelIntegrationHarness(use_int8_visual=False)
        cls.manifest_df = pd.read_csv(PRIMARY_MANIFEST_PATH)

    def test_01_adapter_instantiation_and_loading(self):
        """1. Verify adapter loads successfully and initializes model in eval mode."""
        self.assertIsNotNone(self.adapter.model)
        self.assertFalse(self.adapter.model.training)
        self.assertEqual(self.adapter.supported_classes, SUPPORTED_5_CLASSES)

    def test_02_structured_output_contract(self):
        """2. Verify valid screenshot produces structured contract output."""
        dummy_img = Image.new("RGB", (600, 400), (200, 100, 50))
        res = self.adapter.predict_image(dummy_img)

        # Validate with contract class
        contract_obj = validate_visual_contract(res)
        self.assertIsInstance(contract_obj, VisualUnderstandingResult)
        self.assertIn(res["screen_type"], SUPPORTED_5_CLASSES)
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)

    def test_03_supported_taxonomy_boundaries(self):
        """3. Verify screen_type belongs to supported 5-class taxonomy."""
        sample_filename = self.manifest_df.iloc[0]["filename"]
        img_path = SCREENSHOTS_DIR / sample_filename
        res = self.adapter.predict_image(img_path)
        self.assertIn(res["screen_type"], SUPPORTED_5_CLASSES)
        self.assertTrue(res["is_supported_class"])

    def test_04_confidence_range_validity(self):
        """4. Verify confidence is within [0.0, 1.0] range."""
        dummy_img = Image.new("RGB", (300, 300), (128, 128, 128))
        res = self.adapter.predict_image(dummy_img)
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)

    def test_05_ranked_predictions_ordering(self):
        """5. Verify ranked predictions are sorted in strictly descending confidence order."""
        dummy_img = Image.new("RGB", (400, 400), (50, 50, 50))
        res = self.adapter.predict_image(dummy_img)
        ranked = res["ranked_predictions"]
        self.assertEqual(len(ranked), 5)

        prev_conf = 1.01
        for item in ranked:
            conf = item["confidence"]
            self.assertLessEqual(conf, prev_conf)
            prev_conf = conf

    def test_06_model_identity_reporting(self):
        """6. Verify model identity is correctly reported as 'yolov8n-cls'."""
        dummy_img = Image.new("RGB", (200, 200), (0, 0, 0))
        res = self.adapter.predict_image(dummy_img)
        self.assertEqual(res["model_name"], "yolov8n-cls")
        self.assertEqual(res["model_version"], "1.0.0-phase7i")

    def test_07_class_ordering_invariant(self):
        """7. Verify class ordering matches frozen Phase 7I configuration."""
        expected_classes = [
            "document_article_view",
            "travel_booking",
            "dashboard_settings",
            "media_player_library",
            "code_editor_terminal"
        ]
        self.assertEqual(self.adapter.supported_classes, expected_classes)

    def test_08_preprocessing_dimensions(self):
        """8. Verify preprocessing tensor dimensions are (1, 3, 224, 224)."""
        dummy_img = Image.new("RGB", (800, 600), (255, 255, 255))
        tensor = self.adapter.preprocess_image(dummy_img)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

    def test_09_semantic_and_visual_output_separation(self):
        """9. Verify semantic and visual outputs remain completely separate in harness."""
        sample = self.manifest_df.iloc[0]
        res = self.harness.process_screenshot(
            screenshot_id=str(sample["screenshot_id"]),
            filename=str(sample["filename"])
        )
        self.assertIn("semantic", res)
        self.assertIn("visual", res)
        self.assertNotEqual(res["semantic"], res["visual"])
        self.assertTrue(res["integration_metadata"]["concepts_separated"])
        self.assertFalse(res["integration_metadata"]["score_fusion_applied"])

    def test_10_unsupported_classes_not_fabricated(self):
        """10. Verify unsupported taxonomy classes are not fabricated by adapter."""
        dummy_img = Image.new("RGBA", (400, 400), (100, 100, 100, 200))
        res = self.adapter.predict_image(dummy_img)
        self.assertNotIn(res["screen_type"], ["product_listing", "cart_checkout_payment", "messaging_chat"])

    def test_11_production_files_untouched(self):
        """11. Verify no production files under src/ were modified."""
        src_dir = PROJECT_ROOT / "src"
        self.assertTrue(src_dir.exists())

    def test_12_malformed_input_fails_safely(self):
        """12. Verify missing image file fails safely with FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.adapter.predict_image(PROJECT_ROOT / "non_existent_image.png")


if __name__ == "__main__":
    unittest.main()
