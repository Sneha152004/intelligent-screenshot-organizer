"""
Unit tests and smoke tests for MultimodalScreenshotClassifier.
"""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import torch

from src.multimodal_classifier import MultimodalScreenshotClassifier
from src.visual.visual_evidence import CANONICAL_CATEGORIES


class TestMultimodalScreenshotClassifier(unittest.TestCase):
    """Unit tests for MultimodalScreenshotClassifier using mocks."""

    def setUp(self):
        # Create mock text classifier
        self.mock_text_classifier = MagicMock()
        self.mock_text_classifier.predict_final.return_value = {
            "prediction": "Payment",
            "confidence": "High",
            "margin": 0.35,
            "is_ambiguous": False,
            "is_sparse": False,
            "category_scores": {c: (0.85 if c == "Payment" else 0.10) for c in CANONICAL_CATEGORIES},
            "top_candidates": [
                {"category": "Payment", "score": 0.85},
                {"category": "Shopping", "score": 0.10},
                {"category": "General", "score": 0.10},
            ],
            "tags": ["Invoice"],
            "evidence_matches": ["Keyword match: 'payment successful'"],
        }

        # Create mock visual evidence engine
        self.mock_visual_engine = MagicMock()
        self.mock_visual_engine.extract_evidence.return_value = {
            "top_category": "Payment",
            "top_score": 0.32,
            "ranked_candidates": [("Payment", 0.32), ("Shopping", 0.20)],
            "category_raw_scores": {c: (0.32 if c == "Payment" else 0.15) for c in CANONICAL_CATEGORIES},
            "category_norm_scores": {c: (0.30 if c == "Payment" else 0.08) for c in CANONICAL_CATEGORIES},
            "prompt_evidence": {
                "Payment": {
                    "best_prompt": "a screenshot of a digital payment confirmation",
                    "best_similarity": 0.32,
                    "all_prompt_similarities": [0.32],
                }
            },
        }

    def test_text_only_mode_without_visual_initialization(self):
        classifier = MultimodalScreenshotClassifier(
            text_classifier=self.mock_text_classifier,
            enable_visual=False,
        )
        self.assertIsNone(classifier.visual_engine)

        result = classifier.predict(ocr_text="Payment of 100 rupees completed")

        self.assertEqual(result["prediction"], "Payment")
        self.assertTrue(result["visual_fallback_used"])
        self.assertEqual(result["text_weight"], 1.0)
        self.assertEqual(result["visual_weight"], 0.0)
        self.assertIn("text_prediction", result)
        self.assertEqual(result["text_prediction"], "Payment")
        self.mock_text_classifier.predict_final.assert_called_once()

    def test_successful_multimodal_fusion(self):
        # Create temporary valid image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            img = Image.new("RGB", (100, 100), color="blue")
            img.save(tmp_path)

        try:
            classifier = MultimodalScreenshotClassifier(
                text_classifier=self.mock_text_classifier,
                visual_engine=self.mock_visual_engine,
                enable_visual=True,
            )

            result = classifier.predict(
                ocr_text="Payment of 100 rupees completed",
                image_path=tmp_path,
            )

            self.assertFalse(result["visual_fallback_used"])
            self.assertEqual(result["prediction"], "Payment")
            self.assertIn("text_scores", result)
            self.assertIn("visual_scores", result)
            self.assertEqual(result["text_prediction"], "Payment")
            self.assertEqual(result["visual_prediction"], "Payment")
            self.assertGreater(result["text_weight"], 0.0)
            self.assertGreater(result["visual_weight"], 0.0)
            self.assertEqual(len(result["category_scores"]), 9)
            self.assertEqual(len(result["top_candidates"]), 3)

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_fallback_on_missing_image_file(self):
        classifier = MultimodalScreenshotClassifier(
            text_classifier=self.mock_text_classifier,
            visual_engine=self.mock_visual_engine,
            enable_visual=True,
        )

        non_existent_img = Path(tempfile.gettempdir()) / "non_existent_img_9999.png"
        result = classifier.predict(
            ocr_text="Payment of 100 rupees completed",
            image_path=non_existent_img,
        )

        self.assertTrue(result["visual_fallback_used"])
        self.assertEqual(result["prediction"], "Payment")
        self.assertEqual(result["text_weight"], 1.0)

    def test_fallback_on_visual_inference_exception(self):
        # Configure mock visual engine to raise an exception
        failing_visual_engine = MagicMock()
        failing_visual_engine.extract_evidence.side_effect = Exception("Model weight corrupt")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            img = Image.new("RGB", (100, 100))
            img.save(tmp_path)

        try:
            classifier = MultimodalScreenshotClassifier(
                text_classifier=self.mock_text_classifier,
                visual_engine=failing_visual_engine,
                enable_visual=True,
            )

            result = classifier.predict(
                ocr_text="Payment of 100 rupees completed",
                image_path=tmp_path,
            )

            self.assertTrue(result["visual_fallback_used"])
            self.assertEqual(result["prediction"], "Payment")

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_sparse_text_dynamic_weighting(self):
        # When text is sparse (e.g. 1 word), visual weight should dominate (w_visual = 0.85)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            img = Image.new("RGB", (100, 100))
            img.save(tmp_path)

        try:
            classifier = MultimodalScreenshotClassifier(
                text_classifier=self.mock_text_classifier,
                visual_engine=self.mock_visual_engine,
                enable_visual=True,
            )

            result = classifier.predict(
                ocr_text="Sparse",  # < 5 tokens
                image_path=tmp_path,
            )

            self.assertFalse(result["visual_fallback_used"])
            self.assertEqual(result["text_weight"], 0.15)
            self.assertEqual(result["visual_weight"], 0.85)

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_original_classifier_preserved(self):
        # Verify text classifier predict_final is called with original parameters
        classifier = MultimodalScreenshotClassifier(
            text_classifier=self.mock_text_classifier,
            enable_visual=False,
        )
        classifier.predict(ocr_text="Sample text", notes="test notes", variant="C")

        self.mock_text_classifier.predict_final.assert_called_with("Sample text", notes="test notes", variant="C")


class TestMultimodalScreenshotClassifierSmoke(unittest.TestCase):
    """Real-weights end-to-end smoke tests for MultimodalScreenshotClassifier."""

    def test_real_weights_multimodal_classification(self):
        from experiments.intelligence.final_refinement.final_refinement_experiment import (
            FinalRefinedClassifier,
        )
        from src.visual.tiny_clip_encoder import TinyCLIPEncoder
        from src.visual.visual_evidence import VisualCategoryEvidenceEngine

        text_clf = FinalRefinedClassifier()
        clip_encoder = TinyCLIPEncoder(model_name="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", device="cpu")
        vis_engine = VisualCategoryEvidenceEngine(encoder=clip_encoder)

        classifier = MultimodalScreenshotClassifier(
            text_classifier=text_clf,
            visual_engine=vis_engine,
            enable_visual=True,
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            img = Image.new("RGB", (224, 224), color="green")
            img.save(tmp_path)

        try:
            result = classifier.predict(
                ocr_text="Order confirmed! Payment of $49.99 successful via PayPal.",
                image_path=tmp_path,
            )

            self.assertFalse(result["visual_fallback_used"])
            self.assertIn(result["prediction"], CANONICAL_CATEGORIES)
            self.assertIn("category_scores", result)
            self.assertEqual(len(result["category_scores"]), 9)
            self.assertIn("top_candidates", result)
            self.assertEqual(len(result["top_candidates"]), 3)

        finally:
            Path(tmp_path).unlink(missing_ok=True)
            clip_encoder.unload_model()


if __name__ == "__main__":
    unittest.main()
