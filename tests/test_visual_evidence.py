"""
Unit tests and smoke tests for VisualCategoryEvidenceEngine.
"""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import torch

from src.visual.base_visual_encoder import (
    ModelLoadError,
    PreprocessingError,
    InferenceError,
)
from src.visual.tiny_clip_encoder import TinyCLIPEncoder
from src.visual.visual_evidence import (
    VisualCategoryEvidenceEngine,
    CANONICAL_CATEGORIES,
    DEFAULT_CATEGORY_PROMPTS,
)


class TestVisualCategoryEvidenceEngineMock(unittest.TestCase):
    """Unit tests for VisualCategoryEvidenceEngine using mocks."""

    def test_canonical_categories_spec(self):
        expected = [
            "Education",
            "Shopping",
            "Payment",
            "Travel",
            "Communication",
            "Entertainment",
            "Work & Career",
            "Connectivity",
            "General",
        ]
        self.assertEqual(CANONICAL_CATEGORIES, expected)
        self.assertNotIn("Coding", CANONICAL_CATEGORIES)

    def test_multiple_prompts_per_category(self):
        for cat in CANONICAL_CATEGORIES:
            self.assertIn(cat, DEFAULT_CATEGORY_PROMPTS)
            prompts = DEFAULT_CATEGORY_PROMPTS[cat]
            self.assertGreaterEqual(len(prompts), 3)
            for p in prompts:
                self.assertIsInstance(p, str)
                self.assertTrue(len(p) > 10)

    @patch("transformers.AutoProcessor.from_pretrained")
    @patch("transformers.CLIPModel.from_pretrained")
    def test_prompt_caching_and_evidence_extraction(self, mock_clip, mock_proc):
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_model_instance.config.projection_dim = 512

        # Return synthetic 512-dim embeddings for text and images
        def mock_text_features(**kwargs):
            text_list = kwargs.get("input_ids")
            num_texts = text_list.shape[0] if text_list is not None else 1
            # Create synthetic distinct vectors per prompt
            return torch.ones((num_texts, 512), dtype=torch.float32)

        def mock_image_features(**kwargs):
            return torch.ones((1, 512), dtype=torch.float32)

        mock_model_instance.get_text_features.side_effect = mock_text_features
        mock_model_instance.get_image_features.side_effect = mock_image_features
        mock_clip.return_value = mock_model_instance

        mock_proc_instance = MagicMock()
        mock_proc_instance.return_value = {
            "input_ids": torch.zeros((36, 10), dtype=torch.long),
            "pixel_values": torch.zeros((1, 3, 224, 224), dtype=torch.float32),
        }
        mock_proc.return_value = mock_proc_instance

        mock_encoder = TinyCLIPEncoder(model_name="mock_clip")
        engine = VisualCategoryEvidenceEngine(encoder=mock_encoder)

        # Test initial state
        self.assertFalse(engine._is_initialized)

        # Execute evidence extraction
        img = Image.new("RGB", (224, 224), color="blue")
        result = engine.extract_evidence(img)

        # Verify caching initialization
        self.assertTrue(engine._is_initialized)
        self.assertIsNotNone(engine._prompt_matrix)
        self.assertEqual(engine._prompt_matrix.shape[1], 512)

        # Verify output dictionary schema
        self.assertIn("top_category", result)
        self.assertIn(result["top_category"], CANONICAL_CATEGORIES)
        self.assertIn("top_score", result)
        self.assertIn("ranked_candidates", result)
        self.assertEqual(len(result["ranked_candidates"]), 9)

        # Check raw and normalized scores
        self.assertEqual(len(result["category_raw_scores"]), 9)
        self.assertEqual(len(result["category_norm_scores"]), 9)

        # Check prompt evidence breakdown
        self.assertEqual(len(result["prompt_evidence"]), 9)
        for cat in CANONICAL_CATEGORIES:
            ev = result["prompt_evidence"][cat]
            self.assertIn("best_prompt", ev)
            self.assertIn("best_similarity", ev)
            self.assertIsInstance(ev["all_prompt_similarities"], list)

        # Verify finiteness
        for val in result["category_raw_scores"].values():
            self.assertTrue(np.isfinite(val))
        for val in result["category_norm_scores"].values():
            self.assertTrue(np.isfinite(val))

    def test_invalid_image_handling(self):
        engine = VisualCategoryEvidenceEngine()
        non_existent_path = Path(tempfile.gettempdir()) / "non_existent_ss_999.png"
        with self.assertRaises(PreprocessingError):
            engine.extract_evidence(non_existent_path)


class TestVisualCategoryEvidenceEngineSmoke(unittest.TestCase):
    """Real-weights end-to-end CPU smoke tests for VisualCategoryEvidenceEngine."""

    def test_real_weights_evidence_extraction(self):
        encoder = TinyCLIPEncoder(model_name="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", device="cpu")
        engine = VisualCategoryEvidenceEngine(encoder=encoder)

        img = Image.new("RGB", (224, 224), color="yellow")

        # Execute extraction twice to verify determinism and prompt caching
        res1 = engine.extract_evidence(img)
        res2 = engine.extract_evidence(img)

        # Schema verification
        self.assertIn(res1["top_category"], CANONICAL_CATEGORIES)
        self.assertEqual(len(res1["ranked_candidates"]), 9)

        # Determinism check
        self.assertEqual(res1["top_category"], res2["top_category"])
        self.assertAlmostEqual(res1["top_score"], res2["top_score"], places=5)

        # Verify all scores are finite
        for cat, score in res1["category_raw_scores"].items():
            self.assertTrue(np.isfinite(score), f"Non-finite raw score for category {cat}")
        for cat, score in res1["category_norm_scores"].items():
            self.assertTrue(np.isfinite(score), f"Non-finite normalized score for category {cat}")

        # Clean up
        encoder.unload_model()


if __name__ == "__main__":
    unittest.main()
