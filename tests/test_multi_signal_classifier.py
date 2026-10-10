"""
Unit Tests for Multi-Signal Hierarchical Screenshot Classifier (Phase 4A Architecture)
===========================================================================================
Tests rule, semantic, embedding, and content-type signal scorers, score fusion,
Top-K candidate ranking, confidence & ambiguity detection, contextual case studies,
and metadata contract persistence.
"""

import unittest
from src.embedder import BaseEmbedder
from src.models.metadata import ScreenshotMetadata
from src.storage.metadata_store import SQLiteMetadataStore
from experiments.intelligence.classifier import (
    HybridScreenshotClassifier,
    DEFAULT_SIGNAL_WEIGHTS,
    CATEGORY_PROTOTYPES,
)
from experiments.intelligence.baseline_intelligence import analyze_screenshot


class MockSimpleEmbedder(BaseEmbedder):
    """Deterministic Mock Embedder returning token-frequency vectors for test execution."""

    def __init__(self, vector_dim: int = 128):
        self.vector_dim = vector_dim

    def embed(self, texts: list) -> list:
        results = []
        for t in texts:
            t_str = str(t).lower()
            vec = [0.0] * self.vector_dim
            for word in t_str.split():
                # Map word hash modulo vector_dim
                idx = abs(hash(word)) % self.vector_dim
                vec[idx] += 1.0
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            results.append(vec)
        return results


class TestMultiSignalClassifier(unittest.TestCase):
    """Verifies individual signal scoring, score fusion, top-k ranking, and ambiguity logic."""

    def setUp(self):
        self.embedder = MockSimpleEmbedder()
        self.classifier = HybridScreenshotClassifier(embedder=self.embedder)

    def test_rule_scoring(self):
        """Verify rule-based keyword and pattern scoring."""
        ocr_text = "UPI Payment successful Rs. 500 to Paytm User ref 123456"
        scores = self.classifier.compute_rule_scores(ocr_text)
        self.assertGreater(scores["Payment"], 0.5)
        self.assertEqual(max(scores, key=scores.get), "Payment")

    def test_semantic_scoring(self):
        """Verify semantic text description scoring."""
        ocr_text = "course syllabus curriculum textbook exam lecture assignment notes university college academic study learning"
        scores = self.classifier.compute_semantic_scores(ocr_text)
        self.assertGreater(scores["Education"], 0.0)

    def test_embedding_scoring(self):
        """Verify prototype embedding scoring."""
        ocr_text = "flight boarding pass airline ticket seat gate departure arrival journey travel booking pnr"
        scores = self.classifier.compute_embedding_scores(ocr_text)
        self.assertIn("Travel", scores)
        self.assertGreaterEqual(scores["Travel"], 0.0)

    def test_content_type_scoring(self):
        """Verify content-type taxonomy score matching."""
        scores = self.classifier.compute_content_type_scores("flight_ticket")
        self.assertEqual(scores["Travel"], 1.0)
        self.assertEqual(scores["Education"], 0.0)

        scores_unknown = self.classifier.compute_content_type_scores("unknown")
        self.assertTrue(all(v == 0.0 for v in scores_unknown.values()))

    def test_score_fusion_weights(self):
        """Verify explicit score fusion calculation with weights."""
        ocr_text = "Airtel Wi-Fi Password supersecret123"
        res = self.classifier.classify_multi_signal(ocr_text, content_type="wifi")
        self.assertEqual(res["selected_category"], "Connectivity")
        self.assertIn("candidates", res)
        self.assertGreaterEqual(len(res["candidates"]), 2)

    def test_top_k_ordering(self):
        """Verify top-k candidates are sorted in descending order of final_score."""
        ocr_text = "Order confirmation for black leather boots total Rs. 2499 delivery by Amazon"
        res = self.classifier.classify_multi_signal(ocr_text, content_type="product_page", top_k=5)
        candidates = res["candidates"]
        self.assertEqual(len(candidates), 5)
        scores = [c["final_score"] for c in candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_ambiguity_detection_for_close_candidates(self):
        """Verify ambiguity flag is triggered when top candidate score margin is below threshold."""
        ocr_text = "Generic item summary list text 123"
        res = self.classifier.classify_multi_signal(ocr_text, ambiguity_threshold=0.80)
        self.assertTrue(res["ambiguous"])

    def test_context_case_shopping_vs_payment(self):
        """Verify distinction between Shopping order with price and pure Payment receipt."""
        shopping_text = "Order Summary Menu Item Combo Rs. 499 Serves 2 Add to Cart Zomato"
        res_shop = self.classifier.classify_multi_signal(shopping_text, content_type="shopping_order")
        self.assertEqual(res_shop["selected_category"], "Shopping")

        payment_text = "Paid successfully Rs 499 to Zomato UPI Ref 9876543210 Bank Account"
        res_pay = self.classifier.classify_multi_signal(payment_text, content_type="payment_receipt")
        self.assertEqual(res_pay["selected_category"], "Payment")

    def test_context_case_coding_vs_career(self):
        """Verify distinction between Coding source code and Work & Career hackathon announcement."""
        coding_text = "const express = require('express'); const router = express.Router(); async function handle(req, res)"
        res_code = self.classifier.classify_multi_signal(coding_text, content_type="code_editor")
        self.assertEqual(res_code["selected_category"], "Coding")

        career_text = "OpenAI Hackathon 2026 Registration Open Prize Pool $50000 Register Team Deadline Sep 30"
        res_career = self.classifier.classify_multi_signal(career_text, content_type="job_posting")
        self.assertEqual(res_career["selected_category"], "Work & Career")


class TestMultiSignalMetadataIntegration(unittest.TestCase):
    """Verifies baseline_intelligence coordinator integration and metadata serialization."""

    def test_analyze_screenshot_multi_signal_output(self):
        """Verify analyze_screenshot populates multi-signal candidates, scores, confidence, and margin."""
        res = analyze_screenshot(
            "dummy.jpg",
            provided_ocr_text="Airtel_5G Wi-Fi Password: supersecret123",
            embedder=MockSimpleEmbedder(),
        )
        meta = res["metadata"]
        self.assertEqual(meta["category"], "Connectivity")
        self.assertIn("category_candidates", meta)
        self.assertGreater(len(meta["category_candidates"]), 0)
        self.assertIn("classification_confidence", meta)
        self.assertIn("classification_margin", meta)
        self.assertIn("classification_ambiguous", meta)

    def test_metadata_sqlite_roundtrip(self):
        """Verify multi-signal fields persist through SQLite store without loss or schema errors."""
        import tempfile
        import os
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_metadata.sqlite")
        try:
            store = SQLiteMetadataStore(db_path=db_path)
            meta = ScreenshotMetadata(
                screenshot_id="test_ms_001",
                filename="test_ms_001.jpg",
                image_uri="/path/to/test_ms_001.jpg",
                category="Shopping",
                content_type="product_page",
                category_candidates=[{"category": "Shopping", "final_score": 0.85}],
                category_scores={"Shopping": 0.85, "Payment": 0.20},
                classification_confidence=0.85,
                classification_ambiguous=False,
                classification_margin=0.65,
            )

            store.save_metadata(meta)
            retrieved = store.get_metadata("test_ms_001")
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.category, "Shopping")
            self.assertEqual(retrieved.classification_confidence, 0.85)
            self.assertEqual(retrieved.classification_ambiguous, False)
            self.assertEqual(retrieved.classification_margin, 0.65)
            self.assertEqual(len(retrieved.category_candidates), 1)
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)


if __name__ == "__main__":
    unittest.main()
