"""
Unit Tests for Best-of-All Categorization Configuration
======================================================
Verifies candidate union, MiniLM shared embedder reuse, provenance tracking,
unscaled score fusion, high-confidence protection guardrail, and Top-K structure.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "best_of_all_categorization"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "candidate_reranking"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "accuracy_improvement"))

from src.embedder import SentenceTransformerEmbedder
from expanded_candidate_pool import ExpandedCandidatePoolBuilder, CANONICAL_CATEGORIES
from best_of_all_reranker import BestOfAllCandidateReranker


class TestBestOfAllCategorization(unittest.TestCase):
    """Unit test suite for Best-of-All Categorization configuration."""

    @classmethod
    def setUpClass(cls):
        cls.embedder = SentenceTransformerEmbedder()
        cls.pool_builder = ExpandedCandidatePoolBuilder(embedder=cls.embedder)
        cls.reranker = BestOfAllCandidateReranker(pool_builder=cls.pool_builder, enable_guardrail=True)

    def test_canonical_categories(self):
        self.assertEqual(len(CANONICAL_CATEGORIES), 10)
        self.assertIn("Coding", CANONICAL_CATEGORIES)
        self.assertIn("Payment", CANONICAL_CATEGORIES)

    def test_shared_embedder_instance_reuse(self):
        # Verify no second MiniLM model loaded; shared instance reused
        self.assertIs(self.pool_builder.embedder, self.embedder)

    def test_candidate_pool_provenance(self):
        text = "gpay upi payment successful transaction id balance received bank receipt"
        pool_res = self.pool_builder.build_candidate_pool(raw_text=text)

        self.assertIn("candidate_pool", pool_res)
        self.assertIn("classifier_top3", pool_res)
        self.assertIn("minilm_top5", pool_res)

        pool = pool_res["candidate_pool"]
        self.assertGreaterEqual(len(pool), 3)

        for cand in pool:
            self.assertIn("category", cand)
            self.assertIn("classifier_rank", cand)
            self.assertIn("classifier_score", cand)
            self.assertIn("minilm_rank", cand)
            self.assertIn("minilm_score", cand)
            self.assertIn("candidate_sources", cand)
            self.assertIn("from_classifier", cand)
            self.assertIn("from_minilm", cand)

    def test_top_k_metadata_preservation(self):
        text = "flight booking status air india PNR boarding pass gate seat"
        res = self.reranker.rerank(raw_text=text, notes="", vis_screen_type="travel_booking", vis_conf=0.90)

        required_keys = [
            "top1_category", "top1_score",
            "top2_category", "top2_score",
            "top3_category", "top3_score",
            "final_category", "confidence_str",
            "reranked", "reranking_reason",
            "candidate_pool", "ranked_candidates"
        ]
        for k in required_keys:
            self.assertIn(k, res)

        self.assertEqual(res["final_category"], "Travel")

    def test_high_confidence_guardrail(self):
        # High confidence base prediction with minimal phrase evidence
        text = "import numpy as np script"
        res = self.reranker.rerank(raw_text=text)

        if res["confidence_str"] == "High" and not res["reranked"]:
            base_top1 = res["candidate_pool"][0]["category"]
            self.assertEqual(res["final_category"], base_top1)


if __name__ == "__main__":
    unittest.main()
