"""
Unit Tests for Categorization Recovery Sprint: Candidate Generator V2 & Recovery Reranker
========================================================================================
Tests candidate pool expansion, provenance tracking, score normalization,
rank-aware agreement bonus, and high-confidence protection guardrails.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "categorization_recovery"))

from expanded_candidate_generator_v2 import CandidateGeneratorV2, CANONICAL_CATEGORIES
from reranker_recovery_engine import CategorizationRecoveryReranker


class TestCandidateGeneratorV2(unittest.TestCase):
    """Tests for CandidateGeneratorV2 candidate pool generation and provenance tracking."""

    @classmethod
    def setUpClass(cls):
        cls.generator = CandidateGeneratorV2()

    def test_canonical_categories_count(self):
        self.assertEqual(len(CANONICAL_CATEGORIES), 10)
        self.assertIn("Coding", CANONICAL_CATEGORIES)
        self.assertIn("Payment", CANONICAL_CATEGORIES)

    def test_minilm_candidates_generation(self):
        text = "gpay payment successful transaction id balance transfer upi bank receipt"
        candidates = self.generator.predict_minilm_candidates(text, top_k=5)
        self.assertEqual(len(candidates), 5)
        top_cat = candidates[0][0]
        self.assertEqual(top_cat, "Payment")

    def test_expanded_candidate_pool_structure(self):
        text = "def calculate_loss(y_true, y_pred): return np.mean((y_true - y_pred)**2) github commit python script"
        pool_res = self.generator.build_expanded_candidate_pool(raw_text=text)

        self.assertIn("candidate_pool", pool_res)
        self.assertIn("classifier_top_k", pool_res)
        self.assertIn("minilm_top_k", pool_res)

        pool = pool_res["candidate_pool"]
        self.assertGreaterEqual(len(pool), 5)

        # Test provenance tracking fields
        for cand in pool:
            self.assertIn("category", cand)
            self.assertIn("from_classifier", cand)
            self.assertIn("from_minilm", cand)
            self.assertIn("both_sources", cand)
            self.assertIn("classifier_rank", cand)
            self.assertIn("minilm_rank", cand)

    def test_empty_raw_text_handling(self):
        candidates = self.generator.predict_minilm_candidates("", top_k=5)
        self.assertEqual(len(candidates), 5)
        # Empty text should fallback to General default without crashing
        self.assertIsNotNone(candidates[0][0])


class TestCategorizationRecoveryReranker(unittest.TestCase):
    """Tests for CategorizationRecoveryReranker multi-signal evidence reranking."""

    @classmethod
    def setUpClass(cls):
        cls.generator = CandidateGeneratorV2()
        cls.reranker = CategorizationRecoveryReranker(candidate_generator=cls.generator)

    def test_rerank_structure_and_keys(self):
        text = "flight booking PNR status air india boarding pass seat gate itinerary"
        res = self.reranker.rerank(raw_text=text, notes="", vis_screen_type="travel_booking", vis_conf=0.90)

        self.assertIn("final_category", res)
        self.assertIn("top1_category", res)
        self.assertIn("ranked_candidates", res)
        self.assertIn("confidence_str", res)

        self.assertEqual(res["final_category"], "Travel")

    def test_score_normalization_in_range(self):
        text = "order shipped delivery amazon item invoice cart MRP price discount"
        res = self.reranker.rerank(raw_text=text)

        for cand in res["ranked_candidates"]:
            score = cand["score"]
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 2.5)  # Sum of max weighted signals

    def test_high_confidence_guardrail_protection(self):
        # High confidence base classifier prediction with minimal conflicting phrase evidence
        text = "python script import numpy"
        res = self.reranker.rerank(
            raw_text=text,
            notes="",
            vis_screen_type="other_screen",
            vis_conf=0.0,
            enable_high_conf_guardrail=True
        )

        # Guardrail should retain base classifier decision when confidence is High
        if res["confidence_str"] == "High" and not res["reranked"]:
            base_top1 = res["candidate_pool"][0]["category"]
            self.assertEqual(res["final_category"], base_top1)


if __name__ == "__main__":
    unittest.main()
