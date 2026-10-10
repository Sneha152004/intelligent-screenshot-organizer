"""
Unit Tests for Phase 3G Lightweight Screenshot Intelligence Baseline
======================================================================
Tests rule extraction, hybrid classification, intent inference, pipeline coordination,
metadata schema compliance, and evaluation harness functionality.
"""

import json
from pathlib import Path
import unittest

from src.embedder import BaseEmbedder
from experiments.intelligence.rules import (
    extract_financial_info,
    extract_dates,
    extract_contacts,
    extract_urls,
    detect_sensitive_info,
    extract_entities,
    extract_action_items,
    generate_deterministic_summary,
)
from experiments.intelligence.classifier import (
    HybridScreenshotClassifier,
    cosine_similarity,
)
from experiments.intelligence.baseline_intelligence import (
    analyze_screenshot,
    generate_tags,
)
from experiments.intelligence.evaluation import (
    evaluate_baseline,
    calculate_set_metrics,
)


class DummyMockEmbedder(BaseEmbedder):
    """Deterministic dummy embedder for fast unit testing without model downloads."""

    def embed(self, texts):
        results = []
        for t in texts:
            t_lower = t.lower()
            # Return distinct fixed 4D vectors based on content keywords
            if "wifi" in t_lower or "connectivity" in t_lower or "password" in t_lower:
                vec = [1.0, 0.0, 0.0, 0.0]
            elif "upi" in t_lower or "payment" in t_lower or "paid" in t_lower:
                vec = [0.0, 1.0, 0.0, 0.0]
            elif "syllabus" in t_lower or "education" in t_lower:
                vec = [0.0, 0.0, 1.0, 0.0]
            else:
                vec = [0.1, 0.1, 0.1, 0.1]
            results.append(vec)
        return results


class TestRulesExtraction(unittest.TestCase):
    """Tests pattern and rule extraction functions in rules.py."""

    def test_extract_financial_info(self):
        text = "Paid successfully ₹689.00 to merchant via UPI sinhasneha158-2@okicici. Txn ID: pay_SdleqofNvlmOZg"
        fin = extract_financial_info(text)
        self.assertIn("₹689.00", fin["amounts"])
        self.assertTrue(fin["has_payment_keywords"])
        self.assertIn("sinhasneha158-2@okicici", fin["upi_ids"])
        self.assertIn("pay_SdleqofNvlmOZg", fin["transaction_ids"])

    def test_extract_dates(self):
        text = "Event scheduled on 15th Apr, 2026 17:55 PM IST"
        dates = extract_dates(text)
        self.assertGreater(len(dates), 0)
        self.assertEqual(dates[0]["text"], "15th Apr, 2026 17:55 PM IST")

    def test_extract_contacts(self):
        text = "Contact support at user@example.com or call +91 9881732186"
        contacts = extract_contacts(text)
        self.assertIn("user@example.com", contacts["emails"])
        self.assertEqual(len(contacts["phones"]), 1)

    def test_extract_urls(self):
        text = "Visit https://openai.com or kiit.ac.in for details"
        urls = extract_urls(text)
        self.assertIn("https://openai.com", urls)
        self.assertIn("kiit.ac.in", urls)

    def test_detect_sensitive_info(self):
        text = "Airtel_5G Wi-Fi Password: supersecret123"
        sens = detect_sensitive_info(text)
        self.assertTrue(sens["contains_password"])

    def test_extract_entities(self):
        text = "IndiGo Flight Boarding Pass to Guwahati"
        entities = extract_entities(text)
        ent_texts = [e["text"] for e in entities]
        self.assertIn("IndiGo", ent_texts)
        self.assertIn("Guwahati", ent_texts)

    def test_extract_action_items(self):
        text = "IndiGo Flight Boarding Pass"
        actions = extract_action_items(text)
        self.assertGreater(len(actions), 0)
        self.assertIn("boarding gate", actions[0]["text"].lower())

    def test_generate_deterministic_summary(self):
        entities = [{"text": "Airtel", "type": "network_name"}]
        summary = generate_deterministic_summary("Connectivity", "Wi-Fi password supersecret", entities)
        self.assertIn("Wi-Fi network credentials", summary)


class TestHybridClassifier(unittest.TestCase):
    """Tests classification, cosine similarity, and intent inference in classifier.py."""

    def setUp(self):
        self.dummy_embedder = DummyMockEmbedder()
        self.classifier = HybridScreenshotClassifier(embedder=self.dummy_embedder)

    def test_cosine_similarity(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0)
        self.assertAlmostEqual(cosine_similarity(v1, v3), 0.0)

    def test_rule_scores(self):
        text = "Paid successfully ₹689.00 via UPI ID sinhasneha158-2@okicici"
        scores = self.classifier.compute_rule_scores(text)
        self.assertGreater(scores["Payment"], 0.5)

    def test_classify_payment(self):
        text = "Paid successfully ₹689.00 via UPI ID sinhasneha158-2@okicici"
        category, confidence, scores = self.classifier.classify(text)
        self.assertEqual(category, "Payment")
        self.assertGreater(confidence, 0.0)

    def test_classify_wifi(self):
        text = "Airtel_5G Wi-Fi Password: supersecret123"
        category, confidence, scores = self.classifier.classify(text)
        self.assertEqual(category, "Connectivity")

    def test_classify_empty(self):
        category, confidence, scores = self.classifier.classify("")
        self.assertEqual(category, "General")
        self.assertEqual(confidence, 0.0)

    def test_infer_intent(self):
        intent = self.classifier.infer_intent("Payment", "Paid ₹689")
        self.assertIn("UPI transaction receipt", intent)


class TestBaselineIntelligencePipeline(unittest.TestCase):
    """Tests baseline_intelligence.py coordinator, output schema, and latency breakdown."""

    def setUp(self):
        self.dummy_embedder = DummyMockEmbedder()
        self.classifier = HybridScreenshotClassifier(embedder=self.dummy_embedder)

    def test_analyze_screenshot_contract(self):
        text = "KIIT University Deep Learning Syllabus Unit I"
        res = analyze_screenshot(
            "ss_003.jpg",
            provided_ocr_text=text,
            embedder=self.dummy_embedder,
            classifier=self.classifier,
        )

        metadata = res["metadata"]
        timing = res["timing"]

        # Validate ScreenshotMetadata contract fields
        required_keys = {
            "screenshot_id", "filename", "image_uri", "category", "ocr_available",
            "created_at", "indexed_at", "tags", "intent", "summary", "importance",
            "entities", "dates", "action_items", "sensitive_info"
        }
        self.assertTrue(required_keys.issubset(metadata.keys()))

        # REQUIREMENT 15: importance must be explicitly None
        self.assertIsNone(metadata["importance"])

        self.assertEqual(metadata["category"], "Education")
        self.assertTrue(metadata["ocr_available"])

        # Validate timing breakdown keys
        timing_keys = {
            "ocr_latency_ms", "rules_latency_ms", "embedding_latency_ms",
            "classification_latency_ms", "total_latency_ms", "classification_confidence"
        }
        self.assertTrue(timing_keys.issubset(timing.keys()))

    def test_generate_tags(self):
        tags = generate_tags("Connectivity", "Airtel_5G Wi-Fi Password: secret", [], {"contains_password": True})
        self.assertIn("connectivity", tags)
        self.assertIn("wifi", tags)


class TestIntelligenceEvaluation(unittest.TestCase):
    """Tests evaluation harness and report generation."""

    def test_calculate_set_metrics(self):
        p, r, f1 = calculate_set_metrics(["wifi", "credentials"], ["wifi", "credentials", "airtel"])
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 2.0 / 3.0)
        self.assertGreater(f1, 0.7)

    def test_evaluate_baseline_run(self):
        eval_data = evaluate_baseline(use_mock_ocr=True)
        self.assertIn("metrics", eval_data)
        self.assertIn("category_accuracy", eval_data["metrics"])
        self.assertGreaterEqual(eval_data["metrics"]["category_accuracy"], 0.875)

        # Check output files exist
        results_dir = Path(__file__).resolve().parent.parent / "experiments" / "intelligence" / "results"
        self.assertTrue((results_dir / "lightweight_baseline_results.json").exists())
        self.assertTrue((results_dir / "lightweight_baseline_report.md").exists())


if __name__ == "__main__":
    unittest.main()
