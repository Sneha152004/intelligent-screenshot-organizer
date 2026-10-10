"""
Unit Tests for Phase 3G.1 Unseen Dataset Evaluation Harness
============================================================
Tests dataset loading, zero-data-leakage separation, reference annotation schema validity,
evaluator execution, latency percentile calculation, confidence grouping, and report output.
"""

import json
from pathlib import Path
import unittest

from experiments.intelligence.unseen.evaluation_targets import (
    get_unseen_evaluation_targets,
    UNSEEN_TARGET_FILENAMES,
    DEVELOPMENT_BENCHMARK_FILENAMES,
)
from experiments.intelligence.evaluate_unseen import (
    evaluate_unseen_dataset,
    calculate_percentiles,
)


class TestUnseenDatasetSeparationAndSchema(unittest.TestCase):
    """Verifies dataset separation, anti-leakage rules, and ground-truth reference schemas."""

    def test_unseen_target_count(self):
        """Verify unseen target set contains exactly 22 distinct screenshots."""
        targets = get_unseen_evaluation_targets()
        self.assertEqual(len(targets), 22)
        self.assertEqual(len(UNSEEN_TARGET_FILENAMES), 22)

    def test_strict_dataset_separation(self):
        """Verify 0% overlap between unseen filenames and 8 development benchmark filenames."""
        unseen_set = set(UNSEEN_TARGET_FILENAMES)
        overlap = unseen_set.intersection(DEVELOPMENT_BENCHMARK_FILENAMES)
        self.assertEqual(len(overlap), 0, f"Data leakage detected! Overlapping files: {overlap}")

    def test_ground_truth_reference_schema(self):
        """Verify ground truth annotations contain valid fields matching ScreenshotMetadata contract."""
        targets = get_unseen_evaluation_targets()
        required_keys = {
            "category", "tags", "intent", "summary", "importance",
            "entities", "dates", "action_items", "sensitive_info"
        }
        categories = {
            "Connectivity", "Payment", "Education", "Shopping", "Travel",
            "Communication", "Coding", "Work & Career", "Entertainment", "General"
        }

        for target in targets:
            expected = target["expected"]
            self.assertTrue(required_keys.issubset(expected.keys()))
            self.assertIn(expected["category"], categories)
            self.assertIsInstance(expected["tags"], list)
            self.assertIsInstance(expected["sensitive_info"], dict)
            for flag in ["contains_password", "contains_payment_info", "contains_personal_contact"]:
                self.assertIn(flag, expected["sensitive_info"])


class TestUnseenEvaluatorExecution(unittest.TestCase):
    """Tests evaluate_unseen.py execution, percentile math, and report generation."""

    def test_calculate_percentiles(self):
        """Verify mean, median (p50), and p95 math functions."""
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        res = calculate_percentiles(vals)
        self.assertEqual(res["mean"], 30.0)
        self.assertEqual(res["p50"], 30.0)
        self.assertGreaterEqual(res["p95"], 45.0)

    def test_evaluate_unseen_dataset_run(self):
        """Executes full unseen dataset evaluation run and validates results dictionary and files."""
        data = evaluate_unseen_dataset(use_mock_ocr=True)

        self.assertTrue(data["is_genuinely_unseen"])
        self.assertEqual(data["unseen_dataset_size"], 22)

        # Validate metrics block
        m = data["metrics"]
        self.assertGreaterEqual(m["category_accuracy"], 0.50)
        self.assertGreater(m["tag_f1"], 0.20)
        self.assertEqual(m["schema_validity_rate"], 1.0)

        # Validate latencies block contains mean, p50, p95
        l = data["latency_percentiles"]
        for stage in ["preprocessing_ms", "ocr_ms", "rules_ms", "embedding_ms", "classification_ms", "total_ms"]:
            self.assertIn(stage, l)
            self.assertIn("mean", l[stage])
            self.assertIn("p50", l[stage])
            self.assertIn("p95", l[stage])

        self.assertIn("cold_start_initialization_ms", l)

        # Validate confidence correlation block
        conf = data["confidence_correlation"]
        self.assertIn("high", conf)
        self.assertIn("medium", conf)
        self.assertIn("low", conf)

        # Validate result file creation
        results_dir = Path(__file__).resolve().parent.parent / "experiments" / "intelligence" / "results"
        self.assertTrue((results_dir / "unseen_evaluation_results.json").exists())
        self.assertTrue((results_dir / "unseen_evaluation_report.md").exists())


if __name__ == "__main__":
    unittest.main()
