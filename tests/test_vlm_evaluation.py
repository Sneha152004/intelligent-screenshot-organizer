"""
Unit Tests for Model-Agnostic VLM Evaluation Harness
=====================================================
Tests target loading, adapter abstraction, raw response parsing, schema validation,
metrics calculation, performance tracking, result serialization, and demo evaluation execution.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from experiments.vlm.evaluation_targets import get_evaluation_targets, EVALUATION_TARGET_FILENAMES
from experiments.vlm.evaluate_vlm import (
    BaseVLMAdapter,
    MockVLMAdapter,
    VLMEvaluator,
    VLMInferenceError,
    VLMInitializationError,
    VLMResponseParseError,
    VLMSchemaValidationError,
    check_schema_validity,
    compute_aggregate_metrics,
    compute_item_metrics,
    parse_raw_vlm_response,
)


class TestVLMEvaluationTargets(unittest.TestCase):
    """Verifies benchmark evaluation targets and reference annotation loading."""

    def test_evaluation_targets_count_and_structure(self):
        targets = get_evaluation_targets()
        self.assertEqual(len(targets), 8)
        self.assertEqual(len(EVALUATION_TARGET_FILENAMES), 8)

        for target in targets:
            self.assertIn("filename", target)
            self.assertIn("screenshot_id", target)
            self.assertIn("expected", target)
            expected = target["expected"]
            self.assertIn("category", expected)
            self.assertIn("tags", expected)
            self.assertIn("intent", expected)
            self.assertIn("summary", expected)
            self.assertIn("importance", expected)
            self.assertIn("entities", expected)
            self.assertIn("dates", expected)
            self.assertIn("action_items", expected)
            self.assertIn("sensitive_info", expected)


class TestVLMAdapterAbstraction(unittest.TestCase):
    """Verifies BaseVLMAdapter contract properties and MockVLMAdapter execution."""

    def test_mock_vlm_adapter_interface_and_properties(self):
        adapter = MockVLMAdapter()
        self.assertTrue(issubclass(MockVLMAdapter, BaseVLMAdapter))
        self.assertIsInstance(adapter.model_name, str)
        self.assertEqual(adapter.model_identifier, "mock-vlm-engine")
        self.assertEqual(adapter.model_version, "mock-3d")
        self.assertEqual(adapter.model_footprint_mb, 0.0)
        self.assertEqual(adapter.device, "cpu")
        self.assertEqual(adapter.initialization_time_seconds, 0.0)

    def test_mock_adapter_analysis_returns_valid_dict(self):
        adapter = MockVLMAdapter()
        res = adapter.analyze("sample/ss_001.jpg", ocr_text="sample ocr")
        self.assertIsInstance(res, dict)
        self.assertEqual(res["category"], "Connectivity")
        self.assertIn("wifi", res["tags"])


class TestVLMResponseParsing(unittest.TestCase):
    """Verifies raw VLM response parsing, markdown stripping, and schema enforcement."""

    def setUp(self):
        self.valid_response_dict = {
            "category": "Connectivity",
            "tags": ["wifi", "password"],
            "intent": "Remember network credential",
            "summary": "Wi-Fi password details",
            "importance": 4,
            "entities": [{"text": "Airtel", "type": "organization"}],
            "dates": [],
            "action_items": [],
            "sensitive_info": {
                "contains_password": True,
                "contains_payment_info": False,
                "contains_personal_contact": False,
            },
        }

    def test_parse_valid_dict_response(self):
        res = parse_raw_vlm_response(self.valid_response_dict)
        self.assertEqual(res, self.valid_response_dict)

    def test_parse_valid_json_string_response(self):
        raw_str = json.dumps(self.valid_response_dict)
        res = parse_raw_vlm_response(raw_str)
        self.assertEqual(res["category"], "Connectivity")

    def test_parse_markdown_codeblock_json_response(self):
        markdown_json = f"```json\n{json.dumps(self.valid_response_dict)}\n```"
        res = parse_raw_vlm_response(markdown_json)
        self.assertEqual(res["category"], "Connectivity")

    def test_parse_invalid_json_syntax_raises_parse_error(self):
        bad_json = "```json\n{invalid json syntax}\n```"
        with self.assertRaises(VLMResponseParseError) as cm:
            parse_raw_vlm_response(bad_json)
        self.assertIn("Failed to parse raw VLM text output as JSON", str(cm.exception))

    def test_parse_schema_violation_raises_validation_error(self):
        invalid_schema_dict = {"category": "Payment"}  # Missing required keys
        with self.assertRaises(VLMSchemaValidationError) as cm:
            parse_raw_vlm_response(invalid_schema_dict)
        self.assertIn("failed schema validation", str(cm.exception))
        self.assertTrue(len(cm.exception.errors) > 0)
        self.assertEqual(cm.exception.raw_output, invalid_schema_dict)

    def test_parse_invalid_response_type_raises_parse_error(self):
        with self.assertRaises(VLMResponseParseError):
            parse_raw_vlm_response(12345)

    def test_vlm_exception_hierarchy(self):
        self.assertTrue(issubclass(VLMResponseParseError, ValueError))
        self.assertTrue(issubclass(VLMSchemaValidationError, ValueError))
        self.assertTrue(issubclass(VLMInitializationError, RuntimeError))
        self.assertTrue(issubclass(VLMInferenceError, RuntimeError))


class TestVLMSchemaValidation(unittest.TestCase):
    """Verifies lightweight JSON and structural schema validation logic."""

    def test_valid_dict_passes_validation(self):
        valid_dict = {
            "category": "Payment",
            "tags": ["receipt"],
            "intent": "Record",
            "summary": "UPI Payment",
            "importance": 4,
            "entities": [{"text": "Store", "type": "merchant"}],
            "dates": [],
            "action_items": [],
            "sensitive_info": {
                "contains_password": False,
                "contains_payment_info": True,
                "contains_personal_contact": False,
            },
        }
        is_valid, errors = check_schema_validity(valid_dict)
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_valid_json_string_passes_validation(self):
        valid_json = json.dumps({
            "category": "Education",
            "tags": ["math"],
            "intent": None,
            "summary": None,
            "importance": None,
            "entities": [],
            "dates": [],
            "action_items": [],
            "sensitive_info": {
                "contains_password": False,
                "contains_payment_info": False,
                "contains_personal_contact": False,
            },
        })
        is_valid, errors = check_schema_validity(valid_json)
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_invalid_structures_fail_validation(self):
        # Missing keys
        is_valid, errors = check_schema_validity({"category": "General"})
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

        # Non-integer importance
        is_valid, errors = check_schema_validity({
            "category": "General",
            "tags": [],
            "intent": None,
            "summary": None,
            "importance": "high",
            "entities": [],
            "dates": [],
            "action_items": [],
            "sensitive_info": {},
        })
        self.assertFalse(is_valid)

        # Invalid importance integer range
        is_valid, errors = check_schema_validity({
            "category": "General",
            "tags": [],
            "intent": None,
            "summary": None,
            "importance": 10,
            "entities": [],
            "dates": [],
            "action_items": [],
            "sensitive_info": {},
        })
        self.assertFalse(is_valid)


class TestVLMMetricsCalculation(unittest.TestCase):
    """Verifies item and aggregate evaluation metrics calculation."""

    def test_item_metrics_exact_match(self):
        expected = {
            "category": "Payment",
            "tags": ["upi", "receipt"],
            "intent": "Keep payment record",
            "entities": [{"text": "Store", "type": "merchant"}],
            "dates": [{"text": "2026-09-22", "type": "date"}],
            "action_items": [{"text": "Pay bill", "status": "pending"}],
            "sensitive_info": {
                "contains_password": False,
                "contains_payment_info": True,
                "contains_personal_contact": False,
            },
        }
        metrics = compute_item_metrics(expected, expected)
        self.assertEqual(metrics["category_match"], 1.0)
        self.assertEqual(metrics["tag_precision"], 1.0)
        self.assertEqual(metrics["tag_recall"], 1.0)
        self.assertEqual(metrics["intent_match"], 1.0)
        self.assertEqual(metrics["entities_recall"], 1.0)
        self.assertEqual(metrics["dates_recall"], 1.0)
        self.assertEqual(metrics["action_items_recall"], 1.0)
        self.assertEqual(metrics["sensitive_info_accuracy"], 1.0)

    def test_item_metrics_partial_match(self):
        expected = {
            "category": "Payment",
            "tags": ["upi", "receipt", "bank"],
            "intent": "Keep payment record",
            "entities": [{"text": "Store", "type": "merchant"}],
            "dates": [],
            "action_items": [],
            "sensitive_info": {
                "contains_password": False,
                "contains_payment_info": True,
                "contains_personal_contact": False,
            },
        }
        predicted = {
            "category": "Shopping",  # mismatch
            "tags": ["upi"],        # 1 match out of 3 expected
            "intent": "Buy item",    # mismatch
            "entities": [],         # 0 matches
            "dates": [],
            "action_items": [],
            "sensitive_info": {
                "contains_password": False,
                "contains_payment_info": True,
                "contains_personal_contact": True,  # 1 mismatch out of 3 flags
            },
        }
        metrics = compute_item_metrics(expected, predicted)
        self.assertEqual(metrics["category_match"], 0.0)
        self.assertEqual(metrics["tag_precision"], 1.0)
        self.assertAlmostEqual(metrics["tag_recall"], 1 / 3, places=2)
        self.assertEqual(metrics["intent_match"], 0.0)
        self.assertEqual(metrics["entities_recall"], 0.0)
        self.assertAlmostEqual(metrics["sensitive_info_accuracy"], 2 / 3, places=2)

    def test_aggregate_metrics_calculation(self):
        per_target = [
            {
                "schema_validity": {"is_valid": True},
                "metrics": {
                    "category_match": 1.0,
                    "tag_precision": 1.0,
                    "tag_recall": 0.5,
                    "intent_match": 1.0,
                    "entities_recall": 1.0,
                    "dates_recall": 1.0,
                    "action_items_recall": 1.0,
                    "sensitive_info_accuracy": 1.0,
                },
            },
            {
                "schema_validity": {"is_valid": True},
                "metrics": {
                    "category_match": 0.0,
                    "tag_precision": 0.5,
                    "tag_recall": 0.5,
                    "intent_match": 0.0,
                    "entities_recall": 0.0,
                    "dates_recall": 0.0,
                    "action_items_recall": 0.0,
                    "sensitive_info_accuracy": 0.6667,
                },
            },
        ]
        agg = compute_aggregate_metrics(per_target)
        self.assertEqual(agg["schema_validity_rate"], 1.0)
        self.assertEqual(agg["category_accuracy"], 0.5)
        self.assertEqual(agg["tag_precision"], 0.75)
        self.assertEqual(agg["intent_match_rate"], 0.5)


class TestVLMEvaluatorAndSerialization(unittest.TestCase):
    """Verifies evaluator execution loop, latency tracking, and result output writing."""

    def test_evaluator_demo_execution_and_file_serialization(self):
        adapter = MockVLMAdapter()
        evaluator = VLMEvaluator(adapter=adapter)

        with tempfile.TemporaryDirectory() as tmp_dir:
            res = evaluator.evaluate(limit=2)
            self.assertEqual(res["metadata"]["total_screenshots"], 2)
            self.assertEqual(res["performance"]["successful_evaluations"], 2)
            self.assertEqual(res["performance"]["failed_evaluations"], 0)
            self.assertEqual(len(res["per_target_results"]), 2)

            json_path, md_path = evaluator.save_results(res, output_dir=tmp_dir)

            self.assertTrue(Path(json_path).exists())
            self.assertTrue(Path(md_path).exists())

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.assertEqual(data["metadata"]["model_name"], adapter.model_name)
                self.assertEqual(data["metadata"]["device"], "cpu")

            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()
                self.assertIn("Phase 3F.1: VLM Evaluation Report", md_content)
                self.assertIn(adapter.model_name, md_content)


if __name__ == "__main__":
    unittest.main()
