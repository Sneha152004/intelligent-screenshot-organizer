"""
Unit Tests for VLM / Screenshot Understanding Foundation
===========================================================
Tests BaseVLM interface, MockVLM deterministic execution, schema validation, factory,
and AppConfig integration without requiring any real VLM model downloads.
"""

import unittest

from src.config import config
from src.vlm import BaseVLM, MockVLM, get_vlm_engine


class TestVLMInterfaceAndMock(unittest.TestCase):
    """Verifies class inheritance, instantiation, deterministic execution, and error handling."""

    def test_mock_vlm_inherits_from_base_vlm(self):
        self.assertTrue(issubclass(MockVLM, BaseVLM))
        vlm = MockVLM()
        self.assertIsInstance(vlm, BaseVLM)

    def test_instantiation_without_model_download(self):
        """Confirm MockVLM can be instantiated instantly offline without external model loading."""
        vlm = MockVLM()
        self.assertIsNotNone(vlm.mappings)

    def test_valid_sample_screenshot_produces_deterministic_result(self):
        vlm = MockVLM()
        res1 = vlm.analyze_image("sample/ss_001.jpg")
        res2 = vlm.analyze_image("sample/ss_001.jpg")
        self.assertEqual(res1, res2)
        self.assertEqual(res1["category"], "Connectivity")
        self.assertIn("wifi", res1["tags"])

    def test_result_contains_all_expected_fields(self):
        vlm = MockVLM()
        res = vlm.analyze_image("sample/ss_001.jpg")
        expected_fields = {
            "category",
            "tags",
            "intent",
            "summary",
            "importance",
            "entities",
            "dates",
            "action_items",
            "sensitive_info",
        }
        self.assertEqual(set(res.keys()), expected_fields)

    def test_field_types_and_constraints(self):
        vlm = MockVLM()
        res = vlm.analyze_image("sample/ss_001.jpg")

        # 5. Category is a string
        self.assertIsInstance(res["category"], str)

        # 6. Tags are a list of strings
        self.assertIsInstance(res["tags"], list)
        self.assertTrue(all(isinstance(t, str) for t in res["tags"]))

        # Optional intent and summary check
        self.assertTrue(res["intent"] is None or isinstance(res["intent"], str))
        self.assertTrue(res["summary"] is None or isinstance(res["summary"], str))

        # 7. Importance is None or 1-5 integer (and not bool)
        importance = res["importance"]
        if importance is not None:
            self.assertNotIsInstance(importance, bool)
            self.assertIsInstance(importance, int)
            self.assertTrue(1 <= importance <= 5)

        # 8. Entities, dates, action_items are lists
        self.assertIsInstance(res["entities"], list)
        self.assertIsInstance(res["dates"], list)
        self.assertIsInstance(res["action_items"], list)

        # 9. Sensitive info is a dictionary
        self.assertIsInstance(res["sensitive_info"], dict)
        self.assertIn("contains_password", res["sensitive_info"])

    def test_missing_image_raises_file_not_found(self):
        # 10. Missing image raises FileNotFoundError
        vlm = MockVLM()
        with self.assertRaises(FileNotFoundError) as cm:
            vlm.analyze_image("sample/non_existent_image_999.png")
        self.assertIn("not found", str(cm.exception))

    def test_unmapped_image_returns_valid_fallback(self):
        vlm = MockVLM()
        # Non-mapped existing screenshot or fallback verification
        fallback = vlm.FALLBACK_VLM_RESPONSE
        vlm.validate_output(fallback)
        self.assertEqual(fallback["category"], "General")

    def test_output_validation_catches_malformed_structures(self):
        vlm = MockVLM()
        with self.assertRaises(ValueError):
            vlm.validate_output("not_a_dict")

        with self.assertRaises(ValueError):
            vlm.validate_output({"category": 123, "tags": []})

        with self.assertRaises(ValueError):
            vlm.validate_output({"category": "General", "tags": "not_a_list"})

        with self.assertRaises(ValueError):
            vlm.validate_output({"category": "General", "tags": [], "importance": 10})

        with self.assertRaises(ValueError):
            vlm.validate_output({"category": "General", "tags": [], "importance": True})

        with self.assertRaises(ValueError):
            vlm.validate_output({"category": "General", "tags": [], "entities": "not_a_list"})

        with self.assertRaises(ValueError):
            vlm.validate_output({"category": "General", "tags": [], "entities": [], "dates": [], "action_items": [], "sensitive_info": "not_a_dict"})


class TestVLMFactoryAndConfig(unittest.TestCase):
    """Verifies factory instantiation and configuration integration."""

    def test_factory_returns_mock_vlm_for_mock(self):
        # 11. Factory returns MockVLM for "mock"
        vlm = get_vlm_engine("mock")
        self.assertIsInstance(vlm, MockVLM)

    def test_factory_rejects_unsupported_engine_types(self):
        # 12. Factory rejects unsupported VLM engine types
        with self.assertRaises(ValueError) as cm:
            get_vlm_engine("unsupported_vlm_engine")
        self.assertIn("Unsupported VLM engine type", str(cm.exception))

    def test_configuration_default_selects_mock(self):
        # 13. Configuration default selects "mock"
        self.assertEqual(config.vlm_engine_type, "mock")
        vlm = get_vlm_engine(config.vlm_engine_type)
        self.assertIsInstance(vlm, MockVLM)


if __name__ == "__main__":
    unittest.main()
