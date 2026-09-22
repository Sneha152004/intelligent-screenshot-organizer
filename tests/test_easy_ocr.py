"""
Unit Tests for EasyOCR Integration & AppConfig Selection
==========================================================
Tests EasyOCREngine class, factory function, lazy reader initialization, error handling,
AppConfig engine selection, and OCR pipeline engine swappability without requiring real model downloads.
"""

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from src.config import config
from src.easy_ocr import EasyOCREngine, get_ocr_engine
from src.mock_ocr import BaseOCREngine, MockOCREngine
from src.pipeline import OCRPipeline


class TestEasyOCREngineInterface(unittest.TestCase):
    """Verifies class inheritance, interface compliance, and initialization behavior."""

    def test_easy_ocr_subclass_of_base_ocr(self):
        self.assertTrue(issubclass(EasyOCREngine, BaseOCREngine))
        engine = EasyOCREngine()
        self.assertIsInstance(engine, BaseOCREngine)

    def test_instantiation_without_easyocr_model_load(self):
        """Confirm that creating EasyOCREngine does NOT load/download EasyOCR models until reader property is accessed."""
        engine = EasyOCREngine(languages=["en"], gpu=False)
        self.assertIsNone(engine._reader)
        self.assertEqual(engine.languages, ["en"])
        self.assertFalse(engine.gpu)

    def test_missing_image_file_raises_file_not_found(self):
        """Confirm clear FileNotFoundError when passed non-existent image path."""
        engine = EasyOCREngine()
        with self.assertRaises(FileNotFoundError) as cm:
            engine.extract_text("sample/non_existent_image_9999.jpg")
        self.assertIn("not found", str(cm.exception))


class TestEasyOCRMockedExecution(unittest.TestCase):
    """Tests text extraction logic with mocked EasyOCR reader to avoid model download in unit tests."""

    @patch("easyocr.Reader")
    def test_mocked_reader_text_extraction(self, mock_reader_cls):
        mock_reader_instance = MagicMock()
        mock_reader_instance.readtext.return_value = [
            "Airtel_runu_7550",
            "Password",
            "Air@53054",
        ]
        mock_reader_cls.return_value = mock_reader_instance

        engine = EasyOCREngine(languages=["en"], gpu=False)
        sample_img = "sample/ss_001.jpg"

        text = engine.extract_text(sample_img)

        # Confirm reader was initialized with correct arguments
        mock_reader_cls.assert_called_once_with(["en"], gpu=False, verbose=False)
        # Confirm readtext was called with image path and detail=0
        mock_reader_instance.readtext.assert_called_once_with(str(Path(sample_img)), detail=0)
        # Confirm output is joined by newline
        expected_text = "Airtel_runu_7550\nPassword\nAir@53054"
        self.assertEqual(text, expected_text)

    def test_get_ocr_engine_factory(self):
        mock_eng = get_ocr_engine("mock")
        self.assertIsInstance(mock_eng, MockOCREngine)

        easy_eng = get_ocr_engine("easyocr", languages=["en"], gpu=False)
        self.assertIsInstance(easy_eng, EasyOCREngine)

        easy_eng_alt = get_ocr_engine("easy_ocr")
        self.assertIsInstance(easy_eng_alt, EasyOCREngine)

        with self.assertRaises(ValueError):
            get_ocr_engine("invalid_engine_name")

    def test_pipeline_engine_swappability(self):
        """Verify OCRPipeline can accept different OCR engine instances."""
        mock_engine = MockOCREngine()
        pipeline_mock = OCRPipeline(ocr_engine=mock_engine)
        self.assertIs(pipeline_mock.ocr_engine, mock_engine)

        easy_engine = EasyOCREngine()
        pipeline_easy = OCRPipeline(ocr_engine=easy_engine)
        self.assertIs(pipeline_easy.ocr_engine, easy_engine)

        # Test custom dummy engine swap
        class DummyEngine(BaseOCREngine):
            def extract_text(self, image_path: str) -> str:
                return "Dummy Text"

        dummy_engine = DummyEngine()
        pipeline_dummy = OCRPipeline(ocr_engine=dummy_engine)
        self.assertEqual(pipeline_dummy.ocr_engine.extract_text("any"), "Dummy Text")

    def test_mock_ocr_behavior_remains_unchanged(self):
        """Verify MockOCREngine is preserved and functions deterministically."""
        mock_ocr = MockOCREngine()
        text = mock_ocr.extract_text("sample/ss_001.jpg")
        self.assertIn("Airtel_runu_7550", text)
        self.assertIn("Air@53054", text)


class TestAppConfigEngineSelection(unittest.TestCase):
    """Verifies that AppConfig.ocr_engine_type correctly participates in OCR engine selection."""

    def setUp(self):
        self._orig_type = config.ocr_engine_type

    def tearDown(self):
        config.ocr_engine_type = self._orig_type

    def test_default_config_selects_mock_ocr_engine(self):
        config.ocr_engine_type = "mock"
        pipeline = OCRPipeline()
        self.assertIsInstance(pipeline.ocr_engine, MockOCREngine)

    def test_config_selects_easy_ocr_engine_without_model_load(self):
        config.ocr_engine_type = "easyocr"
        pipeline = OCRPipeline()
        self.assertIsInstance(pipeline.ocr_engine, EasyOCREngine)
        # Ensure model is lazily uninitialized
        self.assertIsNone(pipeline.ocr_engine._reader)

    def test_explicit_injected_engine_takes_precedence_over_config(self):
        config.ocr_engine_type = "easyocr"
        explicit_mock = MockOCREngine()
        pipeline = OCRPipeline(ocr_engine=explicit_mock)
        self.assertIs(pipeline.ocr_engine, explicit_mock)

    def test_pipeline_engine_type_param_overrides_config(self):
        config.ocr_engine_type = "mock"
        pipeline = OCRPipeline(ocr_engine_type="easyocr")
        self.assertIsInstance(pipeline.ocr_engine, EasyOCREngine)
        self.assertIsNone(pipeline.ocr_engine._reader)

    def test_invalid_config_engine_type_raises_value_error(self):
        config.ocr_engine_type = "unsupported_engine"
        with self.assertRaises(ValueError) as cm:
            OCRPipeline()
        self.assertIn("Unsupported OCR engine type", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
