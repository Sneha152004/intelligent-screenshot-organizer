"""
Unit Tests for Mock OCR Pipeline
================================
Tests individual modules (loader, mock_ocr, chunker, pipeline) to ensure robustness.
"""

import os
import unittest

from src.loader import load_dataset
from src.mock_ocr import BaseOCREngine, MockOCREngine
from src.chunker import ParagraphChunker
from src.pipeline import OCRPipeline


class TestLoader(unittest.TestCase):
    def test_load_dataset_success(self):
        docs = load_dataset(metadata_path="sample/metadata.csv")
        self.assertGreater(len(docs), 0)
        self.assertIn("image_path", docs[0])
        self.assertIn("metadata", docs[0])
        self.assertTrue(os.path.exists(docs[0]["image_path"]))

    def test_load_dataset_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset(metadata_path="non_existent_metadata.csv")


class TestMockOCR(unittest.TestCase):
    def setUp(self):
        self.ocr = MockOCREngine()

    def test_deterministic_extraction(self):
        text1 = self.ocr.extract_text("sample/ss_001.jpg")
        text2 = self.ocr.extract_text("sample/ss_001.jpg")
        self.assertEqual(text1, text2)
        self.assertIn("Airtel_runu_7550", text1)
        self.assertIn("Air@53054", text1)

    def test_unmapped_fallback(self):
        fallback_text = self.ocr.extract_text("sample/unknown_image_999.png")
        self.assertEqual(fallback_text, MockOCREngine.FALLBACK_TEXT)

    def test_swappable_interface(self):
        class CustomRealOCREngine(BaseOCREngine):
            def extract_text(self, image_path: str) -> str:
                return "Real OCR Text"

        custom_engine = CustomRealOCREngine()
        self.assertEqual(custom_engine.extract_text("any_path.jpg"), "Real OCR Text")


class TestChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = ParagraphChunker()

    def test_chunking_paragraphs(self):
        text = "Paragraph 1 line A\nParagraph 1 line B\n\nParagraph 2 line A\n\n\nParagraph 3"
        chunks = self.chunker.chunk(text)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "Paragraph 1 line A\nParagraph 1 line B")
        self.assertEqual(chunks[1], "Paragraph 2 line A")
        self.assertEqual(chunks[2], "Paragraph 3")

    def test_empty_text_returns_empty_list(self):
        self.assertEqual(self.chunker.chunk("   "), [])
        self.assertEqual(self.chunker.chunk(""), [])


class TestPipeline(unittest.TestCase):
    def test_pipeline_end_to_end(self):
        pipeline = OCRPipeline()
        docs = pipeline.process(metadata_path="sample/metadata.csv")
        self.assertGreater(len(docs), 0)
        first_doc = docs[0]
        self.assertIn("image_path", first_doc)
        self.assertIn("metadata", first_doc)
        self.assertIn("ocr_text", first_doc)
        self.assertIn("chunks", first_doc)
        self.assertIsInstance(first_doc["chunks"], list)


if __name__ == "__main__":
    unittest.main()
