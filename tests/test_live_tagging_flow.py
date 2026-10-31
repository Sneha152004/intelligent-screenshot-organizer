"""
Regression Tests for Live Tag Generation, OCR Pipeline & Backfill Eligibility
=============================================================================
Verifies:
1. Live pipeline tag generation for shopping, payment, and coding content.
2. Coding content produces Coding tags while preserving canonical categories (Education / Work & Career).
3. Educational programming content remains Education with Coding/Python tags.
4. Shopping content is not mislabeled as Payment/Receipt without settlement evidence.
5. Payment receipts receive evidence-backed Payment and Receipt tags.
6. Empty and generic placeholder text safely produces empty tag list (no hallucination).
7. Backfill eligibility filters out generic synthetic placeholders and selects genuine OCR evidence.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.backfill_tags import is_genuine_ocr_text, evaluate_backfill_eligibility
from src.mock_ocr import MockOCREngine
from src.models.metadata import ScreenshotMetadata
from src.pipeline import OCRPipeline
from src.storage.metadata_store import SQLiteMetadataStore
from src.tag_generator import TagGenerator, generate_tags


class TestLiveTaggingFlow(unittest.TestCase):
    """Verifies end-to-end pipeline tag generation and disambiguation rules."""

    def setUp(self):
        self.tag_gen = TagGenerator()

    def test_shopping_order_tag_generation(self):
        """Verifies shopping order generates Shopping tag and does not generate false Payment tags."""
        shopping_text = (
            "Amazon.in Shopping\n"
            "boAt Rockerz 450 Bluetooth On-Ear Headphones\n"
            "₹1,499.00\n"
            "FREE Delivery by Tomorrow\n"
            "In stock. Add to Cart. Buy Now."
        )
        tags = self.tag_gen.generate_tags(ocr_text=shopping_text, category="Shopping")
        self.assertIn("Shopping", tags)
        self.assertIn("Amazon", tags)
        self.assertNotIn("Payment", tags)
        self.assertNotIn("Receipt", tags)

    def test_payment_receipt_tag_generation(self):
        """Verifies payment confirmation generates Payment, Receipt, and UPI tags."""
        payment_text = (
            "Google Pay\n"
            "Paid successfully to Zomato\n"
            "₹450.00\n"
            "UPI Transaction ID: 412345678901\n"
            "Completed on 16 Oct 2026, 8:30 PM\n"
            "Payment Receipt"
        )
        tags = self.tag_gen.generate_tags(ocr_text=payment_text, category="Payment")
        self.assertIn("Payment", tags)
        self.assertIn("Receipt", tags)
        self.assertIn("UPI", tags)
        self.assertIn("Google Pay", tags)

    def test_educational_python_preserves_education_category_with_coding_tag(self):
        """Verifies Python tutorial keeps Education category and generates Coding/Python tags."""
        edu_text = (
            "CS101: Introduction to Python Programming\n"
            "Lecture 4: Data Structures & Algorithms\n"
            "def binary_search(arr, target):\n"
            "    left, right = 0, len(arr) - 1\n"
            "Homework assignment 2 due Friday."
        )
        tags = self.tag_gen.generate_tags(ocr_text=edu_text, category="Education")
        self.assertIn("Coding", tags)
        self.assertIn("Programming", tags)
        self.assertIn("Python", tags)

    def test_developer_work_receives_coding_and_tool_tags(self):
        """Verifies VS Code and Git screenshot generates Coding, Git, and VS Code tags."""
        dev_text = (
            "Visual Studio Code - screenshot-organizer\n"
            "git status\n"
            "On branch main\n"
            "Changes to be committed: git commit -m 'feat: live tag generation'\n"
            "import os\nimport sys\nfrom fastapi import FastAPI"
        )
        tags = self.tag_gen.generate_tags(ocr_text=dev_text, category="Work & Career")
        self.assertIn("Coding", tags)
        self.assertIn("VS Code", tags)
        self.assertIn("Git", tags)

    def test_generic_synthetic_placeholder_produces_no_hallucinated_tags(self):
        """Verifies generic label summary produces an empty tag list."""
        synthetic_text = "Screenshot showing a product, shopping, offer, order, or delivery page."
        tags = self.tag_gen.generate_tags(ocr_text=synthetic_text, category="Shopping")
        self.assertEqual(tags, [])

    def test_empty_or_whitespace_ocr_produces_no_tags(self):
        """Verifies empty or whitespace OCR text safely returns an empty tag list."""
        self.assertEqual(self.tag_gen.generate_tags(ocr_text="", category="General"), [])
        self.assertEqual(self.tag_gen.generate_tags(ocr_text="   \n\t  ", category="General"), [])
        self.assertEqual(self.tag_gen.generate_tags(ocr_text=None, category="General"), [])

    def test_pipeline_integration_saves_tags_to_sqlite(self):
        """Verifies OCRPipeline processes document, attaches tags, and persists to SQLite."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "test_metadata.sqlite")
            meta_store = SQLiteMetadataStore(db_path=db_path)

            mock_ocr = MockOCREngine(mapping={
                "upload_test.png": (
                    "Paid successfully ₹499 via UPI to Swiggy\n"
                    "Transaction ID: UPI987654321\n"
                    "Payment Receipt"
                )
            })

            # Create dummy image file
            img_path = os.path.join(tmp_dir, "upload_test.png")
            with open(img_path, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82")

            pipeline = OCRPipeline(
                ocr_engine=mock_ocr,
                metadata_store=meta_store,
                enable_visual=False,
            )

            doc = {
                "image_path": img_path,
                "metadata": {
                    "Screenshot ID": "test_up_001",
                    "Actual Category": "Payment",
                    "filename": "upload_test.png",
                },
            }

            processed = pipeline.process_documents([doc])
            self.assertEqual(len(processed), 1)

            saved = meta_store.get_metadata("test_up_001")
            self.assertIsNotNone(saved)
            self.assertEqual(saved.category, "Payment")
            self.assertIn("Payment", saved.tags)
            self.assertIn("Receipt", saved.tags)
            self.assertIn("UPI", saved.tags)


class TestBackfillSafety(unittest.TestCase):
    """Verifies backfill eligibility rules and safety constraints."""

    def test_synthetic_prefixes_rejected_by_is_genuine_ocr(self):
        """Confirm synthetic placeholders are rejected."""
        self.assertFalse(is_genuine_ocr_text("Screenshot showing a product, shopping, offer, order, or delivery page."))
        self.assertFalse(is_genuine_ocr_text("[MOCK OCR] No predefined text."))
        self.assertFalse(is_genuine_ocr_text(""))
        self.assertFalse(is_genuine_ocr_text(None))

    def test_genuine_text_accepted_by_is_genuine_ocr(self):
        """Confirm actual OCR text passes filter."""
        self.assertTrue(is_genuine_ocr_text("Paid successfully ₹689 to Braingrow Eduserv UPI ref 987654321"))
        self.assertTrue(is_genuine_ocr_text("Airtel_runu_7550 Password Air@53054 Show password"))
        self.assertTrue(is_genuine_ocr_text("Amazon.in Order #402-1234567-8901234 Total ₹1,299"))


if __name__ == "__main__":
    unittest.main()
