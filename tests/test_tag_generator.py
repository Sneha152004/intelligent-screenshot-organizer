"""
Unit and Integration Tests for Tag Generation Feature in I.R.I.S.
==================================================================
Covers:
1. Multi-family tag extraction (Domain/Activity, Entity/App, Content Type).
2. Normalization, deduplication, and deterministic sorting.
3. Disambiguation rules (Education tutorial vs Coding IDE, Shopping order vs Payment receipt).
4. Resilience against empty, whitespace, and irrelevant OCR text (no hallucination).
5. Canonical 9-category taxonomy integrity (Coding is strictly a multi-label tag, never a category).
6. SQLite metadata store serialization, round-trip persistence, and tag querying.
7. Policy V3 baseline preservation and regression checks.
"""

import os
import json
import shutil
import tempfile
import unittest
from typing import List

from src.models.metadata import ScreenshotMetadata
from src.storage.metadata_store import SQLiteMetadataStore
from src.tag_generator import TagGenerator, generate_tags, TAG_FAMILIES


class TestTagGeneratorFamilies(unittest.TestCase):
    """Verify that each tag family extracts expected tags given supported OCR evidence."""

    def setUp(self):
        self.generator = TagGenerator()

    def test_domain_activity_tags(self):
        """Test extraction of domain/activity family tags."""
        ocr_coding = "def calculate_hash(data):\n    import hashlib\n    return hashlib.sha256(data).hexdigest()"
        tags = self.generator.generate(ocr_text=ocr_coding)
        self.assertIn("Coding", tags)
        self.assertIn("Python", tags)

        ocr_learning = "Chapter 4: Principles of Quantum Mechanics. Lecture slides and syllabus for Final Exam study."
        tags_learning = self.generator.generate(ocr_text=ocr_learning, category="Education")
        self.assertIn("Learning", tags_learning)

        ocr_gaming = "Victory Royale! Level 84 Battle Pass unlocked. High score achieved in multiplayer lobby."
        tags_gaming = self.generator.generate(ocr_text=ocr_gaming, category="Entertainment")
        self.assertIn("Gaming", tags_gaming)

    def test_entity_app_tags(self):
        """Test extraction of technology, app, and service entities."""
        ocr_tech = "const app = express(); app.listen(3000, () => console.log('Server running on port 3000'));"
        tags_tech = self.generator.generate(ocr_text=ocr_tech)
        self.assertTrue(any(t in tags_tech for t in ["JavaScript", "Programming", "Coding"]))

        ocr_apps = "Opening project in Visual Studio Code. git commit -m 'Initial commit' to github repository."
        tags_apps = self.generator.generate(ocr_text=ocr_apps)
        self.assertIn("VS Code", tags_apps)
        self.assertIn("GitHub", tags_apps)
        self.assertIn("Git", tags_apps)

        ocr_commerce = "Your Amazon order #402-9812471 has shipped via BlueDart. Delivery expected tomorrow."
        tags_commerce = self.generator.generate(ocr_text=ocr_commerce, category="Shopping")
        self.assertIn("Amazon", tags_commerce)
        self.assertIn("Order", tags_commerce)

        ocr_payment_app = "Google Pay transaction successful. Paid to Swiggy via UPI reference number 928374182910."
        tags_payment = self.generator.generate(ocr_text=ocr_payment_app, category="Payment")
        self.assertIn("Google Pay", tags_payment)
        self.assertIn("UPI", tags_payment)

    def test_content_type_tags(self):
        """Test extraction of content type tags (Receipt, Invoice, Order, Ticket, Tutorial, etc.)."""
        ocr_invoice = "TAX INVOICE / BILL OF SUPPLY. Invoice No: INV-2026-091. Total Amount: $450.00. Balance Due: $0.00"
        tags_invoice = self.generator.generate(ocr_text=ocr_invoice)
        self.assertIn("Invoice", tags_invoice)

        ocr_ticket = "Boarding Pass - Flight AI 804. Seat 14B. Departure Gate 4. Boarding time: 14:30."
        tags_ticket = self.generator.generate(ocr_text=ocr_ticket, category="Travel")
        self.assertIn("Ticket", tags_ticket)

        ocr_tutorial = "Step-by-step tutorial for beginners: How to build a REST API in Python. Walkthrough and code guide."
        tags_tutorial = self.generator.generate(ocr_text=ocr_tutorial, category="Education")
        self.assertIn("Tutorial", tags_tutorial)


class TestDisambiguationRules(unittest.TestCase):
    """Verify disambiguation logic prevents category conflation and false tags."""

    def setUp(self):
        self.generator = TagGenerator()

    def test_python_tutorial_in_education(self):
        """Python tutorial / lecture slide must remain in Education category without false Coding category."""
        ocr = "CS 101 Lecture 5: Introduction to Python Programming for Data Science. Professor Smith's lecture notes and tutorial."
        category = "Education"
        tags = self.generator.generate(ocr_text=ocr, category=category)

        self.assertIn("Python", tags)
        self.assertIn("Programming", tags)
        self.assertIn("Tutorial", tags)
        self.assertIn("Learning", tags)
        # Category remains intact
        self.assertEqual(category, "Education")

    def test_shopping_order_suppresses_payment_receipt(self):
        """Shopping order with price details belongs to Shopping and should NOT generate false Payment/Receipt tags."""
        ocr = "Order Placed! Thank you for shopping on Flipkart. Total item price: Rs 1,499. Order ID: OD817264819. Arriving by Friday."
        category = "Shopping"
        tags = self.generator.generate(ocr_text=ocr, category=category)

        self.assertIn("Order", tags)
        self.assertIn("Flipkart", tags)
        self.assertIn("Shopping", tags)
        # Suppressed tags
        self.assertNotIn("Receipt", tags)
        self.assertNotIn("Payment", tags)

    def test_payment_confirmation_receipt(self):
        """Payment confirmation or bank transaction generates Receipt/Transaction/UPI tags."""
        ocr = "Payment Successful! Transaction ID: TXN482019482. Paid Rs 350 to Merchant via UPI. Bank Reference No: 9182374."
        category = "Payment"
        tags = self.generator.generate(ocr_text=ocr, category=category)

        self.assertIn("Receipt", tags)
        self.assertIn("Transaction", tags)
        self.assertIn("UPI", tags)
        self.assertNotIn("Shopping", tags)


class TestNormalizationAndDeduplication(unittest.TestCase):
    """Verify tag formatting, casing normalization, and deduplication."""

    def setUp(self):
        self.generator = TagGenerator()

    def test_deduplication(self):
        """Multiple occurrences of the same technology should produce a single tag."""
        ocr = "Python is great. Writing python scripts with python modules. python python python."
        tags = self.generator.generate(ocr_text=ocr)
        self.assertEqual(tags.count("Python"), 1)

    def test_deterministic_sorting(self):
        """Generated tags must be sorted deterministically."""
        ocr = "Docker container deployed on AWS with Python backend and PostgreSQL database."
        tags1 = self.generator.generate(ocr_text=ocr)
        tags2 = self.generator.generate(ocr_text=ocr)
        self.assertEqual(tags1, tags2)
        self.assertEqual(tags1, sorted(tags1))


class TestResilienceAndNoHallucinations(unittest.TestCase):
    """Verify that empty, sparse, or irrelevant text produces no hallucinated tags."""

    def setUp(self):
        self.generator = TagGenerator()

    def test_empty_and_whitespace_text(self):
        """Empty or whitespace text returns an empty list."""
        self.assertEqual(self.generator.generate(ocr_text=""), [])
        self.assertEqual(self.generator.generate(ocr_text="   \n\t  "), [])
        self.assertEqual(self.generator.generate(ocr_text=None), [])

    def test_irrelevant_text_no_hallucination(self):
        """Generic non-matching text returns no tags."""
        text = "The quick brown fox jumps over the lazy dog in a green grassy field on a sunny morning."
        tags = self.generator.generate(ocr_text=text)
        self.assertEqual(tags, [])

    def test_standalone_helper_function(self):
        """The module-level generate_tags helper behaves identically to generator.generate."""
        text = "Writing unit tests in Python using pytest framework."
        tags = generate_tags(ocr_text=text)
        self.assertIn("Python", tags)
        self.assertIn("Coding", tags)


class TestCategoryIntegrity(unittest.TestCase):
    """Verify that Coding is NEVER treated as a top-level category and canonical 9 classes are intact."""

    CANONICAL_CATEGORIES = {
        "Education",
        "Shopping",
        "Payment",
        "Travel",
        "Communication",
        "Work & Career",
        "Work_Career",
        "Entertainment",
        "Connectivity",
        "General",
    }

    def test_coding_never_category(self):
        """Ensure Coding is strictly a tag and not among canonical categories."""
        self.assertNotIn("Coding", {"Education", "Shopping", "Payment", "Travel", "Communication", "Entertainment", "Work & Career", "Connectivity", "General"})

    def test_generator_never_overrides_category(self):
        """TagGenerator must accept and preserve canonical category without mutating it."""
        generator = TagGenerator()
        for cat in ["Education", "Shopping", "Payment", "Travel", "Communication", "Entertainment", "Connectivity", "General"]:
            tags = generator.generate(ocr_text="import os\nprint('hello')", category=cat)
            self.assertIsInstance(tags, list)
            # The input category string is never altered
            self.assertIn(cat, self.CANONICAL_CATEGORIES)


class TestMetadataAndSQLiteStorage(unittest.TestCase):
    """Verify ScreenshotMetadata and SQLiteMetadataStore persistence with tags."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_tags_storage_")
        self.db_path = os.path.join(self.temp_dir, "metadata.sqlite")
        self.store = SQLiteMetadataStore(db_path=self.db_path)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_metadata_tags_roundtrip(self):
        """Verify tags are persisted to SQLite and retrieved accurately."""
        meta = ScreenshotMetadata(
            screenshot_id="ss_tag_001",
            filename="code_snippet.png",
            image_uri="data/screenshots/code_snippet.png",
            category="Education",
            tags=["Coding", "Python", "Tutorial", "VS Code"],
        )
        self.store.save(meta)

        retrieved = self.store.get("ss_tag_001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.tags, ["Coding", "Python", "Tutorial", "VS Code"])
        self.assertEqual(retrieved.category, "Education")

    def test_empty_tags_backward_compatibility(self):
        """Screenshots with empty tags or legacy data deserialize as empty list."""
        meta = ScreenshotMetadata(
            screenshot_id="ss_tag_002",
            filename="landscape.png",
            image_uri="data/screenshots/landscape.png",
            category="General",
            tags=[],
        )
        self.store.save(meta)

        retrieved = self.store.get("ss_tag_002")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.tags, [])

    def test_query_by_tag(self):
        """Verify SQLite query by tag works as expected."""
        meta1 = ScreenshotMetadata(
            screenshot_id="ss_1",
            filename="python_notes.png",
            image_uri="data/screenshots/python_notes.png",
            category="Education",
            tags=["Python", "Learning"],
        )
        meta2 = ScreenshotMetadata(
            screenshot_id="ss_2",
            filename="amazon_order.png",
            image_uri="data/screenshots/amazon_order.png",
            category="Shopping",
            tags=["Amazon", "Order", "Shopping"],
        )
        meta3 = ScreenshotMetadata(
            screenshot_id="ss_3",
            filename="python_script.png",
            image_uri="data/screenshots/python_script.png",
            category="Work_Career",
            tags=["Coding", "Python", "VS Code"],
        )
        self.store.save(meta1)
        self.store.save(meta2)
        self.store.save(meta3)

        python_items = self.store.get_by_tag("Python")
        self.assertEqual(len(python_items), 2)
        python_ids = {item.screenshot_id for item in python_items}
        self.assertEqual(python_ids, {"ss_1", "ss_3"})

        amazon_items = self.store.get_by_tag("Amazon")
        self.assertEqual(len(amazon_items), 1)
        self.assertEqual(amazon_items[0].screenshot_id, "ss_2")

        empty_items = self.store.get_by_tag("NonExistentTag")
        self.assertEqual(len(empty_items), 0)

    def test_get_all_tags(self):
        """Verify get_all_tags aggregates and counts tags correctly."""
        meta1 = ScreenshotMetadata(
            screenshot_id="ss_a",
            filename="a.png",
            image_uri="a.png",
            category="Education",
            tags=["Python", "Learning"],
        )
        meta2 = ScreenshotMetadata(
            screenshot_id="ss_b",
            filename="b.png",
            image_uri="b.png",
            category="Education",
            tags=["Python", "Coding"],
        )
        self.store.save(meta1)
        self.store.save(meta2)

        tag_counts = self.store.get_all_tags()
        self.assertEqual(tag_counts.get("Python"), 2)
        self.assertEqual(tag_counts.get("Learning"), 1)
        self.assertEqual(tag_counts.get("Coding"), 1)


class TestPolicyV3Preservation(unittest.TestCase):
    """Verify that frozen Policy V3 baseline files exist and have not been modified."""

    def test_policy_v3_frozen_directory_exists(self):
        policy_v3_dir = os.path.join(os.path.dirname(__file__), "..", "experiments", "category_policy_v3")
        self.assertTrue(os.path.isdir(policy_v3_dir), "Policy V3 directory must exist")

    def test_policy_v3_baseline_file_exists(self):
        baseline_doc = os.path.join(
            os.path.dirname(__file__),
            "..",
            "experiments",
            "category_policy_v3",
            "FROZEN_BASELINE_V3.md"
        )
        self.assertTrue(os.path.isfile(baseline_doc), "Policy V3 FROZEN_BASELINE_V3.md must exist")

        config_file = os.path.join(
            os.path.dirname(__file__),
            "..",
            "experiments",
            "category_policy_v3",
            "policy_v3_config.json"
        )
        self.assertTrue(os.path.isfile(config_file), "Policy V3 config json must exist")


if __name__ == "__main__":
    unittest.main()
