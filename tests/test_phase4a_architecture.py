"""
Unit Tests for Phase 4A Intelligence Architecture Refactor
============================================================
Tests the evidence-first architecture: content_type, structured evidence, events, financial_info,
contacts, urls, text_topics, field-level confidence scores, optional intent, SQLite non-destructive
migration, backward-compatible serialization, and 30-record persistence.
"""

import json
import os
from pathlib import Path
import tempfile
import unittest

from src.models.metadata import (
    ScreenshotMetadata,
    CATEGORY_CONTENT_TYPES,
    DEFAULT_SENSITIVE_INFO,
)
from src.storage.metadata_store import SQLiteMetadataStore
from experiments.intelligence.rules import (
    extract_financial_info,
    extract_events,
    extract_text_topics,
    resolve_content_type,
)
from experiments.intelligence.baseline_intelligence import analyze_screenshot


class TestScreenshotMetadataPhase4A(unittest.TestCase):
    """Tests new metadata schema fields, validations, and backward-compatible serialization."""

    def test_metadata_instantiation_with_new_fields(self):
        """Verify ScreenshotMetadata accepts content_type, evidence, events, confidence_scores, and optional intent."""
        meta = ScreenshotMetadata(
            screenshot_id="ss_test_001",
            filename="ss_test_001.jpg",
            image_uri="/path/to/ss_test_001.jpg",
            category="Travel",
            content_type="flight_ticket",
            intent=None,  # Optional intent
            summary="IndiGo flight boarding pass to Guwahati",
            evidence={"flight_number": "6E 6914"},
            events=[{"type": "travel", "status": "booked"}],
            financial_info={"amount": "₹1,800.00"},
            contacts={"emails": ["test@example.com"]},
            urls=["https://indigo.in"],
            text_topics=["flight", "boarding_pass"],
            confidence_scores={"category": 0.85, "content_type": 0.90},
        )

        self.assertEqual(meta.content_type, "flight_ticket")
        self.assertIsNone(meta.intent)
        self.assertEqual(meta.evidence["flight_number"], "6E 6914")
        self.assertEqual(len(meta.events), 1)
        self.assertEqual(meta.confidence_scores["category"], 0.85)

    def test_optional_intent_behavior(self):
        """Verify intent defaults to None when evidence is weak or unassigned."""
        meta = ScreenshotMetadata(
            screenshot_id="ss_no_intent",
            filename="ss_no_intent.jpg",
            image_uri="/path/to/img.jpg",
            category="General",
        )
        self.assertIsNone(meta.intent)

    def test_to_dict_and_from_dict_roundtrip(self):
        """Verify full to_dict and from_dict roundtrip with new evidence fields."""
        original = ScreenshotMetadata(
            screenshot_id="ss_rt_1",
            filename="ss_rt_1.jpg",
            image_uri="/path/to/rt.jpg",
            category="Payment",
            content_type="payment_receipt",
            intent="Save receipt",
            summary="UPI payment of ₹689 to merchant",
            evidence={"upi_id": "user@okicici"},
            events=[{"type": "payment", "status": "successful"}],
            financial_info={"amounts": ["₹689.00"], "payment_method": "UPI"},
            contacts={"emails": ["user@example.com"]},
            urls=["https://razorpay.com"],
            text_topics=["payment", "upi"],
            confidence_scores={"category": 0.95, "content_type": 0.92},
        )

        d = original.to_dict()
        self.assertEqual(d["content_type"], "payment_receipt")
        self.assertEqual(d["evidence"]["upi_id"], "user@okicici")

        reconstructed = ScreenshotMetadata.from_dict(d)
        self.assertEqual(reconstructed.screenshot_id, original.screenshot_id)
        self.assertEqual(reconstructed.content_type, original.content_type)
        self.assertEqual(reconstructed.intent, original.intent)
        self.assertEqual(reconstructed.financial_info["payment_method"], "UPI")

    def test_backward_compatibility_old_dict_without_new_fields(self):
        """Verify older serialized dictionary without new fields loads cleanly with safe defaults."""
        old_data = {
            "screenshot_id": "ss_old_30",
            "filename": "ss_old_30.jpg",
            "image_uri": "/path/to/old.jpg",
            "category": "Education",
            "tags": ["syllabus"],
            "summary": "Old summary",
            "entities": [{"text": "KIIT", "type": "institution"}],
        }

        meta = ScreenshotMetadata.from_dict(old_data)
        self.assertEqual(meta.screenshot_id, "ss_old_30")
        self.assertEqual(meta.category, "Education")
        self.assertEqual(meta.content_type, "unknown")
        self.assertIsNone(meta.intent)
        self.assertEqual(meta.evidence, {})
        self.assertEqual(meta.events, [])
        self.assertEqual(meta.confidence_scores, {})

    def test_taxonomy_content_types(self):
        """Verify taxonomy defines valid content types per category."""
        self.assertIn("flight_ticket", CATEGORY_CONTENT_TYPES["Travel"])
        self.assertIn("payment_receipt", CATEGORY_CONTENT_TYPES["Payment"])
        self.assertIn("course_material", CATEGORY_CONTENT_TYPES["Education"])


class TestSQLiteMigrationAnd30RecordsPersistence(unittest.TestCase):
    """Tests SQLite non-destructive schema migration and 30-record persistence."""

    def test_sqlite_migration_on_temp_db(self):
        """Verify non-destructive migration on SQLite database containing old schema."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
            tmp_db = tmp.name

        try:
            # 1. Manually create an older table schema (without new Phase 4A columns)
            import sqlite3
            conn = sqlite3.connect(tmp_db)
            conn.execute(
                """
                CREATE TABLE screenshot_metadata (
                    screenshot_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    image_uri TEXT NOT NULL,
                    category TEXT DEFAULT 'General',
                    ocr_available INTEGER DEFAULT 1,
                    created_at TEXT,
                    indexed_at TEXT,
                    tags TEXT,
                    intent TEXT,
                    summary TEXT,
                    importance INTEGER,
                    entities TEXT,
                    dates TEXT,
                    action_items TEXT,
                    sensitive_info TEXT
                )
                """
            )
            conn.execute(
                """
                INSERT INTO screenshot_metadata (screenshot_id, filename, image_uri, category, summary)
                VALUES ('ss_mig_1', 'ss_mig_1.jpg', '/uri/1', 'Education', 'Old record')
                """
            )
            conn.commit()
            conn.close()

            # 2. Instantiate SQLiteMetadataStore (triggers non-destructive migration)
            store = SQLiteMetadataStore(db_path=tmp_db)

            # 3. Retrieve pre-existing record
            rec = store.get_metadata("ss_mig_1")
            self.assertIsNotNone(rec)
            self.assertEqual(rec.screenshot_id, "ss_mig_1")
            self.assertEqual(rec.category, "Education")
            self.assertEqual(rec.content_type, "unknown")
            self.assertEqual(rec.evidence, {})

            # 4. Save a new record with new evidence fields
            new_meta = ScreenshotMetadata(
                screenshot_id="ss_mig_2",
                filename="ss_mig_2.jpg",
                image_uri="/uri/2",
                category="Payment",
                content_type="payment_receipt",
                evidence={"upi_id": "test@okicici"},
                financial_info={"amounts": ["₹500.00"]},
            )
            store.save_metadata(new_meta)

            rec2 = store.get_metadata("ss_mig_2")
            self.assertIsNotNone(rec2)
            self.assertEqual(rec2.content_type, "payment_receipt")
            self.assertEqual(rec2.evidence["upi_id"], "test@okicici")

        finally:
            if os.path.exists(tmp_db):
                os.remove(tmp_db)

    def test_existing_30_records_in_repository_database(self):
        """Verify all 30 existing records in data/metadata.sqlite remain intact and readable."""
        db_path = Path(__file__).resolve().parent.parent / "data" / "metadata.sqlite"
        if not db_path.exists():
            self.skipTest("data/metadata.sqlite not found")

        store = SQLiteMetadataStore(db_path=str(db_path))
        records = store.list_all_metadata()
        self.assertGreaterEqual(len(records), 30, f"Expected at least 30 records, found {len(records)}")

        for rec in records:
            self.assertTrue(rec.screenshot_id.startswith("ss_"))
            self.assertIsNotNone(rec.content_type)


class TestEvidenceExtractionLayer(unittest.TestCase):
    """Tests modular evidence extraction functions and derived content_type resolution."""

    def test_extract_events(self):
        text = "Paid successfully ₹689.00 via UPI. Payment Id: pay_123"
        fin = extract_financial_info(text)
        events = extract_events(text, financial_info=fin)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "payment")

    def test_extract_text_topics(self):
        text = "Airtel_5G Wi-Fi Password: supersecret123"
        topics = extract_text_topics(text)
        self.assertIn("connectivity", topics)
        self.assertIn("credentials", topics)

    def test_resolve_content_type(self):
        text_flight = "IndiGo Flight Boarding Pass Gate 4B Seat 12A"
        ct_flight = resolve_content_type("Travel", text_flight, {}, [])
        self.assertEqual(ct_flight, "flight_ticket")

        text_wifi = "Airtel_5G Wi-Fi Password: supersecret123"
        ct_wifi = resolve_content_type("Connectivity", text_wifi, {}, [])
        self.assertEqual(ct_wifi, "wifi_credentials")


if __name__ == "__main__":
    unittest.main()
