"""
Comprehensive Unit Tests for ScreenshotMetadata Data Contract & Storage Layers
==============================================================================
Covers identity validation, importance boundaries (1-5), optional/default fields,
structured JSON serialization/deserialization (tags, entities, dates, action_items, sensitive_info),
and SQLite persistence.
"""

import os
import shutil
import tempfile
import unittest

from src.models.metadata import ScreenshotMetadata, DEFAULT_SENSITIVE_INFO
from src.storage.file_store import LocalFileStore
from src.storage.metadata_store import SQLiteMetadataStore


class TestScreenshotMetadataContract(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_contract_")
        self.screenshots_dir = os.path.join(self.temp_dir, "screenshots")
        self.metadata_db_path = os.path.join(self.temp_dir, "metadata.sqlite")

        self.dummy_image_path = os.path.join(self.temp_dir, "dummy_ss.jpg")
        with open(self.dummy_image_path, "wb") as f:
            f.write(b"\xFF\xD8\xFF\xE0DummyImageData")

        self.file_store = LocalFileStore(storage_dir=self.screenshots_dir)
        self.metadata_store = SQLiteMetadataStore(db_path=self.metadata_db_path)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_metadata_model_creation(self):
        """Test 1: Metadata model creation with basic required fields."""
        meta = ScreenshotMetadata(
            screenshot_id="ss_001",
            filename="ss_001.jpg",
            image_uri="data/screenshots/ss_001.jpg",
            category="Connectivity",
        )
        self.assertEqual(meta.screenshot_id, "ss_001")
        self.assertEqual(meta.filename, "ss_001.jpg")
        self.assertEqual(meta.category, "Connectivity")
        self.assertIsNotNone(meta.created_at)
        self.assertIsNotNone(meta.indexed_at)

    def test_required_identity_fields_validation(self):
        """Test 2: Required identity fields validation raises ValueError on empty strings."""
        with self.assertRaises(ValueError):
            ScreenshotMetadata(screenshot_id="", filename="ss.jpg", image_uri="uri")

        with self.assertRaises(ValueError):
            ScreenshotMetadata(screenshot_id="ss", filename="", image_uri="uri")

        with self.assertRaises(ValueError):
            ScreenshotMetadata(screenshot_id="ss", filename="ss.jpg", image_uri="")

    def test_optional_fields_handling(self):
        """Test 3: Optional intelligence fields default to None or empty structures."""
        meta = ScreenshotMetadata(
            screenshot_id="ss_002",
            filename="ss_002.jpg",
            image_uri="uri_002",
        )
        self.assertIsNone(meta.intent)
        self.assertIsNone(meta.summary)
        self.assertIsNone(meta.importance)
        self.assertEqual(meta.tags, [])
        self.assertEqual(meta.entities, [])
        self.assertEqual(meta.dates, [])
        self.assertEqual(meta.action_items, [])
        self.assertEqual(meta.sensitive_info, DEFAULT_SENSITIVE_INFO)

    def test_importance_valid_range(self):
        """Test 4: Importance validation accepts integers from 1 to 5."""
        for val in [1, 2, 3, 4, 5]:
            meta = ScreenshotMetadata(
                screenshot_id="ss_imp",
                filename="ss.jpg",
                image_uri="uri",
                importance=val,
            )
            self.assertEqual(meta.importance, val)

    def test_importance_invalid_rejection(self):
        """Test 5: Importance rejection for invalid boundaries or types."""
        for invalid_val in [0, 6, -1, 10, "3", 3.5, True]:
            with self.assertRaises(ValueError):
                ScreenshotMetadata(
                    screenshot_id="ss_invalid",
                    filename="ss.jpg",
                    image_uri="uri",
                    importance=invalid_val,  # type: ignore
                )

    def test_tags_serialization(self):
        """Test 6: Tags serialization and deserialization."""
        tags_input = ["wifi", "password", "network"]
        meta = ScreenshotMetadata(
            screenshot_id="ss_tags",
            filename="ss.jpg",
            image_uri="uri",
            tags=tags_input,
        )
        self.metadata_store.save_metadata(meta)
        retrieved = self.metadata_store.get_metadata("ss_tags")
        self.assertIsNotNone(retrieved)
        self.assertIsInstance(retrieved.tags, list)
        self.assertEqual(retrieved.tags, tags_input)

    def test_entities_serialization(self):
        """Test 7: Entities serialization and deserialization."""
        entities_input = [{"text": "IRCTC", "type": "organization"}, {"text": "Guwahati", "type": "location"}]
        meta = ScreenshotMetadata(
            screenshot_id="ss_ent",
            filename="ss.jpg",
            image_uri="uri",
            entities=entities_input,
        )
        self.metadata_store.save_metadata(meta)
        retrieved = self.metadata_store.get_metadata("ss_ent")
        self.assertIsNotNone(retrieved)
        self.assertIsInstance(retrieved.entities, list)
        self.assertEqual(retrieved.entities, entities_input)

    def test_dates_serialization(self):
        """Test 8: Dates serialization and deserialization."""
        dates_input = [{"date": "2026-10-04", "type": "travel"}]
        meta = ScreenshotMetadata(
            screenshot_id="ss_dates",
            filename="ss.jpg",
            image_uri="uri",
            dates=dates_input,
        )
        self.metadata_store.save_metadata(meta)
        retrieved = self.metadata_store.get_metadata("ss_dates")
        self.assertIsNotNone(retrieved)
        self.assertIsInstance(retrieved.dates, list)
        self.assertEqual(retrieved.dates, dates_input)

    def test_action_items_serialization(self):
        """Test 9: Action items serialization and deserialization."""
        actions_input = [{"action": "Pay electricity bill", "due_date": "2026-09-25"}]
        meta = ScreenshotMetadata(
            screenshot_id="ss_actions",
            filename="ss.jpg",
            image_uri="uri",
            action_items=actions_input,
        )
        self.metadata_store.save_metadata(meta)
        retrieved = self.metadata_store.get_metadata("ss_actions")
        self.assertIsNotNone(retrieved)
        self.assertIsInstance(retrieved.action_items, list)
        self.assertEqual(retrieved.action_items, actions_input)

    def test_sensitive_info_serialization(self):
        """Test 10: Sensitive info serialization and deserialization."""
        sensitive_input = {
            "contains_password": True,
            "contains_payment_info": False,
            "contains_personal_contact": True,
        }
        meta = ScreenshotMetadata(
            screenshot_id="ss_sensitive",
            filename="ss.jpg",
            image_uri="uri",
            sensitive_info=sensitive_input,
        )
        self.metadata_store.save_metadata(meta)
        retrieved = self.metadata_store.get_metadata("ss_sensitive")
        self.assertIsNotNone(retrieved)
        self.assertIsInstance(retrieved.sensitive_info, dict)
        self.assertEqual(retrieved.sensitive_info, sensitive_input)

    def test_empty_default_values_semantics(self):
        """Test 11: Empty/default semantics when fields are not provided."""
        meta = ScreenshotMetadata(
            screenshot_id="ss_defaults",
            filename="ss.jpg",
            image_uri="uri",
        )
        self.assertEqual(meta.tags, [])
        self.assertEqual(meta.entities, [])
        self.assertEqual(meta.dates, [])
        self.assertEqual(meta.action_items, [])
        self.assertIsNone(meta.summary)
        self.assertIsNone(meta.intent)
        self.assertIsNone(meta.importance)
        self.assertEqual(meta.sensitive_info["contains_password"], False)

    def test_sqlite_persistence_across_instances(self):
        """Test 12 & 13: SQLite persistence and retrieval returning native Python structures."""
        meta = ScreenshotMetadata(
            screenshot_id="ss_persist",
            filename="ss_persist.jpg",
            image_uri="uri_persist",
            category="Payment",
            tags=["upi", "payment"],
            importance=4,
            entities=[{"text": "Braingrow", "type": "organization"}],
            dates=[{"date": "2026-08-15", "type": "transaction"}],
            action_items=[{"action": "Save receipt"}],
            sensitive_info={"contains_password": False, "contains_payment_info": True, "contains_personal_contact": False},
        )
        self.metadata_store.save_metadata(meta)

        # Re-open database with a new SQLiteMetadataStore instance
        new_store = SQLiteMetadataStore(db_path=self.metadata_db_path)
        retrieved = new_store.get_metadata("ss_persist")

        self.assertIsNotNone(retrieved)
        self.assertIsInstance(retrieved.tags, list)
        self.assertIsInstance(retrieved.entities, list)
        self.assertIsInstance(retrieved.dates, list)
        self.assertIsInstance(retrieved.action_items, list)
        self.assertIsInstance(retrieved.sensitive_info, dict)
        self.assertEqual(retrieved.importance, 4)
        self.assertEqual(retrieved.tags, ["upi", "payment"])
        self.assertEqual(retrieved.entities[0]["text"], "Braingrow")

    def test_existing_30_records_preserved(self):
        """Test 14: Existing 30 screenshot metadata records remain readable."""
        real_store = SQLiteMetadataStore(db_path="data/metadata.sqlite")
        all_records = real_store.list_all_metadata()
        self.assertEqual(len(all_records), 30)

        ss001 = real_store.get_metadata("ss_001")
        self.assertIsNotNone(ss001)
        self.assertEqual(ss001.screenshot_id, "ss_001")
        self.assertEqual(ss001.filename, "ss_001.jpg")
        self.assertEqual(ss001.category, "Connectivity")
        self.assertIsInstance(ss001.tags, list)
        self.assertIsInstance(ss001.sensitive_info, dict)


if __name__ == "__main__":
    unittest.main()
