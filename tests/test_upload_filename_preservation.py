"""
Regression Tests for Upload Original Filename Preservation
===========================================================
Verifies that:
1. When a screenshot upload is staged under a UUID-based temporary path, the original
   uploaded filename is faithfully preserved in the persisted ScreenshotMetadata and SQLite store.
2. When original filename metadata is absent (e.g. legacy/direct callers), the pipeline
   gracefully falls back to the source image filename on disk.
3. When two uploads share the exact same original filename, both screenshots receive distinct
   internal screenshot IDs and distinct physical storage paths without collision or data loss.

Safety:
- Uses isolated temporary directories for SQLite database, file storage, and image staging.
- Never accesses or modifies production data (data/metadata.sqlite, data/screenshots, chroma_db).
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from src.pipeline import OCRPipeline
from src.mock_ocr import MockOCREngine
from src.storage.file_store import LocalFileStore
from src.storage.metadata_store import SQLiteMetadataStore


class TestUploadFilenamePreservation(unittest.TestCase):
    """Isolated regression tests for original uploaded filename preservation."""

    def setUp(self):
        # Create an isolated sandbox environment
        self.temp_dir = tempfile.mkdtemp(prefix="iris_test_fn_")
        self.file_dir = os.path.join(self.temp_dir, "screenshots")
        self.db_path = os.path.join(self.temp_dir, "metadata.sqlite")

        self.file_store = LocalFileStore(storage_dir=self.file_dir)
        self.metadata_store = SQLiteMetadataStore(db_path=self.db_path)

        # Fast, deterministic pipeline without external model loading
        self.pipeline = OCRPipeline(
            file_store=self.file_store,
            metadata_store=self.metadata_store,
            ocr_engine=MockOCREngine(),
            enable_visual=False,
        )

    def tearDown(self):
        # Clean up temporary sandbox directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_preserves_original_filename_with_uuid_staged_path(self):
        """
        Test 1: Preserves an original filename when the staged upload uses a UUID-based temporary filename.
        Simulates: upload endpoint writing bytes to /tmp/iris_upload_xxx/3f9a1b2c4e5d.png
        while metadata carries original filename 'IndiGo_BoardingPass_Guwahati.png'.
        """
        uuid_staged_path = os.path.join(self.temp_dir, "3f9a1b2c4e5d6a7b.png")
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        img.save(uuid_staged_path)

        doc = {
            "image_path": uuid_staged_path,
            "metadata": {
                "Screenshot ID": "test_sid_001",
                "Actual Category": "Travel",
                "filename": "IndiGo_BoardingPass_Guwahati.png",
            },
        }

        processed = self.pipeline.process_documents([doc])
        self.assertEqual(len(processed), 1)

        # 1. In-memory processed document metadata reflects original filename
        self.assertEqual(
            processed[0]["metadata"]["filename"],
            "IndiGo_BoardingPass_Guwahati.png",
            "In-memory metadata filename should match original uploaded filename.",
        )

        # 2. Persisted SQLite metadata reflects original filename
        saved_meta = self.metadata_store.get_metadata("test_sid_001")
        self.assertIsNotNone(saved_meta, "Metadata record must exist in SQLite.")
        self.assertEqual(
            saved_meta.filename,
            "IndiGo_BoardingPass_Guwahati.png",
            "SQLite metadata record must store the original uploaded filename, not the UUID temp filename.",
        )
        self.assertEqual(saved_meta.screenshot_id, "test_sid_001")

        # 3. Physical disk file uses screenshot_id, keeping storage unique and collision-free
        expected_storage_path = os.path.join(self.file_dir, "test_sid_001.png")
        self.assertTrue(
            os.path.exists(expected_storage_path),
            f"Physical storage file must exist at {expected_storage_path}",
        )
        self.assertEqual(saved_meta.image_uri, os.path.abspath(expected_storage_path))

    def test_fallback_to_source_filename_when_metadata_absent(self):
        """
        Test 2: Falls back to the source filename when original filename metadata is absent.
        Ensures backward-compatibility for dataset ingestion (e.g. load_dataset).
        """
        src_img_path = os.path.join(self.temp_dir, "legacy_dataset_img.jpg")
        img = Image.new("RGB", (100, 100), color=(200, 200, 200))
        img.save(src_img_path)

        doc = {
            "image_path": src_img_path,
            "metadata": {
                "Screenshot ID": "test_sid_002",
                "Actual Category": "Shopping",
                # Note: No 'filename' key provided
            },
        }

        processed = self.pipeline.process_documents([doc])
        self.assertEqual(len(processed), 1)

        # Must fall back to Path(src_path).name ("legacy_dataset_img.jpg")
        self.assertEqual(
            processed[0]["metadata"]["filename"],
            "legacy_dataset_img.jpg",
            "Should fall back to source file name when filename metadata is omitted.",
        )

        saved_meta = self.metadata_store.get_metadata("test_sid_002")
        self.assertIsNotNone(saved_meta)
        self.assertEqual(
            saved_meta.filename,
            "legacy_dataset_img.jpg",
            "SQLite metadata record should store the fallback source filename.",
        )

    def test_same_original_filename_retains_distinct_ids_and_storage(self):
        """
        Test 3: Processing two uploads with the same original filename while retaining
        distinct screenshot IDs and physical storage paths.
        Simulates: Two different users or requests uploading 'payment_receipt.png'.
        """
        temp_path_1 = os.path.join(self.temp_dir, "temp_staged_1.png")
        temp_path_2 = os.path.join(self.temp_dir, "temp_staged_2.png")

        img1 = Image.new("RGB", (100, 100), color=(10, 20, 30))
        img1.save(temp_path_1)
        img2 = Image.new("RGB", (100, 100), color=(40, 50, 60))
        img2.save(temp_path_2)

        doc1 = {
            "image_path": temp_path_1,
            "metadata": {
                "Screenshot ID": "upload_sid_alpha",
                "filename": "payment_receipt.png",
            },
        }
        doc2 = {
            "image_path": temp_path_2,
            "metadata": {
                "Screenshot ID": "upload_sid_beta",
                "filename": "payment_receipt.png",  # Same original filename
            },
        }

        processed = self.pipeline.process_documents([doc1, doc2])
        self.assertEqual(len(processed), 2)

        meta1 = self.metadata_store.get_metadata("upload_sid_alpha")
        meta2 = self.metadata_store.get_metadata("upload_sid_beta")

        self.assertIsNotNone(meta1)
        self.assertIsNotNone(meta2)

        # 1. Both retain the original display filename
        self.assertEqual(meta1.filename, "payment_receipt.png")
        self.assertEqual(meta2.filename, "payment_receipt.png")

        # 2. Both retain distinct internal IDs
        self.assertNotEqual(meta1.screenshot_id, meta2.screenshot_id)

        # 3. Both have distinct physical disk paths and neither overwrote the other
        self.assertNotEqual(meta1.image_uri, meta2.image_uri)
        self.assertTrue(os.path.exists(meta1.image_uri))
        self.assertTrue(os.path.exists(meta2.image_uri))
        self.assertTrue(meta1.image_uri.endswith("upload_sid_alpha.png"))
        self.assertTrue(meta2.image_uri.endswith("upload_sid_beta.png"))


if __name__ == "__main__":
    unittest.main()
