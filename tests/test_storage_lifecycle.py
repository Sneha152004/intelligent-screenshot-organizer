"""
Isolated Regression Tests for Storage Lifecycle & Deletion Consistency
======================================================================
Tests:
1. Deletion through the actual FastAPI DELETE /api/screenshots/{id} endpoint.
2. Deleting a screenshot with a single vector chunk in ChromaDB.
3. Deleting a screenshot with multiple vector chunks in ChromaDB.
4. Ensuring unrelated screenshot vectors remain completely intact.
5. Deleting a screenshot with zero vector chunks in ChromaDB.
6. Vector store failure aborts before SQLite or disk modification (fail-fast consistency).
7. SQLite failure after vector deletion explicitly reports partial failure and retains retryability.
8. Missing physical image file handled without failing an otherwise successful deletion.
9. Repeated deletion requests return HTTP 404 cleanly.
10. Accurate vector deletion count reported in response.
11. Two HTTP upload requests with the same original filename receive distinct generated IDs
    and distinct physical storage paths.

Safety:
- Every test uses an isolated temporary directory for SQLite, file storage, and ChromaDB.
- Never touches production databases (data/metadata.sqlite, data/screenshots, chroma_db).
"""

import os
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app
from src.models.metadata import ScreenshotMetadata
from src.mock_ocr import MockOCREngine
from src.pipeline import OCRPipeline
from src.storage.file_store import LocalFileStore
from src.storage.metadata_store import SQLiteMetadataStore
from src.vector_store import VectorStoreManager


class MockEmbedder:
    """Fast deterministic embedder for isolated unit testing."""
    def embed(self, texts):
        return [[0.1] * 384 for _ in texts]


class TestStorageLifecycle(unittest.TestCase):
    """Isolated regression tests for screenshot deletion consistency and upload lifecycle."""

    def setUp(self):
        # Create an isolated temporary sandbox
        self.temp_dir = tempfile.mkdtemp(prefix="iris_lifecycle_test_")
        self.screenshots_dir = os.path.join(self.temp_dir, "screenshots")
        self.db_path = os.path.join(self.temp_dir, "metadata.sqlite")
        self.chroma_dir = os.path.join(self.temp_dir, "chroma")

        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(self.chroma_dir, exist_ok=True)

        self.file_store = LocalFileStore(storage_dir=self.screenshots_dir)
        self.metadata_store = SQLiteMetadataStore(db_path=self.db_path)
        self.vector_store = VectorStoreManager(
            db_dir=self.chroma_dir,
            collection_name="test_screenshot_chunks",
        )

        self.client = TestClient(app)

        # Patch storage accessors in screenshots API router
        self.patcher_stores = patch(
            "backend.app.api.screenshots._get_stores",
            return_value=(self.file_store, self.metadata_store),
        )
        self.patcher_vector = patch(
            "backend.app.api.screenshots._get_vector_store",
            return_value=self.vector_store,
        )
        self.patcher_stores.start()
        self.patcher_vector.start()

    def tearDown(self):
        self.patcher_vector.stop()
        self.patcher_stores.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _insert_test_screenshot(
        self,
        sid: str,
        filename: str,
        chunk_texts: list = None,
        create_file: bool = True,
    ) -> ScreenshotMetadata:
        """Helper to populate isolated test stores with a controlled screenshot."""
        img_path = os.path.join(self.screenshots_dir, f"{sid}.png")
        if create_file:
            img = Image.new("RGB", (60, 60), color=(100, 150, 200))
            img.save(img_path)

        meta = ScreenshotMetadata(
            screenshot_id=sid,
            filename=filename,
            image_uri=os.path.abspath(img_path) if create_file else img_path,
            category="Payment",
            tags=["Receipt"],
            summary=f"Screenshot {filename}",
        )
        self.metadata_store.save_metadata(meta)

        if chunk_texts:
            chunk_ids = [f"{sid}_chunk_{i}" for i in range(len(chunk_texts))]
            embeddings = [[0.1 * (i + 1)] * 384 for i in range(len(chunk_texts))]
            metas = [
                {"screenshot_id": sid, "chunk_index": i, "category": "Payment"}
                for i in range(len(chunk_texts))
            ]
            self.vector_store.add_documents(
                ids=chunk_ids,
                documents=chunk_texts,
                embeddings=embeddings,
                metadatas=metas,
            )

        return meta

    def test_01_delete_single_chunk_via_endpoint(self):
        """Test 1: Deletion of a screenshot with 1 vector chunk through actual FastAPI DELETE endpoint."""
        self._insert_test_screenshot("ss_single_01", "receipt.png", chunk_texts=["Payment confirmation ₹500"])
        self.assertEqual(self.vector_store.count_documents(), 1)

        res = self.client.delete("/api/screenshots/ss_single_01")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertTrue(data["deleted"])
        self.assertEqual(data["screenshot_id"], "ss_single_01")
        self.assertTrue(data["file_deleted"])
        self.assertTrue(data["vectors_deleted"])
        self.assertEqual(data["vector_chunks_deleted"], 1)

        # Verify SQLite record is deleted
        self.assertIsNone(self.metadata_store.get_metadata("ss_single_01"))

        # Verify ChromaDB chunk is deleted
        self.assertEqual(self.vector_store.count_documents(), 0)

        # Verify physical disk file is deleted
        expected_path = os.path.join(self.screenshots_dir, "ss_single_01.png")
        self.assertFalse(os.path.exists(expected_path))

    def test_02_delete_multi_chunk_via_endpoint(self):
        """Test 2: Deleting a screenshot with multiple vector chunks removes all associated chunks."""
        self._insert_test_screenshot(
            "ss_multi_02",
            "long_invoice.png",
            chunk_texts=["Paragraph 1", "Paragraph 2", "Paragraph 3", "Paragraph 4"],
        )
        self.assertEqual(self.vector_store.count_documents(), 4)

        res = self.client.delete("/api/screenshots/ss_multi_02")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertTrue(data["deleted"])
        self.assertEqual(data["vector_chunks_deleted"], 4)
        self.assertEqual(self.vector_store.count_documents(), 0)
        self.assertIsNone(self.metadata_store.get_metadata("ss_multi_02"))

    def test_03_unrelated_vectors_remain_intact(self):
        """Test 3: Deleting a target screenshot never touches or deletes chunks belonging to other screenshots."""
        self._insert_test_screenshot("ss_target", "target.png", chunk_texts=["Target Chunk 1", "Target Chunk 2"])
        self._insert_test_screenshot(
            "ss_preserve",
            "preserve.png",
            chunk_texts=["Keep Chunk 1", "Keep Chunk 2", "Keep Chunk 3"],
        )
        self.assertEqual(self.vector_store.count_documents(), 5)

        res = self.client.delete("/api/screenshots/ss_target")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["vector_chunks_deleted"], 2)

        # ChromaDB must retain exactly the 3 chunks belonging to ss_preserve
        self.assertEqual(self.vector_store.count_documents(), 3)
        remaining = self.vector_store.collection.get()
        self.assertEqual(len(remaining["ids"]), 3)
        self.assertTrue(all("ss_preserve" in cid for cid in remaining["ids"]))

        # SQLite must preserve ss_preserve
        self.assertIsNotNone(self.metadata_store.get_metadata("ss_preserve"))
        self.assertIsNone(self.metadata_store.get_metadata("ss_target"))

    def test_04_delete_zero_chunk_screenshot(self):
        """Test 4: Screenshots with zero vector chunks (e.g. text-poor screens) delete cleanly without error."""
        self._insert_test_screenshot("ss_zero_chunk", "empty_screen.png", chunk_texts=[])
        self.assertEqual(self.vector_store.count_documents(), 0)

        res = self.client.delete("/api/screenshots/ss_zero_chunk")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertTrue(data["deleted"])
        self.assertEqual(data["vector_chunks_deleted"], 0)
        self.assertTrue(data["file_deleted"])
        self.assertIsNone(self.metadata_store.get_metadata("ss_zero_chunk"))

    def test_05_missing_image_file_handled_cleanly(self):
        """Test 5: Missing physical image file does not fail an otherwise successful deletion."""
        self._insert_test_screenshot("ss_missing_file", "missing.png", chunk_texts=["Some text"], create_file=False)

        res = self.client.delete("/api/screenshots/ss_missing_file")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertTrue(data["deleted"])
        self.assertFalse(data["file_deleted"], "file_deleted must be False when image file did not exist.")
        self.assertTrue(data["vectors_deleted"])
        self.assertEqual(data["vector_chunks_deleted"], 1)
        self.assertIsNone(self.metadata_store.get_metadata("ss_missing_file"))
        self.assertEqual(self.vector_store.count_documents(), 0)

    def test_06_chromadb_failure_aborts_before_modifying_sqlite_or_disk(self):
        """Test 6: If vector deletion fails, SQLite metadata and disk image remain untouched (fail-fast consistency)."""
        self._insert_test_screenshot("ss_abort_test", "abort_test.png", chunk_texts=["Sensitive data"])
        expected_file = os.path.join(self.screenshots_dir, "ss_abort_test.png")
        self.assertTrue(os.path.exists(expected_file))

        with patch.object(
            self.vector_store,
            "delete_by_screenshot_id",
            side_effect=RuntimeError("ChromaDB IO error / disk lock"),
        ):
            res = self.client.delete("/api/screenshots/ss_abort_test")
            self.assertEqual(res.status_code, 500)
            self.assertIn("Vector store deletion failed", res.json()["detail"])

        # Consistency verification: SQLite metadata MUST still exist
        self.assertIsNotNone(
            self.metadata_store.get_metadata("ss_abort_test"),
            "SQLite metadata must not be deleted when vector deletion fails.",
        )
        # Disk image MUST still exist
        self.assertTrue(
            os.path.exists(expected_file),
            "Physical file must not be deleted when vector deletion fails.",
        )

    def test_07_sqlite_failure_reports_partial_failure_with_safe_state(self):
        """Test 7: If SQLite deletion fails after vector deletion, reports partial failure explicitly."""
        self._insert_test_screenshot(
            "ss_partial_test",
            "partial.png",
            chunk_texts=["Invoice chunk 1", "Invoice chunk 2"],
        )

        with patch.object(
            self.metadata_store,
            "delete_metadata",
            side_effect=sqlite3.OperationalError("Database disk image is locked"),
        ):
            res = self.client.delete("/api/screenshots/ss_partial_test")
            self.assertEqual(res.status_code, 500)
            detail = res.json()["detail"]

            self.assertEqual(detail["error"], "Partial deletion failure")
            self.assertEqual(detail["screenshot_id"], "ss_partial_test")
            self.assertEqual(detail["vector_chunks_deleted"], 2)
            self.assertFalse(detail["metadata_deleted"])

        # ChromaDB chunks were removed
        self.assertEqual(self.vector_store.count_documents(), 0)
        # SQLite metadata is still present
        self.assertIsNotNone(self.metadata_store.get_metadata("ss_partial_test"))

        # Subsequent retry should successfully clean up without error
        retry_res = self.client.delete("/api/screenshots/ss_partial_test")
        self.assertEqual(retry_res.status_code, 200)
        self.assertEqual(retry_res.json()["vector_chunks_deleted"], 0)
        self.assertIsNone(self.metadata_store.get_metadata("ss_partial_test"))

    def test_08_repeated_deletion_requests_return_404(self):
        """Test 8: First deletion request succeeds; second request returns HTTP 404."""
        self._insert_test_screenshot("ss_repeat_test", "repeat.png", chunk_texts=["Some content"])

        # Call 1: Success
        res1 = self.client.delete("/api/screenshots/ss_repeat_test")
        self.assertEqual(res1.status_code, 200)
        self.assertTrue(res1.json()["deleted"])

        # Call 2: Repeated request returns 404
        res2 = self.client.delete("/api/screenshots/ss_repeat_test")
        self.assertEqual(res2.status_code, 404)
        self.assertEqual(res2.json()["detail"], "Screenshot not found")

    def test_09_non_existent_screenshot_returns_404(self):
        """Test 9: Attempting to delete a non-existent screenshot returns 404 immediately."""
        res = self.client.delete("/api/screenshots/ss_does_not_exist_99999")
        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.json()["detail"], "Screenshot not found")

    def test_10_upload_endpoint_duplicate_filenames_generate_distinct_ids_and_paths(self):
        """
        Test 10: Two HTTP upload requests with the exact same original filename
        generate distinct screenshot IDs and distinct physical storage paths.
        """
        isolated_pipeline = OCRPipeline(
            file_store=self.file_store,
            metadata_store=self.metadata_store,
            ocr_engine=MockOCREngine(),
            enable_visual=False,
        )

        with patch("backend.app.core.pipeline_instance.get_pipeline", return_value=isolated_pipeline), \
             patch("backend.app.core.pipeline_instance.get_embedder", return_value=MockEmbedder()):

            # Create dummy image bytes
            img_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\r\xef\x8a\x19\x00\x00\x00\x00IEND\xaeB`\x82"

            # Upload Request 1
            res1 = self.client.post(
                "/api/screenshots/upload",
                files=[("files", ("receipt_october.png", img_bytes, "image/png"))],
            )
            self.assertEqual(res1.status_code, 200)
            data1 = res1.json()["results"][0]
            sid1 = data1["screenshot_id"]

            # Upload Request 2 (identical original filename)
            res2 = self.client.post(
                "/api/screenshots/upload",
                files=[("files", ("receipt_october.png", img_bytes, "image/png"))],
            )
            self.assertEqual(res2.status_code, 200)
            data2 = res2.json()["results"][0]
            sid2 = data2["screenshot_id"]

            # 1. Internal screenshot IDs must be distinct
            self.assertNotEqual(sid1, sid2, "Two uploads must generate distinct screenshot IDs.")

            # 2. Both must faithfully retain the user-uploaded display filename
            m1 = self.metadata_store.get_metadata(sid1)
            m2 = self.metadata_store.get_metadata(sid2)
            self.assertIsNotNone(m1)
            self.assertIsNotNone(m2)
            self.assertEqual(m1.filename, "receipt_october.png")
            self.assertEqual(m2.filename, "receipt_october.png")

            # 3. Physical disk paths must be distinct and exist
            self.assertNotEqual(m1.image_uri, m2.image_uri)
            self.assertTrue(os.path.exists(m1.image_uri))
            self.assertTrue(os.path.exists(m2.image_uri))
            self.assertTrue(m1.image_uri.endswith(f"{sid1}.png"))
            self.assertTrue(m2.image_uri.endswith(f"{sid2}.png"))


if __name__ == "__main__":
    unittest.main()
