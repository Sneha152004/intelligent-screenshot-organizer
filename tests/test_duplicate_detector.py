"""
Unit and Integration Tests for Duplicate Screenshot Detection (Feature 2)
==========================================================================
Tests:
- Exact duplicate detection (SHA-256)
- Different image detection
- Visually similar image detection (pHash / Hamming distance)
- Missing and corrupt image handling
- Database persistence in SQLiteMetadataStore
- Backfill idempotency
- API endpoint response
- Safe deletion of single duplicate preserving remaining group members
"""

import io
import os
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from fastapi.testclient import TestClient

from backend.app.main import app
from src.backfill_hashes import run_hash_backfill
from src.duplicate_detector import DuplicateDetector
from src.models.metadata import ScreenshotMetadata
from src.storage.metadata_store import SQLiteMetadataStore


class TestDuplicateDetector(unittest.TestCase):
    def setUp(self):
        self.detector = DuplicateDetector(default_max_distance=6)

    def _create_test_image(self, color=(100, 150, 200), text="Sample Image"):
        img = Image.new("RGB", (200, 200), color=color)
        d = ImageDraw.Draw(img)
        d.text((20, 20), text, fill=(255, 255, 255))
        d.rectangle([50, 50, 150, 150], outline=(255, 0, 0), width=3)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_exact_duplicate_detection(self):
        """Test 1: Identical images produce identical SHA-256 and are grouped as exact duplicates."""
        img_bytes = self._create_test_image()
        sha1 = self.detector.compute_sha256(img_bytes)
        sha2 = self.detector.compute_sha256(img_bytes)
        self.assertEqual(sha1, sha2)
        self.assertIsNotNone(sha1)
        self.assertEqual(len(sha1), 64)

        phash = self.detector.compute_phash(img_bytes)

        screenshots = [
            {"screenshot_id": "shot_1", "filename": "shot_1.png", "file_hash": sha1, "perceptual_hash": phash, "created_at": "2026-01-01T10:00:00"},
            {"screenshot_id": "shot_2", "filename": "shot_2.png", "file_hash": sha2, "perceptual_hash": phash, "created_at": "2026-01-01T10:05:00"},
        ]

        result = self.detector.find_duplicates(screenshots)
        self.assertEqual(result["total_groups"], 1)
        self.assertEqual(result["exact_groups_count"], 1)
        self.assertEqual(result["visual_groups_count"], 0)
        self.assertEqual(result["total_duplicate_candidates"], 1)

        group = result["groups"][0]
        self.assertEqual(group["type"], "exact")
        self.assertEqual(group["similarity_score"], 1.0)
        self.assertEqual(group["primary"]["screenshot_id"], "shot_1")
        self.assertEqual(group["duplicates"][0]["screenshot_id"], "shot_2")

    def test_different_image_detection(self):
        """Test 2: Two completely different images produce different hashes and are not grouped."""
        img_a = self._create_test_image(color=(0, 0, 0), text="Black Screen")
        img_b = self._create_test_image(color=(255, 255, 255), text="White Bright Screen")

        sha_a = self.detector.compute_sha256(img_a)
        sha_b = self.detector.compute_sha256(img_b)
        self.assertNotEqual(sha_a, sha_b)

        phash_a = self.detector.compute_phash(img_a)
        phash_b = self.detector.compute_phash(img_b)
        dist = self.detector.hamming_distance(phash_a, phash_b)
        self.assertGreater(dist, 6, "Different images should have Hamming distance > 6")

        screenshots = [
            {"screenshot_id": "shot_a", "filename": "shot_a.png", "file_hash": sha_a, "perceptual_hash": phash_a},
            {"screenshot_id": "shot_b", "filename": "shot_b.png", "file_hash": sha_b, "perceptual_hash": phash_b},
        ]
        result = self.detector.find_duplicates(screenshots, max_distance=6)
        self.assertEqual(result["total_groups"], 0)
        self.assertEqual(result["total_duplicate_candidates"], 0)

    def test_visually_similar_image_detection(self):
        """Test 3: Near-identical images (slight brightness change) are detected as visual duplicates."""
        img = Image.new("RGB", (250, 250), color=(80, 120, 160))
        d = ImageDraw.Draw(img)
        d.rectangle([30, 30, 220, 220], fill=(200, 50, 50))
        d.ellipse([70, 70, 180, 180], fill=(255, 255, 0))

        # Modified version: slight brightness enhancement (simulating re-compression / gamma shift)
        enhancer = ImageEnhance.Brightness(img)
        img_modified = enhancer.enhance(1.05)

        phash1 = self.detector.compute_phash(img)
        phash2 = self.detector.compute_phash(img_modified)

        dist = self.detector.hamming_distance(phash1, phash2)
        self.assertLessEqual(dist, 6, f"Expected near-identical images to have distance <= 6, got {dist}")

        # They have different SHA-256
        buf1, buf2 = io.BytesIO(), io.BytesIO()
        img.save(buf1, format="PNG")
        img_modified.save(buf2, format="JPEG", quality=90)

        sha1 = self.detector.compute_sha256(buf1.getvalue())
        sha2 = self.detector.compute_sha256(buf2.getvalue())
        self.assertNotEqual(sha1, sha2)

        screenshots = [
            {"screenshot_id": "vshot_1", "filename": "vshot_1.png", "file_hash": sha1, "perceptual_hash": phash1, "created_at": "2026-01-01T10:00:00"},
            {"screenshot_id": "vshot_2", "filename": "vshot_2.jpg", "file_hash": sha2, "perceptual_hash": phash2, "created_at": "2026-01-01T10:02:00"},
        ]
        result = self.detector.find_duplicates(screenshots, max_distance=6)
        self.assertEqual(result["total_groups"], 1)
        self.assertEqual(result["visual_groups_count"], 1)
        group = result["groups"][0]
        self.assertEqual(group["type"], "visual")
        self.assertGreaterEqual(group["similarity_score"], 0.90)

    def test_missing_and_corrupt_image_handling(self):
        """Test 4: Missing or corrupted images return None without unhandled exceptions."""
        non_existent_file = "non_existent_image_12345.png"
        self.assertIsNone(self.detector.compute_sha256(non_existent_file))
        self.assertIsNone(self.detector.compute_phash(non_existent_file))
        self.assertIsNone(self.detector.compute_dhash(non_existent_file))

        # Corrupted bytes
        corrupt_bytes = b"NOT_A_VALID_IMAGE_HEADER_RANDOM_GARBAGE_12345678"
        self.assertIsNotNone(self.detector.compute_sha256(corrupt_bytes))
        self.assertIsNone(self.detector.compute_phash(corrupt_bytes))
        self.assertIsNone(self.detector.compute_dhash(corrupt_bytes))

    def test_database_persistence(self):
        """Test 5: file_hash and perceptual_hash persist and reload cleanly in SQLite."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "test_hashes.sqlite")
            store = SQLiteMetadataStore(db_path=db_path)

            meta = ScreenshotMetadata(
                screenshot_id="hash_test_1",
                filename="hash_test_1.png",
                image_uri="/data/screenshots/hash_test_1.png",
                category="General",
                file_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                perceptual_hash="8f0f07e0f0f0f83c",
            )
            store.save_metadata(meta)

            loaded = store.get_metadata("hash_test_1")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.file_hash, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
            self.assertEqual(loaded.perceptual_hash, "8f0f07e0f0f0f83c")
            self.assertEqual(loaded.sha256, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
            self.assertEqual(loaded.phash, "8f0f07e0f0f0f83c")

    def test_backfill_idempotency(self):
        """Test 6: Backfill script runs idempotently without corrupting or altering existing records."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "test_backfill.sqlite")
            img_dir = os.path.join(tmp_dir, "screenshots")
            os.makedirs(img_dir, exist_ok=True)

            img_file = os.path.join(img_dir, "sample.png")
            with open(img_file, "wb") as f:
                f.write(self._create_test_image())

            store = SQLiteMetadataStore(db_path=db_path)
            meta = ScreenshotMetadata(
                screenshot_id="backfill_01",
                filename="sample.png",
                image_uri=img_file,
                category="Shopping",
            )
            store.save_metadata(meta)

            # First run: updates hashes
            stats1 = run_hash_backfill(db_path=db_path)
            self.assertEqual(stats1["total_records"], 1)
            self.assertEqual(stats1["updated_records"], 1)

            # Second run: already hashed, updates 0 records
            stats2 = run_hash_backfill(db_path=db_path)
            self.assertEqual(stats2["total_records"], 1)
            self.assertEqual(stats2["updated_records"], 0)

    def test_api_response(self):
        """Test 7: GET /api/screenshots/duplicates returns expected schema and status 200."""
        client = TestClient(app)
        response = client.get("/api/screenshots/duplicates")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_groups", data)
        self.assertIn("exact_groups_count", data)
        self.assertIn("visual_groups_count", data)
        self.assertIn("groups", data)

        # Test filtering by exact
        res_exact = client.get("/api/screenshots/duplicates?type=exact")
        self.assertEqual(res_exact.status_code, 200)
        data_exact = res_exact.json()
        for g in data_exact["groups"]:
            self.assertEqual(g["type"], "exact")

    def test_deleting_one_duplicate_preserves_group_members(self):
        """Test 8: Deleting one duplicate candidate does NOT delete the primary or other group members."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "test_del.sqlite")
            store = SQLiteMetadataStore(db_path=db_path)

            sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            meta_primary = ScreenshotMetadata(screenshot_id="m_prim", filename="m_prim.png", image_uri="/data/1.png", file_hash=sha)
            meta_dup1 = ScreenshotMetadata(screenshot_id="m_dup1", filename="m_dup1.png", image_uri="/data/2.png", file_hash=sha)
            meta_dup2 = ScreenshotMetadata(screenshot_id="m_dup2", filename="m_dup2.png", image_uri="/data/3.png", file_hash=sha)

            store.save_metadata(meta_primary)
            store.save_metadata(meta_dup1)
            store.save_metadata(meta_dup2)

            self.assertEqual(len(store.list_all_metadata()), 3)

            # Delete only m_dup1
            store.delete_metadata("m_dup1")

            remaining = store.list_all_metadata()
            remaining_ids = {m.screenshot_id for m in remaining}
            self.assertEqual(len(remaining), 2)
            self.assertIn("m_prim", remaining_ids, "Primary screenshot must be preserved")
            self.assertIn("m_dup2", remaining_ids, "Sibling duplicate must be preserved")
            self.assertNotIn("m_dup1", remaining_ids, "Deleted duplicate must be removed")


if __name__ == "__main__":
    unittest.main()
