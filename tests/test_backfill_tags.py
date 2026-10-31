"""
Unit Tests for I.R.I.S. Tag Backfill Utility
=============================================
Tests all safety guarantees:
1. Dry-run mode produces zero database writes.
2. True cryptographic SHA-256 hash verification detects any database mutation.
3. Synthetic descriptions are never used as genuine OCR evidence.
4. Existing tags are preserved and deduplicated (idempotency).
5. Separation of tags-only backfill vs canonical category remapping.
6. Phase 1 --apply remaps all coding records to canonical categories.
7. Optional --hold-flagged preserves flagged records if requested.
8. Mutually exclusive record partitioning without double counting.
9. Phase 2 EasyOCR dry-run mode and image validation.
"""

import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.backfill_tags import (
    apply_approved_backfill,
    audit_database,
    compute_file_sha256,
    compute_table_sha256,
    execute_dry_run_audit,
    is_genuine_ocr_text,
    plan_backfill_actions,
    reprocess_legacy_images_easyocr,
    MANUAL_REVIEW_FLAGS,
)


class TestBackfillTagsUtility(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_metadata.sqlite")
        self.screenshots_dir = os.path.join(self.test_dir, "screenshots")
        os.makedirs(self.screenshots_dir, exist_ok=True)

        # Create test database
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE screenshot_metadata (
                screenshot_id TEXT PRIMARY KEY,
                filename TEXT,
                image_uri TEXT,
                category TEXT,
                ocr_available INTEGER DEFAULT 1,
                created_at TEXT,
                indexed_at TEXT,
                tags TEXT,
                intent TEXT,
                summary TEXT,
                importance TEXT,
                entities TEXT,
                action_items TEXT,
                sensitive_info TEXT,
                dates TEXT,
                content_type TEXT,
                evidence TEXT,
                events TEXT,
                financial_info TEXT,
                contacts TEXT,
                urls TEXT,
                text_topics TEXT,
                confidence_scores TEXT,
                category_candidates TEXT,
                category_scores TEXT,
                classification_confidence TEXT,
                classification_ambiguous INTEGER,
                classification_margin REAL,
                classification_evidence TEXT
            )
        """)

        # Insert representative test records
        # 1. Genuine OCR, untagged -> eligible for tag backfill
        cur.execute("""
            INSERT INTO screenshot_metadata (screenshot_id, filename, image_uri, category, tags, summary, evidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            "test_ocr_01",
            "test_ocr_01.jpg",
            os.path.join(self.screenshots_dir, "test_ocr_01.jpg"),
            "Payment",
            "[]",
            "Payment receipt",
            json.dumps({"text_snippet": "Paid successfully INR 500 via UPI transaction id 123456"}),
        ))

        # 2. Genuine OCR, already tagged -> preserved
        cur.execute("""
            INSERT INTO screenshot_metadata (screenshot_id, filename, image_uri, category, tags, summary, evidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            "test_ocr_02",
            "test_ocr_02.jpg",
            os.path.join(self.screenshots_dir, "test_ocr_02.jpg"),
            "Payment",
            json.dumps(["Payment", "UPI"]),
            "Payment receipt",
            json.dumps({"text_snippet": "Paid successfully INR 500 via UPI transaction id 123456"}),
        ))

        # 3. Synthetic description record -> skipped in Phase 1
        dummy_img = os.path.join(self.screenshots_dir, "test_syn_01.jpg")
        with open(dummy_img, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 20)  # minimal jpeg header

        cur.execute("""
            INSERT INTO screenshot_metadata (screenshot_id, filename, image_uri, category, tags, summary, evidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            "test_syn_01",
            "test_syn_01.jpg",
            dummy_img,
            "Shopping",
            None,
            "Screenshot showing a product, shopping, offer, order, or delivery page.",
            None,
        ))

        # 4. Legacy Coding record -> eligible with Coding tag
        cur.execute("""
            INSERT INTO screenshot_metadata (screenshot_id, filename, image_uri, category, tags, summary, evidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            "ss_033",
            "ss_033.jpg",
            os.path.join(self.screenshots_dir, "ss_033.jpg"),
            "Coding",
            None,
            "Screenshot showing programming or software-development content.",
            None,
        ))

        # 5. Flagged legacy Coding record (e.g. ss_280) -> remapped to General (fallback)
        cur.execute("""
            INSERT INTO screenshot_metadata (screenshot_id, filename, image_uri, category, tags, summary, evidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            "ss_280",
            "ss_280.png",
            os.path.join(self.screenshots_dir, "ss_280.png"),
            "Coding",
            None,
            "Screenshot showing programming or software-development content.",
            None,
        ))

        conn.commit()
        conn.close()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_is_genuine_ocr_text(self):
        self.assertFalse(is_genuine_ocr_text(None))
        self.assertFalse(is_genuine_ocr_text(""))
        self.assertFalse(is_genuine_ocr_text("abc"))  # too short (<5)
        self.assertFalse(is_genuine_ocr_text("Screenshot showing a product page"))
        self.assertFalse(is_genuine_ocr_text("[MOCK OCR] Sample text"))
        self.assertFalse(is_genuine_ocr_text("No predefined text available"))
        self.assertTrue(is_genuine_ocr_text("Paid successfully INR 500 via UPI"))
        self.assertTrue(is_genuine_ocr_text("Boarding Pass Indian Railways PNR 245124"))

    def test_dry_run_audit_unmodified(self):
        res = execute_dry_run_audit(db_path=self.db_path)
        self.assertTrue(res["database_unmodified"])
        self.assertEqual(res["file_sha256_before"], res["file_sha256_after"])
        self.assertEqual(res["content_sha256_before"], res["content_sha256_after"])

        # Check database rows directly
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT tags FROM screenshot_metadata WHERE screenshot_id = 'test_ocr_01'")
        self.assertEqual(cur.fetchone()[0], "[]")
        conn.close()

    def test_cryptographic_hash_detects_mutation(self):
        """Verifies that SHA-256 content hash detects any single-field row mutation."""
        h_before = compute_table_sha256(self.db_path)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        # Mutate a single tag inside an existing record
        cur.execute("UPDATE screenshot_metadata SET tags = ? WHERE screenshot_id = 'test_ocr_02'", (json.dumps(["Payment", "Invoice"]),))
        conn.commit()
        conn.close()
        h_after = compute_table_sha256(self.db_path)
        self.assertNotEqual(h_before, h_after)

    def test_mutually_exclusive_partitioning(self):
        """Verifies that eligible + skipped equals total records with zero double counting."""
        plan = plan_backfill_actions(db_path=self.db_path)
        audit = plan["audit"]
        total_records = audit["total_records"]
        total_eligible = plan["total_eligible"]
        skipped_sum = sum(plan["skipped_summary"].values())
        self.assertEqual(total_eligible + skipped_sum, total_records)

    def test_tags_only_mode_preserves_categories(self):
        # Apply with tags_only=True
        res = apply_approved_backfill(db_path=self.db_path, tags_only=True, dry_run=False)
        self.assertFalse(res["dry_run"])
        self.assertEqual(res["updated_categories_count"], 0)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Check test_ocr_01 received tags
        cur.execute("SELECT tags, category FROM screenshot_metadata WHERE screenshot_id = 'test_ocr_01'")
        tags_raw, cat = cur.fetchone()
        tags = json.loads(tags_raw)
        self.assertIn("Payment", tags)
        self.assertIn("UPI", tags)
        self.assertEqual(cat, "Payment")

        # Check legacy coding record ss_033: tags updated, category preserved
        cur.execute("SELECT tags, category FROM screenshot_metadata WHERE screenshot_id = 'ss_033'")
        tags_raw, cat = cur.fetchone()
        tags = json.loads(tags_raw)
        self.assertIn("Coding", tags)
        self.assertEqual(cat, "Coding")  # Category untouched

        conn.close()

    def test_apply_remaps_all_coding_to_canonical(self):
        # Apply with default Phase 1 behavior (tags + canonical categories)
        res = apply_approved_backfill(
            db_path=self.db_path,
            tags_only=False,
            update_categories=True,
            hold_flagged=False,
            dry_run=False,
        )
        self.assertFalse(res["dry_run"])
        self.assertEqual(res["updated_categories_count"], 2)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Check no Coding category remains in the entire table
        cur.execute("SELECT COUNT(*) FROM screenshot_metadata WHERE category = 'Coding'")
        self.assertEqual(cur.fetchone()[0], 0)

        # ss_033 is Work & Career
        cur.execute("SELECT tags, category FROM screenshot_metadata WHERE screenshot_id = 'ss_033'")
        tags_raw, cat = cur.fetchone()
        self.assertIn("Coding", json.loads(tags_raw))
        self.assertEqual(cat, "Work & Career")

        # ss_280 is General (canonical fallback from metadata_v2.csv)
        cur.execute("SELECT tags, category FROM screenshot_metadata WHERE screenshot_id = 'ss_280'")
        tags_raw, cat = cur.fetchone()
        self.assertIn("Coding", json.loads(tags_raw))
        self.assertEqual(cat, "General")

        conn.close()

    def test_hold_flagged_option(self):
        # Apply with hold_flagged=True
        res = apply_approved_backfill(
            db_path=self.db_path,
            tags_only=False,
            update_categories=True,
            hold_flagged=True,
            dry_run=False,
        )
        self.assertEqual(res["held_for_manual_review"], 1)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT category FROM screenshot_metadata WHERE screenshot_id = 'ss_280'")
        self.assertEqual(cur.fetchone()[0], "Coding")
        conn.close()

    def test_idempotent_backfill(self):
        # First pass
        apply_approved_backfill(db_path=self.db_path, tags_only=False, dry_run=False)
        # Second pass
        plan2 = plan_backfill_actions(db_path=self.db_path)
        # Eligible count should be 0 because candidate tags are already present and categories are no longer Coding
        self.assertEqual(plan2["eligible_text_snippet_count"], 0)
        self.assertEqual(plan2["eligible_legacy_coding_count"], 0)

    def test_phase2_ocr_dry_run(self):
        res = reprocess_legacy_images_easyocr(db_path=self.db_path, dry_run=True)
        self.assertTrue(res["dry_run"])
        self.assertEqual(res["total_eligible"], 1)  # test_syn_01 has dummy image
        self.assertIn("Dry run complete", res["status"])


if __name__ == "__main__":
    unittest.main()
