"""
Unit Tests for Legacy Screenshot Vector Indexer
===============================================
Verifies:
1. Dry-run mode makes zero changes to ChromaDB or SQLite.
2. Already-indexed screenshots are detected and skipped.
3. Chunk IDs are deterministic: {screenshot_id}_chunk_{chunk_index}.
4. Screenshot metadata (category, tags) is preserved in ChromaDB.
5. Duplicate physical aliases (ss_0010–ss_0030) are excluded.
6. Repeated indexing is strictly idempotent and does not duplicate chunks.
7. Partial indexing is detected and flagged.
8. SQLite remains cryptographically unmodified.
"""

import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.index_legacy_embeddings import (
    KNOWN_DUPLICATE_ALIASES,
    audit_indexing_plan,
    compute_table_sha256,
    index_legacy_screenshots,
)
from src.vector_store import VectorStoreManager


class TestIndexLegacyEmbeddings(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_legacy_indexer_")
        self.db_path = os.path.join(self.test_dir, "test_metadata.sqlite")
        self.chroma_dir = os.path.join(self.test_dir, "chroma_db")
        self.collection_name = "test_chunks"

        # Create isolated test SQLite database with mock schema matching screenshot_metadata
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE screenshot_metadata (
                screenshot_id TEXT PRIMARY KEY,
                filename TEXT,
                image_uri TEXT,
                category TEXT,
                ocr_available INTEGER,
                created_at TEXT,
                indexed_at TEXT,
                tags TEXT,
                intent TEXT,
                summary TEXT,
                importance TEXT,
                entities TEXT,
                action_items TEXT,
                sensitive_info INTEGER,
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
                classification_confidence REAL,
                classification_ambiguous INTEGER,
                classification_margin REAL,
                classification_evidence TEXT
            )
        """)

        # Insert test records
        # Record 1: Already indexed seed record
        cur.execute("""
            INSERT INTO screenshot_metadata (screenshot_id, filename, category, tags, evidence, ocr_available)
            VALUES (?, ?, ?, ?, ?, 1)
        """, (
            "ss_001",
            "ss_001.jpg",
            "Connectivity",
            json.dumps(["Networking"]),
            json.dumps({"text_snippet": "Wi-Fi Network connected successfully\n\nSSID: OfficeNet", "reprocessed": True}),
        ))

        # Record 2: Eligible unindexed record
        cur.execute("""
            INSERT INTO screenshot_metadata (screenshot_id, filename, category, tags, evidence, ocr_available)
            VALUES (?, ?, ?, ?, ?, 1)
        """, (
            "ss_372",
            "ss_372.png",
            "Shopping",
            json.dumps(["Shopping", "Order"]),
            json.dumps({"text_snippet": "SPEAKERS 40% OFF\n\nOrder total: $45.99 free delivery", "reprocessed": True}),
        ))

        # Record 3: Duplicate alias of ss_010
        cur.execute("""
            INSERT INTO screenshot_metadata (screenshot_id, filename, category, tags, evidence, ocr_available)
            VALUES (?, ?, ?, ?, ?, 1)
        """, (
            "ss_0010",
            "ss_0010.jpg",
            "Shopping",
            json.dumps(["Shopping"]),
            json.dumps({"text_snippet": "Flipkart order delivery tracking details", "reprocessed": True}),
        ))

        # Record 4: Lacks genuine OCR text (synthetic placeholder)
        cur.execute("""
            INSERT INTO screenshot_metadata (screenshot_id, filename, category, tags, evidence, ocr_available)
            VALUES (?, ?, ?, ?, ?, 1)
        """, (
            "ss_999",
            "ss_999.jpg",
            "General",
            json.dumps([]),
            json.dumps({"text_snippet": "Screenshot showing a generic user interface page", "reprocessed": False}),
        ))

        conn.commit()
        conn.close()

        # Initialize vector store and pre-populate ss_001 chunk (simulating already-indexed record)
        self.vsm = VectorStoreManager(db_dir=self.chroma_dir, collection_name=self.collection_name)
        self.vsm.add_documents(
            ids=["ss_001_chunk_0", "ss_001_chunk_1"],
            documents=["Wi-Fi Network connected successfully", "SSID: OfficeNet"],
            embeddings=[[0.05] * 384, [0.06] * 384],
            metadatas=[
                {"screenshot_id": "ss_001", "chunk_index": 0, "category": "Connectivity", "tags": "Networking"},
                {"screenshot_id": "ss_001", "chunk_index": 1, "category": "Connectivity", "tags": "Networking"},
            ],
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir, ignore_errors=True)
            except Exception:
                pass

    def test_dry_run_makes_no_changes(self):
        """Test 1: Dry run mode modifies zero ChromaDB documents and leaves SQLite unchanged."""
        initial_chunks = self.vsm.count_documents()
        initial_sha = compute_table_sha256(self.db_path)

        res = index_legacy_screenshots(
            db_path=self.db_path,
            chroma_dir=self.chroma_dir,
            collection_name=self.collection_name,
            dry_run=True,
        )

        self.assertTrue(res["dry_run"])
        self.assertEqual(res["initial_chroma_count"], initial_chunks)
        self.assertEqual(res["final_chroma_count"], initial_chunks)
        self.assertTrue(res["sqlite_unmodified"])
        self.assertEqual(compute_table_sha256(self.db_path), initial_sha)
        self.assertEqual(self.vsm.count_documents(), initial_chunks)

    def test_skips_already_indexed_screenshots(self):
        """Test 2: Already indexed screenshots (ss_001) are skipped."""
        plan = audit_indexing_plan(
            db_path=self.db_path,
            vector_store=self.vsm,
        )
        self.assertEqual(plan["already_indexed_count"], 1)
        selected_sids = [r["screenshot_id"] for r in plan["selected_records"]]
        self.assertNotIn("ss_001", selected_sids)

    def test_excludes_duplicate_aliases(self):
        """Test 3: Known duplicate alias ss_0010 is excluded from selection."""
        plan = audit_indexing_plan(
            db_path=self.db_path,
            vector_store=self.vsm,
        )
        self.assertEqual(plan["excluded_aliases_count"], 1)
        excluded_sids = [a["screenshot_id"] for a in plan["excluded_aliases"]]
        self.assertIn("ss_0010", excluded_sids)
        selected_sids = [r["screenshot_id"] for r in plan["selected_records"]]
        self.assertNotIn("ss_0010", selected_sids)

    def test_excludes_synthetic_placeholder(self):
        """Test 4: Record lacking genuine OCR text (ss_999) is excluded."""
        plan = audit_indexing_plan(
            db_path=self.db_path,
            vector_store=self.vsm,
        )
        self.assertEqual(plan["lacking_genuine_ocr_count"], 1)
        selected_sids = [r["screenshot_id"] for r in plan["selected_records"]]
        self.assertNotIn("ss_999", selected_sids)

    def test_indexing_execution_and_metadata_preservation(self):
        """Test 5 & 6: Applying index creates deterministic chunk IDs and preserves metadata."""
        initial_sha = compute_table_sha256(self.db_path)

        res = index_legacy_screenshots(
            db_path=self.db_path,
            chroma_dir=self.chroma_dir,
            collection_name=self.collection_name,
            dry_run=False,
        )

        self.assertFalse(res["dry_run"])
        self.assertEqual(res["indexed_screenshots_count"], 1)  # ss_372
        self.assertTrue(res["sqlite_unmodified"])
        self.assertEqual(compute_table_sha256(self.db_path), initial_sha)

        # Verify chunk IDs and metadata for ss_372
        doc_0 = self.vsm.get_document("ss_372_chunk_0")
        self.assertIsNotNone(doc_0)
        self.assertEqual(doc_0["id"], "ss_372_chunk_0")
        self.assertIn("SPEAKERS", doc_0["document"])
        self.assertEqual(doc_0["metadata"]["screenshot_id"], "ss_372")
        self.assertEqual(doc_0["metadata"]["category"], "Shopping")
        self.assertIn("Order", doc_0["metadata"]["tags"])

    def test_idempotency_repeated_indexing(self):
        """Test 7: Repeated execution does not duplicate chunks or records."""
        # First execution: apply
        res1 = index_legacy_screenshots(
            db_path=self.db_path,
            chroma_dir=self.chroma_dir,
            collection_name=self.collection_name,
            dry_run=False,
        )
        count_after_first = self.vsm.count_documents()

        # Second execution: apply again
        res2 = index_legacy_screenshots(
            db_path=self.db_path,
            chroma_dir=self.chroma_dir,
            collection_name=self.collection_name,
            dry_run=False,
        )
        count_after_second = self.vsm.count_documents()

        self.assertEqual(count_after_first, count_after_second)
        self.assertEqual(res2["indexed_screenshots_count"], 0)
        self.assertEqual(res2["chunks_added"], 0)

    def test_partial_indexing_detection(self):
        """Test 8: Partial indexing (missing chunks) is detected in the audit plan."""
        # Add another record with 2 expected chunks
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO screenshot_metadata (screenshot_id, filename, category, tags, evidence, ocr_available)
            VALUES (?, ?, ?, ?, ?, 1)
        """, (
            "ss_500",
            "ss_500.png",
            "Education",
            json.dumps(["Learning"]),
            json.dumps({"text_snippet": "First paragraph of notes\n\nSecond paragraph of notes", "reprocessed": True}),
        ))
        conn.commit()
        conn.close()

        # Pre-populate only chunk 0 (leaving chunk 1 missing)
        self.vsm.add_documents(
            ids=["ss_500_chunk_0"],
            documents=["First paragraph of notes"],
            embeddings=[[0.01] * 384],
            metadatas=[{"screenshot_id": "ss_500", "chunk_index": 0, "category": "Education"}],
        )

        plan = audit_indexing_plan(db_path=self.db_path, vector_store=self.vsm)
        self.assertEqual(plan["partially_indexed_count"], 1)
        partial = plan["partially_indexed_records"][0]
        self.assertEqual(partial["screenshot_id"], "ss_500")
        self.assertEqual(partial["actual_count"], 1)
        self.assertEqual(partial["expected_count"], 2)


if __name__ == "__main__":
    unittest.main()
