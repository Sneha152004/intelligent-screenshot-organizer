"""
Intelligence Metadata Backfill Test Suite
==========================================
Validates:
1. Dry run makes zero database changes (SHA-256 verified)
2. Live backfill updates eligible records
3. Existing metadata values (category, tags, summary, filename) are strictly preserved
4. Repeated execution is idempotent (second run updates 0 records, skips all)
5. Empty OCR records are handled safely with safe defaults
6. Malformed records do not abort the full run
7. Sensitive information is never persisted as cleartext passwords
8. Database backup and transaction handling are safe and verifiable
9. Category assignments and unrelated fields remain completely unchanged
"""

import json
from pathlib import Path
import sqlite3
import pytest

from src.backfill_intelligence import (
    IntelligenceBackfillManager,
    compute_file_sha256,
    compute_table_sha256,
    create_sqlite_backup,
)
from src.extractor import StructuredExtractor
from src.intent_detector import ScreenshotIntentDetector


@pytest.fixture
def test_db_setup(tmp_path):
    """Sets up an isolated test SQLite database populated with realistic synthetic records."""
    db_file = tmp_path / "test_metadata.sqlite"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE screenshot_metadata (
            screenshot_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            image_uri TEXT NOT NULL,
            category TEXT DEFAULT 'General',
            content_type TEXT DEFAULT 'unknown',
            ocr_available INTEGER DEFAULT 1,
            created_at TEXT,
            indexed_at TEXT,
            tags TEXT,
            intent TEXT,
            summary TEXT,
            importance INTEGER,
            evidence TEXT,
            events TEXT,
            financial_info TEXT,
            contacts TEXT,
            urls TEXT,
            text_topics TEXT,
            entities TEXT,
            dates TEXT,
            action_items TEXT,
            sensitive_info TEXT,
            confidence_scores TEXT,
            category_candidates TEXT,
            category_scores TEXT,
            classification_confidence REAL DEFAULT 0.0,
            classification_ambiguous INTEGER DEFAULT 0,
            classification_margin REAL DEFAULT 0.0,
            classification_evidence TEXT
        )
        """
    )

    records = [
        # Record 1: Shopping order with amount & order ID
        (
            "ss_shop_01",
            "amazon_order.png",
            "/data/screenshots/amazon_order.png",
            "Shopping",
            "unknown",
            1,
            "2026-10-01T10:00:00",
            "2026-10-01T10:05:00",
            json.dumps(["amazon", "order"]),
            None,
            "Amazon order confirmation",
            None,
            json.dumps({
                "text_snippet": "Amazon.in Order # 402-1234567-8901234\nTotal: ₹1,499.00\nDelivered Monday",
                "visual_evidence": {"screen_type": "product_page"},
            }),
            None, None, None, None, None, None, None, None, None, None,
            None, None, 0.95, 0, 0.5, None,
        ),
        # Record 2: Payment receipt with UPI
        (
            "ss_pay_02",
            "gpay_receipt.png",
            "/data/screenshots/gpay_receipt.png",
            "Payment",
            "unknown",
            1,
            "2026-10-02T11:00:00",
            "2026-10-02T11:05:00",
            json.dumps(["gpay", "upi"]),
            None,
            "UPI payment to Ramesh",
            None,
            json.dumps({
                "text_snippet": "Paid to Ramesh Kumar\nUPI ID: ramesh@oksbi\nAmount: ₹450.00\nStatus: Paid Successfully",
                "visual_evidence": None,
            }),
            None, None, None, None, None, None, None, None, None, None,
            None, None, 0.95, 0, 0.5, None,
        ),
        # Record 3: Sensitive Wi-Fi credential
        (
            "ss_wifi_03",
            "wifi_details.png",
            "/data/screenshots/wifi_details.png",
            "Connectivity",
            "unknown",
            1,
            "2026-10-03T12:00:00",
            "2026-10-03T12:05:00",
            json.dumps(["wifi"]),
            None,
            "Wi-Fi credentials for office",
            None,
            json.dumps({
                "text_snippet": "Wi-Fi Network: GuestNet\nPassword: TopSecretPassword99!\nShow password",
            }),
            None, None, None, None, None, None, None, None, None, None,
            None, None, 0.90, 0, 0.4, None,
        ),
        # Record 4: Empty / No OCR text record
        (
            "ss_empty_04",
            "empty_screen.png",
            "/data/screenshots/empty_screen.png",
            "General",
            "unknown",
            0,
            "2026-10-04T13:00:00",
            "2026-10-04T13:05:00",
            json.dumps([]),
            None,
            "General screenshot",
            None,
            None,  # Null evidence
            None, None, None, None, None, None, None, None, None, None,
            None, None, 0.50, 0, 0.0, None,
        ),
        # Record 5: Malformed JSON evidence
        (
            "ss_malformed_05",
            "bad_json.png",
            "/data/screenshots/bad_json.png",
            "Work & Career",
            "unknown",
            1,
            "2026-10-05T14:00:00",
            "2026-10-05T14:05:00",
            json.dumps(["work"]),
            None,
            "Corrupt evidence record",
            None,
            "INVALID_NOT_JSON{[[",
            None, None, None, None, None, None, None, None, None, None,
            None, None, 0.70, 0, 0.2, None,
        ),
    ]

    cur.executemany(
        """
        INSERT INTO screenshot_metadata (
            screenshot_id, filename, image_uri, category, content_type,
            ocr_available, created_at, indexed_at, tags, intent, summary,
            importance, evidence, events, financial_info, contacts, urls,
            text_topics, entities, dates, action_items, sensitive_info,
            confidence_scores, category_candidates, category_scores,
            classification_confidence, classification_ambiguous,
            classification_margin, classification_evidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )
    conn.commit()
    conn.close()

    return db_file


# ── Test 1: Dry Run makes zero database changes ───────────────────────────────

def test_dry_run_makes_zero_database_changes(test_db_setup):
    manager = IntelligenceBackfillManager(db_path=test_db_setup)
    res = manager.run_dry_run()

    assert res["dry_run"] is True
    assert res["database_unmodified"] is True
    assert res["file_sha256_before"] == res["file_sha256_after"]
    assert res["table_sha256_before"] == res["table_sha256_after"]

    # Verify directly via SQLite
    conn = sqlite3.connect(str(test_db_setup))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM screenshot_metadata WHERE intent IS NOT NULL")
    assert cur.fetchone()[0] == 0
    conn.close()


# ── Test 2: Live backfill updates eligible records ────────────────────────────

def test_live_backfill_updates_eligible_records(test_db_setup):
    manager = IntelligenceBackfillManager(db_path=test_db_setup)
    res = manager.apply_backfill(create_backup=False)

    assert res["dry_run"] is False
    assert res["updated_count"] >= 4
    assert res["failure_count"] == 1  # ss_malformed_05 failed cleanly

    conn = sqlite3.connect(str(test_db_setup))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Verify ss_shop_01
    shop = dict(cur.execute("SELECT * FROM screenshot_metadata WHERE screenshot_id = 'ss_shop_01'").fetchone())
    assert shop["intent"] == "Shopping / Purchase"
    assert shop["content_type"] in ["shopping_order", "product_page"]
    fin = json.loads(shop["financial_info"])
    assert fin.get("amount") == 1499.0

    # Verify ss_pay_02
    pay = dict(cur.execute("SELECT * FROM screenshot_metadata WHERE screenshot_id = 'ss_pay_02'").fetchone())
    assert pay["intent"] == "Payment / Transaction / Bill"
    assert pay["content_type"] in ["payment_receipt", "transaction"]
    pay_fin = json.loads(pay["financial_info"])
    assert pay_fin.get("upi_id") == "ramesh@oksbi"

    conn.close()


# ── Test 3: Existing metadata values are preserved ────────────────────────────

def test_existing_metadata_values_are_preserved(test_db_setup):
    manager = IntelligenceBackfillManager(db_path=test_db_setup)
    manager.apply_backfill(create_backup=False)

    conn = sqlite3.connect(str(test_db_setup))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    shop = dict(cur.execute("SELECT * FROM screenshot_metadata WHERE screenshot_id = 'ss_shop_01'").fetchone())
    # Crucial preservation assertions
    assert shop["category"] == "Shopping"
    assert json.loads(shop["tags"]) == ["amazon", "order"]
    assert shop["summary"] == "Amazon order confirmation"
    assert shop["filename"] == "amazon_order.png"
    assert shop["image_uri"] == "/data/screenshots/amazon_order.png"
    assert shop["created_at"] == "2026-10-01T10:00:00"

    # Evidence visual evidence preserved
    ev = json.loads(shop["evidence"])
    assert "visual_evidence" in ev
    assert ev["visual_evidence"]["screen_type"] == "product_page"

    conn.close()


# ── Test 4: Repeated execution is idempotent ──────────────────────────────────

def test_repeated_execution_is_idempotent(test_db_setup):
    manager = IntelligenceBackfillManager(db_path=test_db_setup)
    # Run 1
    res1 = manager.apply_backfill(create_backup=False)
    assert res1["updated_count"] >= 4

    # Run 2: idempotent
    res2 = manager.apply_backfill(create_backup=False)
    # All previously updated records are now skipped
    assert res2["updated_count"] == 0
    assert res2["skipped_count"] >= 4


# ── Test 5: Empty OCR records handled safely ──────────────────────────────────

def test_empty_ocr_records_handled_safely(test_db_setup):
    manager = IntelligenceBackfillManager(db_path=test_db_setup)
    manager.apply_backfill(create_backup=False)

    conn = sqlite3.connect(str(test_db_setup))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    empty_row = dict(cur.execute("SELECT * FROM screenshot_metadata WHERE screenshot_id = 'ss_empty_04'").fetchone())
    assert empty_row["category"] == "General"
    assert empty_row["intent"] is None
    assert empty_row["content_type"] == "unknown"
    assert json.loads(empty_row["financial_info"]) == {}
    assert json.loads(empty_row["entities"]) == []

    conn.close()


# ── Test 6: Malformed record does not abort full run ──────────────────────────

def test_malformed_record_does_not_abort_full_run(test_db_setup):
    manager = IntelligenceBackfillManager(db_path=test_db_setup)
    plan = manager.plan_backfill()

    assert plan["failure_count"] == 1
    assert plan["failures"][0]["screenshot_id"] == "ss_malformed_05"
    assert "Malformed evidence JSON" in plan["failures"][0]["reason"]
    # Other 4 records are still eligible
    assert plan["eligible_count"] == 4


# ── Test 7: Sensitive info not persisted as cleartext ─────────────────────────

def test_sensitive_info_not_persisted_as_cleartext(test_db_setup):
    manager = IntelligenceBackfillManager(db_path=test_db_setup)
    manager.apply_backfill(create_backup=False)

    conn = sqlite3.connect(str(test_db_setup))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    wifi_row = dict(cur.execute("SELECT * FROM screenshot_metadata WHERE screenshot_id = 'ss_wifi_03'").fetchone())
    assert wifi_row["intent"] == "Credential / Network Access"
    sens = json.loads(wifi_row["sensitive_info"])
    assert sens.get("contains_password") is True

    raw_secret = "TopSecretPassword99!"
    assert raw_secret not in wifi_row["entities"]
    assert raw_secret not in wifi_row["financial_info"]
    assert raw_secret not in wifi_row["contacts"]

    entities = json.loads(wifi_row["entities"])
    for ent in entities:
        assert raw_secret not in ent["value"]
        if ent.get("type") == "PASSWORD" or ent.get("label") == "PASSWORD":
            assert ent["value"] == "[REDACTED_PASSWORD]"

    conn.close()


# ── Test 8: Safe SQLite backup creation and verification ──────────────────────

def test_safe_sqlite_backup_creation_and_verification(test_db_setup, tmp_path):
    backup_dir = tmp_path / "backups"
    backup_path = create_sqlite_backup(test_db_setup, backup_dir=backup_dir)

    assert backup_path.exists()
    assert backup_path.stat().st_size > 0

    with sqlite3.connect(str(backup_path)) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA integrity_check")
        assert cur.fetchone()[0] == "ok"
        cur.execute("SELECT COUNT(*) FROM screenshot_metadata")
        assert cur.fetchone()[0] == 5


# ── Test 9: Category assignments remain unchanged ─────────────────────────────

def test_category_assignments_remain_unchanged(test_db_setup):
    conn = sqlite3.connect(str(test_db_setup))
    cur = conn.cursor()
    cur.execute("SELECT screenshot_id, category FROM screenshot_metadata ORDER BY screenshot_id")
    categories_before = dict(cur.fetchall())
    conn.close()

    manager = IntelligenceBackfillManager(db_path=test_db_setup)
    manager.apply_backfill(create_backup=False)

    conn = sqlite3.connect(str(test_db_setup))
    cur = conn.cursor()
    cur.execute("SELECT screenshot_id, category FROM screenshot_metadata ORDER BY screenshot_id")
    categories_after = dict(cur.fetchall())
    conn.close()

    assert categories_before == categories_after
