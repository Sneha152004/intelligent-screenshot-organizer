"""
Safe, Idempotent Intelligence Metadata Backfill Utility
========================================================
Populates Phase 1 structured intelligence metadata (intent, content_type,
entities, dates, financial_info, contacts, urls, action_items, sensitive_info,
events, text_topics, confidence_scores) for existing screenshot records in SQLite.

Safety & Architecture Guarantees:
1. Dry-run by default: no database records are modified without --apply.
2. Cryptographic SHA-256 hash verification (raw file and table rows) verifies zero mutations on dry-run.
3. Consistent SQLite backup: creates a verified backup via SQLite's online backup API before any live write.
4. Reads existing OCR text stored in SQLite (evidence.text_snippet); does NOT rerun OCR.
5. Does NOT regenerate embeddings or modify ChromaDB.
6. Does NOT alter category assignments (preserves Category Decision Policy V3).
7. Preserves filenames, image URIs, IDs, timestamps, existing tags, and existing human summaries.
8. Strict secret redaction: passwords matched in OCR are flagged in sensitive_info, and entities
   strictly use [REDACTED_PASSWORD]; cleartext secrets are never persisted or printed.
9. Fully idempotent: subsequent runs detect already populated records and skip them.
10. Parameterized SQL with atomic transaction handling (commits on success, rolls back on error).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.extractor import StructuredExtractor
from src.intent_detector import ScreenshotIntentDetector

logger = logging.getLogger("backfill_intelligence")


def compute_file_sha256(file_path: str | Path) -> str:
    """Computes SHA-256 hash of a file on disk."""
    p = Path(file_path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_table_sha256(db_path: str | Path) -> str:
    """Computes a deterministic SHA-256 hash across all rows and columns in screenshot_metadata."""
    p = Path(db_path)
    if not p.exists():
        return ""
    conn = sqlite3.connect(str(p))
    cur = conn.cursor()
    cur.execute("SELECT * FROM screenshot_metadata ORDER BY screenshot_id ASC")
    h = hashlib.sha256()
    for row in cur.fetchall():
        row_str = "|".join("" if v is None else str(v) for v in row)
        h.update(row_str.encode("utf-8"))
    conn.close()
    return h.hexdigest()


def create_sqlite_backup(
    db_path: str | Path,
    backup_dir: Optional[str | Path] = None,
) -> Path:
    """
    Creates a consistent SQLite backup using the official SQLite online backup API.
    Safe under WAL mode and concurrent access.
    Verifies integrity and row count before returning.
    """
    src_path = Path(db_path).resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Database to backup does not exist: {src_path}")

    b_dir = Path(backup_dir) if backup_dir else src_path.parent / "backups"
    b_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = b_dir / f"metadata_backup_{timestamp}.sqlite"

    with sqlite3.connect(str(src_path)) as src_conn:
        with sqlite3.connect(str(backup_path)) as dst_conn:
            src_conn.backup(dst_conn)

    # Verify backup integrity and row count
    with sqlite3.connect(str(backup_path)) as verify_conn:
        cur = verify_conn.cursor()
        cur.execute("PRAGMA integrity_check;")
        check_res = cur.fetchone()[0]
        if check_res != "ok":
            raise RuntimeError(f"Backup verification failed integrity check: {check_res}")

        cur.execute("SELECT COUNT(*) FROM screenshot_metadata;")
        backup_count = cur.fetchone()[0]

    with sqlite3.connect(str(src_path)) as orig_conn:
        orig_count = orig_conn.execute("SELECT COUNT(*) FROM screenshot_metadata;").fetchone()[0]

    if backup_count != orig_count:
        raise RuntimeError(
            f"Backup row count mismatch: backup={backup_count}, original={orig_count}"
        )

    return backup_path


class IntelligenceBackfillManager:
    """
    Coordinates planning and safe execution of intelligence metadata backfill.
    """

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        extractor: Optional[StructuredExtractor] = None,
        intent_detector: Optional[ScreenshotIntentDetector] = None,
    ):
        self.db_path = Path(db_path) if db_path else Path(config.metadata_db_path)
        self.extractor = extractor if extractor else StructuredExtractor()
        self.intent_detector = intent_detector if intent_detector else ScreenshotIntentDetector()

    def plan_backfill(
        self,
        limit: Optional[int] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Reads existing metadata and OCR snippets, generates extracted fields in memory,
        and determines update eligibility without mutating the database.
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                screenshot_id, filename, image_uri, category, content_type,
                tags, intent, summary, evidence, entities, dates,
                financial_info, contacts, urls, sensitive_info,
                action_items, events, text_topics, confidence_scores
            FROM screenshot_metadata
            ORDER BY screenshot_id ASC
            """
        )
        rows = cur.fetchall()
        conn.close()

        total_records = len(rows)
        eligible_records: List[Dict[str, Any]] = []
        skipped_records: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []

        for row in rows:
            sid = row["screenshot_id"]
            cat = row["category"] or "General"
            existing_intent = row["intent"]
            existing_content_type = row["content_type"] or "unknown"
            existing_entities = row["entities"]
            raw_tags = row["tags"]
            raw_evidence = row["evidence"]

            # Parse tags
            tags = []
            if raw_tags:
                try:
                    tags = json.loads(raw_tags) if isinstance(raw_tags, str) else list(raw_tags)
                except Exception:
                    tags = []

            # Parse evidence
            evidence = {}
            if raw_evidence:
                try:
                    evidence = json.loads(raw_evidence) if isinstance(raw_evidence, str) else dict(raw_evidence)
                except Exception as e:
                    failures.append({
                        "screenshot_id": sid,
                        "reason": f"Malformed evidence JSON: {e}",
                    })
                    continue

            # Idempotence check: if record already has intent_evidence in evidence, skip unless force
            is_already_populated = isinstance(evidence, dict) and "intent_evidence" in evidence
            if is_already_populated and not force:
                skipped_records.append({
                    "screenshot_id": sid,
                    "reason": "already_populated",
                })
                continue

            # Read existing OCR text from evidence snippet
            ocr_text = evidence.get("text_snippet", "") if isinstance(evidence, dict) else ""

            try:
                # 1. Structured content extraction
                extracted = self.extractor.extract_all(ocr_text=ocr_text, category=cat)

                # 2. Intent detection
                intent, conf, intent_ev = self.intent_detector.detect_intent(
                    ocr_text=ocr_text,
                    category=cat,
                    tags=tags,
                    content_type=extracted["content_type"],
                    financial_info=extracted["financial_info"],
                    dates=extracted["dates"],
                    entities=extracted["entities"],
                    sensitive_info=extracted["sensitive_info"],
                )

                # Strict Secret Redaction verification
                for ent in extracted["entities"]:
                    ent_val = ent.get("value", "") or ent.get("text", "")
                    if ent.get("type") == "PASSWORD" or ent.get("label") == "PASSWORD":
                        if ent_val != "[REDACTED_PASSWORD]":
                            raise ValueError(f"Unredacted password entity found: {ent}")

                # Prepare updated evidence dict preserving existing keys
                updated_evidence = dict(evidence) if isinstance(evidence, dict) else {}
                if intent:
                    updated_evidence["intent_evidence"] = intent_ev
                else:
                    updated_evidence["intent_evidence"] = {"evaluated": True, "reason": "insufficient_evidence"}

                # Determine final content_type (preserve existing if non-unknown and not force)
                final_content_type = extracted["content_type"]
                if existing_content_type != "unknown" and not force:
                    final_content_type = existing_content_type

                conf_scores = {"intent": conf} if intent else {}

                eligible_records.append({
                    "screenshot_id": sid,
                    "category": cat,
                    "content_type": final_content_type,
                    "intent": intent,
                    "entities": extracted["entities"],
                    "dates": extracted["dates"],
                    "financial_info": extracted["financial_info"],
                    "contacts": extracted["contacts"],
                    "urls": extracted["urls"],
                    "action_items": extracted["action_items"],
                    "sensitive_info": extracted["sensitive_info"],
                    "events": extracted["events"],
                    "text_topics": extracted["text_topics"],
                    "confidence_scores": conf_scores,
                    "evidence": updated_evidence,
                    "has_ocr": bool(ocr_text and ocr_text.strip()),
                })

            except Exception as e:
                failures.append({
                    "screenshot_id": sid,
                    "reason": f"Extraction error: {e}",
                })

        if limit is not None and limit > 0:
            eligible_records = eligible_records[:limit]

        return {
            "total_records": total_records,
            "eligible_records": eligible_records,
            "eligible_count": len(eligible_records),
            "skipped_count": len(skipped_records),
            "failure_count": len(failures),
            "failures": failures,
        }

    def run_dry_run(
        self,
        limit: Optional[int] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes dry run, verifying that database file and content SHA-256 remain strictly unchanged.
        """
        file_sha_before = compute_file_sha256(self.db_path)
        table_sha_before = compute_table_sha256(self.db_path)

        plan = self.plan_backfill(limit=limit, force=force)

        file_sha_after = compute_file_sha256(self.db_path)
        table_sha_after = compute_table_sha256(self.db_path)

        db_unmodified = (
            file_sha_before == file_sha_after
            and table_sha_before == table_sha_after
        )

        return {
            "dry_run": True,
            "database_path": str(self.db_path),
            "database_unmodified": db_unmodified,
            "file_sha256_before": file_sha_before,
            "file_sha256_after": file_sha_after,
            "table_sha256_before": table_sha_before,
            "table_sha256_after": table_sha_after,
            "total_records": plan["total_records"],
            "eligible_count": plan["eligible_count"],
            "skipped_count": plan["skipped_count"],
            "failure_count": plan["failure_count"],
            "failures": plan["failures"],
            "sample_eligible": [
                {
                    "screenshot_id": item["screenshot_id"],
                    "category": item["category"],
                    "content_type": item["content_type"],
                    "intent": item["intent"],
                    "has_ocr": item["has_ocr"],
                }
                for item in plan["eligible_records"][:5]
            ],
        }

    def apply_backfill(
        self,
        limit: Optional[int] = None,
        force: bool = False,
        create_backup: bool = True,
        backup_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """
        Applies intelligence metadata updates directly to SQLite within an atomic transaction.
        Creates a verified backup before writing if create_backup=True.
        """
        backup_path: Optional[Path] = None
        if create_backup:
            backup_path = create_sqlite_backup(self.db_path, backup_dir=backup_dir)

        plan = self.plan_backfill(limit=limit, force=force)
        eligible = plan["eligible_records"]

        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        updated_count = 0

        try:
            with conn:
                for item in eligible:
                    cur.execute(
                        """
                        UPDATE screenshot_metadata SET
                            content_type = ?,
                            intent = ?,
                            entities = ?,
                            dates = ?,
                            financial_info = ?,
                            contacts = ?,
                            urls = ?,
                            action_items = ?,
                            sensitive_info = ?,
                            events = ?,
                            text_topics = ?,
                            confidence_scores = ?,
                            evidence = ?
                        WHERE screenshot_id = ?
                        """,
                        (
                            item["content_type"],
                            item["intent"],
                            json.dumps(item["entities"]),
                            json.dumps(item["dates"]),
                            json.dumps(item["financial_info"]),
                            json.dumps(item["contacts"]),
                            json.dumps(item["urls"]),
                            json.dumps(item["action_items"]),
                            json.dumps(item["sensitive_info"]),
                            json.dumps(item["events"]),
                            json.dumps(item["text_topics"]),
                            json.dumps(item["confidence_scores"]),
                            json.dumps(item["evidence"]),
                            item["screenshot_id"],
                        ),
                    )
                    updated_count += 1

        except Exception as e:
            logger.error("Backfill transaction failed: %s", e)
            conn.close()
            raise

        conn.close()

        return {
            "dry_run": False,
            "database_path": str(self.db_path),
            "backup_path": str(backup_path) if backup_path else None,
            "total_records": plan["total_records"],
            "eligible_count": plan["eligible_count"],
            "updated_count": updated_count,
            "skipped_count": plan["skipped_count"],
            "failure_count": plan["failure_count"],
        }


def main():
    parser = argparse.ArgumentParser(
        description="Safe, Idempotent Intelligence Metadata Backfill for I.R.I.S."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply database changes (default is dry-run mode).",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=config.metadata_db_path,
        help="Path to SQLite metadata database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of eligible records to process.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation of already populated records.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip database backup before apply (not recommended).",
    )

    args = parser.parse_args()

    manager = IntelligenceBackfillManager(db_path=args.db_path)

    if not args.apply:
        print("\n=== DRY RUN MODE: No database changes will be made ===")
        res = manager.run_dry_run(limit=args.limit, force=args.force)
        print(f"Database: {res['database_path']}")
        print(f"Database Unmodified: {res['database_unmodified']}")
        print(f"File SHA-256 Before/After: {res['file_sha256_before'][:16]}... == {res['file_sha256_after'][:16]}...")
        print(f"Total Records: {res['total_records']}")
        print(f"Eligible Records: {res['eligible_count']}")
        print(f"Skipped Records: {res['skipped_count']}")
        print(f"Failures: {res['failure_count']}")
        print("\nSample Eligible Records:")
        for sample in res["sample_eligible"]:
            print(f"  [{sample['screenshot_id']}] Category: {sample['category']} | Content-Type: {sample['content_type']} | Intent: {sample['intent']}")
        print("\nTo apply these updates, rerun with --apply.")
    else:
        print("\n=== LIVE APPLY MODE: Updating database ===")
        res = manager.apply_backfill(
            limit=args.limit,
            force=args.force,
            create_backup=not args.no_backup,
        )
        print(f"Database: {res['database_path']}")
        print(f"Backup created: {res['backup_path']}")
        print(f"Total Records: {res['total_records']}")
        print(f"Eligible Records: {res['eligible_count']}")
        print(f"Updated Records: {res['updated_count']}")
        print(f"Skipped Records: {res['skipped_count']}")
        print(f"Failures: {res['failure_count']}")
        print("\nBackfill successfully completed.")


if __name__ == "__main__":
    main()
