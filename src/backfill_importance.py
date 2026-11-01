"""
Backfill Importance Script
==========================
Populates importance_score (1-5) and importance_reason for all existing screenshots
in SQLite using ImportanceScorer.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any

from src.config import config
from src.importance_scorer import ImportanceScorer
from src.storage.metadata_store import SQLiteMetadataStore


def run_importance_backfill(db_path: str = None) -> Dict[str, Any]:
    """
    Backfills importance scores and reasons for all records in SQLite.
    Returns summary statistics.
    """
    target_db = db_path or config.metadata_db_path
    db_file = Path(target_db)
    if not db_file.exists():
        raise FileNotFoundError(f"Database not found at {target_db}")

    # Ensure schema migrations run
    store = SQLiteMetadataStore(db_path=target_db)

    scorer = ImportanceScorer()
    start_time = time.time()

    conn = sqlite3.connect(target_db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT * FROM screenshot_metadata")
    rows = cur.fetchall()
    total_records = len(rows)

    score_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    updated_count = 0

    for r in rows:
        row_dict = dict(r)
        sid = row_dict["screenshot_id"]
        category = row_dict.get("category")
        content_type = row_dict.get("content_type")
        intent = row_dict.get("intent")

        # Parse JSON fields safely
        def _parse(val, default):
            if not val:
                return default
            if isinstance(val, (dict, list)):
                return val
            try:
                return json.loads(val)
            except Exception:
                return default

        evidence = _parse(row_dict.get("evidence"), {})
        ocr_text = evidence.get("text_snippet") or row_dict.get("summary") or ""
        financial_info = _parse(row_dict.get("financial_info"), {})
        dates = _parse(row_dict.get("dates"), [])
        entities = _parse(row_dict.get("entities"), [])
        sensitive_info = _parse(row_dict.get("sensitive_info"), {})
        action_items = _parse(row_dict.get("action_items"), [])
        urls = _parse(row_dict.get("urls"), [])
        contacts = _parse(row_dict.get("contacts"), {})

        score, reason = scorer.score_screenshot(
            ocr_text=ocr_text,
            category=category,
            intent=intent,
            content_type=content_type,
            financial_info=financial_info,
            dates=dates,
            entities=entities,
            sensitive_info=sensitive_info,
            action_items=action_items,
            urls=urls,
            contacts=contacts,
        )

        cur.execute(
            """
            UPDATE screenshot_metadata
            SET importance = ?, importance_score = ?, importance_reason = ?
            WHERE screenshot_id = ?
            """,
            (score, score, reason, sid),
        )
        score_counts[score] = score_counts.get(score, 0) + 1
        updated_count += 1

    conn.commit()
    conn.close()

    elapsed = time.time() - start_time
    stats = {
        "total_records": total_records,
        "updated_records": updated_count,
        "score_distribution": score_counts,
        "elapsed_seconds": round(elapsed, 3),
    }

    print(f"Importance backfill complete in {elapsed:.2f}s:")
    print(f"  Total records: {total_records}")
    print(f"  Score 5 (Critical):    {score_counts[5]}")
    print(f"  Score 4 (Actionable):  {score_counts[4]}")
    print(f"  Score 3 (Reference):   {score_counts[3]}")
    print(f"  Score 2 (General):     {score_counts[2]}")
    print(f"  Score 1 (Low-value):   {score_counts[1]}")

    return stats


if __name__ == "__main__":
    run_importance_backfill()
