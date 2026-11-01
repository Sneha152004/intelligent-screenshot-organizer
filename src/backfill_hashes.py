"""
Backfill Hashes Script
======================
Computes SHA-256 and pHash for all stored screenshots and persists them
in SQLite for exact and near-duplicate detection.
"""

import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import config
from src.duplicate_detector import DuplicateDetector
from src.storage.metadata_store import SQLiteMetadataStore


def run_hash_backfill(db_path: Optional[str] = None, force_recompute: bool = False) -> Dict[str, Any]:
    """
    Computes and populates file_hash (SHA-256) and perceptual_hash (pHash)
    for all screenshot records in SQLite.
    """
    target_db = db_path or config.metadata_db_path
    if not os.path.exists(target_db):
        raise FileNotFoundError(f"Database not found at {target_db}")

    # Ensure schema migrations run
    store = SQLiteMetadataStore(db_path=target_db)

    detector = DuplicateDetector()
    start_time = time.time()

    conn = sqlite3.connect(target_db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT screenshot_id, filename, image_uri, file_hash, perceptual_hash, category, created_at FROM screenshot_metadata")
    rows = cur.fetchall()
    total_records = len(rows)

    updated_count = 0
    missing_files_count = 0
    records_for_clustering = []

    for r in rows:
        sid = r["screenshot_id"]
        fname = r["filename"]
        img_uri = r["image_uri"]
        existing_sha = r["file_hash"]
        existing_phash = r["perceptual_hash"]

        sha = existing_sha
        phash = existing_phash

        if force_recompute or not sha or not phash:
            # Resolve image file location
            target_file = None
            candidate_paths = [
                img_uri,
                os.path.join(os.getcwd(), img_uri.lstrip("/\\")) if img_uri else None,
                os.path.join(config.screenshots_dir, fname) if fname else None,
                os.path.join(config.screenshots_dir, f"{sid}.jpg"),
                os.path.join(config.screenshots_dir, f"{sid}.png"),
                os.path.join(config.screenshots_dir, f"{sid}.webp"),
            ]

            for cp in candidate_paths:
                if cp and os.path.isfile(cp):
                    target_file = cp
                    break

            if target_file:
                try:
                    if not sha or force_recompute:
                        sha = detector.compute_sha256(target_file)
                    if not phash or force_recompute:
                        phash = detector.compute_phash(target_file)
                    cur.execute(
                        "UPDATE screenshot_metadata SET file_hash = ?, perceptual_hash = ? WHERE screenshot_id = ?",
                        (sha, phash, sid),
                    )
                    updated_count += 1
                except Exception as ex:
                    print(f"Warning: Failed hashing {target_file}: {ex}")
            else:
                missing_files_count += 1

        records_for_clustering.append({
            "screenshot_id": sid,
            "filename": fname,
            "image_url": f"/data/screenshots/{fname}" if fname else None,
            "category": r["category"],
            "created_at": r["created_at"],
            "file_hash": sha,
            "perceptual_hash": phash,
        })

    conn.commit()
    conn.close()

    # Detect duplicates across all backfilled records
    dup_stats = detector.find_duplicates(records_for_clustering)
    elapsed = time.time() - start_time

    stats = {
        "total_records": total_records,
        "updated_records": updated_count,
        "missing_files_count": missing_files_count,
        "total_duplicate_groups": dup_stats["total_groups"],
        "exact_duplicate_groups": dup_stats["exact_groups_count"],
        "visual_duplicate_groups": dup_stats["visual_groups_count"],
        "total_duplicate_candidates": dup_stats["total_duplicate_candidates"],
        "elapsed_seconds": round(elapsed, 3),
    }

    print(f"Hash backfill complete in {elapsed:.2f}s:")
    print(f"  Total records:             {total_records}")
    print(f"  Hashes updated:            {updated_count}")
    print(f"  Missing image files:       {missing_files_count}")
    print(f"  Total duplicate groups:    {dup_stats['total_groups']}")
    print(f"  Exact groups (100% SHA):   {dup_stats['exact_groups_count']}")
    print(f"  Visual groups (pHash):     {dup_stats['visual_groups_count']}")
    print(f"  Total duplicate copies:    {dup_stats['total_duplicate_candidates']}")

    return stats


if __name__ == "__main__":
    run_hash_backfill()
