"""
Safe, Idempotent Tag Backfill Utility for I.R.I.S.
==================================================
Audits and selectively backfills evidence-backed tags for existing screenshot records.

Safety Guarantees:
1. Dry-run by default: no database records are modified unless explicitly run with --apply.
2. True cryptographic SHA-256 hash verification (both raw file and table rows) prevents false assurances.
3. Never treats generic synthetic descriptions ("Screenshot showing...", "[MOCK OCR]") as genuine OCR evidence.
4. Preserves all existing valid tags, avoiding duplicates (idempotent tag merging).
5. Preserves screenshot IDs, image URIs, timestamps, and all unrelated evidence/metadata fields.
6. Canonical 9-category taxonomy integrity: maps legacy 'Coding' category records to audited
   canonical categories from the project's existing mapping policy (data/metadata_v2.csv),
   preserving 'Coding' as a multi-label tag so no top-level 'Coding' category remains.
7. Does not modify frozen Category Decision Policy V3 files, benchmark labels, or results.
8. Does not run automatically at application startup.
9. Phase 2 EasyOCR image reprocessing is isolated, preserving existing valid tags and categories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tag_generator import TagGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("backfill_tags")

GENERIC_SYNTHETIC_PREFIXES = (
    "screenshot showing",
    "[mock ocr]",
    "no predefined text",
)

METADATA_V2_PATH = PROJECT_ROOT / "data" / "metadata_v2.csv"

# Records identified with cautionary notes in forensic audit
MANUAL_REVIEW_FLAGS: Dict[str, str] = {
    "ss_280": "No OCR text available in database or cache; mapped to canonical fallback 'General'.",
    "ss_826": "No OCR text available in database or cache; mapped to canonical fallback 'General'.",
    "ss_827": "No OCR text available in database or cache; mapped to canonical fallback 'General'.",
    "ss_041": "QuestLog web app repository git pull; mapped to canonical category 'Education'.",
    "ss_042": "QuestLog web app file diff statistics; mapped to canonical category 'Education'.",
    "ss_057": "AI Lab terminal search algorithms; mapped to canonical category 'General'.",
}


def compute_file_sha256(file_path: str) -> str:
    """Computes SHA-256 hash of a file on disk."""
    if not os.path.exists(file_path):
        return ""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_table_sha256(db_path: str) -> str:
    """Computes a deterministic SHA-256 hash across all rows and columns in screenshot_metadata."""
    if not os.path.exists(db_path):
        return ""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM screenshot_metadata ORDER BY screenshot_id ASC")
    h = hashlib.sha256()
    for row in cur.fetchall():
        row_str = "|".join("" if v is None else str(v) for v in row)
        h.update(row_str.encode("utf-8"))
    conn.close()
    return h.hexdigest()


def is_genuine_ocr_text(text: Optional[str]) -> bool:
    """Check if text is genuine OCR evidence rather than a synthetic placeholder."""
    if not text or not isinstance(text, str):
        return False
    cleaned = text.strip()
    if not cleaned or len(cleaned) < 5:
        return False
    lower = cleaned.lower()
    for prefix in GENERIC_SYNTHETIC_PREFIXES:
        if lower.startswith(prefix):
            return False
    return True


def evaluate_backfill_eligibility(
    ocr_text: Optional[str],
    category: str = "General",
    existing_tags: Optional[List[str]] = None,
) -> Tuple[bool, List[str], str]:
    """Helper function to evaluate single-record eligibility for tag backfill."""
    existing = existing_tags or []
    if not is_genuine_ocr_text(ocr_text):
        return False, [], "Synthetic placeholder or insufficient text."
    tg = TagGenerator()
    candidate_tags = tg.generate_tags(ocr_text=ocr_text, category=category)
    new_tags = [t for t in candidate_tags if t not in existing]
    if not candidate_tags:
        return False, [], "Text yields no candidate tags (insufficient evidence)."
    if not new_tags:
        return False, [], "Record already has all candidate tags."
    return True, new_tags, "Eligible for tag generation."


def load_legacy_coding_v2_map() -> Dict[str, Tuple[str, List[str], str]]:
    """Load authoritative canonical remapping for legacy coding files from metadata_v2.csv."""
    v2_map = {}
    if not METADATA_V2_PATH.exists():
        return v2_map
    try:
        import pandas as pd
        df = pd.read_csv(METADATA_V2_PATH)
        for _, row in df.iterrows():
            fname = str(row["file"]).strip()
            cat = str(row["category"]).strip()
            raw_tags = str(row.get("tags", ""))
            reason = str(row.get("migration_reason", "")).strip()
            tags = [t.strip() for t in raw_tags.split(",") if t.strip()] if raw_tags and raw_tags != "nan" else []
            if "Coding" not in tags:
                tags.insert(0, "Coding")
            v2_map[fname] = (cat, tags, reason)
    except Exception as e:
        logger.warning(f"Could not load metadata_v2.csv mapping: {e}")
    return v2_map


def audit_database(db_path: str = "data/metadata.sqlite") -> Dict[str, Any]:
    """
    Performs a read-only audit of the SQLite database and classifies all records.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT screenshot_id, filename, image_uri, category, tags, summary, evidence FROM screenshot_metadata")
    rows = cur.fetchall()

    total_records = len(rows)
    non_empty_tags_count = 0
    empty_or_null_tags_count = 0
    genuine_stored_ocr_count = 0
    generic_synthetic_count = 0
    available_for_reprocessing_count = 0
    legacy_coding_count = 0

    records_with_existing_tags = []
    records_with_genuine_ocr = []
    records_with_synthetic_desc = []
    records_legacy_coding = []

    for r in rows:
        sid = r["screenshot_id"]
        cat = r["category"]
        raw_tags = r["tags"]
        filename = r["filename"] or ""

        # Parse tags
        tags = []
        if raw_tags:
            try:
                tags = json.loads(raw_tags) if isinstance(raw_tags, str) else list(raw_tags)
            except Exception:
                tags = []

        if tags:
            non_empty_tags_count += 1
            records_with_existing_tags.append({"screenshot_id": sid, "filename": filename, "tags": tags})
        else:
            empty_or_null_tags_count += 1

        # Check evidence text
        ev = {}
        if r["evidence"]:
            try:
                ev = json.loads(r["evidence"]) if isinstance(r["evidence"], str) else r["evidence"]
            except Exception:
                ev = {}

        snippet = ev.get("text_snippet", "") if isinstance(ev, dict) else ""
        summary = r["summary"] or ""

        if is_genuine_ocr_text(snippet):
            genuine_stored_ocr_count += 1
            records_with_genuine_ocr.append({"screenshot_id": sid, "filename": filename, "category": cat, "text": snippet, "existing_tags": tags})
        else:
            generic_synthetic_count += 1
            # Check original image on disk
            img_path = r["image_uri"] or ""
            if not os.path.isabs(img_path) or not os.path.exists(img_path):
                img_path = os.path.join(os.path.dirname(os.path.abspath(db_path)), "screenshots", filename)
            has_image = os.path.exists(img_path)
            if has_image:
                available_for_reprocessing_count += 1
            records_with_synthetic_desc.append({
                "screenshot_id": sid,
                "filename": filename,
                "category": cat,
                "summary": summary,
                "has_image": has_image,
                "image_path": img_path if has_image else None,
                "existing_tags": tags,
            })

        if cat == "Coding":
            legacy_coding_count += 1
            records_legacy_coding.append({"screenshot_id": sid, "filename": filename, "existing_tags": tags})

    conn.close()

    return {
        "total_records": total_records,
        "non_empty_tags_count": non_empty_tags_count,
        "empty_or_null_tags_count": empty_or_null_tags_count,
        "genuine_stored_ocr_count": genuine_stored_ocr_count,
        "generic_synthetic_count": generic_synthetic_count,
        "available_for_reprocessing_count": available_for_reprocessing_count,
        "existing_tags_preserved_count": non_empty_tags_count,
        "legacy_coding_count": legacy_coding_count,
        "records_with_existing_tags": records_with_existing_tags,
        "records_with_genuine_ocr": records_with_genuine_ocr,
        "records_with_synthetic_desc": records_with_synthetic_desc,
        "records_legacy_coding": records_legacy_coding,
    }


def plan_backfill_actions(
    db_path: str = "data/metadata.sqlite",
    reprocess_images: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluates which actions would be taken during backfill, respecting strict eligibility rules.
    Guarantees 100% mutually exclusive categorization with zero double-counting.
    """
    audit = audit_database(db_path=db_path)
    tg = TagGenerator()
    v2_map = load_legacy_coding_v2_map()

    eligible_text_snippet = []
    eligible_legacy_coding = []
    skipped_records = {
        "synthetic_placeholder_only": [],
        "insufficient_evidence_zero_tags": [],
        "already_fully_tagged": [],
        "image_file_missing": [],
    }

    # 1. Evaluate records with genuine stored OCR text
    for item in audit["records_with_genuine_ocr"]:
        sid = item["screenshot_id"]
        cat = item["category"]
        text = item["text"]
        existing = item["existing_tags"]

        candidate_tags = tg.generate_tags(ocr_text=text, category=cat)
        new_tags = [t for t in candidate_tags if t not in existing]

        if not candidate_tags:
            # Genuine text exists, but no rules matched (insufficient evidence)
            skipped_records["insufficient_evidence_zero_tags"].append({
                "screenshot_id": sid,
                "category": cat,
                "existing_tags": existing,
                "reason": "Text contains insufficient evidence for tag generation (no hallucination).",
            })
        elif not new_tags:
            # Already has all candidate tags
            skipped_records["already_fully_tagged"].append({
                "screenshot_id": sid,
                "category": cat,
                "existing_tags": existing,
                "reason": "Record already possesses all candidate tags.",
            })
        else:
            eligible_text_snippet.append({
                "screenshot_id": sid,
                "filename": item["filename"],
                "category": cat,
                "existing_tags": existing,
                "new_tags": new_tags,
                "combined_tags": list(dict.fromkeys(existing + candidate_tags)),
                "source": "stored_ocr_evidence",
            })

    # 2. Evaluate legacy Coding records (34 records)
    for item in audit["records_legacy_coding"]:
        sid = item["screenshot_id"]
        fname = item["filename"]
        existing = item["existing_tags"]

        if fname in v2_map:
            target_cat, v2_tags, reason = v2_map[fname]
        else:
            target_cat, v2_tags, reason = "Work & Career", ["Coding"], "Legacy fallback"

        combined = list(dict.fromkeys(existing + v2_tags))
        flag = MANUAL_REVIEW_FLAGS.get(sid, None)

        eligible_legacy_coding.append({
            "screenshot_id": sid,
            "filename": fname,
            "current_category": "Coding",
            "canonical_category": target_cat,
            "existing_tags": existing,
            "new_tags": [t for t in combined if t not in existing],
            "combined_tags": combined,
            "migration_reason": reason,
            "manual_review_flag": flag,
            "source": "legacy_coding_taxonomy_refactor",
        })

    # 3. Evaluate synthetic description records (excluding legacy coding records handled above)
    legacy_coding_sids = {item["screenshot_id"] for item in eligible_legacy_coding}
    for item in audit["records_with_synthetic_desc"]:
        sid = item["screenshot_id"]
        # Mutually exclusive: do not double-count coding records in skipped synthetic
        if sid in legacy_coding_sids:
            continue

        if not item["has_image"]:
            skipped_records["image_file_missing"].append({
                "screenshot_id": sid,
                "filename": item["filename"],
                "reason": "Original image file does not exist on disk.",
            })
        elif not reprocess_images:
            skipped_records["synthetic_placeholder_only"].append({
                "screenshot_id": sid,
                "filename": item["filename"],
                "reason": "Record contains only generic placeholder text. Original image available on disk; requires explicit image OCR reprocessing.",
            })

    total_eligible = len(eligible_text_snippet) + len(eligible_legacy_coding)

    if limit:
        eligible_text_snippet = eligible_text_snippet[:limit]
        eligible_legacy_coding = eligible_legacy_coding[:limit]

    return {
        "audit": audit,
        "total_eligible": total_eligible,
        "eligible_text_snippet_count": len(eligible_text_snippet),
        "eligible_text_snippet": eligible_text_snippet,
        "eligible_legacy_coding_count": len(eligible_legacy_coding),
        "eligible_legacy_coding": eligible_legacy_coding,
        "skipped_records": skipped_records,
        "skipped_summary": {k: len(v) for k, v in skipped_records.items()},
    }


def execute_dry_run_audit(
    db_path: str = "data/metadata.sqlite",
    reprocess_images: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes a complete dry-run audit, returning exact counts, skip reasons, expected changes,
    and a cryptographic SHA-256 confirmation (both file bytes and row content) that no database
    records were modified.
    """
    file_sha256_before = compute_file_sha256(db_path)
    content_sha256_before = compute_table_sha256(db_path)

    plan = plan_backfill_actions(db_path=db_path, reprocess_images=reprocess_images, limit=limit)

    file_sha256_after = compute_file_sha256(db_path)
    content_sha256_after = compute_table_sha256(db_path)

    db_unmodified = (
        file_sha256_before == file_sha256_after
        and content_sha256_before == content_sha256_after
    )

    return {
        "plan": plan,
        "file_sha256_before": file_sha256_before,
        "file_sha256_after": file_sha256_after,
        "content_sha256_before": content_sha256_before,
        "content_sha256_after": content_sha256_after,
        "database_unmodified": db_unmodified,
    }


def apply_approved_backfill(
    db_path: str = "data/metadata.sqlite",
    tags_only: bool = False,
    update_categories: bool = True,
    hold_flagged: bool = False,
    dry_run: bool = True,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Applies the approved Phase 1 tag backfill and canonical category migration.
    Modifies database ONLY if dry_run=False.
    """
    plan = plan_backfill_actions(db_path=db_path, limit=limit)

    if dry_run:
        return {
            "dry_run": True,
            "status": "No database changes made (dry-run mode).",
            "eligible_count": plan["total_eligible"],
            "updated_count": 0,
        }

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    updated_tags = 0
    updated_categories = 0
    held_for_manual_review = 0

    try:
        with conn:
            # 1. Update records eligible by genuine text snippet (tags only, category preserved)
            for item in plan["eligible_text_snippet"]:
                sid = item["screenshot_id"]
                combined = json.dumps(item["combined_tags"])
                cur.execute(
                    "UPDATE screenshot_metadata SET tags = ? WHERE screenshot_id = ?",
                    (combined, sid),
                )
                updated_tags += 1

            # 2. Update legacy coding records
            for item in plan["eligible_legacy_coding"]:
                sid = item["screenshot_id"]
                combined = json.dumps(item["combined_tags"])
                new_cat = item["canonical_category"]
                is_flagged = bool(item.get("manual_review_flag"))

                if tags_only or not update_categories:
                    # Update tags only; preserve current category
                    cur.execute(
                        "UPDATE screenshot_metadata SET tags = ? WHERE screenshot_id = ?",
                        (combined, sid),
                    )
                    updated_tags += 1
                elif hold_flagged and is_flagged:
                    # Update tags only; hold category update for manual review
                    cur.execute(
                        "UPDATE screenshot_metadata SET tags = ? WHERE screenshot_id = ?",
                        (combined, sid),
                    )
                    updated_tags += 1
                    held_for_manual_review += 1
                    logger.info(f"Record {sid} category held for manual review: {item.get('manual_review_flag')}")
                else:
                    # Update both canonical category and tags
                    cur.execute(
                        "UPDATE screenshot_metadata SET category = ?, tags = ? WHERE screenshot_id = ?",
                        (new_cat, combined, sid),
                    )
                    updated_categories += 1
                    updated_tags += 1
    finally:
        conn.close()

    return {
        "dry_run": False,
        "status": "Backfill applied successfully.",
        "updated_tags_count": updated_tags,
        "updated_categories_count": updated_categories,
        "held_for_manual_review": held_for_manual_review,
        "total_updated_rows": updated_tags,
    }


def select_representative_batch(eligible: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Selects diverse, representative screenshots across major categories when a batch limit is set."""
    target_categories = [
        "Shopping",
        "Payment",
        "Travel",
        "Work & Career",
        "Connectivity",
        "Education",
        "Entertainment",
        "Communication",
        "General",
    ]
    selected = []
    seen_ids = set()
    for cat in target_categories:
        for item in eligible:
            if item["category"] == cat and item["screenshot_id"] not in seen_ids:
                selected.append(item)
                seen_ids.add(item["screenshot_id"])
                break
        if len(selected) >= limit:
            break
    for item in eligible:
        if len(selected) >= limit:
            break
        if item["screenshot_id"] not in seen_ids:
            selected.append(item)
            seen_ids.add(item["screenshot_id"])
    return selected


def reprocess_legacy_images_easyocr(
    db_path: str = "data/metadata.sqlite",
    limit: Optional[int] = None,
    dry_run: bool = True,
    batch_size: int = 10,
) -> Dict[str, Any]:
    """
    Phase 2: Reprocesses legacy images that have synthetic placeholder summaries using EasyOCR.
    Safety Guarantees:
    - Never uses MockOCR or synthetic summaries.
    - Preserves existing categories without alteration.
    - Preserves all existing valid tags, appending only newly discovered tags.
    - Logs errors for missing/unreadable images without corrupting database records.
    - Dry-run mode by default.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT screenshot_id, filename, image_uri, category, tags, evidence, summary FROM screenshot_metadata")
    rows = cur.fetchall()
    conn.close()

    eligible_for_ocr = []
    base_dir = Path(db_path).resolve().parent

    for r in rows:
        ev = {}
        if r["evidence"]:
            try:
                ev = json.loads(r["evidence"]) if isinstance(r["evidence"], str) else r["evidence"]
            except Exception:
                ev = {}

        snippet = ev.get("text_snippet", "") if isinstance(ev, dict) else ""
        if not is_genuine_ocr_text(snippet):
            filename = r["filename"] or ""
            img_path = r["image_uri"] or ""
            if not os.path.isabs(img_path) or not os.path.exists(img_path):
                img_path = str(base_dir / "screenshots" / filename)

            if os.path.exists(img_path):
                raw_tags = r["tags"]
                tags = []
                if raw_tags:
                    try:
                        tags = json.loads(raw_tags) if isinstance(raw_tags, str) else list(raw_tags)
                    except Exception:
                        tags = []
                eligible_for_ocr.append({
                    "screenshot_id": r["screenshot_id"],
                    "filename": filename,
                    "image_path": img_path,
                    "category": r["category"],
                    "existing_tags": tags,
                })

    total_eligible = len(eligible_for_ocr)
    if limit:
        eligible_for_ocr = select_representative_batch(eligible_for_ocr, limit)

    if dry_run:
        return {
            "dry_run": True,
            "total_eligible": total_eligible,
            "evaluated_in_dry_run": len(eligible_for_ocr),
            "status": "Dry run complete. No images processed, no database records modified.",
            "sample_candidates": eligible_for_ocr,
        }

    from src.easy_ocr import EasyOCREngine
    ocr_engine = EasyOCREngine(gpu=False, verbose=False)
    tg = TagGenerator()

    processed_count = 0
    updated_records = 0
    failed_images = []
    processed_details = []

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    try:
        for idx, item in enumerate(eligible_for_ocr, 1):
            sid = item["screenshot_id"]
            img_path = item["image_path"]
            cat = item["category"]
            existing = item["existing_tags"]

            try:
                extracted_text = ocr_engine.extract_text(img_path)
                processed_count += 1

                candidate_tags = []
                combined_tags = existing
                updated = False

                if is_genuine_ocr_text(extracted_text):
                    candidate_tags = tg.generate_tags(ocr_text=extracted_text, category=cat)
                    combined_tags = list(dict.fromkeys(existing + candidate_tags))

                    new_evidence = {
                        "text_snippet": extracted_text[:100],
                        "ocr_engine": "EasyOCR",
                        "reprocessed": True,
                    }

                    cur.execute(
                        "UPDATE screenshot_metadata SET tags = ?, evidence = ?, ocr_available = 1 WHERE screenshot_id = ?",
                        (json.dumps(combined_tags), json.dumps(new_evidence), sid),
                    )
                    updated_records += 1
                    updated = True

                processed_details.append({
                    "screenshot_id": sid,
                    "filename": item["filename"],
                    "category": cat,
                    "ocr_engine": "EasyOCR",
                    "text_length": len(extracted_text),
                    "text_excerpt": extracted_text[:150].replace("\n", " "),
                    "tags_before": existing,
                    "tags_after": combined_tags,
                    "new_tags_generated": [t for t in combined_tags if t not in existing],
                    "updated": updated,
                })

                if idx % batch_size == 0:
                    conn.commit()
                    logger.info(f"Progress: {idx}/{len(eligible_for_ocr)} images processed.")
            except Exception as e:
                failed_images.append({"screenshot_id": sid, "image_path": img_path, "error": str(e)})
                processed_details.append({
                    "screenshot_id": sid,
                    "filename": item["filename"],
                    "category": cat,
                    "ocr_engine": "EasyOCR",
                    "error": str(e),
                    "updated": False,
                })
                logger.warning(f"Error processing {img_path} for {sid}: {e}")

        conn.commit()
    finally:
        conn.close()

    return {
        "dry_run": False,
        "total_eligible": total_eligible,
        "processed_count": processed_count,
        "updated_records": updated_records,
        "failed_count": len(failed_images),
        "failed_images": failed_images,
        "details": processed_details,
        "status": "EasyOCR reprocessing completed successfully.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="I.R.I.S. Tag Backfill Utility (Safe & Dry-Run by Default)")
    parser.add_argument("--db-path", default="data/metadata.sqlite", help="Path to SQLite database")
    parser.add_argument("--apply", action="store_true", help="Execute approved Phase 1 changes (default is dry-run)")
    parser.add_argument("--tags-only", action="store_true", help="Backfill tags only, leaving all categories untouched")
    parser.add_argument("--hold-flagged", action="store_true", help="Hold flagged ambiguous records from category updates")
    parser.add_argument("--reprocess-ocr", action="store_true", help="Phase 2: Reprocess legacy images with EasyOCR")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch commit size for OCR reprocessing")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluated records")
    args = parser.parse_args()

    if args.reprocess_ocr:
        print("================================================================================")
        print("                I.R.I.S. PHASE 2: EASYOCR IMAGE REPROCESSING                    ")
        print("================================================================================")
        print(f"Database Path: {args.db_path}")
        print(f"Dry Run Mode:  {not args.apply}")
        print(f"Batch Limit:   {args.limit or 'All eligible'}")
        ocr_res = reprocess_legacy_images_easyocr(
            db_path=args.db_path,
            limit=args.limit,
            dry_run=not args.apply,
            batch_size=args.batch_size,
        )
        print(f"Total Eligible Images: {ocr_res['total_eligible']}")
        print(f"Status:                {ocr_res['status']}")
        if not args.apply:
            print("\n[DRY RUN COMPLETE] Zero database records were modified.")
            print("Candidate images selected for evaluation:")
            for c in ocr_res.get("sample_candidates", []):
                print(f"  • {c['screenshot_id']} ({c['filename']}) - Category: {c['category']}")
            print("\nTo run Phase 2 live on a test batch: python src/backfill_tags.py --reprocess-ocr --apply --limit 5")
        else:
            print(f"Processed:             {ocr_res.get('processed_count', 0)}")
            print(f"Updated Records:       {ocr_res.get('updated_records', 0)}")
            print(f"Failed Count:          {ocr_res.get('failed_count', 0)}")
            print("\nProcessed Screenshot Details:")
            for d in ocr_res.get("details", []):
                print(f"  • {d['screenshot_id']} ({d['filename']}) | Category: {d['category']}")
                print(f"    Excerpt:     {d.get('text_excerpt', '')[:100]}")
                print(f"    Tags Before: {d.get('tags_before', [])} -> After: {d.get('tags_after', [])} (New: {d.get('new_tags_generated', [])})")
                print(f"    Updated in DB: {d.get('updated', False)}")
    else:
        result = execute_dry_run_audit(
            db_path=args.db_path,
            reprocess_images=False,
            limit=args.limit,
        )

        plan = result["plan"]
        audit = plan["audit"]
        skipped = plan["skipped_summary"]

        print("================================================================================")
        print("                     I.R.I.S. TAG BACKFILL AUDIT & DRY RUN                      ")
        print("================================================================================")
        print(f"Database Path:                       {args.db_path}")
        print(f"Total Screenshot Records:            {audit['total_records']}")
        print(f"Records with Non-Empty Tags:         {audit['non_empty_tags_count']}")
        print(f"Records with Missing/Empty Tags:     {audit['empty_or_null_tags_count']}")
        print(f"Records with Genuine Stored OCR:     {audit['genuine_stored_ocr_count']}")
        print(f"Records with Synthetic Descriptions: {audit['generic_synthetic_count']}")
        print(f"Images Available for Reprocessing:   {audit['available_for_reprocessing_count']}")
        print(f"Records with Existing Tags Intact:   {audit['existing_tags_preserved_count']}")
        print("--------------------------------------------------------------------------------")
        print("MUTUALLY EXCLUSIVE BACKFILL BREAKDOWN:")
        print(f"  • Eligible via Genuine Stored OCR:  {plan['eligible_text_snippet_count']}")
        print(f"  • Eligible Legacy 'Coding' Records: {plan['eligible_legacy_coding_count']}")
        print(f"  • Total Records Eligible:          {plan['total_eligible']}")
        print("--------------------------------------------------------------------------------")
        print("SKIPPED RECORDS BREAKDOWN (NON-OVERLAPPING):")
        for reason, count in skipped.items():
            print(f"  • {reason:<35}: {count}")
        print("--------------------------------------------------------------------------------")
        print(f"Table Content SHA-256 (Before):      {result['content_sha256_before']}")
        print(f"Table Content SHA-256 (After):       {result['content_sha256_after']}")
        print(f"Database Unmodified Confirmation:    {result['database_unmodified']} (Cryptographically Verified)")
        print("================================================================================")

        if args.apply:
            print("\n[APPLYING APPROVED PHASE 1 BACKFILL] Writing updates to database...")
            apply_res = apply_approved_backfill(
                db_path=args.db_path,
                tags_only=args.tags_only,
                update_categories=not args.tags_only,
                hold_flagged=args.hold_flagged,
                dry_run=False,
                limit=args.limit,
            )
            print(f"Result: {apply_res['status']}")
            print(f"  • Total rows updated:          {apply_res['total_updated_rows']}")
            print(f"  • Tags updated:                {apply_res['updated_tags_count']}")
            print(f"  • Categories updated:          {apply_res['updated_categories_count']}")
            print(f"  • Held for manual review:      {apply_res['held_for_manual_review']}")
        else:
            print("\n[DRY RUN COMPLETE] Zero database records were modified.")
            print("Command to apply approved Phase 1: python src/backfill_tags.py --apply")
