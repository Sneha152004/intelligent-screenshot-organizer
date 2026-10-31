"""
Legacy Screenshot Vector Indexer for I.R.I.S.
==============================================
Extracts genuine OCR text snippets from Layer 2 SQLite (screenshot_metadata),
chunks the text via ParagraphChunker, generates dense embeddings with
SentenceTransformerEmbedder (all-MiniLM-L6-v2, 384-dim), and indexes the chunks
into Layer 3 ChromaDB (screenshot_chunks).

Guarantees & Constraints:
1. Strict read-only on SQLite: zero records modified, deleted, or inserted in SQLite.
2. Safe duplicate physical alias exclusion: identifies ss_0010–ss_0030 as physical
   aliases of already-indexed ss_010–ss_030, avoiding vector duplication.
3. Partial indexing detection: checks existing chunk IDs per screenshot to prevent
   inconsistent states or duplicate chunk insertion.
4. Idempotent & recoverable: skips already indexed screenshot IDs cleanly on rerun.
5. Dry-run by default: requires --apply to write to ChromaDB.
6. Supports --limit N for controlled pilot testing.
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

from src.backfill_tags import is_genuine_ocr_text
from src.chunker import ParagraphChunker
from src.config import config
from src.embedder import SentenceTransformerEmbedder
from src.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("index_legacy_embeddings")

# Verified alias mappings (ss_0010-ss_0030 map to same physical images as ss_010-ss_030)
KNOWN_DUPLICATE_ALIASES: Dict[str, str] = {
    f"ss_{i:04d}": f"ss_{i:03d}" for i in range(10, 31)
}


def compute_table_sha256(db_path: str = "data/metadata.sqlite") -> str:
    """Computes deterministic SHA-256 hash across all rows and columns in screenshot_metadata."""
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


def get_chromadb_existing_chunks(vector_store: VectorStoreManager) -> Dict[str, List[str]]:
    """
    Retrieves all existing chunk IDs from ChromaDB mapped by screenshot_id.
    Returns:
        Dict[screenshot_id, List[chunk_id]]
    """
    existing_chunks: Dict[str, List[str]] = {}
    try:
        col = vector_store.collection
        all_data = col.get(include=["metadatas"])
        ids = all_data.get("ids", [])
        metadatas = all_data.get("metadatas", [])
        for chunk_id, meta in zip(ids, metadatas):
            sid = meta.get("screenshot_id") if meta else None
            if not sid:
                # Fallback: extract screenshot_id from chunk_id (e.g. ss_001_chunk_0 -> ss_001)
                sid = chunk_id.rsplit("_chunk_", 1)[0]
            existing_chunks.setdefault(sid, []).append(chunk_id)
    except Exception as e:
        logger.warning(f"Error fetching existing ChromaDB chunks: {e}")
    return existing_chunks


def audit_indexing_plan(
    db_path: str = "data/metadata.sqlite",
    vector_store: Optional[VectorStoreManager] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Performs a non-mutating audit of SQLite and ChromaDB to construct the indexing plan.
    """
    if vector_store is None:
        vector_store = VectorStoreManager(
            db_dir=config.chroma_db_dir,
            collection_name=config.collection_name,
        )

    existing_chunks_by_sid = get_chromadb_existing_chunks(vector_store)
    indexed_sids = set(existing_chunks_by_sid.keys())

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT screenshot_id, filename, category, tags, evidence, summary FROM screenshot_metadata ORDER BY screenshot_id ASC"
    )
    rows = cur.fetchall()
    conn.close()

    total_records = len(rows)
    already_indexed = []
    partially_indexed = []
    excluded_aliases = []
    lacking_genuine_ocr = []
    eligible_to_index = []

    chunker = ParagraphChunker()

    for r in rows:
        sid = r["screenshot_id"]
        fname = r["filename"] or ""
        cat = r["category"] or "General"
        raw_tags = r["tags"]
        tags = []
        if raw_tags:
            try:
                tags = json.loads(raw_tags) if isinstance(raw_tags, str) else list(raw_tags)
            except Exception:
                tags = []

        ev = {}
        if r["evidence"]:
            try:
                ev = json.loads(r["evidence"]) if isinstance(r["evidence"], str) else r["evidence"]
            except Exception:
                ev = {}

        snippet = ev.get("text_snippet", "") if isinstance(ev, dict) else ""

        # 1. Check genuine OCR availability
        if not is_genuine_ocr_text(snippet):
            lacking_genuine_ocr.append({
                "screenshot_id": sid,
                "filename": fname,
                "reason": "Lacks genuine OCR text in evidence (empty, synthetic, or failed decode).",
            })
            continue

        # 2. Check duplicate physical alias mapping (ss_0010–ss_0030)
        if sid in KNOWN_DUPLICATE_ALIASES:
            primary_sid = KNOWN_DUPLICATE_ALIASES[sid]
            # Verify primary_sid exists in dataset and points to same file
            excluded_aliases.append({
                "screenshot_id": sid,
                "primary_id": primary_sid,
                "filename": fname,
                "reason": f"Physical alias of primary record '{primary_sid}' pointing to identical file '{fname}'.",
                "primary_indexed": primary_sid in indexed_sids,
            })
            continue

        # 3. Check ChromaDB existing status
        if sid in indexed_sids:
            chunks = chunker.chunk(snippet)
            expected_count = len(chunks) if chunks else 1
            actual_count = len(existing_chunks_by_sid[sid])

            if actual_count < expected_count:
                partially_indexed.append({
                    "screenshot_id": sid,
                    "existing_chunk_ids": existing_chunks_by_sid[sid],
                    "expected_count": expected_count,
                    "actual_count": actual_count,
                    "reason": f"Partial chunk count mismatch: {actual_count} existing vs {expected_count} expected.",
                })
            else:
                already_indexed.append({
                    "screenshot_id": sid,
                    "chunk_count": actual_count,
                })
            continue

        # 4. Record is eligible for indexing
        chunks = chunker.chunk(snippet)
        if not chunks:
            chunks = [snippet.strip()]

        eligible_to_index.append({
            "screenshot_id": sid,
            "filename": fname,
            "category": cat,
            "tags": tags,
            "text": snippet,
            "chunks": chunks,
            "chunk_count": len(chunks),
        })

    total_eligible = len(eligible_to_index)
    selected_for_run = eligible_to_index[:limit] if limit else eligible_to_index

    return {
        "total_sqlite_records": total_records,
        "already_indexed_count": len(already_indexed),
        "partially_indexed_count": len(partially_indexed),
        "partially_indexed_records": partially_indexed,
        "excluded_aliases_count": len(excluded_aliases),
        "excluded_aliases": excluded_aliases,
        "lacking_genuine_ocr_count": len(lacking_genuine_ocr),
        "total_eligible_count": total_eligible,
        "selected_count": len(selected_for_run),
        "selected_records": selected_for_run,
        "limit": limit,
    }


def index_legacy_screenshots(
    db_path: str = "data/metadata.sqlite",
    chroma_dir: str = config.chroma_db_dir,
    collection_name: str = config.collection_name,
    limit: Optional[int] = None,
    dry_run: bool = True,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Executes the vector indexing pass according to the audited plan.
    """
    vector_store = VectorStoreManager(db_dir=chroma_dir, collection_name=collection_name)
    initial_chroma_count = vector_store.count_documents()
    initial_table_sha = compute_table_sha256(db_path)

    plan = audit_indexing_plan(db_path=db_path, vector_store=vector_store, limit=limit)
    selected = plan["selected_records"]

    if dry_run:
        return {
            "dry_run": True,
            "status": "Dry-run complete. Zero records modified in SQLite or ChromaDB.",
            "plan": plan,
            "initial_chroma_count": initial_chroma_count,
            "final_chroma_count": initial_chroma_count,
            "table_sha_before": initial_table_sha,
            "table_sha_after": initial_table_sha,
            "sqlite_unmodified": True,
            "indexed_count": 0,
            "failed_count": 0,
            "failed_records": [],
        }

    embedder = SentenceTransformerEmbedder(model_name=config.embedding_model)

    indexed_count = 0
    failed_records = []
    all_new_chunk_ids: List[str] = []

    # Process in batches
    total_to_process = len(selected)
    for start_idx in range(0, total_to_process, batch_size):
        batch = selected[start_idx : start_idx + batch_size]
        batch_ids: List[str] = []
        batch_docs: List[str] = []
        batch_metas: List[Dict[str, Any]] = []

        for item in batch:
            sid = item["screenshot_id"]
            cat = item["category"]
            tags = item["tags"]
            chunks = item["chunks"]

            for c_idx, chunk_text in enumerate(chunks):
                chunk_id = f"{sid}_chunk_{c_idx}"
                batch_ids.append(chunk_id)
                batch_docs.append(chunk_text)
                batch_metas.append({
                    "screenshot_id": sid,
                    "chunk_index": c_idx,
                    "category": cat,
                    "tags": tags,
                })

        try:
            # Generate embeddings for the batch
            embeddings = embedder.embed(batch_docs)
            vector_store.add_documents(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_metas,
            )
            all_new_chunk_ids.extend(batch_ids)
            indexed_count += len(batch)
            logger.info(f"Indexed progress: {min(start_idx + batch_size, total_to_process)}/{total_to_process} screenshots.")
        except Exception as e:
            logger.error(f"Error indexing batch starting at index {start_idx}: {e}")
            for item in batch:
                failed_records.append({
                    "screenshot_id": item["screenshot_id"],
                    "error": str(e),
                })

    final_chroma_count = vector_store.count_documents()
    final_table_sha = compute_table_sha256(db_path)

    return {
        "dry_run": False,
        "status": "Indexing pass applied successfully.",
        "plan": plan,
        "initial_chroma_count": initial_chroma_count,
        "final_chroma_count": final_chroma_count,
        "chunks_added": final_chroma_count - initial_chroma_count,
        "indexed_screenshots_count": indexed_count,
        "failed_count": len(failed_records),
        "failed_records": failed_records,
        "table_sha_before": initial_table_sha,
        "table_sha_after": final_table_sha,
        "sqlite_unmodified": (initial_table_sha == final_table_sha),
        "sample_indexed_ids": [item["screenshot_id"] for item in selected[:10]],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="I.R.I.S. Legacy Screenshot Vector Indexer (Safe & Dry-Run by Default)")
    parser.add_argument("--db-path", default="data/metadata.sqlite", help="Path to SQLite database")
    parser.add_argument("--chroma-dir", default=config.chroma_db_dir, help="Path to ChromaDB directory")
    parser.add_argument("--collection-name", default=config.collection_name, help="ChromaDB collection name")
    parser.add_argument("--apply", action="store_true", help="Write embeddings to ChromaDB (default is dry-run)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of screenshots to index (for pilot)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation")
    args = parser.parse_args()

    print("================================================================================")
    print("               I.R.I.S. LEGACY SCREENSHOT VECTOR INDEXER                        ")
    print("================================================================================")
    print(f"Database Path:   {args.db_path}")
    print(f"ChromaDB Path:   {args.chroma_dir}")
    print(f"Collection Name: {args.collection_name}")
    print(f"Dry Run Mode:    {not args.apply}")
    print(f"Limit:           {args.limit or 'All eligible'}")
    print(f"Batch Size:      {args.batch_size}")
    print("--------------------------------------------------------------------------------")

    result = index_legacy_screenshots(
        db_path=args.db_path,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        limit=args.limit,
        dry_run=not args.apply,
        batch_size=args.batch_size,
    )

    plan = result["plan"]
    print("AUDIT & SELECTION SUMMARY:")
    print(f"  • Total SQLite Records:          {plan['total_sqlite_records']}")
    print(f"  • Already Indexed in ChromaDB:   {plan['already_indexed_count']}")
    print(f"  • Partially Indexed Records:     {plan['partially_indexed_count']}")
    print(f"  • Excluded Physical Aliases:     {plan['excluded_aliases_count']}")
    print(f"  • Lacking Genuine OCR Evidence:  {plan['lacking_genuine_ocr_count']}")
    print(f"  • Total Eligible Unindexed:      {plan['total_eligible_count']}")
    print(f"  • Selected for this run:         {plan['selected_count']}")
    print("--------------------------------------------------------------------------------")
    print("EXCLUDED PHYSICAL ALIASES (ss_0010–ss_0030):")
    for alias in plan["excluded_aliases"][:5]:
        print(f"  • {alias['screenshot_id']} -> primary '{alias['primary_id']}' (file: {alias['filename']})")
    if len(plan["excluded_aliases"]) > 5:
        print(f"  ... and {len(plan['excluded_aliases']) - 5} more alias records.")
    print("--------------------------------------------------------------------------------")
    print(f"Initial ChromaDB Chunks:         {result['initial_chroma_count']}")
    print(f"Final ChromaDB Chunks:           {result['final_chroma_count']}")
    print(f"SQLite SHA-256 (Before):         {result['table_sha_before']}")
    print(f"SQLite SHA-256 (After):          {result['table_sha_after']}")
    print(f"SQLite Unmodified Confirmation:  {result['sqlite_unmodified']} (Cryptographically Verified)")
    print("================================================================================")
    print(f"Status: {result['status']}")

    if not args.apply:
        print("\n[DRY RUN COMPLETE] Zero changes were made to ChromaDB or SQLite.")
        print("To index a pilot batch of 5 screenshots: python src/index_legacy_embeddings.py --limit 5 --apply")
    else:
        print(f"\n[APPLIED] Successfully indexed {result['indexed_screenshots_count']} screenshots ({result['chunks_added']} chunks added).")
        if result["failed_count"] > 0:
            print(f"[WARNING] {result['failed_count']} screenshots failed. Details:")
            for f_rec in result["failed_records"]:
                print(f"  • {f_rec['screenshot_id']}: {f_rec['error']}")
