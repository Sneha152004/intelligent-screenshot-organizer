"""
Screenshots API router.

Endpoints
---------
GET  /api/screenshots          List all screenshots, optional ?category= and ?q= filters
GET  /api/screenshots/{id}     Get single screenshot detail
POST /api/screenshots/upload   Upload one or more images and run the pipeline
DELETE /api/screenshots/{id}   Delete a screenshot record (and optionally its file)
"""

import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/screenshots", tags=["screenshots"])


def _image_url_from_uri(image_uri: str) -> Optional[str]:
    """Convert absolute path to /data/... relative URL."""
    if not image_uri:
        return None
    if os.path.isabs(image_uri):
        try:
            rel = Path(image_uri).relative_to(Path(os.getcwd()))
            return "/" + str(rel).replace("\\", "/")
        except ValueError:
            return None
    return image_uri


def _meta_to_dict(meta) -> dict:
    """Safely serialise a ScreenshotMetadata object."""
    d = meta.to_dict()
    d["image_url"] = _image_url_from_uri(d.get("image_uri", ""))
    return d


def _get_stores():
    from ..core.pipeline_instance import file_store, metadata_store
    return file_store, metadata_store


def _get_vector_store():
    from ..core.pipeline_instance import get_vector_store
    return get_vector_store()


@router.get("")
def list_screenshots(
    category: Optional[str] = Query(None, description="Filter by category name"),
    tag: Optional[str] = Query(None, description="Filter by tag name"),
    q: Optional[str] = Query(None, description="Simple text filter on filename / OCR text"),
    min_importance: Optional[int] = Query(None, ge=1, le=5, description="Filter by minimum importance score (1-5)"),
    importance: Optional[int] = Query(None, ge=1, le=5, description="Filter by exact importance score (1-5)"),
    sort_by: Optional[str] = Query(None, description="Sort order: 'importance_desc', 'importance_asc', 'newest', 'oldest'"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Return a paginated list of screenshots with optional filters and sorting."""
    file_store, metadata_store = _get_stores()
    vector_store = _get_vector_store()
    all_meta = metadata_store.list_all_metadata()

    if category:
        all_meta = [m for m in all_meta if m.category.lower() == category.lower()]

    if tag:
        tag_lower = tag.lower()
        all_meta = [m for m in all_meta if any(t.lower() == tag_lower for t in (m.tags or []))]

    if q:
        q_lower = q.lower()
        all_meta = [
            m for m in all_meta
            if q_lower in m.filename.lower()
            or q_lower in (m.summary or "").lower()
            or q_lower in m.category.lower()
            or any(q_lower in t.lower() for t in (m.tags or []))
        ]

    if importance is not None:
        all_meta = [m for m in all_meta if m.importance == importance]
    elif min_importance is not None:
        all_meta = [m for m in all_meta if (m.importance or 0) >= min_importance]

    if sort_by == "importance_desc":
        all_meta.sort(key=lambda m: (m.importance or 0, m.created_at or ""), reverse=True)
    elif sort_by == "importance_asc":
        all_meta.sort(key=lambda m: (m.importance if m.importance is not None else 999, m.created_at or ""))
    elif sort_by == "oldest":
        all_meta.sort(key=lambda m: m.created_at or "")
    elif sort_by == "newest":
        all_meta.sort(key=lambda m: m.created_at or "", reverse=True)

    total = len(all_meta)
    page = all_meta[offset: offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": [_meta_to_dict(m) for m in page],
    }


@router.get("/tags")
def list_tags():
    """Return all unique tags present across stored screenshots with counts."""
    from collections import Counter
    file_store, metadata_store = _get_stores()
    vector_store = _get_vector_store()
    all_meta = metadata_store.list_all_metadata()
    tag_counts = Counter()
    for m in all_meta:
        for t in (m.tags or []):
            tag_counts[t] += 1
    return {
        "total_tags": len(tag_counts),
        "tags": [{"name": k, "count": v} for k, v in tag_counts.most_common()],
    }


@router.get("/duplicates")
def get_duplicates(
    type: Optional[str] = Query(None, description="Filter by duplicate type: 'exact' or 'visual'"),
    max_distance: int = Query(6, ge=0, le=20, description="Max Hamming distance threshold for visual similarity"),
    category: Optional[str] = Query(None, description="Filter duplicates to a specific category"),
):
    """
    Identifies exact and near-duplicate screenshots across the collection.
    Returns grouped primary screenshots and their duplicate candidates.
    Never automatically deletes files.
    """
    from src.duplicate_detector import DuplicateDetector
    file_store, metadata_store = _get_stores()
    all_meta = metadata_store.list_all_metadata()

    detector = DuplicateDetector(default_max_distance=max_distance)

    screenshot_dicts = []
    for m in all_meta:
        d = _meta_to_dict(m)
        screenshot_dicts.append(d)

    res = detector.find_duplicates(
        screenshots=screenshot_dicts,
        max_distance=max_distance,
        category_filter=category,
    )

    if type:
        filtered_groups = [g for g in res["groups"] if g["type"].lower() == type.lower()]
        res["groups"] = filtered_groups
        res["total_groups"] = len(filtered_groups)
        res["total_duplicate_candidates"] = sum(len(g["duplicates"]) for g in filtered_groups)

    return res


@router.get("/timeline")
def get_timeline(
    year: Optional[int] = Query(None, description="Filter by year (e.g. 2026)"),
    month: Optional[int] = Query(None, ge=1, le=12, description="Filter by month (1-12)"),
    category: Optional[str] = Query(None, description="Filter by category"),
    intent: Optional[str] = Query(None, description="Filter by intent"),
    sort: str = Query("desc", description="Sort order: 'desc' (newest first) or 'asc'"),
):
    """
    Returns screenshots organized as a chronological timeline grouped by year and month.
    Prioritizes reliable extracted event dates, falling back to creation dates.
    """
    from src.timeline_service import TimelineService
    file_store, metadata_store = _get_stores()
    all_meta = metadata_store.list_all_metadata()
    service = TimelineService()
    return service.build_timeline(
        screenshots=all_meta,
        year=year,
        month=month,
        category=category,
        intent=intent,
        sort_order=sort,
    )


@router.get("/{screenshot_id}")
def get_screenshot(screenshot_id: str):
    """Return full detail for a single screenshot, including OCR text."""
    file_store, metadata_store = _get_stores()
    vector_store = _get_vector_store()
    meta = metadata_store.get_metadata(screenshot_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Screenshot not found")

    d = _meta_to_dict(meta)

    # Attach raw OCR text from evidence if available
    evidence = meta.evidence or {}
    ocr_snippet = evidence.get("text_snippet", "")
    d["ocr_text"] = ocr_snippet

    return d


@router.get("/{screenshot_id}/importance")
def get_screenshot_importance(screenshot_id: str):
    """Return importance score and reason for a single screenshot."""
    file_store, metadata_store = _get_stores()
    meta = metadata_store.get_metadata(screenshot_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Screenshot not found")

    return {
        "screenshot_id": screenshot_id,
        "importance": meta.importance,
        "importance_score": meta.importance,
        "importance_reason": meta.importance_reason or "Assigned by I.R.I.S. Importance Scorer",
    }


@router.post("/upload")
async def upload_screenshots(files: List[UploadFile] = File(...)):
    """
    Upload one or more images.
    Each image is saved to a temp dir, run through the full OCR + classification
    pipeline, then stored in the 3-layer architecture.
    Returns per-file results with detected category, OCR snippet, and screenshot_id.
    """
    from ..core.pipeline_instance import get_pipeline, get_embedder, get_vector_store
    from src.chunker import ParagraphChunker

    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not available — ensure OCR dependencies are installed.",
        )

    embedder = get_embedder()
    vector_store = _get_vector_store()
    _chunker = ParagraphChunker()
    results = []
    tmp_dir = tempfile.mkdtemp(prefix="iris_upload_")

    try:
        for upload in files:
            # Validate MIME type
            if upload.content_type not in (
                "image/png", "image/jpeg", "image/webp", "image/gif",
                "image/bmp", "image/tiff",
            ):
                results.append({
                    "filename": upload.filename,
                    "status": "error",
                    "error": f"Unsupported file type: {upload.content_type}",
                })
                continue

            # Save to temp path
            suffix = Path(upload.filename).suffix or ".png"
            tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}{suffix}")
            contents = await upload.read()
            with open(tmp_path, "wb") as f:
                f.write(contents)

            # Build minimal document dict the pipeline expects
            doc = {
                "image_path": tmp_path,
                "metadata": {
                    "Screenshot ID": uuid.uuid4().hex[:12],
                    "Actual Category": "General",
                    "notes": "",
                    "tags": [],
                    "filename": upload.filename,
                },
            }

            try:
                processed = pipeline.process_documents([doc])
                if not processed:
                    raise RuntimeError("Pipeline returned no result")

                p = processed[0]
                sid = p["screenshot_id"]
                category = p["metadata"]["category"]
                tags = p["metadata"].get("tags", [])
                ocr_text = p.get("ocr_text", "")
                evidence = p["metadata"].get("evidence", {})
                fusion_applied = p.get("fusion_applied", False)

                # Index chunks into ChromaDB
                chunks = _chunker.chunk(ocr_text)
                chunk_ids, chunk_texts, chunk_metas = [], [], []
                for idx, chunk in enumerate(chunks):
                    cleaned = chunk.strip()
                    if cleaned:
                        chunk_ids.append(f"{sid}_chunk_{idx}")
                        chunk_texts.append(cleaned)
                        chunk_metas.append({
                            "screenshot_id": sid,
                            "chunk_index": idx,
                            "category": category,
                            "tags": tags,
                        })

                if chunk_texts:
                    embeddings = embedder.embed(chunk_texts)
                    vector_store.add_documents(
                        ids=chunk_ids,
                        documents=chunk_texts,
                        embeddings=embeddings,
                        metadatas=chunk_metas,
                    )

                # Check for duplicate warning against existing collection
                from src.duplicate_detector import DuplicateDetector
                _detector = DuplicateDetector()
                f_hash = p["metadata"].get("file_hash")
                p_hash = p["metadata"].get("perceptual_hash")
                is_dup = False
                dup_type = None
                dup_of = None

                target_meta_store = pipeline.metadata_store if (pipeline and hasattr(pipeline, "metadata_store") and pipeline.metadata_store) else _get_stores()[1]
                if (f_hash or p_hash) and target_meta_store:
                    all_existing = target_meta_store.list_all_metadata()
                    for em in all_existing:
                        if em.screenshot_id == sid:
                            continue
                        if f_hash and em.file_hash == f_hash:
                            is_dup = True
                            dup_type = "exact"
                            dup_of = em.screenshot_id
                            break
                        if p_hash and em.perceptual_hash:
                            dist = _detector.hamming_distance(p_hash, em.perceptual_hash)
                            if dist <= 4:
                                is_dup = True
                                dup_type = "visual"
                                dup_of = em.screenshot_id
                                break

                # Intelligent reminder detection
                detected_reminders = []
                try:
                    from src.reminder_service import ReminderService
                    from src.storage.reminder_store import ReminderStore
                    _r_store = ReminderStore("data/metadata.sqlite")
                    _r_service = ReminderService(reminder_store=_r_store)
                    _meta_for_rem = {
                        "screenshot_id": sid,
                        "filename": upload.filename,
                        "category": category,
                        "intent": p["metadata"].get("intent"),
                        "content_type": p["metadata"].get("content_type"),
                        "summary": p["metadata"].get("summary"),
                        "ocr_text": ocr_text,
                        "evidence": evidence,
                        "dates": p["metadata"].get("dates", []),
                        "action_items": p["metadata"].get("action_items", []),
                        "importance_score": p["metadata"].get("importance_score") or p["metadata"].get("importance", 3),
                        "sensitive_info": p["metadata"].get("sensitive_info", {}),
                    }
                    detected_reminders = _r_service.scan_and_create_for_screenshot(_meta_for_rem)
                except Exception as _r_err:
                    logger.warning(f"Reminder detection failed for {upload.filename}: {_r_err}")
                    detected_reminders = []

                results.append({
                    "filename": upload.filename,
                    "status": "success",
                    "screenshot_id": sid,
                    "category": category,
                    "tags": tags,
                    "ocr_text": ocr_text[:500],
                    "fusion_applied": fusion_applied,
                    "chunks_indexed": len(chunk_ids),
                    "is_duplicate": is_dup,
                    "duplicate_type": dup_type,
                    "duplicate_of": dup_of,
                    "reminders_created": len(detected_reminders),
                    "reminders": detected_reminders,
                })

            except Exception as e:
                results.append({
                    "filename": upload.filename,
                    "status": "error",
                    "error": str(e),
                })

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {"results": results}


@router.delete("/{screenshot_id}")
def delete_screenshot(screenshot_id: str):
    """Delete screenshot metadata (and best-effort file removal)."""
    file_store, metadata_store = _get_stores()
    vector_store = _get_vector_store()
    meta = metadata_store.get_metadata(screenshot_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Screenshot not found")

    image_uri = meta.image_uri
    chunks_deleted = 0

    # 1. Delete associated vector chunks from ChromaDB (fail-fast before modifying SQLite or disk)
    if vector_store is not None:
        try:
            chunks_deleted = vector_store.delete_by_screenshot_id(screenshot_id)
        except Exception as e:
            logger.error("Vector deletion failed for %s: %s", screenshot_id, e)
            raise HTTPException(
                status_code=500,
                detail=f"Vector store deletion failed for '{screenshot_id}': {e}. Deletion aborted to preserve storage consistency.",
            )

    # 2. Delete metadata record from SQLite
    try:
        metadata_store.delete_metadata(screenshot_id)
    except Exception as e:
        logger.critical(
            "PARTIAL DELETION FAILURE: Vector chunks (%d) were deleted from ChromaDB for '%s', "
            "but SQLite metadata deletion failed: %s",
            chunks_deleted, screenshot_id, e
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Partial deletion failure",
                "screenshot_id": screenshot_id,
                "vector_chunks_deleted": chunks_deleted,
                "metadata_deleted": False,
                "message": f"Vector chunks were removed from search index, but SQLite metadata deletion failed: {e}. Please retry deletion.",
            },
        )

    # Best-effort file deletion
    deleted_file = False
    if image_uri and os.path.isfile(image_uri):
        try:
            os.remove(image_uri)
            deleted_file = True
        except OSError:
            pass

    return {
        "deleted": True,
        "screenshot_id": screenshot_id,
        "file_deleted": deleted_file,
        "vectors_deleted": True,
        "vector_chunks_deleted": chunks_deleted,
    }
