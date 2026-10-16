"""
Screenshots API router.

Endpoints
---------
GET  /api/screenshots          List all screenshots, optional ?category= and ?q= filters
GET  /api/screenshots/{id}     Get single screenshot detail
POST /api/screenshots/upload   Upload one or more images and run the pipeline
DELETE /api/screenshots/{id}   Delete a screenshot record (and optionally its file)
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

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


@router.get("")
def list_screenshots(
    category: Optional[str] = Query(None, description="Filter by category name"),
    q: Optional[str] = Query(None, description="Simple text filter on filename / OCR text"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Return a paginated list of screenshots with optional filters."""
    _, metadata_store = _get_stores()
    all_meta = metadata_store.list_all_metadata()

    if category:
        all_meta = [m for m in all_meta if m.category.lower() == category.lower()]

    if q:
        q_lower = q.lower()
        all_meta = [
            m for m in all_meta
            if q_lower in m.filename.lower()
            or q_lower in (m.summary or "").lower()
            or q_lower in m.category.lower()
        ]

    total = len(all_meta)
    page = all_meta[offset: offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": [_meta_to_dict(m) for m in page],
    }


@router.get("/{screenshot_id}")
def get_screenshot(screenshot_id: str):
    """Return full detail for a single screenshot, including OCR text."""
    _, metadata_store = _get_stores()
    meta = metadata_store.get_metadata(screenshot_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Screenshot not found")

    d = _meta_to_dict(meta)

    # Attach raw OCR text from evidence if available
    evidence = meta.evidence or {}
    ocr_snippet = evidence.get("text_snippet", "")
    d["ocr_text"] = ocr_snippet

    return d


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
    vector_store = get_vector_store()
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
                        })

                if chunk_texts:
                    embeddings = embedder.embed(chunk_texts)
                    vector_store.add_documents(
                        ids=chunk_ids,
                        documents=chunk_texts,
                        embeddings=embeddings,
                        metadatas=chunk_metas,
                    )

                results.append({
                    "filename": upload.filename,
                    "status": "success",
                    "screenshot_id": sid,
                    "category": category,
                    "ocr_text": ocr_text[:500],
                    "fusion_applied": fusion_applied,
                    "chunks_indexed": len(chunk_ids),
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
    _, metadata_store = _get_stores()
    meta = metadata_store.get_metadata(screenshot_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Screenshot not found")

    image_uri = meta.image_uri
    metadata_store.delete_metadata(screenshot_id)

    # Best-effort file deletion
    deleted_file = False
    if image_uri and os.path.isfile(image_uri):
        try:
            os.remove(image_uri)
            deleted_file = True
        except OSError:
            pass

    return {"deleted": True, "screenshot_id": screenshot_id, "file_deleted": deleted_file}
