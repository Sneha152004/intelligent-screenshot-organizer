"""
Semantic Search API router.

Endpoints
---------
GET /api/search?q=<text>&top_k=5&category=<optional>

Uses ChromaDB vector store + MiniLM embedder for dense retrieval,
then enriches results with SQLite metadata.
Falls back gracefully if the vector store or embedder is not available.
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/search", tags=["search"])


def _image_url(image_uri: str) -> Optional[str]:
    """Convert an absolute image_uri to a relative /data/... URL."""
    if not image_uri:
        return None
    if os.path.isabs(image_uri):
        try:
            rel = Path(image_uri).relative_to(Path(os.getcwd()))
            return "/" + str(rel).replace("\\", "/")
        except ValueError:
            return None
    return image_uri


@router.get("")
def semantic_search(
    q: str = Query(..., min_length=1, description="Search query text"),
    top_k: int = Query(10, ge=1, le=50, description="Max results to return"),
    category: Optional[str] = Query(None, description="Optionally filter results to a category"),
):
    """
    Perform semantic vector search over indexed screenshot chunks.
    Returns enriched results with metadata from SQLite.
    """
    from ..core.pipeline_instance import get_embedder, get_vector_store, metadata_store, file_store

    embedder = get_embedder()
    vector_store = get_vector_store()

    if embedder is None or vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Search is temporarily unavailable: embedding model or vector store is loading. Please try again shortly.",
        )

    if not q.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    try:
        results = _pipeline_search(q, top_k, category, embedder, vector_store, metadata_store, file_store)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

    return {"query": q, "total": len(results), "results": results}


def _pipeline_search(query_text, top_k, category_filter, embedder, vector_store, metadata_store, file_store):
    """Run vector search and enrich results with SQLite metadata."""
    query_vec = embedder.embed([query_text])[0]
    fetch_k = top_k * 3 if category_filter else top_k
    raw_results = vector_store.similarity_search(query_embedding=query_vec, top_k=fetch_k)

    enriched = []
    seen_ids = set()

    for res in raw_results:
        sid = res.get("screenshot_id")
        if not sid or sid in seen_ids:
            continue

        meta_obj = metadata_store.get_metadata(sid)
        image_uri = file_store.get_image_uri(sid) if sid else None

        category = meta_obj.category if meta_obj else res.get("category", "General")

        if category_filter and category.lower() != category_filter.lower():
            continue

        seen_ids.add(sid)
        resolved_uri = image_uri or (meta_obj.image_uri if meta_obj else "")
        enriched.append({
            "screenshot_id": sid,
            "chunk_index": res.get("chunk_index", 0),
            "text": res.get("document", ""),
            "similarity_score": round(float(res.get("similarity_score", 0.0)), 4),
            "distance": round(float(res.get("distance", 0.0)), 4),
            "filename": meta_obj.filename if meta_obj else "",
            "category": category,
            "summary": meta_obj.summary if meta_obj else "",
            "tags": meta_obj.tags if meta_obj else [],
            "image_uri": resolved_uri,
            "image_url": _image_url(resolved_uri),
            "created_at": meta_obj.created_at if meta_obj else None,
        })

        if len(enriched) >= top_k:
            break

    return enriched
