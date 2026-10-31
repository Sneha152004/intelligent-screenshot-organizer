"""
Tags API router.

Endpoints
---------
GET /api/tags    Return all unique tags present across stored screenshots with counts
"""

from collections import Counter
from fastapi import APIRouter

router = APIRouter(prefix="/api/tags", tags=["tags"])


def _get_metadata_store():
    from ..core.pipeline_instance import metadata_store
    return metadata_store


@router.get("")
def list_tags():
    """Return all unique tags present in the database with counts."""
    store = _get_metadata_store()
    all_meta = store.list_all_metadata()
    counts = Counter()
    for m in all_meta:
        for t in (m.tags or []):
            counts[t] += 1

    tags = [{"name": k, "count": v} for k, v in counts.most_common()]
    return {
        "total_tags": len(tags),
        "tags": tags,
    }
