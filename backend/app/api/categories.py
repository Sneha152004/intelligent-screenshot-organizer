"""
Categories API router.

Endpoints
---------
GET /api/categories    Return all known categories with screenshot counts and icons
"""

from collections import Counter
from fastapi import APIRouter

router = APIRouter(prefix="/api/categories", tags=["categories"])

# Human-friendly metadata for each category the pipeline can produce
CATEGORY_META = {
    "Shopping":      {"icon": "🛒", "color": "#f59e0b", "description": "E-commerce, product listings, receipts"},
    "Travel":        {"icon": "✈️", "color": "#06b6d4", "description": "Bookings, maps, itineraries"},
    "Education":     {"icon": "📚", "color": "#8b5cf6", "description": "Articles, lectures, study material"},
    "Payment":       {"icon": "💳", "color": "#10b981", "description": "Invoices, transactions, bills"},
    "Communication": {"icon": "💬", "color": "#3b82f6", "description": "Chats, emails, notifications"},
    "Work_Career":   {"icon": "💼", "color": "#6366f1", "description": "Resumes, job listings, work docs"},
    "Coding":        {"icon": "💻", "color": "#00d4ff", "description": "Code, terminals, IDEs"},
    "Entertainment": {"icon": "🎮", "color": "#ec4899", "description": "Gaming, streaming, media"},
    "Connectivity":  {"icon": "📶", "color": "#f97316", "description": "Network, settings, dashboards"},
    "General":       {"icon": "📁", "color": "#9ca3af", "description": "Uncategorised screenshots"},
}

DEFAULT_META = {"icon": "🗂️", "color": "#64748b", "description": "Other"}


@router.get("")
def list_categories():
    """Return all categories present in the database with counts and metadata."""
    from ..core.pipeline_instance import metadata_store
    all_meta = metadata_store.list_all_metadata()
    counts: Counter = Counter(m.category for m in all_meta)

    # Include all known categories (even if count is 0) plus any unseen ones in DB
    all_cats = set(CATEGORY_META.keys()) | set(counts.keys())

    categories = []
    for cat in sorted(all_cats):
        meta = CATEGORY_META.get(cat, DEFAULT_META)
        categories.append({
            "name": cat,
            "label": cat.replace("_", " & "),
            "count": counts.get(cat, 0),
            "icon": meta["icon"],
            "color": meta["color"],
            "description": meta["description"],
        })

    return {
        "total_screenshots": len(all_meta),
        "categories": categories,
    }
