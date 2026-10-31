"""
Categories API router.

Endpoints
---------
GET /api/categories    Return all known categories with screenshot counts and icons
"""

from collections import Counter
from fastapi import APIRouter

router = APIRouter(prefix="/api/categories", tags=["categories"])

# Canonical 9 categories supported by the production taxonomy
CATEGORY_META = {
    "Education":     {"icon": "📚", "color": "#8b5cf6", "description": "Articles, lectures, study material"},
    "Shopping":      {"icon": "🛒", "color": "#f59e0b", "description": "E-commerce, product listings, orders"},
    "Payment":       {"icon": "💳", "color": "#10b981", "description": "Invoices, transactions, receipts"},
    "Travel":        {"icon": "✈️", "color": "#06b6d4", "description": "Bookings, tickets, maps, itineraries"},
    "Communication": {"icon": "💬", "color": "#3b82f6", "description": "Chats, emails, notifications"},
    "Entertainment": {"icon": "🎮", "color": "#ec4899", "description": "Gaming, streaming, media"},
    "Work & Career": {"icon": "💼", "color": "#6366f1", "description": "Resumes, job listings, work docs"},
    "Connectivity":  {"icon": "📶", "color": "#f97316", "description": "Network, settings, dashboards"},
    "General":       {"icon": "📁", "color": "#9ca3af", "description": "General & uncategorised screenshots"},
}

# Aliases for legacy formatting in historical databases
CATEGORY_ALIASES = {
    "Work_Career": "Work & Career",
}

DEFAULT_META = {"icon": "🗂️", "color": "#64748b", "description": "Other"}


@router.get("")
def list_categories():
    """Return all categories present in the database with counts and metadata."""
    from ..core.pipeline_instance import metadata_store
    all_meta = metadata_store.list_all_metadata()
    # Count screenshots across canonical categories, safely resolving aliases and legacy values
    counts: Counter = Counter()
    for m in all_meta:
        cat = m.category
        if cat in CATEGORY_ALIASES:
            cat = CATEGORY_ALIASES[cat]
        elif cat == "Coding":
            # Historical legacy value: attribute count to Work & Career without inflating top-level taxonomy
            cat = "Work & Career"
        elif cat not in CATEGORY_META:
            cat = "General"
        counts[cat] += 1

    # Return strictly the 9 canonical categories in defined order
    categories = []
    for cat in CATEGORY_META:
        meta = CATEGORY_META[cat]
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
