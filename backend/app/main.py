"""
FastAPI Application Entry Point
================================
Intelligent Screenshot Organizer — REST API

Run from the intelligent-screenshot-organizer/ root:
    uvicorn backend.app.main:app --reload --port 8000
"""

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.screenshots import router as screenshots_router
from .api.categories import router as categories_router
from .api.search import router as search_router
from .api.tags import router as tags_router
from .api.chat import router as chat_router
from .api.reminders import router as reminders_router
from .api.alerts import router as alerts_router

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="I.R.I.S. Screenshot Organizer API",
    description="Intelligent Screenshot Organizer — OCR, multimodal classification & semantic search.",
    version="1.0.0",
)

# ── CORS (allow the Vite dev server and any localhost origin) ─────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(screenshots_router)
app.include_router(categories_router)
app.include_router(search_router)
app.include_router(tags_router)
app.include_router(chat_router)
app.include_router(reminders_router)
app.include_router(alerts_router)

# ── Static files: serve screenshots stored in data/screenshots ────────────────
_data_dir = Path(os.getcwd()) / "data"
_screenshots_static = _data_dir / "screenshots"
_screenshots_static.mkdir(parents=True, exist_ok=True)

app.mount("/data", StaticFiles(directory=str(_data_dir)), name="data")

# ── Duplicates alias ──────────────────────────────────────────────────────────
@app.get("/api/duplicates")
def api_duplicates_alias(
    type: str = None,
    max_distance: int = 6,
    category: str = None,
):
    """Direct alias for /api/screenshots/duplicates."""
    from .api.screenshots import get_duplicates
    return get_duplicates(type=type, max_distance=max_distance, category=category)


# ── Timeline alias ──────────────────────────────────────────────────────────
@app.get("/api/timeline")
def api_timeline_alias(
    year: Optional[int] = None,
    month: Optional[int] = None,
    category: Optional[str] = None,
    intent: Optional[str] = None,
    sort: str = "desc",
):
    """Direct alias for /api/screenshots/timeline."""
    from .api.screenshots import get_timeline
    return get_timeline(year=year, month=month, category=category, intent=intent, sort=sort)


# ── Root health check ─────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "online", "service": "I.R.I.S. Screenshot Organizer API", "version": "1.0.0"}


@app.get("/api/health")
def health():
    """Quick liveness check for the frontend to verify the API is reachable."""
    return {"status": "ok"}
