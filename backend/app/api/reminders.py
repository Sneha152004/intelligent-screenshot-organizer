"""
I.R.I.S. Reminder API Endpoints
===============================
Provides REST endpoints for managing screenshot reminders,
querying summary metrics, and triggering idempotent library scans.
"""

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.reminder_service import ReminderService
from src.storage.reminder_store import ReminderStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/reminders", tags=["reminders"])


def _get_reminder_store() -> ReminderStore:
    return ReminderStore("data/metadata.sqlite")


def _get_metadata_store():
    from ..core.pipeline_instance import metadata_store
    return metadata_store


# ── Pydantic Request Models ──────────────────────────────────────────────────
class CreateReminderRequest(BaseModel):
    title: str = Field(..., min_length=1, description="Title of the reminder")
    reminder_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Date in YYYY-MM-DD format")
    reminder_time: Optional[str] = Field(None, description="Optional time string (HH:MM)")
    reminder_type: Optional[str] = Field("general", description="Type: bill_payment, travel, event, task, subscription, general")
    priority: Optional[str] = Field("medium", description="Priority: high, medium, low")
    importance_score: Optional[int] = Field(3, ge=1, le=5, description="Importance score 1-5")
    description: Optional[str] = Field("", description="Optional details or instructions")
    screenshot_id: Optional[str] = Field(None, description="Optional associated screenshot ID")
    source_text: Optional[str] = Field("", description="Evidence excerpt or trigger text")


class UpdateReminderRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    reminder_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    reminder_time: Optional[str] = None
    reminder_type: Optional[str] = None
    priority: Optional[str] = None
    importance_score: Optional[int] = Field(None, ge=1, le=5)
    status: Optional[str] = Field(None, pattern=r"^(pending|completed|dismissed)$")


# ── Endpoints ────────────────────────────────────────────────────────────────
@router.get("")
def list_reminders(
    status: Optional[str] = Query(None, description="Filter status: pending, completed, dismissed, all"),
    type: Optional[str] = Query(None, description="Filter type: bill_payment, travel, event, task, subscription, general"),
    date_filter: Optional[str] = Query(None, description="Filter horizon: overdue, today, upcoming, completed"),
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    sort: str = Query("asc", description="Sort order: 'asc' or 'desc'"),
):
    """Lists reminders with filtering and summary metrics."""
    store = _get_reminder_store()
    items = store.list_reminders(
        status=status,
        reminder_type=type,
        date_filter=date_filter,
        from_date=from_date,
        to_date=to_date,
        sort=sort,
    )
    summary = store.get_summary_metrics()

    return {
        "total": len(items),
        "summary": summary,
        "filters_applied": {
            "status": status,
            "type": type,
            "date_filter": date_filter,
            "from_date": from_date,
            "to_date": to_date,
            "sort": sort,
        },
        "reminders": items,
    }


@router.post("/scan")
def scan_library_for_reminders():
    """
    Scans stored screenshot metadata for actionable reminder candidates.
    Idempotent: skips existing reminders and creates only new unique candidates.
    """
    r_store = _get_reminder_store()
    m_store = _get_metadata_store()
    service = ReminderService(reminder_store=r_store, metadata_store=m_store)
    res = service.scan_all_screenshots()
    summary = r_store.get_summary_metrics()
    res["summary"] = summary

    # Automatically evaluate alerts after scanning reminders
    try:
        from src.alert_service import AlertService
        from src.storage.alert_store import AlertStore
        _a_store = AlertStore("data/metadata.sqlite")
        _a_service = AlertService(alert_store=_a_store, reminder_store=r_store, metadata_store=m_store)
        _a_service.scan_all_for_alerts()
    except Exception as a_err:
        logger.warning(f"Failed to sync alerts on scan: {a_err}")

    return res


@router.post("", status_code=status.HTTP_201_CREATED)
def create_reminder(req: CreateReminderRequest):
    """Manually creates a new reminder."""
    store = _get_reminder_store()

    # Check for duplicate
    existing = store.find_duplicate_reminder(
        screenshot_id=req.screenshot_id,
        reminder_date=req.reminder_date,
        reminder_type=req.reminder_type or "general",
        title=req.title,
    )
    if existing:
        return existing

    reminder_id = f"rem_man_{uuid.uuid4().hex[:8]}"
    created = store.create_reminder({
        "reminder_id": reminder_id,
        "screenshot_id": req.screenshot_id,
        "title": req.title,
        "description": req.description or "",
        "reminder_date": req.reminder_date,
        "reminder_time": req.reminder_time,
        "reminder_type": req.reminder_type or "general",
        "priority": req.priority or "medium",
        "importance_score": req.importance_score or 3,
        "status": "pending",
        "source_text": req.source_text or "Manually created",
    })

    # Synchronize alert for the newly created reminder
    try:
        from src.alert_service import AlertService
        from src.storage.alert_store import AlertStore
        _a_store = AlertStore("data/metadata.sqlite")
        _a_service = AlertService(alert_store=_a_store, reminder_store=store)
        _a_service.sync_reminder_to_alerts(created)
    except Exception as a_err:
        logger.warning(f"Failed to sync alert on create_reminder: {a_err}")

    return created


@router.get("/{reminder_id}")
def get_reminder(reminder_id: str):
    """Retrieves a single reminder and attached screenshot details."""
    store = _get_reminder_store()
    rem = store.get_reminder(reminder_id)
    if not rem:
        raise HTTPException(status_code=404, detail="Reminder not found")
    return rem


@router.patch("/{reminder_id}")
def update_reminder(reminder_id: str, req: UpdateReminderRequest):
    """Updates fields or status for an existing reminder."""
    store = _get_reminder_store()
    updates = req.model_dump(exclude_unset=True) if hasattr(req, "model_dump") else req.dict(exclude_unset=True)
    if not updates:
        rem = store.get_reminder(reminder_id)
        if not rem:
            raise HTTPException(status_code=404, detail="Reminder not found")
        return rem

    updated = store.update_reminder(reminder_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Reminder not found")

    # Synchronize alerts (e.g. resolve if completed/dismissed, or recalculate)
    try:
        from src.alert_service import AlertService
        from src.storage.alert_store import AlertStore
        _a_store = AlertStore("data/metadata.sqlite")
        _a_service = AlertService(alert_store=_a_store, reminder_store=store)
        _a_service.sync_reminder_to_alerts(updated)
    except Exception as a_err:
        logger.warning(f"Failed to sync alert on update_reminder: {a_err}")

    return updated


@router.delete("/{reminder_id}")
def delete_reminder(reminder_id: str):
    """Deletes a reminder."""
    store = _get_reminder_store()
    deleted = store.delete_reminder(reminder_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Reminder not found")
    return {"deleted": True, "reminder_id": reminder_id}
