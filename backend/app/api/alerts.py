"""
I.R.I.S. Alert API Endpoints
============================
Provides REST endpoints for managing system alerts, querying summary metrics,
updating alert status, and triggering idempotent alert scans.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.alert_service import AlertService
from src.storage.alert_store import AlertStore
from src.storage.reminder_store import ReminderStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


def _get_alert_store() -> AlertStore:
    return AlertStore("data/metadata.sqlite")


def _get_reminder_store() -> ReminderStore:
    return ReminderStore("data/metadata.sqlite")


def _get_metadata_store():
    from ..core.pipeline_instance import metadata_store
    return metadata_store


# ── Pydantic Request Models ──────────────────────────────────────────────────
class UpdateAlertRequest(BaseModel):
    status: Optional[str] = Field(None, pattern=r"^(active|read|resolved|dismissed)$")
    severity: Optional[str] = Field(None, pattern=r"^(critical|high|medium|low)$")
    title: Optional[str] = None
    message: Optional[str] = None


# ── Endpoints ────────────────────────────────────────────────────────────────
@router.get("")
def list_alerts(
    status: Optional[str] = Query(None, description="Filter by status: active, read, resolved, dismissed, all"),
    severity: Optional[str] = Query(None, description="Filter by severity: critical, high, medium, low, all"),
    type: Optional[str] = Query(None, description="Filter by alert type"),
    sort: str = Query("desc", description="Sort order: 'desc' or 'asc'"),
):
    """Lists alerts with multi-dimensional filtering and summary metrics."""
    store = _get_alert_store()
    items = store.list_alerts(
        status=status,
        severity=severity,
        alert_type=type,
        sort=sort,
    )
    summary = store.get_summary_metrics()

    return {
        "total": len(items),
        "summary": summary,
        "filters_applied": {
            "status": status,
            "severity": severity,
            "type": type,
            "sort": sort,
        },
        "alerts": items,
    }


@router.get("/summary")
def get_alerts_summary():
    """Returns active, unread, critical, high-severity, and resolution counts."""
    store = _get_alert_store()
    return store.get_summary_metrics()


@router.post("/scan")
def scan_alerts():
    """
    Evaluates reminders and high-importance screenshots to generate or update alerts.
    Idempotent: updates existing active alerts without creating duplicates.
    """
    a_store = _get_alert_store()
    r_store = _get_reminder_store()
    m_store = _get_metadata_store()
    service = AlertService(alert_store=a_store, reminder_store=r_store, metadata_store=m_store)
    res = service.scan_all_for_alerts()
    return res


@router.get("/{alert_id}")
def get_alert(alert_id: str):
    """Retrieves a single alert and its joined reminder/screenshot metadata."""
    store = _get_alert_store()
    alert = store.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.patch("/{alert_id}")
def update_alert(alert_id: str, req: UpdateAlertRequest):
    """Updates status or details for an existing alert."""
    store = _get_alert_store()
    updates = req.model_dump(exclude_unset=True) if hasattr(req, "model_dump") else req.dict(exclude_unset=True)
    if not updates:
        alert = store.get_alert(alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        return alert

    updated = store.update_alert(alert_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Alert not found")
    return updated


@router.delete("/{alert_id}")
def delete_alert(alert_id: str):
    """Deletes an alert."""
    store = _get_alert_store()
    deleted = store.delete_alert(alert_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"deleted": True, "alert_id": alert_id}
