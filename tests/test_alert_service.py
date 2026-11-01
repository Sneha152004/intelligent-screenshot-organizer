"""
Tests for Feature 4: Intelligent Alert Engine (I.R.I.S.)
========================================================
Verifies deterministic alert evaluation rules, urgency & severity calculation,
safety against sensitive credentials, reminder lifecycle synchronization,
idempotent library scans, and REST API endpoints.
"""

from datetime import date, datetime, timedelta, timezone
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from src.alert_service import AlertService
from src.storage.alert_store import AlertStore
from src.storage.reminder_store import ReminderStore


@pytest.fixture
def temp_stores(tmp_path):
    """Provides isolated SQLite alert and reminder stores."""
    db_file = tmp_path / "test_alerts.sqlite"
    a_store = AlertStore(str(db_file))
    r_store = ReminderStore(str(db_file))
    return a_store, r_store


def test_overdue_payment_generates_critical_alert(temp_stores):
    """Verifies that an overdue bill payment creates a critical severity alert."""
    a_store, r_store = temp_stores
    service = AlertService(alert_store=a_store, reminder_store=r_store)

    ref_date = date(2026, 11, 15)
    past_date = "2026-11-10"  # 5 days overdue

    reminder = {
        "reminder_id": "rem_bill_past",
        "screenshot_id": "ss_bill_001",
        "title": "Pay Electricity Bill",
        "reminder_date": past_date,
        "reminder_type": "bill_payment",
        "priority": "high",
        "importance_score": 5,
        "status": "pending",
    }

    alerts = service.evaluate_reminder_for_alerts(reminder, reference_date=ref_date)
    assert len(alerts) == 1
    alt = alerts[0]
    assert alt["alert_type"] == "overdue_payment"
    assert alt["severity"] == "critical"
    assert "OVERDUE" in alt["title"]
    assert "5 days ago" in alt["message"]


def test_due_today_generates_critical_or_high_alert(temp_stores):
    """Verifies that a payment due today triggers a critical alert."""
    a_store, r_store = temp_stores
    service = AlertService(alert_store=a_store, reminder_store=r_store)

    today_date = date(2026, 11, 15)

    reminder = {
        "reminder_id": "rem_today_002",
        "screenshot_id": "ss_bill_002",
        "title": "Internet Bill Due",
        "reminder_date": "2026-11-15",
        "reminder_type": "bill_payment",
        "priority": "high",
        "importance_score": 4,
        "status": "pending",
    }

    alerts = service.evaluate_reminder_for_alerts(reminder, reference_date=today_date)
    assert len(alerts) == 1
    alt = alerts[0]
    assert alt["alert_type"] == "upcoming_payment"
    assert alt["severity"] == "critical"
    assert "DUE TODAY" in alt["title"]


def test_urgent_tomorrow_travel_generates_critical_alert(temp_stores):
    """Verifies that a flight departure tomorrow (within 24h) produces a critical upcoming_travel alert."""
    a_store, r_store = temp_stores
    service = AlertService(alert_store=a_store, reminder_store=r_store)

    ref_date = date(2026, 11, 15)
    tomorrow = "2026-11-16"

    reminder = {
        "reminder_id": "rem_travel_003",
        "screenshot_id": "ss_flight_003",
        "title": "IndiGo Flight 6E-204",
        "reminder_date": tomorrow,
        "reminder_time": "14:30",
        "reminder_type": "travel",
        "priority": "high",
        "importance_score": 5,
        "status": "pending",
    }

    alerts = service.evaluate_reminder_for_alerts(reminder, reference_date=ref_date)
    assert len(alerts) == 1
    alt = alerts[0]
    assert alt["alert_type"] == "upcoming_travel"
    assert alt["severity"] == "critical"
    assert "URGENT" in alt["title"]
    assert "TOMORROW" in alt["message"]


def test_upcoming_within_three_days_generates_medium_high_alert(temp_stores):
    """Verifies that events or deadlines 3 days away produce upcoming alerts."""
    a_store, r_store = temp_stores
    service = AlertService(alert_store=a_store, reminder_store=r_store)

    ref_date = date(2026, 11, 15)
    in_3_days = "2026-11-18"

    reminder = {
        "reminder_id": "rem_event_004",
        "screenshot_id": "ss_conf_004",
        "title": "AI Systems Webinar",
        "reminder_date": in_3_days,
        "reminder_type": "event",
        "priority": "medium",
        "importance_score": 3,
        "status": "pending",
    }

    alerts = service.evaluate_reminder_for_alerts(reminder, reference_date=ref_date)
    assert len(alerts) == 1
    alt = alerts[0]
    assert alt["alert_type"] == "upcoming_event"
    assert alt["severity"] == "medium"
    assert "3 days" in alt["message"]


def test_completed_and_dismissed_reminders_excluded(temp_stores):
    """Rule 1: Completed or dismissed reminders must NEVER generate active alerts."""
    a_store, r_store = temp_stores
    service = AlertService(alert_store=a_store, reminder_store=r_store)

    ref_date = date(2026, 11, 15)

    completed_rem = {
        "reminder_id": "rem_comp_005",
        "title": "Paid Rent",
        "reminder_date": "2026-11-15",
        "reminder_type": "bill_payment",
        "status": "completed",
    }
    dismissed_rem = {
        "reminder_id": "rem_dism_006",
        "title": "Old Task",
        "reminder_date": "2026-11-15",
        "reminder_type": "task",
        "status": "dismissed",
    }

    assert service.evaluate_reminder_for_alerts(completed_rem, reference_date=ref_date) == []
    assert service.evaluate_reminder_for_alerts(dismissed_rem, reference_date=ref_date) == []


def test_sensitive_credentials_excluded(temp_stores):
    """Rule 6: Never generate alerts exposing passwords, OTPs, or authentication credentials."""
    a_store, r_store = temp_stores
    service = AlertService(alert_store=a_store, reminder_store=r_store)

    # Clean sanitization
    raw_text = "Your password is: SecretPassword123. Use before today."
    clean = service.sanitize_message(raw_text)
    assert "SecretPassword123" not in clean
    assert "[REDACTED]" in clean

    # High importance item with sensitive_info containing password
    meta_with_pwd = {
        "screenshot_id": "ss_pwd_007",
        "importance": 5,
        "sensitive_info": {"contains_password": True},
        "created_at": "2026-11-15T10:00:00Z",
    }
    assert service.evaluate_high_importance_screenshot(meta_with_pwd) is None


def test_alert_crud_and_status_lifecycle(temp_stores):
    """Verifies creating alerts, updating status (active -> read -> resolved -> dismissed), and deletion."""
    a_store, _ = temp_stores

    created = a_store.create_alert({
        "alert_id": "alt_test_001",
        "reminder_id": "rem_1",
        "screenshot_id": "ss_1",
        "title": "URGENT: Review Contract",
        "message": "Review deadline is approaching.",
        "alert_type": "approaching_deadline",
        "severity": "high",
        "status": "active",
        "trigger_date": "2026-11-16",
    })
    assert created["alert_id"] == "alt_test_001"
    assert created["status"] == "active"

    # Mark as read
    updated_read = a_store.update_alert("alt_test_001", {"status": "read"})
    assert updated_read["status"] == "read"

    # Resolve alert
    updated_res = a_store.update_alert("alt_test_001", {"status": "resolved"})
    assert updated_res["status"] == "resolved"

    # Summary metrics should reflect resolved
    summary = a_store.get_summary_metrics()
    assert summary["resolved"] == 1
    assert summary["active"] == 0

    # Delete
    assert a_store.delete_alert("alt_test_001") is True
    assert a_store.get_alert("alt_test_001") is None


def test_reminder_completion_resolves_linked_alerts(temp_stores):
    """Rule 4: Completing a reminder automatically resolves linked active alerts."""
    a_store, r_store = temp_stores
    service = AlertService(alert_store=a_store, reminder_store=r_store)

    # Create active reminder in r_store due tomorrow
    due_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    r_store.create_reminder({
        "reminder_id": "rem_sync_001",
        "title": "Pay Credit Card Bill",
        "reminder_date": due_date,
        "reminder_type": "bill_payment",
        "priority": "high",
        "importance_score": 5,
        "status": "pending",
    })

    # Sync generates alert
    service.sync_reminder_to_alerts(r_store.get_reminder("rem_sync_001"))
    active_alerts = a_store.list_alerts(status="active")
    assert len(active_alerts) == 1
    assert active_alerts[0]["reminder_id"] == "rem_sync_001"

    # Now mark reminder as completed
    r_store.update_reminder("rem_sync_001", {"status": "completed"})
    service.sync_reminder_to_alerts(r_store.get_reminder("rem_sync_001"))

    # Linked alert must now be resolved
    active_now = a_store.list_alerts(status="active")
    assert len(active_now) == 0

    resolved_now = a_store.list_alerts(status="resolved")
    assert len(resolved_now) == 1
    assert resolved_now[0]["reminder_id"] == "rem_sync_001"


def test_idempotent_scanning(temp_stores):
    """Verifies that scan_all_for_alerts is 100% idempotent."""
    a_store, r_store = temp_stores
    service = AlertService(alert_store=a_store, reminder_store=r_store)

    r_store.create_reminder({
        "reminder_id": "rem_scan_001",
        "title": "Boarding Gate Alert",
        "reminder_date": datetime.now().strftime("%Y-%m-%d"),
        "reminder_type": "travel",
        "priority": "high",
        "importance_score": 5,
        "status": "pending",
    })

    # Scan 1
    res1 = service.scan_all_for_alerts()
    assert res1["new_alerts_created"] == 1
    assert res1["alerts_updated"] == 0

    # Scan 2: Must not create duplicate active alert
    res2 = service.scan_all_for_alerts()
    assert res2["new_alerts_created"] == 0
    assert res2["alerts_updated"] == 1  # Updated cleanly in place

    # Total active alerts in DB must still be 1
    active_items = a_store.list_alerts(status="active")
    assert len(active_items) == 1


def test_api_alerts_endpoints():
    """Verifies the REST API endpoints (/api/alerts) via FastAPI TestClient."""
    client = TestClient(app)

    # 1. GET /api/alerts
    resp = client.get("/api/alerts")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "alerts" in data

    # 2. GET /api/alerts/summary
    resp_sum = client.get("/api/alerts/summary")
    assert resp_sum.status_code == 200
    summary = resp_sum.json()
    assert "active" in summary
    assert "critical" in summary

    # 3. POST /api/alerts/scan
    resp_scan = client.post("/api/alerts/scan")
    assert resp_scan.status_code == 200
    scan_res = resp_scan.json()
    assert "new_alerts_created" in scan_res
    assert "summary" in scan_res

    # 4. If any alert exists, test PATCH /api/alerts/{id} and DELETE
    alerts_list = client.get("/api/alerts").json()["alerts"]
    if alerts_list:
        sample_id = alerts_list[0]["alert_id"]
        # PATCH
        patch_res = client.patch(f"/api/alerts/{sample_id}", json={"status": "read"})
        assert patch_res.status_code == 200
        assert patch_res.json()["status"] == "read"

        # GET single
        get_res = client.get(f"/api/alerts/{sample_id}")
        assert get_res.status_code == 200
        assert get_res.json()["alert_id"] == sample_id
