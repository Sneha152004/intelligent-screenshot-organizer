"""
Tests for Feature: Intelligent Screenshot Reminder Engine
==========================================================
Verifies reminder detection rules, strict safety against sensitive credentials,
deduplication, manual creation, status updating, idempotent scanning,
and API endpoints.
"""

from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from src.models.metadata import ScreenshotMetadata
from src.reminder_service import ReminderService
from src.storage.reminder_store import ReminderStore


@pytest.fixture
def temp_reminder_store(tmp_path):
    """Provides a fresh isolated SQLite reminder store for unit tests."""
    db_file = tmp_path / "test_reminders.sqlite"
    return ReminderStore(str(db_file))


def test_detect_explicit_payment_deadline(temp_reminder_store):
    """Verifies that an unpaid bill with an explicit due date produces a bill_payment reminder."""
    service = ReminderService(reminder_store=temp_reminder_store)
    meta = ScreenshotMetadata(
        screenshot_id="t_bill_001",
        filename="electricity_bill.jpg",
        image_uri="data/screenshots/electricity_bill.jpg",
        category="Payment",
        intent="Payment / Transaction / Bill",
        content_type="bill_invoice",
        importance=5,
        summary="Electricity bill of Rs 1450 due before 25 Nov 2026",
        dates=[{"raw_text": "25th Nov, 2026", "text": "2026-11-25", "normalized": "2026-11-25", "type": "due_date"}],
        evidence={"text_snippet": "Electricity Bill Due Date: 2026-11-25. Please pay before deadline to avoid disconnection."},
    )

    reminders = service.detect_reminders(meta)
    assert len(reminders) == 1
    rem = reminders[0]
    assert rem["reminder_type"] == "bill_payment"
    assert rem["reminder_date"] == "2026-11-25"
    assert rem["priority"] == "high"
    assert "Due" in rem["title"] or "Bill" in rem["title"]


def test_detect_travel_departure_date(temp_reminder_store):
    """Verifies that travel bookings with boarding/departure dates produce travel reminders."""
    service = ReminderService(reminder_store=temp_reminder_store)
    meta = ScreenshotMetadata(
        screenshot_id="t_flight_002",
        filename="indigo_boarding_pass.png",
        image_uri="data/screenshots/indigo_boarding_pass.png",
        category="Travel",
        intent="Travel Planning / Booking",
        content_type="flight_ticket",
        importance=5,
        summary="IndiGo flight 6E-204 departure on 15 Dec 2026",
        dates=[{"raw_text": "15 Dec 2026", "text": "2026-12-15", "normalized": "2026-12-15", "type": "travel_date"}],
        evidence={"text_snippet": "IndiGo Boarding Pass. Flight 6E-204 Departs Date: 15 Dec 2026. Boarding Gate closes at 14:30."},
    )

    reminders = service.detect_reminders(meta)
    assert len(reminders) == 1
    rem = reminders[0]
    assert rem["reminder_type"] == "travel"
    assert rem["reminder_date"] == "2026-12-15"
    assert rem["priority"] == "high"


def test_detect_event_appointment_and_webinar(temp_reminder_store):
    """Verifies scheduled events, conferences, and webinars produce event reminders."""
    service = ReminderService(reminder_store=temp_reminder_store)
    meta = ScreenshotMetadata(
        screenshot_id="t_event_003",
        filename="tech_webinar.png",
        image_uri="data/screenshots/tech_webinar.png",
        category="Education",
        intent="General Information",
        content_type="webinar",
        importance=4,
        summary="AI systems architecture webinar scheduled on 2026-11-20",
        dates=[{"raw_text": "2026-11-20", "text": "2026-11-20", "normalized": "2026-11-20", "type": "event_date"}],
        evidence={"text_snippet": "Webinar on Advanced AI Architectures scheduled for 2026-11-20 at 18:00."},
    )

    reminders = service.detect_reminders(meta)
    assert len(reminders) == 1
    rem = reminders[0]
    assert rem["reminder_type"] == "event"
    assert rem["reminder_date"] == "2026-11-20"


def test_detect_actionable_task_deadlines(temp_reminder_store):
    """Verifies action items (e.g. assignments, registrations) generate task reminders."""
    service = ReminderService(reminder_store=temp_reminder_store)
    meta = ScreenshotMetadata(
        screenshot_id="t_task_004",
        filename="assignment_deadline.jpg",
        image_uri="data/screenshots/assignment_deadline.jpg",
        category="Education",
        intent="Task / Reminder",
        content_type="course_notes",
        importance=4,
        summary="Course assignment submission",
        action_items=[{"text": "Submit academic assignment on time", "status": "pending"}],
        dates=[{"raw_text": "2026-11-18", "normalized": "2026-11-18"}],
        evidence={"text_snippet": "Assignment 4. Submission deadline: 2026-11-18 before 23:59."},
    )

    reminders = service.detect_reminders(meta)
    assert len(reminders) >= 1
    task_rem = next(r for r in reminders if r["reminder_type"] == "task")
    assert task_rem["reminder_date"] == "2026-11-18"
    assert "assignment" in task_rem["title"].lower()


def test_rejection_of_historical_paid_receipts(temp_reminder_store):
    """Verifies that past, fulfilled payment receipts without future deadlines do not produce reminders."""
    service = ReminderService(reminder_store=temp_reminder_store)
    meta = ScreenshotMetadata(
        screenshot_id="t_paid_005",
        filename="old_receipt.jpg",
        image_uri="data/screenshots/old_receipt.jpg",
        category="Payment",
        intent="Payment / Transaction / Bill",
        content_type="payment_receipt",
        importance=3,
        summary="Payment transaction of Rs 450",
        dates=[{"raw_text": "12/04/2026", "text": "2026-04-12", "normalized": "2026-04-12", "type": "transaction_date"}],
        evidence={"text_snippet": "Paid Successfully. UPI Transaction ID: 94829348. Thank you."},
    )

    reminders = service.detect_reminders(meta)
    # Already fulfilled with no future due date -> should produce 0 reminders
    assert len(reminders) == 0


def test_rejection_of_sensitive_credentials(temp_reminder_store):
    """STRICT SAFETY: Never generates reminders from credentials or password screenshots."""
    service = ReminderService(reminder_store=temp_reminder_store)
    meta = ScreenshotMetadata(
        screenshot_id="t_pwd_006",
        filename="wifi_password.png",
        image_uri="data/screenshots/wifi_password.png",
        category="Connectivity",
        intent="Credential / Password Storage",
        content_type="wifi_credential",
        importance=5,
        sensitive_info={"contains_password": True, "contains_payment_info": False},
        dates=[{"raw_text": "2026-11-15", "normalized": "2026-11-15"}],
        evidence={"text_snippet": "Network: Office_5G, Password: SuperSecretPassword123, Date: 2026-11-15"},
    )

    reminders = service.detect_reminders(meta)
    assert len(reminders) == 0


def test_invalid_date_noise_handling(temp_reminder_store):
    """Verifies corrupted OCR dates (> 2040 or gibberish) are safely rejected without crashing."""
    service = ReminderService(reminder_store=temp_reminder_store)
    meta = ScreenshotMetadata(
        screenshot_id="t_noisy_007",
        filename="noisy.jpg",
        image_uri="data/screenshots/noisy.jpg",
        category="General",
        intent="General Information",
        dates=[{"raw_text": "6\nZIMvNZ\n2573", "text": "6\nZIMvNZ\n2573"}],
        evidence={"text_snippet": "Some unreadable noisy text"},
    )

    reminders = service.detect_reminders(meta)
    assert len(reminders) == 0


def test_crud_and_status_lifecycle(temp_reminder_store):
    """Verifies manual creation, retrieval, status updating, and deletion."""
    # Create manual reminder
    rem = temp_reminder_store.create_reminder({
        "reminder_id": "rem_unit_001",
        "title": "Renew Vehicle Insurance",
        "description": "Annual policy renewal deadline",
        "reminder_date": "2026-12-01",
        "reminder_time": "11:00",
        "reminder_type": "subscription",
        "priority": "high",
        "importance_score": 4,
        "status": "pending",
    })
    assert rem["reminder_id"] == "rem_unit_001"
    assert rem["status"] == "pending"

    # Update status to completed
    updated = temp_reminder_store.update_reminder("rem_unit_001", {"status": "completed"})
    assert updated["status"] == "completed"

    # List completed
    completed_items = temp_reminder_store.list_reminders(status="completed")
    assert len(completed_items) == 1
    assert completed_items[0]["reminder_id"] == "rem_unit_001"

    # Delete reminder
    deleted = temp_reminder_store.delete_reminder("rem_unit_001")
    assert deleted is True
    assert temp_reminder_store.get_reminder("rem_unit_001") is None


def test_idempotent_scanning(temp_reminder_store):
    """Verifies that repeatedly scanning screenshots creates items only once without duplicating."""
    class MockMetaStore:
        def list_all_metadata(self):
            return [
                ScreenshotMetadata(
                    screenshot_id="scan_ss_1",
                    filename="upcoming_event.png",
                    image_uri="data/screenshots/upcoming_event.png",
                    category="Education",
                    intent="General Information",
                    content_type="webinar",
                    importance=4,
                    summary="Tech conference talk",
                    dates=[{"raw_text": "2026-12-10", "normalized": "2026-12-10", "type": "event_date"}],
                    evidence={"text_snippet": "Webinar on 2026-12-10"},
                )
            ]

    service = ReminderService(reminder_store=temp_reminder_store, metadata_store=MockMetaStore())

    # Scan 1
    res1 = service.scan_all_screenshots()
    assert res1["new_reminders_created"] == 1
    assert res1["existing_skipped"] == 0

    # Scan 2: Must be 100% idempotent
    res2 = service.scan_all_screenshots()
    assert res2["new_reminders_created"] == 0
    assert res2["existing_skipped"] == 1

    # Check total in DB
    items = temp_reminder_store.list_reminders()
    assert len(items) == 1


def test_api_reminders_endpoints():
    """Verifies the REST API endpoints via FastAPI TestClient."""
    client = TestClient(app)

    # 1. GET /api/reminders
    resp = client.get("/api/reminders")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "reminders" in data

    # 2. POST /api/reminders (Manual creation)
    create_payload = {
        "title": "File Quarterly Tax Returns",
        "reminder_date": "2026-12-31",
        "reminder_time": "17:00",
        "reminder_type": "task",
        "priority": "high",
        "importance_score": 5,
        "description": "Submit Q4 tax returns before deadline",
    }
    resp_create = client.post("/api/reminders", json=create_payload)
    assert resp_create.status_code == 201
    created_id = resp_create.json()["reminder_id"]
    assert resp_create.json()["title"] == "File Quarterly Tax Returns"

    # 3. GET /api/reminders/{reminder_id}
    resp_get = client.get(f"/api/reminders/{created_id}")
    assert resp_get.status_code == 200
    assert resp_get.json()["reminder_id"] == created_id

    # 4. PATCH /api/reminders/{reminder_id}
    resp_patch = client.patch(f"/api/reminders/{created_id}", json={"status": "completed"})
    assert resp_patch.status_code == 200
    assert resp_patch.json()["status"] == "completed"

    # 5. POST /api/reminders/scan
    resp_scan = client.post("/api/reminders/scan")
    assert resp_scan.status_code == 200
    scan_data = resp_scan.json()
    assert "total_screenshots_scanned" in scan_data

    # 6. DELETE /api/reminders/{reminder_id}
    resp_del = client.delete(f"/api/reminders/{created_id}")
    assert resp_del.status_code == 200
    assert resp_del.json()["deleted"] is True
