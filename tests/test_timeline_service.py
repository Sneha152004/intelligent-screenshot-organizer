"""
Tests for Feature 3: Intelligent Screenshot Timeline
====================================================
Verifies chronological ordering, month grouping, event date preference,
creation date fallback, invalid date handling, filtering, and API endpoints.
"""

from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from src.models.metadata import ScreenshotMetadata
from src.timeline_service import TimelineService


@pytest.fixture
def sample_screenshots():
    """Builds synthetic screenshots spanning diverse dates, categories, and intents."""
    return [
        # ss_1: Event date in April 2026, created in November 2026
        ScreenshotMetadata(
            screenshot_id="t_ss_001",
            filename="flight_booking.jpg",
            image_uri="data/screenshots/flight_booking.jpg",
            category="Travel",
            intent="Travel Planning / Booking",
            content_type="flight_ticket",
            importance=5,
            importance_reason="Flight booking ticket",
            created_at="2026-11-01T10:00:00+00:00",
            summary="Flight booking from DEL to BLR on 15 Apr 2026",
            dates=[{"raw_text": "15th Apr, 2026", "text": "2026-04-15", "normalized": "2026-04-15", "type": "travel_date"}],
        ),
        # ss_2: Fallback to created_at in November 2026
        ScreenshotMetadata(
            screenshot_id="t_ss_002",
            filename="wifi_password.png",
            image_uri="data/screenshots/wifi_password.png",
            category="Connectivity",
            intent="Credential / Password Storage",
            content_type="wifi_credential",
            importance=4,
            importance_reason="Wifi credentials",
            created_at="2026-11-10T08:30:00+00:00",
            summary="Airtel office wifi credentials",
            dates=[],
        ),
        # ss_3: Older creation date in September 2026
        ScreenshotMetadata(
            screenshot_id="t_ss_003",
            filename="grocery_order.jpg",
            image_uri="data/screenshots/grocery_order.jpg",
            category="Shopping",
            intent="Shopping / Purchase",
            content_type="shopping_order",
            importance=3,
            importance_reason="Order receipt",
            created_at="2026-09-15T14:20:00+00:00",
            summary="BigBasket grocery order confirmation",
            dates=[],
        ),
        # ss_4: Invalid OCR date artifact ("6\nZIMvNZ\n2573") -> falls back to created_at (September 2026)
        ScreenshotMetadata(
            screenshot_id="t_ss_004",
            filename="noisy_receipt.jpg",
            image_uri="data/screenshots/noisy_receipt.jpg",
            category="Payment",
            intent="Payment / Transaction / Bill",
            content_type="payment_receipt",
            importance=3,
            importance_reason="Receipt",
            created_at="2026-09-20T12:00:00+00:00",
            summary="Payment receipt with noisy text",
            dates=[{"raw_text": "6\nZIMvNZ\n2573", "text": "6\nZIMvNZ\n2573", "normalized": "2573-01-01"}],
        ),
        # ss_5: Event date in 2025 (October 2025)
        ScreenshotMetadata(
            screenshot_id="t_ss_005",
            filename="past_conference.png",
            image_uri="data/screenshots/past_conference.png",
            category="Education",
            intent="General Information",
            content_type="course_notes",
            importance=2,
            importance_reason="Notes",
            created_at="2026-10-01T00:00:00+00:00",
            summary="Conference schedule notes",
            dates=[{"raw_text": "2025-10-27", "normalized": "2025-10-27"}],
        ),
    ]


def test_event_date_preference_over_creation_date(sample_screenshots):
    """Verifies that reliable event date (April 2026) is chosen over created_at (November 2026)."""
    service = TimelineService()
    t_ss_001 = sample_screenshots[0]
    entry = service.build_entry(t_ss_001)

    assert entry["is_event_date"] is True
    assert entry["event_date"] == "2026-04-15"
    assert entry["timeline_date"] == "2026-04-15"
    assert entry["year"] == 2026
    assert entry["month"] == 4
    assert "Apr 15, 2026" in entry["display_date"]


def test_creation_date_fallback(sample_screenshots):
    """Verifies fallback to created_at when dates list is empty."""
    service = TimelineService()
    t_ss_002 = sample_screenshots[1]
    entry = service.build_entry(t_ss_002)

    assert entry["is_event_date"] is False
    assert entry["event_date"] is None
    assert entry["year"] == 2026
    assert entry["month"] == 11
    assert entry["timeline_date"] == "2026-11-10"


def test_invalid_date_graceful_handling(sample_screenshots):
    """Verifies that an implausible or noisy date (> 2040) is safely ignored, falling back to created_at."""
    service = TimelineService()
    t_ss_004 = sample_screenshots[3]
    entry = service.build_entry(t_ss_004)

    assert entry["is_event_date"] is False
    assert entry["year"] == 2026
    assert entry["month"] == 9
    assert entry["timeline_date"] == "2026-09-20"


def test_chronological_ordering_newest_first(sample_screenshots):
    """Verifies that timeline entries and month groups are ordered descending (newest first)."""
    service = TimelineService()
    res = service.build_timeline(screenshots=sample_screenshots, sort_order="desc")

    assert res["total_entries"] == 5
    assert len(res["groups"]) > 0

    # First group should be November 2026, last should be October 2025
    group_months = [g["month_name"] for g in res["groups"]]
    assert group_months[0] == "November 2026"
    assert group_months[-1] == "October 2025"

    # Verify order of groups: (2026, 11) > (2026, 9) > (2026, 4) > (2025, 10)
    expected_order = [
        (2026, 11),
        (2026, 9),
        (2026, 4),
        (2025, 10),
    ]
    actual_order = [(g["year"], g["month"]) for g in res["groups"]]
    assert actual_order == expected_order


def test_month_grouping(sample_screenshots):
    """Verifies screenshots in the same year-month are grouped together."""
    service = TimelineService()
    res = service.build_timeline(screenshots=sample_screenshots)

    # September 2026 should have 2 entries (t_ss_003 and t_ss_004)
    sep_group = next(g for g in res["groups"] if g["month_name"] == "September 2026")
    assert sep_group["count"] == 2
    entry_ids = [e["screenshot_id"] for e in sep_group["entries"]]
    assert "t_ss_003" in entry_ids
    assert "t_ss_004" in entry_ids


def test_empty_timeline():
    """Verifies empty list produces valid empty response structure without errors."""
    service = TimelineService()
    res = service.build_timeline(screenshots=[])

    assert res["total_entries"] == 0
    assert res["total_groups"] == 0
    assert res["groups"] == []
    assert res["available_years"] == []
    assert res["available_categories"] == []


def test_category_and_intent_filtering(sample_screenshots):
    """Verifies filtering by category, intent, year, and month."""
    service = TimelineService()

    # Filter category = "Travel"
    travel_res = service.build_timeline(screenshots=sample_screenshots, category="Travel")
    assert travel_res["total_entries"] == 1
    assert travel_res["groups"][0]["entries"][0]["category"] == "Travel"

    # Filter intent = "Shopping / Purchase"
    shop_res = service.build_timeline(screenshots=sample_screenshots, intent="Shopping")
    assert shop_res["total_entries"] == 1
    assert shop_res["groups"][0]["entries"][0]["screenshot_id"] == "t_ss_003"

    # Filter year = 2025
    y25_res = service.build_timeline(screenshots=sample_screenshots, year=2025)
    assert y25_res["total_entries"] == 1
    assert y25_res["groups"][0]["entries"][0]["screenshot_id"] == "t_ss_005"

    # Filter month = 4
    m4_res = service.build_timeline(screenshots=sample_screenshots, month=4)
    assert m4_res["total_entries"] == 1
    assert m4_res["groups"][0]["entries"][0]["screenshot_id"] == "t_ss_001"


def test_existing_screenshot_metadata_compatibility():
    """Verifies compatibility with real SQLite database records."""
    from src.storage.metadata_store import SQLiteMetadataStore

    store = SQLiteMetadataStore("data/metadata.sqlite")
    all_meta = store.list_all_metadata()
    assert len(all_meta) >= 864

    service = TimelineService()
    res = service.build_timeline(screenshots=all_meta)

    assert res["total_entries"] == len(all_meta)
    assert res["total_groups"] > 0
    assert len(res["available_years"]) > 0
    assert len(res["available_categories"]) == 9

    # Check that every entry has mandatory timeline keys
    sample_entry = res["groups"][0]["entries"][0]
    mandatory_keys = [
        "screenshot_id",
        "filename",
        "image_url",
        "thumbnail_url",
        "display_date",
        "timeline_date",
        "category",
        "intent",
        "content_type",
        "importance",
        "short_description",
    ]
    for key in mandatory_keys:
        assert key in sample_entry, f"Missing key {key} in timeline entry"


def test_api_timeline_endpoints():
    """Verifies both /api/timeline and /api/screenshots/timeline endpoints via TestClient."""
    client = TestClient(app)

    # 1. GET /api/timeline
    resp1 = client.get("/api/timeline")
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert "total_entries" in data1
    assert "total_groups" in data1
    assert "groups" in data1
    assert data1["total_entries"] >= 864

    # 2. GET /api/screenshots/timeline with category filter
    resp2 = client.get("/api/screenshots/timeline?category=Shopping")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["filters_applied"]["category"] == "Shopping"
    assert data2["total_entries"] > 0
    for g in data2["groups"]:
        for e in g["entries"]:
            assert e["category"] == "Shopping"

    # 3. GET /api/timeline with year filter
    resp3 = client.get("/api/timeline?year=2026")
    assert resp3.status_code == 200
    data3 = resp3.json()
    assert data3["filters_applied"]["year"] == 2026
    for g in data3["groups"]:
        assert g["year"] == 2026
