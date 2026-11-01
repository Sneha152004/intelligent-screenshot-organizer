"""
Phase 1 Intelligence Test Suite
================================
Validates:
1. Shopping intent detection & extraction
2. Payment / financial intent detection & extraction
3. Event / appointment / travel intent detection & extraction
4. Coding / technical work intent detection & extraction
5. Empty / uninformative OCR text handling
6. Dates, contacts, URLs extraction & normalization
7. Missing fields & malformed input safety
8. Sensitive info & password redaction safety (no cleartext passwords)
9. OCRPipeline metadata creation & SQLite persistence round-trip
10. Category taxonomy preservation (9 canonical categories untouched)
"""

import os
import tempfile
from pathlib import Path
import pytest

from src.extractor import StructuredExtractor
from src.intent_detector import ScreenshotIntentDetector
from src.models.metadata import ScreenshotMetadata, SUPPORTED_CATEGORIES
from src.pipeline import OCRPipeline
from src.mock_ocr import MockOCREngine
from src.storage.file_store import LocalFileStore
from src.storage.metadata_store import SQLiteMetadataStore


@pytest.fixture
def extractor():
    return StructuredExtractor()


@pytest.fixture
def intent_detector():
    return ScreenshotIntentDetector()


# ── Scenario 1: Shopping Intent Detection & Extraction ────────────────────────

def test_shopping_intent_and_extraction(extractor, intent_detector):
    ocr_text = """
    Amazon.in - Order Confirmation
    Order # 402-8392018-9182391
    Placed on 12 Oct 2026
    Items: Sony WH-1000XM5 Wireless Headphones
    Order Total: ₹26,990.00
    Delivery Status: Arriving Wednesday
    Shipping to: Bangalore, Karnataka
    """
    extracted = extractor.extract_all(ocr_text, category="Shopping")

    assert extracted["content_type"] in ["shopping_order", "product_page", "delivery_tracking"]
    assert extracted["financial_info"]["amount"] == 26990.0
    assert extracted["financial_info"]["currency"] in ["INR", "₹"]

    # Check entities
    order_id_entities = [e for e in extracted["entities"] if e.get("type") == "ORDER_ID" or e.get("label") == "ORDER_ID"]
    assert len(order_id_entities) > 0
    assert any("402-8392018-9182391" in e["value"] for e in order_id_entities)

    # Detect intent
    intent, conf, evidence = intent_detector.detect_intent(
        ocr_text=ocr_text,
        category="Shopping",
        tags=["amazon", "order", "headphones"],
        content_type=extracted["content_type"],
        financial_info=extracted["financial_info"],
        dates=extracted["dates"],
        entities=extracted["entities"],
        sensitive_info=extracted["sensitive_info"],
    )
    assert intent == "Shopping / Purchase"
    assert conf >= 0.70
    assert "signal" in evidence or "merchant" in evidence or "order" in str(evidence)


# ── Scenario 2: Payment / Financial Intent & Extraction ───────────────────────

def test_payment_intent_and_extraction(extractor, intent_detector):
    ocr_text = """
    Google Pay
    Payment Successful!
    Paid to Ramesh Kumar
    UPI ID: ramesh.kumar@oksbi
    UPI Transaction ID: 429184719284
    Amount: ₹750.50
    Date: 24 Oct 2026, 08:30 PM
    Debited from: State Bank of India XX1234
    """
    extracted = extractor.extract_all(ocr_text, category="Payment")

    assert extracted["content_type"] in ["payment_receipt", "transaction"]
    assert extracted["financial_info"]["amount"] == 750.50
    assert extracted["financial_info"]["upi_id"] == "ramesh.kumar@oksbi"
    assert extracted["financial_info"]["transaction_id"] == "429184719284"
    assert extracted["financial_info"]["status"] == "success"

    intent, conf, evidence = intent_detector.detect_intent(
        ocr_text=ocr_text,
        category="Payment",
        tags=["payment", "upi", "google pay"],
        content_type=extracted["content_type"],
        financial_info=extracted["financial_info"],
        dates=extracted["dates"],
        entities=extracted["entities"],
        sensitive_info=extracted["sensitive_info"],
    )
    assert intent == "Payment / Transaction / Bill"
    assert conf >= 0.75


# ── Scenario 3: Event / Appointment / Travel Intent ───────────────────────────

def test_event_and_travel_intent(extractor, intent_detector):
    ocr_text = """
    IndiGo Airlines - Boarding Pass & Itinerary
    Flight: 6E-204 from DEL to BLR
    Departure: 18 Nov 2026 at 06:45 AM
    Gate: 12B | Seat: 14F
    Passenger: John Doe
    PNR: W4K9LP
    """
    extracted = extractor.extract_all(ocr_text, category="Travel")

    assert extracted["content_type"] in ["flight_ticket", "travel_itinerary"]
    assert len(extracted["dates"]) > 0
    # ISO date normalized
    assert any("2026-11-18" in d.get("normalized", "") for d in extracted["dates"])

    intent, conf, evidence = intent_detector.detect_intent(
        ocr_text=ocr_text,
        category="Travel",
        tags=["flight", "travel"],
        content_type=extracted["content_type"],
        financial_info=extracted["financial_info"],
        dates=extracted["dates"],
        entities=extracted["entities"],
        sensitive_info=extracted["sensitive_info"],
    )
    assert intent in ["Travel / Booking", "Event / Appointment"]
    assert conf >= 0.70


# ── Scenario 4: Coding / Technical Work Intent ────────────────────────────────

def test_coding_technical_intent(extractor, intent_detector):
    ocr_text = """
    Traceback (most recent call last):
      File "backend/server.py", line 124, in handle_request
        response = database.query(user_id)
      File "backend/database.py", line 56, in query
        raise ConnectionRefusedError("Could not connect to Redis server on port 6379")
    ConnectionRefusedError: Could not connect to Redis server on port 6379
    git commit -m "fix redis connection pool"
    """
    extracted = extractor.extract_all(ocr_text, category="Work & Career")

    assert extracted["content_type"] in ["error_log", "code_editor", "terminal"]
    assert "error" in extracted["text_topics"] or "coding" in extracted["text_topics"] or "terminal" in extracted["text_topics"]

    intent, conf, evidence = intent_detector.detect_intent(
        ocr_text=ocr_text,
        category="Work & Career",
        tags=["python", "error", "traceback"],
        content_type=extracted["content_type"],
        financial_info=extracted["financial_info"],
        dates=extracted["dates"],
        entities=extracted["entities"],
        sensitive_info=extracted["sensitive_info"],
    )
    assert intent == "Coding / Technical Work"
    assert conf >= 0.75


# ── Scenario 5: Empty / Uninformative Text Handling ───────────────────────────

def test_empty_or_uninformative_ocr(extractor, intent_detector):
    # Empty string
    extracted_empty = extractor.extract_all("", category="General")
    assert extracted_empty["content_type"] == "unknown"
    assert extracted_empty["financial_info"] == {}
    assert extracted_empty["entities"] == []

    intent_empty, conf_empty, _ = intent_detector.detect_intent(
        ocr_text="",
        category="General",
        tags=[],
    )
    assert intent_empty is None
    assert conf_empty == 0.0

    # Very short uninformative text (< 2 words)
    intent_short, conf_short, _ = intent_detector.detect_intent(
        ocr_text="ok",
        category="General",
        tags=[],
    )
    assert intent_short is None
    assert conf_short == 0.0


# ── Scenario 6: Dates, Contacts, URLs Extraction & Normalization ──────────────

def test_dates_contacts_urls_extraction(extractor):
    ocr_text = """
    Please contact our support team at support@iris-assistant.ai or sales.desk@company.org.
    Helpline: +91-9876543210 or 1800-123-4567.
    Documentation available at https://iris-organizer.dev/docs/getting-started.
    The meeting is scheduled on 2026-11-20 at 14:00.
    """
    extracted = extractor.extract_all(ocr_text)

    # Emails
    emails = extracted["contacts"].get("emails", [])
    assert "support@iris-assistant.ai" in emails
    assert "sales.desk@company.org" in emails

    # Phones
    phones = extracted["contacts"].get("phones", [])
    assert any("9876543210" in p for p in phones)

    # URLs
    urls = extracted["urls"]
    assert any("https://iris-organizer.dev" in u for u in urls)

    # Dates
    dates = extracted["dates"]
    assert any("2026-11-20" in d.get("normalized", "") for d in dates)


# ── Scenario 7: Missing Fields & Malformed Input Safety ───────────────────────

def test_missing_fields_and_malformed_input(extractor, intent_detector):
    # Malformed / None inputs should not crash
    assert extractor.extract_all(None) is not None
    assert extractor.extract_financial_info(None) == {}
    assert extractor.extract_dates(None) == []
    assert extractor.extract_contacts(None) == {"emails": [], "phones": []}
    assert extractor.extract_urls(None) == []
    assert extractor.detect_sensitive_info(None) == {
        "contains_password": False,
        "contains_payment_info": False,
        "contains_personal_contact": False,
    }

    intent, conf, ev = intent_detector.detect_intent(
        ocr_text=None,
        category=None,
        tags=None,
        content_type=None,
        financial_info=None,
        dates=None,
        entities=None,
        sensitive_info=None,
    )
    assert intent is None
    assert conf == 0.0
    assert isinstance(ev, dict)


# ── Scenario 8: Sensitive Info & Password Redaction Safety ───────────────────

def test_sensitive_info_and_password_redaction_safety(extractor, intent_detector):
    ocr_text = """
    Wi-Fi Configuration:
    SSID: Office_HighSpeed_5G
    Password: MySecretPassword123!
    Card Details:
    Card Number: 4532 8912 3456 7890
    CVV: 789
    Expiry: 08/29
    """
    extracted = extractor.extract_all(ocr_text, category="Connectivity")

    # Sensitive flags must be raised
    assert extracted["sensitive_info"]["contains_password"] is True
    assert extracted["sensitive_info"]["contains_payment_info"] is True

    # CRITICAL: Cleartext password MUST NOT be stored in entities or any field
    raw_secret = "MySecretPassword123!"
    for entity in extracted["entities"]:
        assert raw_secret not in entity.get("value", ""), f"Cleartext password leaked in entity value: {entity}"
        assert raw_secret not in entity.get("text", ""), f"Cleartext password leaked in entity text: {entity}"
        if entity.get("label") == "PASSWORD" or entity.get("type") == "PASSWORD":
            assert entity.get("value") == "[REDACTED_PASSWORD]"
            assert entity.get("text") == "[REDACTED_PASSWORD]"

    # Intent detection should classify as Credential / Network Access
    intent, conf, _ = intent_detector.detect_intent(
        ocr_text=ocr_text,
        category="Connectivity",
        tags=["wifi", "password"],
        sensitive_info=extracted["sensitive_info"],
    )
    assert intent == "Credential / Network Access"


# ── Scenario 9: OCRPipeline Metadata Creation & SQLite Persistence ────────────

def test_pipeline_metadata_creation_and_sqlite_persistence(tmp_path):
    # Setup temporary isolated SQLite db and file store
    db_file = tmp_path / "test_metadata.sqlite"
    storage_dir = tmp_path / "screenshots"
    storage_dir.mkdir()

    dummy_img = tmp_path / "receipt_001.png"
    dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")

    receipt_ocr = (
        "Zomato Order Received! Restaurant: Biryani Blues. "
        "Total Paid: Rs 540.00 via UPI ID user@okaxis. Transaction ID: 98127391823. "
        "Estimated delivery by 8:45 PM on 15 Nov 2026."
    )
    mock_ocr = MockOCREngine(mapping={"receipt_001.png": receipt_ocr})

    file_store = LocalFileStore(storage_dir=str(storage_dir))
    metadata_store = SQLiteMetadataStore(db_path=str(db_file))

    pipeline = OCRPipeline(
        ocr_engine=mock_ocr,
        file_store=file_store,
        metadata_store=metadata_store,
        enable_visual=False,
    )

    doc = {
        "image_path": str(dummy_img),
        "metadata": {
            "Screenshot ID": "test_shot_123",
            "Actual Category": "Payment",
            "notes": "Dinner order receipt",
            "filename": "receipt_001.png",
        }
    }

    results = pipeline.process_documents([doc])
    assert len(results) == 1

    # Verify metadata created in memory
    res_meta = results[0]["metadata"]
    assert res_meta["screenshot_id"] == "test_shot_123"
    assert res_meta["category"] == "Payment"
    assert res_meta["intent"] in ["Payment / Transaction / Bill", "Shopping / Purchase"]
    assert res_meta["financial_info"]["amount"] == 540.0
    assert res_meta["financial_info"]["upi_id"] == "user@okaxis"
    assert res_meta["ocr_available"] is True

    # Verify persistence to SQLite
    loaded_meta = metadata_store.get_metadata("test_shot_123")
    assert loaded_meta is not None
    assert loaded_meta.screenshot_id == "test_shot_123"
    assert loaded_meta.category == "Payment"
    assert loaded_meta.intent == res_meta["intent"]
    assert loaded_meta.financial_info["amount"] == 540.0
    assert loaded_meta.financial_info["upi_id"] == "user@okaxis"
    assert loaded_meta.financial_info["transaction_id"] == "98127391823"
    assert isinstance(loaded_meta.dates, list)
    assert isinstance(loaded_meta.entities, list)
    assert isinstance(loaded_meta.evidence, dict)


# ── Scenario 10: Category Taxonomy Preservation ──────────────────────────────

def test_category_taxonomy_preservation():
    # Verify the canonical 9 categories remain strictly intact
    expected_categories = [
        "Education",
        "Payment",
        "Shopping",
        "Travel",
        "Communication",
        "Entertainment",
        "Work & Career",
        "Connectivity",
        "General",
    ]
    assert sorted(SUPPORTED_CATEGORIES) == sorted(expected_categories)
    assert len(SUPPORTED_CATEGORIES) == 9
