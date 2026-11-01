"""
Unit and Integration Tests for Importance Scorer (Feature 1)
============================================================
Tests:
- Critical credentials (Score 5)
- Flight tickets (Score 5)
- Payment proofs (Score 5)
- Upcoming deadlines (Score 5)
- Actionable appointments and bills (Score 4)
- Educational references (Score 3)
- Shopping and work references (Score 3)
- General screenshots (Score 2)
- Invalid or empty OCR (Score 1)
- Score range validation (1-5 integer)
- Database persistence in SQLiteMetadataStore
- Existing metadata compatibility
"""

import os
import tempfile
import unittest

from src.importance_scorer import ImportanceScorer
from src.models.metadata import ScreenshotMetadata
from src.storage.metadata_store import SQLiteMetadataStore


class TestImportanceScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = ImportanceScorer()

    def test_critical_credentials(self):
        """Test 1: Critical security credentials and access keys score 5."""
        cases = [
            ("Your login password is: SecretP@ssw0rd123", {}),
            ("AWS API_KEY = akia_test_1234567890abcdef", {}),
            ("Your one-time password OTP is 482910 for HDFC Bank", {}),
            ("Login screen", {"has_password": True}),
        ]
        for text, sens in cases:
            score, reason = self.scorer.score_screenshot(ocr_text=text, sensitive_info=sens)
            self.assertEqual(score, 5, f"Failed for text: {text}")
            self.assertIn("credential", reason.lower())

    def test_flight_tickets(self):
        """Test 2: Flight tickets and boarding passes score 5."""
        cases = [
            ("IndiGo Boarding Pass - Seat 14B - Gate 4 - Flight 6E 204 - GAU to BBI", "Travel", "flight_ticket"),
            ("Air India E-Ticket Confirmation - PNR: AB12CD - Departure: 12-Nov-2026", "Travel", "Travel / Navigation"),
            ("Boarding Pass Airline: Emirates Seat: 22A Terminal 3 Gate B14", "Travel", "unknown"),
        ]
        for text, cat, ctype in cases:
            score, reason = self.scorer.score_screenshot(
                ocr_text=text,
                category=cat,
                content_type=ctype,
            )
            self.assertEqual(score, 5, f"Failed for text: {text}")
            self.assertIn("flight", reason.lower())

    def test_payment_proofs(self):
        """Test 3: Completed payment proofs and verified transaction receipts score 5."""
        cases = [
            (
                "Payment Successful! Rs 1,499.00 paid to TechStore. UPI Ref No: 421098765432",
                "Payment",
                "payment_receipt",
                {"amount": 1499.0, "transaction_ids": ["TXN98765"], "upi_ids": ["tech@okhdfc"]},
            ),
            (
                "Paid successfully to Swiggy. Debited from A/c XX4021. UTR: 998877665544",
                "Payment",
                "payment_receipt",
                {"amounts": ["Rs 450"], "transaction_ids": ["998877665544"]},
            ),
        ]
        for text, cat, ctype, fin in cases:
            score, reason = self.scorer.score_screenshot(
                ocr_text=text,
                category=cat,
                content_type=ctype,
                financial_info=fin,
            )
            self.assertEqual(score, 5, f"Failed for text: {text}")
            self.assertIn("payment", reason.lower())

    def test_upcoming_deadlines(self):
        """Test 4: Strict deadlines and urgent expirations score 5."""
        cases = [
            ("Project Submission Deadline: Strictly before 15th November 2026, 11:59 PM", []),
            ("Payment Due Date: Today! Avoid late penalty after 5:00 PM", []),
            ("Assignment Portal", [{"action": "Submit report", "deadline": "submission deadline tomorrow"}]),
        ]
        for text, act in cases:
            score, reason = self.scorer.score_screenshot(
                ocr_text=text,
                action_items=act,
            )
            self.assertEqual(score, 5, f"Failed for text: {text}")
            self.assertIn("deadline", reason.lower())

    def test_actionable_appointments_and_bills(self):
        """Test 5: Appointments, bookings, and bills score 4."""
        cases = [
            ("Doctor Appointment Confirmed with Dr. Sharma for tomorrow at 10:30 AM", "Health", "appointment"),
            ("Hotel Booking Confirmation: Grand Palace Hotel, Check-in Date: 20-Nov-2026", "Travel", "hotel_booking"),
            ("Electricity Bill Statement - Total Due: Rs 2,340 - Pay by date: 25-Nov-2026", "Work & Career", "utility_bill"),
            ("To-Do List: 1. Finalize pitch deck 2. Review sprint board 3. Meeting agenda with team", "Work & Career", "unknown"),
        ]
        for text, cat, ctype in cases:
            score, reason = self.scorer.score_screenshot(
                ocr_text=text,
                category=cat,
                content_type=ctype,
            )
            self.assertEqual(score, 4, f"Failed for text: {text}")
            self.assertTrue(any(w in reason.lower() for w in ["actionable", "booking", "bill", "official", "task"]))

    def test_educational_references(self):
        """Test 6: Educational notes and study material score 3."""
        cases = [
            ("Lecture 4: Introduction to Convolutional Neural Networks, Theorem 2.1 Definition of Gradient Descent", "Education"),
            ("Python Data Structures: Understanding binary search trees and algorithm time complexity O(log n)", "Education"),
        ]
        for text, cat in cases:
            score, reason = self.scorer.score_screenshot(
                ocr_text=text,
                category=cat,
            )
            self.assertEqual(score, 3, f"Failed for text: {text}")
            self.assertIn("educational", reason.lower())

    def test_shopping_and_work_references(self):
        """Test 7: Shopping orders and work references score 3."""
        cases = [
            ("Your Amazon Order has shipped: Wireless Earbuds, Expected Delivery: Friday", "Shopping", "shopping_order"),
            ("def calculate_metrics(y_true, y_pred):\n    return f1_score(y_true, y_pred)", "Technical & Development", "code_snippet"),
        ]
        for text, cat, ctype in cases:
            score, reason = self.scorer.score_screenshot(
                ocr_text=text,
                category=cat,
                content_type=ctype,
            )
            self.assertEqual(score, 3, f"Failed for text: {text}")
            self.assertIn("reference", reason.lower())

    def test_general_screenshots(self):
        """Test 8: General chats, media, and connectivity score 2."""
        cases = [
            ("Hey bro, are we meeting up tonight for dinner? Let me know!", "Communication", "Social / Chat"),
            ("Now playing: Cyberpunk 2077 Night City theme song - Episode 4", "Entertainment", "Entertainment / Media"),
            ("Wi-Fi connected: 5GHz Network, Battery at 85%, Bluetooth active", "Connectivity", "General Info / None"),
        ]
        for text, cat, intent in cases:
            score, reason = self.scorer.score_screenshot(
                ocr_text=text,
                category=cat,
                intent=intent,
            )
            self.assertEqual(score, 2, f"Failed for text: {text}")
            self.assertIn("general", reason.lower())

    def test_invalid_or_empty_ocr(self):
        """Test 9: Invalid or minimal OCR scores 1."""
        cases = ["", "   ", "abc", "..."]
        for text in cases:
            score, reason = self.scorer.score_screenshot(ocr_text=text)
            self.assertEqual(score, 1)
            self.assertIn("low-value", reason.lower())

    def test_score_range_validation(self):
        """Test 10: Scores are strictly integers between 1 and 5."""
        samples = [
            "Random text with random words",
            "Critical password secret",
            "Flight ticket gate 5",
            "Doctor appointment",
            "Study notes chapter 5",
            "Hello there friend",
            "",
        ]
        for s in samples:
            score, reason = self.scorer.score_screenshot(ocr_text=s)
            self.assertIsInstance(score, int)
            self.assertNotIsInstance(score, bool)
            self.assertTrue(1 <= score <= 5, f"Score out of range: {score}")
            self.assertIsInstance(reason, str)
            self.assertTrue(len(reason) > 5)

    def test_database_persistence(self):
        """Test 11: Importance score and reason persist and reload correctly from SQLite."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "test_importance.sqlite")
            store = SQLiteMetadataStore(db_path=db_path)

            meta = ScreenshotMetadata(
                screenshot_id="imp_test_001",
                filename="imp_test_001.png",
                image_uri="/data/screenshots/imp_test_001.png",
                category="Travel",
                importance=5,
                importance_reason="Critical travel document: Flight ticket",
            )
            store.save_metadata(meta)

            loaded = store.get_metadata("imp_test_001")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.importance, 5)
            self.assertEqual(loaded.importance_score, 5)
            self.assertEqual(loaded.importance_reason, "Critical travel document: Flight ticket")

    def test_existing_metadata_compatibility(self):
        """Test 12: Existing metadata dictionaries without importance_score/reason load safely."""
        legacy_data = {
            "screenshot_id": "legacy_001",
            "filename": "legacy_001.png",
            "image_uri": "/data/legacy_001.png",
            "category": "Shopping",
        }
        meta = ScreenshotMetadata.from_dict(legacy_data)
        self.assertIsNone(meta.importance)
        self.assertIsNone(meta.importance_score)
        self.assertIsNone(meta.importance_reason)

        # Dictionary export includes keys safely
        d = meta.to_dict()
        self.assertIn("importance", d)
        self.assertIn("importance_score", d)
        self.assertIn("importance_reason", d)
        self.assertIsNone(d["importance"])


if __name__ == "__main__":
    unittest.main()
