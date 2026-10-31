"""
Targeted Regression & Unit Tests for I.R.I.S. Chatbot API (POST /api/chat)
==========================================================================
Covers all 15 required deterministic test specifications:
1. Travel category listing contains only Travel screenshots.
2. Shopping category listing contains only Shopping screenshots.
3. Payment category listing contains only Payment screenshots.
4. Connectivity category listing contains only Connectivity screenshots.
5. General category listing contains only General screenshots.
6. Coding tags do not override canonical category filtering.
7. Category names in ordinary semantic queries do not automatically activate category-listing mode.
8. Visiting-card search retrieves the relevant visiting-card screenshot.
9. Visiting-card search does not accept a financial-card screenshot solely because of the shared word "card".
10. Amazon order-confirmation search prioritizes actual Amazon/order evidence.
11. Generic payment receipts are not automatically described as Amazon order confirmations.
12. A nonexistent-subject query does not return arbitrary weak matches as relevant evidence.
13. Empty results do not fabricate screenshot references.
14. Retrieval failures are handled gracefully.
15. Frontend evidence cards correspond exactly to the current backend response, without stale results.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.api.chat import (
    DEFAULT_GEMINI_MODEL,
    call_gemini,
    detect_category_intent,
    detect_conversational_intent,
    GREETING_RESPONSE,
    CAPABILITY_RESPONSE,
    UNCLEAR_RESPONSE,
    format_offline_category_response,
    format_offline_response,
    retrieve_category_screenshots,
    _build_grounded_prompt,
    _format_references,
)
from backend.app.api.search import _calculate_rerank_score, _extract_query_keywords
from src.models.metadata import ScreenshotMetadata


class TestCategoryIntentDetection(unittest.TestCase):
    """Unit tests for the high-precision category-intent detector."""

    def test_phrasing_variations(self):
        cases = [
            ("Show me my travel screenshots", "Travel"),
            ("Show me my Shopping screenshots", "Shopping"),
            ("Show me my Payment screenshots", "Payment"),
            ("Show me my Connectivity screenshots", "Connectivity"),
            ("Show me my General screenshots", "General"),
            ("Show/list my Shopping screenshots", "Shopping"),
            ("Show/list my Payment screenshots", "Payment"),
            ("Show/list my General screenshots", "General"),
            ("Show / list my Shopping screenshots", "Shopping"),
            ("Show my Shopping screenshots", "Shopping"),
            ("List my Shopping screenshots", "Shopping"),
            ("Show payment screenshots", "Payment"),
            ("List general screenshots", "General"),
            ("Payment screenshots", "Payment"),
            ("Shopping screenshots", "Shopping"),
            ("General screenshots", "General"),
            ("Connectivity screenshots", "Connectivity"),
            ("My Payment screenshots", "Payment"),
            ("Show me Payment", "Payment"),
            ("Show Payment", "Payment"),
            ("List Payment", "Payment"),
            ("View Shopping", "Shopping"),
            ("Browse Connectivity", "Connectivity"),
            ("screenshots in Travel", "Travel"),
            ("show screenshots for Education", "Education"),
            ("SHOW ME MY TRAVEL SCREENSHOTS", "Travel"),
            ("list my shopping screenshots", "Shopping"),
        ]
        for q, expected in cases:
            self.assertEqual(detect_category_intent(q), expected, f"Failed on phrasing: {q}")

    def test_ordinary_semantic_queries_do_not_trigger_category_intent(self):
        semantic_queries = [
            "Find my travel booking confirmation for Delhi",
            "Find the payment confirmation for my train booking",
            "Find my payment receipt for electricity bill",
            "Show me my Amazon shopping order",
            "Find my visiting card in Kanpur",
            "Find my visiting card",
            "Show my flight or train tickets",
            "Find my screenshot about an imaginary underwater city on Mars",
            "Find my Amazon order confirmation",
            "Find credit card or bank statements",
            "Find recharge or bill receipts",
            "extraterrestrial alien encounter",
        ]
        for q in semantic_queries:
            self.assertIsNone(detect_category_intent(q), f"Semantic query incorrectly matched category intent: {q}")


class TestDeterministicChatAndSearch(unittest.TestCase):
    """Deterministic unit tests covering all 15 required specifications."""

    def setUp(self):
        self.client = TestClient(app)

    # 1. Travel category listing contains only Travel screenshots
    def test_1_travel_category_listing_contains_only_travel(self):
        mock_meta = [
            ScreenshotMetadata(screenshot_id="ss_019", filename="ss_019.jpg", image_uri="d/ss_019.jpg", category="Travel", created_at="2026-10-10"),
            ScreenshotMetadata(screenshot_id="ss_016", filename="ss_016.jpg", image_uri="d/ss_016.jpg", category="Travel", created_at="2026-10-09"),
            ScreenshotMetadata(screenshot_id="ss_021", filename="ss_021.jpg", image_uri="d/ss_021.jpg", category="Communication", created_at="2026-10-08"),
        ]
        with patch("backend.app.api.chat.metadata_store.list_all_metadata", return_value=mock_meta):
            res = self.client.post("/api/chat", json={"message": "Show me my travel screenshots", "top_k": 5})
            self.assertEqual(res.status_code, 200)
            refs = res.json()["referenced_screenshots"]
            self.assertEqual(len(refs), 2)
            for r in refs:
                self.assertEqual(r["category"], "Travel")
                self.assertNotEqual(r["screenshot_id"], "ss_021")

    # 2. Shopping category listing contains only Shopping screenshots
    def test_2_shopping_category_listing_contains_only_shopping(self):
        mock_meta = [
            ScreenshotMetadata(screenshot_id="ss_022", filename="ss_022.jpg", image_uri="d/ss_022.jpg", category="Shopping", created_at="2026-10-10"),
            ScreenshotMetadata(screenshot_id="ss_051", filename="ss_051.jpg", image_uri="d/ss_051.jpg", category="Education", tags=["Coding", "Shopping"], created_at="2026-10-09"),
        ]
        with patch("backend.app.api.chat.metadata_store.list_all_metadata", return_value=mock_meta):
            res = self.client.post("/api/chat", json={"message": "Show/list my Shopping screenshots", "top_k": 5})
            self.assertEqual(res.status_code, 200)
            refs = res.json()["referenced_screenshots"]
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0]["screenshot_id"], "ss_022")
            self.assertEqual(refs[0]["category"], "Shopping")

    # 3. Payment category listing contains only Payment screenshots
    def test_3_payment_category_listing_contains_only_payment(self):
        mock_meta = [
            ScreenshotMetadata(screenshot_id="ss_017", filename="ss_017.jpg", image_uri="d/ss_017.jpg", category="Payment", created_at="2026-10-10"),
            ScreenshotMetadata(screenshot_id="ss_478", filename="ss_478.jpg", image_uri="d/ss_478.jpg", category="Work & Career", created_at="2026-10-09"),
        ]
        with patch("backend.app.api.chat.metadata_store.list_all_metadata", return_value=mock_meta):
            res = self.client.post("/api/chat", json={"message": "Show/list my Payment screenshots", "top_k": 5})
            self.assertEqual(res.status_code, 200)
            refs = res.json()["referenced_screenshots"]
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0]["screenshot_id"], "ss_017")
            self.assertEqual(refs[0]["category"], "Payment")

    # 4. Connectivity category listing contains only Connectivity screenshots
    def test_4_connectivity_category_listing_contains_only_connectivity(self):
        mock_meta = [
            ScreenshotMetadata(screenshot_id="ss_001", filename="ss_001.jpg", image_uri="d/ss_001.jpg", category="Connectivity", created_at="2026-10-10"),
            ScreenshotMetadata(screenshot_id="ss_010", filename="ss_010.jpg", image_uri="d/ss_010.jpg", category="Shopping", created_at="2026-10-09"),
        ]
        with patch("backend.app.api.chat.metadata_store.list_all_metadata", return_value=mock_meta):
            res = self.client.post("/api/chat", json={"message": "Show me my Connectivity screenshots", "top_k": 5})
            self.assertEqual(res.status_code, 200)
            refs = res.json()["referenced_screenshots"]
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0]["screenshot_id"], "ss_001")
            self.assertEqual(refs[0]["category"], "Connectivity")

    # 5. General category listing contains only General screenshots
    def test_5_general_category_listing_contains_only_general(self):
        mock_meta = [
            ScreenshotMetadata(screenshot_id="ss_006", filename="ss_006.jpg", image_uri="d/ss_006.jpg", category="General", created_at="2026-10-10"),
            ScreenshotMetadata(screenshot_id="ss_827", filename="ss_827.jpg", image_uri="d/ss_827.jpg", category="General", tags=["Coding"], created_at="2026-10-09"),
            ScreenshotMetadata(screenshot_id="ss_033", filename="ss_033.jpg", image_uri="d/ss_033.jpg", category="Work & Career", tags=["Coding"], created_at="2026-10-08"),
        ]
        with patch("backend.app.api.chat.metadata_store.list_all_metadata", return_value=mock_meta):
            res = self.client.post("/api/chat", json={"message": "Show/list my General screenshots", "top_k": 5})
            self.assertEqual(res.status_code, 200)
            refs = res.json()["referenced_screenshots"]
            self.assertEqual(len(refs), 2)
            for r in refs:
                self.assertEqual(r["category"], "General")

    def test_5b_general_category_excludes_synthetic_mock_items(self):
        mock_meta = [
            ScreenshotMetadata(screenshot_id="ss_006", filename="ss_006.jpg", image_uri="d/ss_006.jpg", category="General", created_at="2026-10-10"),
            ScreenshotMetadata(screenshot_id="11ef54d0f838", filename="mock.png", image_uri="d/mock.png", category="General", evidence={"text_snippet": "[MOCK OCR] No predefined text."}, created_at="2026-10-09"),
        ]
        with patch("backend.app.api.chat.metadata_store.list_all_metadata", return_value=mock_meta):
            res = self.client.post("/api/chat", json={"message": "General screenshots", "top_k": 5})
            self.assertEqual(res.status_code, 200)
            refs = res.json()["referenced_screenshots"]
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0]["screenshot_id"], "ss_006")

    # 6. Coding tags do not override canonical category filtering
    def test_6_coding_tags_do_not_override_canonical_category_filtering(self):
        mock_meta = [
            ScreenshotMetadata(screenshot_id="ss_051", filename="ss_051.jpg", image_uri="d/ss_051.jpg", category="Education", tags=["Coding", "Shopping"], created_at="2026-10-10"),
            ScreenshotMetadata(screenshot_id="ss_362", filename="ss_362.jpg", image_uri="d/ss_362.jpg", category="Shopping", tags=["Coding", "Shopping"], created_at="2026-10-09"),
        ]
        with patch("backend.app.api.chat.metadata_store.list_all_metadata", return_value=mock_meta):
            res = self.client.post("/api/chat", json={"message": "Shopping screenshots", "top_k": 5})
            self.assertEqual(res.status_code, 200)
            refs = res.json()["referenced_screenshots"]
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0]["screenshot_id"], "ss_362")
            self.assertEqual(refs[0]["category"], "Shopping")

    # 7. Category names in ordinary semantic queries do not automatically activate category-listing mode
    def test_7_category_names_in_ordinary_semantic_queries_route_to_semantic(self):
        with patch("backend.app.api.chat._pipeline_search", return_value=[]) as mock_search:
            res = self.client.post("/api/chat", json={"message": "Find my travel booking confirmation for Delhi", "top_k": 3})
            self.assertEqual(res.status_code, 200)
            mock_search.assert_called_once()

    # 8. Visiting-card search retrieves the relevant visiting-card screenshot
    def test_8_visiting_card_search_retrieves_ss478(self):
        meta_visiting = ScreenshotMetadata(screenshot_id="ss_478", filename="ss_478.jpg", image_uri="d/ss_478.jpg", category="Work & Career", tags=[], summary="Work profile")
        score = _calculate_rerank_score(
            query_text="Find my visiting card",
            base_sim=0.25,
            doc_text="apnaCard English Edit Card Aahnik Dayama Graduate",
            meta_obj=meta_visiting,
        )
        self.assertGreaterEqual(score, 0.35, "Visiting card evidence should boost score above threshold")

    # 9. Visiting-card search does not accept a financial-card screenshot solely because of the word 'card'
    def test_9_visiting_card_search_penalizes_credit_cards(self):
        meta_financial = ScreenshotMetadata(screenshot_id="ss_010", filename="ss_010.jpg", image_uri="d/ss_010.jpg", category="Shopping", tags=["Credit Card"], summary="Flipkart order")
        score = _calculate_rerank_score(
            query_text="Find my visiting card in Kanpur",
            base_sim=0.45,
            doc_text="GET 1250 Vouchers With Flipkart SBI Credit Card",
            meta_obj=meta_financial,
        )
        self.assertLess(score, 0.35, "Financial credit card must be penalized and drop below threshold")

    # 10. Amazon order-confirmation search prioritizes actual Amazon/order evidence
    def test_10_amazon_order_confirmation_prioritizes_amazon_evidence(self):
        meta_amazon = ScreenshotMetadata(screenshot_id="ss_010", filename="ss_010.jpg", image_uri="d/ss_010.jpg", category="Shopping", tags=["Amazon", "Order"], summary="Amazon order page")
        score = _calculate_rerank_score(
            query_text="Find my Amazon order confirmation",
            base_sim=0.48,
            doc_text="Amazon order confirmation Delivered by Amazon Order #402-1234",
            meta_obj=meta_amazon,
        )
        self.assertGreaterEqual(score, 0.65, "Amazon order confirmation must be boosted prominently")

    # 10b. Amazon order search rejects competing brands (Flipkart, Instamart)
    def test_10b_amazon_order_rejects_competing_brands(self):
        meta_flipkart = ScreenshotMetadata(screenshot_id="ss_010", filename="ss_010.jpg", image_uri="d/ss_010.jpg", category="Shopping", tags=["Flipkart", "Order"], summary="Flipkart order")
        score_flipkart = _calculate_rerank_score(
            query_text="Find my Amazon order confirmation",
            base_sim=0.60,
            doc_text="My Orders With Flipkart SBI Credit Card Delivery expected by Aug 12",
            meta_obj=meta_flipkart,
        )
        self.assertLess(score_flipkart, 0.0, "Flipkart order must be rejected when Amazon was explicitly requested")

        meta_instamart = ScreenshotMetadata(screenshot_id="ss_015", filename="ss_015.jpg", image_uri="d/ss_015.jpg", category="Payment", tags=["Payment"], summary="Instamart order")
        score_instamart = _calculate_rerank_score(
            query_text="Find my Amazon order confirmation",
            base_sim=0.55,
            doc_text="Instamart order 05:30 PM 5 items cashback on UPI spends",
            meta_obj=meta_instamart,
        )
        self.assertLess(score_instamart, 0.0, "Instamart order must be rejected when Amazon was explicitly requested")

    # 10c. Generic order search preserves orders across brands
    def test_10c_generic_order_search_preserves_orders(self):
        meta_flipkart = ScreenshotMetadata(screenshot_id="ss_010", filename="ss_010.jpg", image_uri="d/ss_010.jpg", category="Shopping", tags=["Flipkart", "Order"], summary="Flipkart order")
        score_orders = _calculate_rerank_score(
            query_text="Find my orders",
            base_sim=0.75,
            doc_text="My Orders With Flipkart SBI Credit Card Delivery expected by Aug 12",
            meta_obj=meta_flipkart,
        )
        self.assertGreaterEqual(score_orders, 0.70, "Generic order search must retain valid order records")

    # 11. Generic payment receipts are not automatically described as Amazon order confirmations
    def test_11_generic_payment_receipt_penalized_for_amazon_query(self):
        meta_payment = ScreenshotMetadata(screenshot_id="ss_017", filename="ss_017.jpg", image_uri="d/ss_017.jpg", category="Payment", tags=["Payment", "Receipt"], summary="Bank payment receipt")
        score = _calculate_rerank_score(
            query_text="Find my Amazon order confirmation",
            base_sim=0.55,
            doc_text="KIIT INDIAN BANK PAYMENT GATEWAY Order ID: BILL-SAP2606297704460",
            meta_obj=meta_payment,
        )
        self.assertLess(score, 0.35, "Generic payment receipt without Amazon evidence must drop below threshold")

    # 12. A nonexistent-subject query does not return arbitrary weak matches as relevant evidence
    def test_12_nonexistent_subject_query_returns_no_weak_matches(self):
        meta_shopping = ScreenshotMetadata(screenshot_id="ss_190", filename="ss_190.jpg", image_uri="d/ss_190.jpg", category="Shopping", tags=[], summary="Delivery location")
        score = _calculate_rerank_score(
            query_text="Find my screenshot about an imaginary underwater city on Mars",
            base_sim=0.20,
            doc_text="Change delivery location Search for your city Use your current location",
            meta_obj=meta_shopping,
        )
        self.assertLess(score, 0.35, "Single token overlap in nonexistent query must drop below threshold")

    # 13. Empty results do not fabricate screenshot references
    def test_13_empty_results_do_not_fabricate_references(self):
        with patch("backend.app.api.chat._pipeline_search", return_value=[]):
            res = self.client.post("/api/chat", json={"message": "Find my screenshot about an imaginary underwater city on Mars", "top_k": 3})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["referenced_screenshots"], [])
            self.assertNotIn("ss_", data["reply"])
            self.assertIn("did not find", data["reply"].lower())

    # 14. Retrieval failures are handled gracefully
    def test_14_retrieval_failures_handled_gracefully(self):
        with patch("backend.app.api.chat._pipeline_search", side_effect=RuntimeError("Index lock error")):
            res = self.client.post("/api/chat", json={"message": "any semantic query"})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["referenced_screenshots"], [])
            self.assertIn("retrieval error", data["reply"].lower())

    # 15. Frontend evidence cards correspond exactly to current backend response
    def test_15_frontend_evidence_cards_correspond_exactly_without_stale_results(self):
        mock_results = [
            {
                "screenshot_id": "ss_478",
                "filename": "ss_478.jpg",
                "category": "Work & Career",
                "tags": ["visiting card"],
                "similarity_score": 0.617,
                "text": "apnaCard English Edit Card",
                "image_url": "/data/screenshots/ss_478.jpg",
            }
        ]
        with patch("backend.app.api.chat._pipeline_search", return_value=mock_results):
            res = self.client.post("/api/chat", json={"message": "Find my visiting card in Kanpur"})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            refs = data["referenced_screenshots"]
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0]["screenshot_id"], "ss_478")
            self.assertEqual(refs[0]["category"], "Work & Career")
            self.assertEqual(refs[0]["similarity_score"], 0.617)
            self.assertIn("ss_478", data["reply"])

    # 16. Context follow-up for details uses previously referenced screenshot
    def test_16_followup_pronoun_details_uses_previous_result(self):
        mock_meta = ScreenshotMetadata(
            screenshot_id="ss_478",
            filename="ss_478.jpg",
            image_uri="data/screenshots/ss_478.jpg",
            category="Work & Career",
            tags=["Visiting Card"],
            evidence={"text_snippet": "apnaCard English Edit Card Aahnik to Dayama"},
        )
        history = [
            {"role": "user", "content": "Find my visiting card in Kanpur"},
            {"role": "assistant", "content": "Found 1 screenshot", "referenced_ids": ["ss_478"]},
        ]
        with patch("backend.app.api.chat.metadata_store.get_metadata", return_value=mock_meta):
            res = self.client.post("/api/chat", json={"message": "Show me its details.", "history": history})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            refs = data["referenced_screenshots"]
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0]["screenshot_id"], "ss_478")
            self.assertIn("ss_478", data["reply"])
            self.assertIn("Work & Career", data["reply"])

    # 17. Context follow-up ordinal selection resolves the requested index
    def test_17_followup_ordinal_second_one_resolves_second_result(self):
        mock_meta = ScreenshotMetadata(
            screenshot_id="ss_014",
            filename="ss_0014.jpg",
            image_uri="data/screenshots/ss_0014.jpg",
            category="Travel",
            tags=["Ticket"],
            evidence={"text_snippet": "Boarding Pass 6Eskai GUWAHATI to BHUBANESWAR"},
        )
        history = [
            {"role": "user", "content": "Find my boarding pass"},
            {"role": "assistant", "content": "Found boarding passes", "referenced_ids": ["ss_011", "ss_014", "ss_019"]},
        ]
        with patch("backend.app.api.chat.metadata_store.get_metadata", return_value=mock_meta):
            res = self.client.post("/api/chat", json={"message": "Show me the second one.", "history": history})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            refs = data["referenced_screenshots"]
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0]["screenshot_id"], "ss_014")
            self.assertIn("ss_014", data["reply"])
            self.assertIn("Travel", data["reply"])

    # 18. Topic changes within same conversation execute fresh retrieval without leaking prior references
    def test_18_topic_change_triggers_fresh_search_without_leakage(self):
        history = [
            {"role": "user", "content": "Find my visiting card in Kanpur"},
            {"role": "assistant", "content": "Found visiting card", "referenced_ids": ["ss_478"]},
        ]
        with patch("backend.app.api.chat._pipeline_search", return_value=[]):
            res = self.client.post("/api/chat", json={"message": "Find my Amazon order confirmation", "history": history})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["referenced_screenshots"], [])
            self.assertNotIn("ss_478", data["reply"])
            self.assertIn("did not find", data["reply"].lower())

    # 19. Non-order brand mentions (like Amazon Music) are penalized for order confirmation queries
    def test_19_amazon_order_penalizes_music_service_screens(self):
        meta_music = ScreenshotMetadata(
            screenshot_id="ss_194",
            filename="ss_194.png",
            image_uri="d/ss_194.png",
            category="Entertainment",
            tags=["Amazon", "Entertainment", "Spotify"],
            summary="Import your existing music to Spotify Transfer from Apple Music, YouTube, Amazon Music",
        )
        score = _calculate_rerank_score(
            query_text="Find my Amazon order confirmation",
            base_sim=0.18,
            doc_text="Transfer from Apple Music, YouTube, Amazon Music",
            meta_obj=meta_music,
        )
        self.assertLess(score, 0.35, "Screenshots mentioning Amazon Music without order evidence must drop below threshold")

    # 20. "hello" receives a greeting and does not invoke retrieval
    def test_20_greeting_does_not_invoke_retrieval(self):
        with patch("backend.app.api.chat._pipeline_search") as mock_pipeline, \
             patch("backend.app.api.chat.retrieve_category_screenshots") as mock_cat:
            for greeting in ["hello", "hi", "hey", "good morning", "how are you"]:
                res = self.client.post("/api/chat", json={"message": greeting})
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["referenced_screenshots"], [])
                self.assertIn("I.R.I.S.", data["reply"])
                mock_pipeline.assert_not_called()
                mock_cat.assert_not_called()

    # 21. "what can you do" receives a capability response and does not invoke retrieval
    def test_21_capability_does_not_invoke_retrieval(self):
        with patch("backend.app.api.chat._pipeline_search") as mock_pipeline, \
             patch("backend.app.api.chat.retrieve_category_screenshots") as mock_cat:
            for cap in ["what can you do", "help", "how can you help me", "who are you"]:
                res = self.client.post("/api/chat", json={"message": cap})
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["referenced_screenshots"], [])
                self.assertIn("search", data["reply"].lower())
                mock_pipeline.assert_not_called()
                mock_cat.assert_not_called()

    # 22. "Show me my shopping screenshots" still invokes category retrieval
    def test_22_shopping_category_query_invokes_category_retrieval(self):
        with patch("backend.app.api.chat.retrieve_category_screenshots", return_value=[]) as mock_cat, \
             patch("backend.app.api.chat._pipeline_search") as mock_pipeline:
            res = self.client.post("/api/chat", json={"message": "Show me my shopping screenshots"})
            self.assertEqual(res.status_code, 200)
            mock_cat.assert_called_once()
            self.assertEqual(mock_cat.call_args[1]["target_category"], "Shopping")
            mock_pipeline.assert_not_called()

    # 23. A genuine search for a visiting card still invokes semantic retrieval
    def test_23_genuine_visiting_card_search_invokes_pipeline_search(self):
        with patch("backend.app.api.chat._pipeline_search", return_value=[]) as mock_pipeline, \
             patch("backend.app.api.chat.retrieve_category_screenshots") as mock_cat:
            res = self.client.post("/api/chat", json={"message": "Find my visiting card in Kanpur"})
            self.assertEqual(res.status_code, 200)
            mock_pipeline.assert_called_once()
            self.assertEqual(mock_pipeline.call_args[1]["query_text"], "Find my visiting card in Kanpur")
            mock_cat.assert_not_called()

    # 24. An unrecognized message does not automatically trigger a broad screenshot search
    def test_24_unrecognized_message_does_not_invoke_retrieval(self):
        with patch("backend.app.api.chat._pipeline_search") as mock_pipeline, \
             patch("backend.app.api.chat.retrieve_category_screenshots") as mock_cat:
            for unclear in ["???", "...", "ok", "asdf", "test"]:
                res = self.client.post("/api/chat", json={"message": unclear})
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["referenced_screenshots"], [])
                self.assertIn("not sure what you'd like to find", data["reply"].lower())
                mock_pipeline.assert_not_called()
                mock_cat.assert_not_called()

    # 25. Pre-checkout button on Travel screens is demoted for generic payment queries
    def test_25_travel_precheckout_button_demoted_for_payment_query(self):
        meta_travel = ScreenshotMetadata(
            screenshot_id="ss_0016",
            filename="ss_0016.jpg",
            image_uri="d/ss_0016.jpg",
            category="Travel",
            tags=["Ticket"],
            summary="Train ticket review",
            evidence={"text_snippet": "Review 15611 Rny Scl Express Your Boarding Details Proceed to payment"}
        )
        score = _calculate_rerank_score(
            query_text="List my Payment screenshots",
            base_sim=0.4051,
            doc_text="Proceed to payment",
            meta_obj=meta_travel,
        )
        self.assertLess(score, 0.35, "Pre-checkout button on travel screen must drop below threshold for payment query")

    # 26. Pre-checkout button on Travel screens is NOT demoted when travel intent is present
    def test_26_travel_precheckout_button_not_demoted_for_travel_query(self):
        meta_travel = ScreenshotMetadata(
            screenshot_id="ss_0016",
            filename="ss_0016.jpg",
            image_uri="d/ss_0016.jpg",
            category="Travel",
            tags=["Ticket"],
            summary="Train ticket review",
            evidence={"text_snippet": "Review 15611 Rny Scl Express Your Boarding Details Proceed to payment"}
        )
        score = _calculate_rerank_score(
            query_text="Find my train ticket",
            base_sim=0.45,
            doc_text="Review 15611 Rny Scl Express Your Boarding Details",
            meta_obj=meta_travel,
        )
        self.assertGreaterEqual(score, 0.35, "Travel query must not demote travel ticket screenshots")

    # 27. Credible completed payment confirmations are not demoted
    def test_27_completed_payment_not_demoted(self):
        meta_payment = ScreenshotMetadata(
            screenshot_id="ss_0013",
            filename="ss_0013.jpg",
            image_uri="d/ss_0013.jpg",
            category="Payment",
            tags=["Payment", "Receipt"],
            summary="Payment receipt",
            evidence={"text_snippet": "KIITConnect Payment Successful Ref No 12345"}
        )
        score = _calculate_rerank_score(
            query_text="Find my payment receipt",
            base_sim=0.55,
            doc_text="KIITConnect Payment Successful",
            meta_obj=meta_payment,
        )
        self.assertGreaterEqual(score, 0.50, "Genuine completed payment confirmation must remain highly scored")


if __name__ == "__main__":
    unittest.main()
