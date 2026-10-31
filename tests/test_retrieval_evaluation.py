"""
I.R.I.S. Screenshot Retrieval & Quality Evaluation Test Suite
============================================================
Comprehensive, automated regression and evaluation suite covering all six
evaluation groups on the actual screenshot library, SQLite metadata, and ChromaDB:

Group A: Conversational Intent Detection (greetings, capabilities, unclear noise)
Group B: All Nine Canonical Categories (Education, Shopping, Payment, Travel,
         Communication, Entertainment, Work & Career, Connectivity, General)
Group C: Specific Screenshot Queries (visiting cards, brand orders, tickets)
Group D: Known Retrieval Regressions (Shopping in Payment, brand leaks, etc.)
Group E: Negative Queries (unsupported subjects, non-existent brands)
Group F: Multi-turn Follow-up Queries & Context Boundaries
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.api.chat import (
    detect_category_intent,
    detect_conversational_intent,
    retrieve_category_screenshots,
    _pipeline_search,
)
from backend.app.core.pipeline_instance import (
    get_embedder,
    get_vector_store,
    metadata_store,
    file_store,
)


class TestGroupAConversationalIntent(unittest.TestCase):
    """
    GROUP A: Verify that conversational pleasantries, capability inquiries,
    and low-information inputs receive conversational replies without invoking
    screenshot retrieval.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_a1_greetings_do_not_trigger_retrieval(self):
        """Greetings receive friendly greeting replies without invoking retrieval."""
        with patch("backend.app.api.chat._pipeline_search") as mock_search, \
             patch("backend.app.api.chat.retrieve_category_screenshots") as mock_cat:
            for greeting in ["hello", "good morning", "hi", "hey"]:
                res = self.client.post("/api/chat", json={"message": greeting})
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["referenced_screenshots"], [], f"Greeting '{greeting}' returned screenshot references.")
                self.assertIn("I.R.I.S.", data["reply"])
                mock_search.assert_not_called()
                mock_cat.assert_not_called()

    def test_a2_capabilities_do_not_trigger_retrieval(self):
        """Capability questions explain system features without invoking retrieval."""
        with patch("backend.app.api.chat._pipeline_search") as mock_search, \
             patch("backend.app.api.chat.retrieve_category_screenshots") as mock_cat:
            for cap in ["what can you do", "who are you", "help", "how can you help me"]:
                res = self.client.post("/api/chat", json={"message": cap})
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["referenced_screenshots"], [], f"Capability query '{cap}' returned screenshot references.")
                self.assertIn("search", data["reply"].lower())
                mock_search.assert_not_called()
                mock_cat.assert_not_called()

    def test_a3_low_information_unclear_queries_do_not_trigger_retrieval(self):
        """Unclear/noise messages return clarification prompts without invoking retrieval."""
        with patch("backend.app.api.chat._pipeline_search") as mock_search, \
             patch("backend.app.api.chat.retrieve_category_screenshots") as mock_cat:
            for noise in ["asdf", "???", "...", "test", "ok"]:
                res = self.client.post("/api/chat", json={"message": noise})
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["referenced_screenshots"], [], f"Noise query '{noise}' returned screenshot references.")
                self.assertIn("not sure what you'd like to find", data["reply"].lower())
                mock_search.assert_not_called()
                mock_cat.assert_not_called()


class TestGroupBCanonicalCategories(unittest.TestCase):
    """
    GROUP B: Verify category-listing intent detection and category retrieval
    across all nine canonical categories on the live SQLite database.
    Ensures zero cross-category leakage.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def _verify_category_retrieval(self, query: str, expected_category: str):
        # 1. Verify intent detection
        detected = detect_category_intent(query)
        self.assertEqual(detected, expected_category, f"Failed category intent on query: '{query}'")

        # 2. Run through live chat endpoint
        res = self.client.post("/api/chat", json={"message": query, "top_k": 3})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        refs = data.get("referenced_screenshots", [])

        self.assertGreaterEqual(len(refs), 1, f"No screenshots returned for category '{expected_category}'")

        # 3. Verify zero cross-category leakage in actual SQLite metadata
        for r in refs:
            actual_cat = r.get("category")
            self.assertEqual(
                actual_cat.lower(),
                expected_category.lower(),
                f"Cross-category leakage! Expected '{expected_category}', but got '{actual_cat}' for {r['screenshot_id']}",
            )

    def test_b1_education_category(self):
        self._verify_category_retrieval("Show me my education screenshots", "Education")

    def test_b2_shopping_category(self):
        self._verify_category_retrieval("Show me my shopping screenshots", "Shopping")

    def test_b3_payment_category(self):
        self._verify_category_retrieval("Find my payment screenshots", "Payment")

    def test_b4_travel_category(self):
        self._verify_category_retrieval("Show me my travel screenshots", "Travel")

    def test_b5_communication_category(self):
        self._verify_category_retrieval("Find my communication screenshots", "Communication")

    def test_b6_entertainment_category(self):
        self._verify_category_retrieval("Show me my entertainment screenshots", "Entertainment")

    def test_b7_work_and_career_category(self):
        self._verify_category_retrieval("Find my work and career screenshots", "Work & Career")

    def test_b8_connectivity_category(self):
        self._verify_category_retrieval("Show me my connectivity screenshots", "Connectivity")

    def test_b9_general_category(self):
        self._verify_category_retrieval("Show me my general screenshots", "General")


class TestGroupCSpecificQueries(unittest.TestCase):
    """
    GROUP C: Evaluate retrieval quality on specific semantic screenshot queries
    against the actual screenshot library, embeddings, and ChromaDB vector store.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Warm up embedder and vector store
        get_embedder()
        get_vector_store()

    def test_c1_visiting_card_kanpur_retrieves_ss478(self):
        """Query for visiting card in Kanpur retrieves apnaCard ss_478."""
        res = self.client.post("/api/chat", json={"message": "Find my visiting card in Kanpur", "top_k": 3})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        refs = data.get("referenced_screenshots", [])
        self.assertGreaterEqual(len(refs), 1)
        ref_ids = [r["screenshot_id"] for r in refs]
        self.assertIn("ss_478", ref_ids, "Expected ss_478 (apnaCard Kanpur) in visiting card search results.")
        self.assertEqual(refs[0]["screenshot_id"], "ss_478")
        self.assertEqual(refs[0]["category"], "Work & Career")

    def test_c2_visiting_card_general_retrieves_ss478(self):
        """Query for visiting card without city still retrieves ss_478."""
        res = self.client.post("/api/chat", json={"message": "Find my visiting card", "top_k": 3})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        refs = data.get("referenced_screenshots", [])
        self.assertGreaterEqual(len(refs), 1)
        ref_ids = [r["screenshot_id"] for r in refs]
        self.assertIn("ss_478", ref_ids, "Expected ss_478 even when city is omitted from query.")

    def test_c3_amazon_order_screenshot_returns_honest_no_match(self):
        """
        The screenshot library contains NO Amazon order confirmations (only Spotify music
        transfer mentioning Amazon Music and LinkedIn recruiter screens).
        The query must NOT return Flipkart or Instamart screenshots as fake Amazon orders.
        """
        res = self.client.post("/api/chat", json={"message": "Find my Amazon order screenshot", "top_k": 3})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        refs = data.get("referenced_screenshots", [])
        ref_ids = [r["screenshot_id"] for r in refs]
        # Must not fabricate or return competing shopping brands (ss_010 Flipkart, ss_015 Instamart)
        self.assertNotIn("ss_010", ref_ids, "Flipkart order ss_010 must not be returned for Amazon order query.")
        self.assertNotIn("ss_015", ref_ids, "Instamart order ss_015 must not be returned for Amazon order query.")
        self.assertEqual(len(refs), 0, "No Amazon orders exist in library; expected 0 references.")
        self.assertIn("did not find", data["reply"].lower())

    def test_c4_flipkart_screenshot_retrieves_orders(self):
        """Query for Flipkart retrieves genuine Flipkart order screens."""
        res = self.client.post("/api/chat", json={"message": "Find my Flipkart screenshot", "top_k": 3})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        refs = data.get("referenced_screenshots", [])
        self.assertGreaterEqual(len(refs), 1)
        ref_ids = [r["screenshot_id"] for r in refs]
        self.assertTrue("ss_010" in ref_ids or "ss_808" in ref_ids, f"Expected Flipkart orders (ss_010/ss_808), got: {ref_ids}")
        for r in refs:
            self.assertEqual(r["category"], "Shopping")

    def test_c5_metro_shoes_shopping_screenshot(self):
        """Query for Metro Shoes retrieves ss_022."""
        res = self.client.post("/api/chat", json={"message": "Find my Metro Shoes shopping screenshot", "top_k": 3})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        refs = data.get("referenced_screenshots", [])
        self.assertGreaterEqual(len(refs), 1)
        ref_ids = [r["screenshot_id"] for r in refs]
        self.assertIn("ss_022", ref_ids, f"Expected Metro Shoes ss_022, got: {ref_ids}")
        self.assertEqual(refs[0]["category"], "Shopping")

    def test_c6_zapvi_order_screenshot(self):
        """Query for Zapvi order retrieves ss_020."""
        res = self.client.post("/api/chat", json={"message": "Find my Zapvi order screenshot", "top_k": 3})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        refs = data.get("referenced_screenshots", [])
        self.assertGreaterEqual(len(refs), 1)
        ref_ids = [r["screenshot_id"] for r in refs]
        self.assertIn("ss_020", ref_ids, f"Expected Zapvi order ss_020, got: {ref_ids}")
        self.assertEqual(refs[0]["category"], "Shopping")

    def test_c7_boarding_pass_retrieves_travel_tickets(self):
        """Query for boarding pass retrieves verified IndiGo boarding passes."""
        res = self.client.post("/api/chat", json={"message": "Find my boarding pass", "top_k": 3})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        refs = data.get("referenced_screenshots", [])
        self.assertGreaterEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r["category"], "Travel")
            self.assertIn("Boarding Pass", r["ocr_excerpt"])


class TestGroupDKnownRegressions(unittest.TestCase):
    """
    GROUP D: Verify that previously observed quality regressions remain permanently fixed.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        get_embedder()
        get_vector_store()

    def test_d1_shopping_screenshots_do_not_leak_into_payment(self):
        """Shopping screenshots must not appear in Payment category listings."""
        res = self.client.post("/api/chat", json={"message": "Find my payment screenshots", "top_k": 5})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        refs = data.get("referenced_screenshots", [])
        for r in refs:
            self.assertEqual(r["category"], "Payment")
            self.assertNotEqual(r["category"], "Shopping")

    def test_d2_coding_screenshots_preserved_as_tag_not_canonical_category(self):
        """
        Under frozen Category Decision Policy V3, 'Coding' is a multi-label tag,
        not a canonical category. General category queries should return screenshots
        whose canonical category is General, even if tagged with #Coding.
        """
        res = self.client.post("/api/chat", json={"message": "Show me my general screenshots", "top_k": 5})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        refs = data.get("referenced_screenshots", [])
        for r in refs:
            self.assertEqual(r["category"], "General")

    def test_d3_unrelated_screenshots_do_not_leak_into_travel(self):
        """Travel listings must strictly contain Travel screenshots."""
        res = self.client.post("/api/chat", json={"message": "Show me my travel screenshots", "top_k": 5})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        refs = data.get("referenced_screenshots", [])
        for r in refs:
            self.assertEqual(r["category"], "Travel")

    def test_d4_amazon_search_does_not_return_flipkart_or_instamart(self):
        """Brand constraint enforcement blocks competing brands."""
        res = self.client.post("/api/chat", json={"message": "Find my Amazon order screenshot", "top_k": 5})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        refs = data.get("referenced_screenshots", [])
        ref_ids = [r["screenshot_id"] for r in refs]
        self.assertNotIn("ss_010", ref_ids)
        self.assertNotIn("ss_015", ref_ids)
        self.assertNotIn("ss_808", ref_ids)

    def test_d5_visiting_card_retrieval_reliability(self):
        """Visiting card query does not promote financial cards."""
        res = self.client.post("/api/chat", json={"message": "Find my visiting card in Kanpur", "top_k": 3})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        refs = data.get("referenced_screenshots", [])
        self.assertEqual(refs[0]["screenshot_id"], "ss_478")
        # Ensure financial credit card screens are not returned
        for r in refs:
            self.assertNotIn("Credit Card", r.get("tags", []))

    def test_d6_travel_screenshot_ss0016_demoted_in_payment_semantic_search(self):
        """
        Travel screenshot ss_0016 (containing 'Proceed to payment') must be demoted
        and excluded from generic payment semantic search results.
        """
        results = _pipeline_search(
            query_text="List my Payment screenshots",
            top_k=10,
            category_filter=None,
            tag_filter=None,
            embedder=get_embedder(),
            vector_store=get_vector_store(),
            metadata_store=metadata_store,
            file_store=file_store,
        )
        returned_ids = [r["screenshot_id"] for r in results]
        self.assertNotIn("ss_0016", returned_ids, "ss_0016 must not appear in payment search results.")
        self.assertNotIn("ss_016", returned_ids, "ss_016 must not appear in payment search results.")

    def test_d7_positive_travel_train_ticket_retrieves_ss0016(self):
        """
        Positive travel query ('Find my train ticket') must continue to retrieve
        ss_016 / ss_0016 as a top relevant result.
        """
        results = _pipeline_search(
            query_text="Find my train ticket",
            top_k=5,
            category_filter=None,
            tag_filter=None,
            embedder=get_embedder(),
            vector_store=get_vector_store(),
            metadata_store=metadata_store,
            file_store=file_store,
        )
        returned_ids = [r["screenshot_id"] for r in results]
        self.assertTrue(
            "ss_016" in returned_ids or "ss_0016" in returned_ids,
            f"Expected ss_016/ss_0016 in train ticket search results, got: {returned_ids}"
        )

    def test_d8_positive_payment_receipts_remain_retrievable(self):
        """
        Positive payment cases (ss_013 KIITConnect, ss_017 Indian Bank, ss_015 Instamart)
        must remain retrievable and not be demoted.
        """
        results = _pipeline_search(
            query_text="Find my payment receipt",
            top_k=10,
            category_filter=None,
            tag_filter=None,
            embedder=get_embedder(),
            vector_store=get_vector_store(),
            metadata_store=metadata_store,
            file_store=file_store,
        )
        returned_ids = [r["screenshot_id"] for r in results]
        # At least one of the canonical payment receipts must be retrieved
        self.assertTrue(
            any(pid in returned_ids for pid in ["ss_017", "ss_0017", "ss_013", "ss_0013", "ss_015", "ss_0015"]),
            f"Expected genuine payment receipts in results, got: {returned_ids}"
        )


class TestGroupENegativeQueries(unittest.TestCase):
    """
    GROUP E: Verify that nonexistent subjects and unsupported brands return honest
    no-match responses without fabricating references or hallucinating weak matches.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        get_embedder()
        get_vector_store()

    def test_e1_imaginary_city_on_mars_returns_zero_matches(self):
        """Nonexistent subject query drops below threshold and returns honest no-match."""
        res = self.client.post(
            "/api/chat",
            json={"message": "Find my screenshot about an imaginary underwater city on Mars", "top_k": 3},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["referenced_screenshots"], [])
        self.assertIn("did not find", data["reply"].lower())

    def test_e2_absent_brand_nike_returns_zero_matches(self):
        """Brand query for brand not in library returns honest no-match."""
        res = self.client.post(
            "/api/chat",
            json={"message": "Find my Nike sneakers order screenshot", "top_k": 3},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["referenced_screenshots"], [])
        self.assertIn("did not find", data["reply"].lower())

    def test_e3_absent_brand_ikea_returns_zero_matches(self):
        """Brand query for Ikea furniture receipt returns honest no-match."""
        res = self.client.post(
            "/api/chat",
            json={"message": "Find my Ikea furniture receipt", "top_k": 3},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["referenced_screenshots"], [])
        self.assertIn("did not find", data["reply"].lower())


class TestGroupFFollowUpQueries(unittest.TestCase):
    """
    GROUP F: Multi-turn conversation history and topic separation.
    Tests the 3-step sequence:
    1. 'Show me my shopping screenshots'
    2. 'Show me the second one'
    3. 'Now find my travel screenshots'
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        get_embedder()
        get_vector_store()

    def test_f1_three_turn_sequence_followup_and_topic_transition(self):
        # Turn 1: Category query for Shopping
        res1 = self.client.post("/api/chat", json={"message": "Show me my shopping screenshots", "top_k": 3})
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        refs1 = data1["referenced_screenshots"]
        self.assertGreaterEqual(len(refs1), 2)
        ref_ids_1 = [r["screenshot_id"] for r in refs1]
        second_screenshot_id = ref_ids_1[1]

        history_turn1 = [
            {"role": "user", "content": "Show me my shopping screenshots"},
            {"role": "assistant", "content": data1["reply"], "referenced_ids": ref_ids_1},
        ]

        # Turn 2: Follow-up ordinal request "Show me the second one"
        res2 = self.client.post(
            "/api/chat",
            json={"message": "Show me the second one", "history": history_turn1},
        )
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        refs2 = data2["referenced_screenshots"]
        self.assertEqual(len(refs2), 1)
        self.assertEqual(refs2[0]["screenshot_id"], second_screenshot_id)
        self.assertIn(second_screenshot_id, data2["reply"])

        history_turn2 = history_turn1 + [
            {"role": "user", "content": "Show me the second one"},
            {"role": "assistant", "content": data2["reply"], "referenced_ids": [second_screenshot_id]},
        ]

        # Turn 3: Topic change "Now find my travel screenshots"
        res3 = self.client.post(
            "/api/chat",
            json={"message": "Now find my travel screenshots", "history": history_turn2, "top_k": 3},
        )
        self.assertEqual(res3.status_code, 200)
        data3 = res3.json()
        refs3 = data3["referenced_screenshots"]
        self.assertGreaterEqual(len(refs3), 1)
        ref_ids_3 = [r["screenshot_id"] for r in refs3]

        # Verify that topic transition executed fresh retrieval without stale Shopping context
        self.assertNotIn(second_screenshot_id, ref_ids_3, "Turn 3 should not retain previous turn screenshot.")
        for r in refs3:
            self.assertEqual(r["category"], "Travel", f"Turn 3 result {r['screenshot_id']} is not Travel.")

    def test_f2_followup_without_usable_history_does_not_invent_results(self):
        """
        When user asks for 'the second one' without usable references in history:
        1. With non-empty history lacking references: returns clarification that previous screenshot
           could not be identified, with 0 references.
        2. With empty history: does not invent a screenshot, returns 0 references.
        """
        # Case A: Conversation history exists, but no assistant turn contained referenced_ids
        history_no_refs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hello! How can I help you?"},
        ]
        res_a = self.client.post("/api/chat", json={"message": "Show me the second one", "history": history_no_refs})
        self.assertEqual(res_a.status_code, 200)
        data_a = res_a.json()
        self.assertEqual(data_a["referenced_screenshots"], [])
        self.assertIn("couldn't identify", data_a["reply"].lower())

        # Case B: Completely empty history
        res_b = self.client.post("/api/chat", json={"message": "Show me the second one", "history": []})
        self.assertEqual(res_b.status_code, 200)
        data_b = res_b.json()
        self.assertEqual(data_b["referenced_screenshots"], [], "Must not invent a result with empty history.")


if __name__ == "__main__":
    unittest.main()
