"""
Isolated Regression Test Suite for Persistent Conversations & Context-Aware Follow-ups
========================================================================================
Validates all requirements for Stage 5:
1. Creating a conversation and persisting messages.
2. Reloading messages from the same conversation.
3. Isolation between two conversations.
4. Correct chronological ordering.
5. A fresh conversation not inheriting old context.
6. Resolving "the second one" against the preceding assistant result set.
7. Correct behavior when the preceding assistant turn has no results.
8. Topic changes preventing stale-result reuse.
9. A new explicit query overriding prior context.
10. A screenshot reference resolving to the correct screenshot ID (unambiguous vs. ambiguous).
11. Restoring messages after simulated frontend remount/reload.
12. Existing greeting, capability, category, and semantic-search behavior remaining intact.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.main import app
from src.models.metadata import ScreenshotMetadata
from src.storage.conversation_store import SQLiteConversationStore
from src.storage.metadata_store import SQLiteMetadataStore


class TestConversationPersistence(unittest.TestCase):
    """Isolated regression tests for conversation persistence and context awareness."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_metadata.sqlite")

        # Isolated test stores
        self.conv_store = SQLiteConversationStore(db_path=self.db_path)
        self.meta_store = SQLiteMetadataStore(db_path=self.db_path)

        # Seed sample metadata for follow-up testing
        self.sample_screens = [
            ScreenshotMetadata(
                screenshot_id="ss_010",
                filename="flipkart_order.jpg",
                image_uri="data/screenshots/flipkart_order.jpg",
                category="Shopping",
                tags=["Order", "Shopping"],
                summary="Flipkart order confirmation",
                evidence={"text_snippet": "Order placed on Flipkart for Bluetooth headphones Rs 1299"},
            ),
            ScreenshotMetadata(
                screenshot_id="ss_014",
                filename="indigo_boarding_pass.jpg",
                image_uri="data/screenshots/indigo_boarding_pass.jpg",
                category="Travel",
                tags=["Ticket", "Boarding Pass"],
                summary="Indigo Boarding Pass Guwahati to Bhubaneswar",
                evidence={"text_snippet": "IndiGo 6E-241 Guwahati GAU to Bhubaneswar BBI Seat 12D"},
            ),
            ScreenshotMetadata(
                screenshot_id="ss_019",
                filename="irctc_train_ticket.jpg",
                image_uri="data/screenshots/irctc_train_ticket.jpg",
                category="Travel",
                tags=["Ticket", "Train"],
                summary="IRCTC train reservation",
                evidence={"text_snippet": "PNR 2415891024 Train 12822 Dhauli Express"},
            ),
            ScreenshotMetadata(
                screenshot_id="ss_478",
                filename="visiting_card_kanpur.jpg",
                image_uri="data/screenshots/visiting_card_kanpur.jpg",
                category="Work & Career",
                tags=["Visiting Card"],
                summary="Visiting card Aahnik Dayama Kanpur",
                evidence={"text_snippet": "apnaCard Aahnik Dayama Software Engineer Kanpur UP"},
            ),
        ]
        for m in self.sample_screens:
            self.meta_store.save_metadata(m)

        # Patches to guarantee complete test isolation from production DB and vector store
        self.patch_conv = patch("backend.app.api.chat.conversation_store", self.conv_store)
        self.patch_meta = patch("backend.app.api.chat.metadata_store", self.meta_store)

        self.patch_conv.start()
        self.patch_meta.start()

        self.client = TestClient(app)

    def tearDown(self):
        self.patch_conv.stop()
        self.patch_meta.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. Creating a conversation and persisting messages
    def test_01_create_conversation_and_persist_messages(self):
        conv = self.conv_store.create_conversation("Travel Research")
        cid = conv["conversation_id"]

        self.conv_store.add_message(cid, "user", "Show me flight tickets")
        self.conv_store.add_message(
            cid,
            "assistant",
            "Found 1 ticket",
            referenced_ids=["ss_014"],
            references=[{"screenshot_id": "ss_014", "filename": "indigo.jpg", "category": "Travel", "tags": [], "similarity_score": 1.0, "ocr_excerpt": "IndiGo"}]
        )

        messages = self.conv_store.get_messages(cid)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "Show me flight tickets")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["referenced_ids"], ["ss_014"])

    # 2. Reloading messages from the same conversation
    def test_02_reload_messages_from_same_conversation(self):
        conv = self.conv_store.create_conversation("My Orders")
        cid = conv["conversation_id"]

        self.conv_store.add_message(cid, "user", "Find orders")
        self.conv_store.add_message(
            cid,
            "assistant",
            "Found Flipkart order",
            referenced_ids=["ss_010"],
            references=[{"screenshot_id": "ss_010", "filename": "flipkart.jpg", "category": "Shopping", "tags": [], "similarity_score": 0.95, "ocr_excerpt": "Order placed"}]
        )

        res = self.client.get(f"/api/chat/conversations/{cid}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["conversation_id"], cid)
        self.assertEqual(data["title"], "My Orders")
        self.assertEqual(len(data["messages"]), 2)
        self.assertEqual(data["messages"][1]["references"][0]["screenshot_id"], "ss_010")

    # 3. Isolation between two conversations
    def test_03_isolation_between_two_conversations(self):
        conv_a = self.conv_store.create_conversation("Session A")["conversation_id"]
        conv_b = self.conv_store.create_conversation("Session B")["conversation_id"]

        self.conv_store.add_message(conv_a, "user", "Message in A")
        self.conv_store.add_message(conv_b, "user", "Message in B")

        msgs_a = self.conv_store.get_messages(conv_a)
        msgs_b = self.conv_store.get_messages(conv_b)

        self.assertEqual(len(msgs_a), 1)
        self.assertEqual(msgs_a[0]["content"], "Message in A")

        self.assertEqual(len(msgs_b), 1)
        self.assertEqual(msgs_b[0]["content"], "Message in B")

    # 4. Correct chronological ordering
    def test_04_correct_chronological_ordering(self):
        cid = self.conv_store.create_conversation("Ordering Test")["conversation_id"]
        self.conv_store.add_message(cid, "user", "First")
        self.conv_store.add_message(cid, "assistant", "Second")
        self.conv_store.add_message(cid, "user", "Third")
        self.conv_store.add_message(cid, "assistant", "Fourth")

        msgs = self.conv_store.get_messages(cid)
        contents = [m["content"] for m in msgs]
        self.assertEqual(contents, ["First", "Second", "Third", "Fourth"])

    # 5. A fresh conversation not inheriting old context
    def test_05_fresh_conversation_not_inheriting_old_context(self):
        # Conversation A has results
        cid_a = self.conv_store.create_conversation("Trip A")["conversation_id"]
        self.conv_store.add_message(cid_a, "user", "Show travel")
        self.conv_store.add_message(cid_a, "assistant", "Found ticket", referenced_ids=["ss_014"])

        # Conversation B is fresh
        cid_b = self.conv_store.create_conversation("Fresh B")["conversation_id"]

        res = self.client.post("/api/chat", json={"message": "Show me the second one", "conversation_id": cid_b})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["referenced_screenshots"], [])
        self.assertIn("couldn't identify", data["reply"].lower())

    # 6. Resolving "the second one" against preceding assistant result set
    def test_06_resolving_second_one_against_preceding_assistant_result_set(self):
        cid = self.conv_store.create_conversation("Tickets")["conversation_id"]
        self.conv_store.add_message(cid, "user", "Show my travel tickets")
        self.conv_store.add_message(
            cid,
            "assistant",
            "Here are your travel tickets:",
            referenced_ids=["ss_014", "ss_019"],
        )

        res = self.client.post("/api/chat", json={"message": "Show me the second one", "conversation_id": cid})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        refs = data["referenced_screenshots"]
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["screenshot_id"], "ss_019")
        self.assertIn("ss_019", data["reply"])
        self.assertIn("Dhauli Express", data["reply"])

    # 7. Correct behavior when preceding assistant turn has no results
    def test_07_correct_behavior_when_preceding_assistant_turn_has_no_results(self):
        cid = self.conv_store.create_conversation("Empty Preceding Turn")["conversation_id"]
        # Turn 1: has results
        self.conv_store.add_message(cid, "user", "Show tickets")
        self.conv_store.add_message(cid, "assistant", "Found tickets", referenced_ids=["ss_014", "ss_019"])

        # Turn 2: returns 0 results
        self.conv_store.add_message(cid, "user", "Find Martian city")
        self.conv_store.add_message(cid, "assistant", "No matching screenshots found.", referenced_ids=[])

        # Turn 3: User says "Show me the second one"
        res = self.client.post("/api/chat", json={"message": "Show me the second one", "conversation_id": cid})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["referenced_screenshots"], [], "Must not resurrect results from Turn 1")
        self.assertIn("no screenshots were returned in the last response", data["reply"])

    # 8. Topic changes preventing stale-result reuse
    def test_08_topic_changes_preventing_stale_result_reuse(self):
        cid = self.conv_store.create_conversation("Topic Switch")["conversation_id"]
        self.conv_store.add_message(cid, "user", "Show my shopping screenshots")
        self.conv_store.add_message(cid, "assistant", "Here is Shopping:", referenced_ids=["ss_010"])

        # Now user switches topic to travel screenshots
        res = self.client.post("/api/chat", json={"message": "Show me my Travel screenshots", "conversation_id": cid})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        refs = data["referenced_screenshots"]
        self.assertGreaterEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r["category"], "Travel")
            self.assertNotEqual(r["screenshot_id"], "ss_010")

    # 9. A new explicit query overriding prior context
    def test_09_new_explicit_query_overrides_prior_context(self):
        cid = self.conv_store.create_conversation("Explicit Query")["conversation_id"]
        self.conv_store.add_message(cid, "user", "Show travel")
        self.conv_store.add_message(cid, "assistant", "Found ticket", referenced_ids=["ss_014"])

        # User enters an explicit search for Flipkart order
        with patch("backend.app.api.chat._pipeline_search") as mock_pipeline:
            mock_pipeline.return_value = [{
                "screenshot_id": "ss_010",
                "filename": "flipkart.jpg",
                "category": "Shopping",
                "tags": ["Shopping"],
                "similarity_score": 0.88,
                "text": "Flipkart order",
                "image_url": "/data/screenshots/flipkart.jpg",
            }]
            res = self.client.post("/api/chat", json={"message": "Find my Flipkart order", "conversation_id": cid})
            self.assertEqual(res.status_code, 200)
            mock_pipeline.assert_called_once()
            data = res.json()
            self.assertEqual(data["referenced_screenshots"][0]["screenshot_id"], "ss_010")

    # 10. Screenshot reference resolving to correct ID (unambiguous vs. ambiguous)
    def test_10_screenshot_reference_resolving_unambiguous_vs_ambiguous(self):
        # Case A: Unambiguous (exactly 1 preceding result)
        cid_a = self.conv_store.create_conversation("Visiting Card")["conversation_id"]
        self.conv_store.add_message(cid_a, "user", "Find visiting card")
        self.conv_store.add_message(cid_a, "assistant", "Found visiting card", referenced_ids=["ss_478"])

        res_a = self.client.post("/api/chat", json={"message": "Show me that screenshot", "conversation_id": cid_a})
        self.assertEqual(res_a.status_code, 200)
        data_a = res_a.json()
        self.assertEqual(len(data_a["referenced_screenshots"]), 1)
        self.assertEqual(data_a["referenced_screenshots"][0]["screenshot_id"], "ss_478")
        self.assertIn("apnaCard", data_a["reply"])

        # Case B: Ambiguous (multiple preceding results)
        cid_b = self.conv_store.create_conversation("Multiple Tickets")["conversation_id"]
        self.conv_store.add_message(cid_b, "user", "Show tickets")
        self.conv_store.add_message(cid_b, "assistant", "Found tickets", referenced_ids=["ss_014", "ss_019"])

        res_b = self.client.post("/api/chat", json={"message": "Show me that screenshot", "conversation_id": cid_b})
        self.assertEqual(res_b.status_code, 200)
        data_b = res_b.json()
        self.assertEqual(data_b["referenced_screenshots"], [], "Ambiguous request must not guess a result")
        self.assertIn("multiple screenshots were returned", data_b["reply"].lower())
        self.assertIn("ss_014", data_b["reply"])
        self.assertIn("ss_019", data_b["reply"])

    # 11. Restoring messages after simulated frontend remount/reload
    def test_11_simulated_reload_and_remount_restores_messages(self):
        cid = self.conv_store.create_conversation("Trip Planning")["conversation_id"]

        # User interacts with assistant
        res1 = self.client.post(
            "/api/chat",
            json={"message": "Show me my travel screenshots", "conversation_id": cid, "top_k": 2}
        )
        self.assertEqual(res1.status_code, 200)

        # Simulate tab switch / page reload: query the restored conversation
        res_restore = self.client.get(f"/api/chat/conversations/{cid}")
        self.assertEqual(res_restore.status_code, 200)
        conv_detail = res_restore.json()

        self.assertEqual(conv_detail["conversation_id"], cid)
        self.assertEqual(len(conv_detail["messages"]), 2)
        self.assertEqual(conv_detail["messages"][0]["role"], "user")
        self.assertEqual(conv_detail["messages"][1]["role"], "assistant")

        # Verify screenshot cards are preserved in the restored message
        restored_refs = conv_detail["messages"][1]["references"]
        self.assertGreaterEqual(len(restored_refs), 1)
        self.assertIn("screenshot_id", restored_refs[0])
        self.assertIn("ocr_excerpt", restored_refs[0])

    # 12. Existing greeting, capability, category, and session deletion behavior intact
    def test_12_existing_greeting_capability_and_delete_session(self):
        cid = self.conv_store.create_conversation("General Q")["conversation_id"]

        # Greeting
        res_g = self.client.post("/api/chat", json={"message": "hello", "conversation_id": cid})
        self.assertEqual(res_g.status_code, 200)
        self.assertEqual(res_g.json()["referenced_screenshots"], [])
        self.assertIn("I.R.I.S.", res_g.json()["reply"])

        # Capability
        res_c = self.client.post("/api/chat", json={"message": "what can you do", "conversation_id": cid})
        self.assertEqual(res_c.status_code, 200)
        self.assertEqual(res_c.json()["referenced_screenshots"], [])
        self.assertIn("search", res_c.json()["reply"].lower())

        # Verify messages persisted
        msgs = self.conv_store.get_messages(cid)
        self.assertEqual(len(msgs), 4)

        # Delete conversation
        res_d = self.client.delete(f"/api/chat/conversations/{cid}")
        self.assertEqual(res_d.status_code, 200)
        self.assertTrue(res_d.json()["deleted"])

        # Verifying deleted
        res_404 = self.client.get(f"/api/chat/conversations/{cid}")
        self.assertEqual(res_404.status_code, 404)


if __name__ == "__main__":
    unittest.main()
