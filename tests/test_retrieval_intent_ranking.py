"""
Isolated Regression and Ranking Tests for Retrieval Intent Disambiguation
========================================================================
Verifies that I.R.I.S. distinguishes between:
1. Completed orders / order details
2. Payments / UPI transactions
3. Product or restaurant discovery screens
4. Checkout / pre-payment screens

Tests run against controlled in-memory metadata objects without modifying
production SQLite or ChromaDB records.
"""

import unittest
from backend.app.api.search import _calculate_rerank_score
from src.models.metadata import ScreenshotMetadata


class TestRetrievalIntentRanking(unittest.TestCase):
    """Isolated regression tests for intent-aware screenshot ranking."""

    def test_1_generic_query_grocery_order_ranks_above_discovery(self):
        """Generic query 'Find my grocery order' ranks completed order above discovery and checkout."""
        q = "Find my grocery order"
        base_sim = 0.45

        meta_order = ScreenshotMetadata(
            screenshot_id="mock_order",
            filename="mock_order.jpg",
            image_uri="data/mock_order.jpg",
            category="Shopping",
            tags=["Grocery", "Order"],
            summary="Blinkit grocery order confirmation",
            evidence={"text_snippet": "Blinkit Order #90210\nDelivered on 14 Oct\nTotal: Rs 420\n3 items ordered"},
        )
        meta_discovery = ScreenshotMetadata(
            screenshot_id="mock_disc",
            filename="mock_disc.jpg",
            image_uri="data/mock_disc.jpg",
            category="Shopping",
            tags=["Grocery"],
            summary="Blinkit search screen",
            evidence={"text_snippet": "Search for groceries\nExplore fresh fruits and vegetables\nAdd to cart"},
        )
        meta_checkout = ScreenshotMetadata(
            screenshot_id="mock_chk",
            filename="mock_chk.jpg",
            image_uri="data/mock_chk.jpg",
            category="Shopping",
            tags=["Cart"],
            summary="Blinkit checkout cart",
            evidence={"text_snippet": "View cart\nItems in cart: 3\nDelivery address: Home\nProceed to checkout"},
        )

        score_order = _calculate_rerank_score(q, base_sim, "Blinkit Order #90210 Delivered on 14 Oct 3 items", meta_order)
        score_disc = _calculate_rerank_score(q, base_sim, "Search for groceries Explore fresh fruits Add to cart", meta_discovery)
        score_chk = _calculate_rerank_score(q, base_sim, "View cart Items in cart Proceed to checkout", meta_checkout)

        self.assertGreater(score_order, score_disc, "Completed order must rank strictly above discovery screen")
        self.assertGreater(score_order, score_chk, "Completed order must rank strictly above pre-checkout cart screen")
        self.assertGreaterEqual(score_order, 0.65, "Completed order should receive intent boost >= 0.65")
        self.assertLess(score_disc, 0.35, "Discovery screen should be penalized and drop below 0.35 threshold")

    def test_2_brand_instamart_order_ranks_above_instamart_discovery(self):
        """Brand query 'Find my Instamart order' ranks actual order above Instamart discovery screen."""
        q = "Find my Instamart order"
        base_sim = 0.50

        # Similar to ss_015
        meta_instamart_order = ScreenshotMetadata(
            screenshot_id="ss_015",
            filename="ss_015.jpg",
            image_uri="data/ss_015.jpg",
            category="Payment",
            tags=["Instamart", "Payment", "UPI"],
            summary="Instamart order cashback summary",
            evidence={"text_snippet": "Instamart order\n05:30 PM - 5 items\n\n5% cashback on all UPI spends"},
        )
        # Similar to ss_476
        meta_instamart_discovery = ScreenshotMetadata(
            screenshot_id="ss_476",
            filename="ss_476.png",
            image_uri="data/ss_476.png",
            category="Shopping",
            tags=[],
            summary="Swiggy Instamart home and search page",
            evidence={"text_snippet": "Search for dishes, restaurants & groceries\nBIRYANI\nFOOD\nINSTAMART\nDINEOUT\ndelivery\ngroceries"},
        )

        score_order = _calculate_rerank_score(q, base_sim, "Instamart order 05:30 PM 5 items", meta_instamart_order)
        score_disc = _calculate_rerank_score(q, base_sim, "Search for dishes restaurants groceries INSTAMART DINEOUT", meta_instamart_discovery)

        self.assertGreater(score_order, score_disc, "Instamart order must rank strictly above Instamart discovery")
        self.assertGreaterEqual(score_order, 0.90, "Instamart order should receive both brand and order boosts")
        self.assertLess(score_disc, 0.35, "Instamart discovery screen must be penalized below 0.35 cutoff for order query")

    def test_3_instamart_upi_transaction_ranks_above_discovery(self):
        """UPI query 'Find my Instamart UPI transaction' ranks UPI receipt above discovery screen."""
        q = "Find my Instamart UPI transaction"
        base_sim = 0.50

        meta_upi_order = ScreenshotMetadata(
            screenshot_id="ss_015",
            filename="ss_015.jpg",
            image_uri="data/ss_015.jpg",
            category="Payment",
            tags=["Payment", "UPI"],
            summary="Instamart order UPI payment",
            evidence={"text_snippet": "Instamart order\n5% cashback on all UPI spends\nUPI Ref No 4291881729"},
        )
        meta_discovery = ScreenshotMetadata(
            screenshot_id="ss_456",
            filename="ss_456.png",
            image_uri="data/ss_456.png",
            category="Shopping",
            tags=[],
            summary="Instamart search screen",
            evidence={"text_snippet": "Search for 'Biryani'\nINTRODUCING MEDICINES On Instamart"},
        )

        score_upi = _calculate_rerank_score(q, base_sim, "Instamart order 5% cashback on all UPI spends UPI Ref No", meta_upi_order)
        score_disc = _calculate_rerank_score(q, base_sim, "Search for Biryani INTRODUCING MEDICINES On Instamart", meta_discovery)

        self.assertGreater(score_upi, score_disc, "UPI payment evidence must rank strictly above discovery screen")
        self.assertGreaterEqual(score_upi, 0.90, "UPI payment evidence should receive both brand and payment boosts")
        self.assertLess(score_disc, 0.35, "Discovery screen without payment evidence must be penalized below 0.35 threshold")

    def test_4_discovery_query_restaurants_near_me_ranks_discovery_above_past_orders(self):
        """Discovery query 'Show me restaurants near me' boosts restaurant discovery and demotes past orders."""
        q = "Show me restaurants near me"
        base_sim = 0.45

        meta_disc = ScreenshotMetadata(
            screenshot_id="ss_484",
            filename="ss_484.png",
            image_uri="data/ss_484.png",
            category="Shopping",
            tags=[],
            summary="Restaurant discovery screen",
            evidence={"text_snippet": "Search for dishes, restaurants near me\nBIRYANI\nFOOD\nDINEOUT\nRated 4+"},
        )
        meta_past_order = ScreenshotMetadata(
            screenshot_id="ss_010",
            filename="ss_010.jpg",
            image_uri="data/ss_010.jpg",
            category="Shopping",
            tags=["Order"],
            summary="Past order list",
            evidence={"text_snippet": "My Orders\nFlipkart Order #98124 Delivered on Aug 12"},
        )

        score_disc = _calculate_rerank_score(q, base_sim, "Search for dishes restaurants near me BIRYANI DINEOUT", meta_disc)
        score_order = _calculate_rerank_score(q, base_sim, "My Orders Flipkart Order #98124 Delivered on Aug 12", meta_past_order)

        self.assertGreater(score_disc, score_order, "Restaurant discovery screen must rank strictly above past order history")
        self.assertGreaterEqual(score_disc, 0.70, "Discovery query should boost discovery screens >= 0.70")
        self.assertLess(score_order, 0.35, "Past orders should be demoted below 0.35 threshold for discovery queries")

    def test_5_product_discovery_query_ranks_browsing_above_past_orders(self):
        """Product discovery query 'Find grocery products' ranks product catalog above past order receipts."""
        q = "Find grocery products"
        base_sim = 0.45

        meta_catalog = ScreenshotMetadata(
            screenshot_id="mock_cat",
            filename="mock_cat.jpg",
            image_uri="data/mock_cat.jpg",
            category="Shopping",
            tags=[],
            summary="Grocery product catalog",
            evidence={"text_snippet": "Search for groceries & products\nFresh farm vegetables\nIn stock\nAdd to cart"},
        )
        meta_past_order = ScreenshotMetadata(
            screenshot_id="ss_010",
            filename="ss_010.jpg",
            image_uri="data/ss_010.jpg",
            category="Shopping",
            tags=["Order"],
            summary="Past order list",
            evidence={"text_snippet": "My Orders\nDelivered on Aug 12\nOrder total: Rs 1,250"},
        )

        score_catalog = _calculate_rerank_score(q, base_sim, "Search for groceries products Fresh farm vegetables Add to cart", meta_catalog)
        score_order = _calculate_rerank_score(q, base_sim, "My Orders Delivered on Aug 12 Order total: Rs 1,250", meta_past_order)

        self.assertGreater(score_catalog, score_order, "Product catalog must rank strictly above past completed orders")
        self.assertGreaterEqual(score_catalog, 0.70, "Product discovery screen should receive discovery boost")
        self.assertLess(score_order, 0.35, "Completed past orders must be demoted below 0.35 threshold for product discovery queries")

    def test_6_checkout_query_ranks_precheckout_above_completed_payment(self):
        """Checkout query 'Find the screen where I was about to pay' ranks pre-checkout above completed payment."""
        q = "Find the screen where I was about to pay"
        base_sim = 0.45

        # Similar to ss_016
        meta_chk = ScreenshotMetadata(
            screenshot_id="ss_016",
            filename="ss_016.jpg",
            image_uri="data/ss_016.jpg",
            category="Travel",
            tags=["Ticket"],
            summary="Train booking review before payment",
            evidence={"text_snippet": "Review\n15611 Express AC 3 Tier\nYour Boarding Details\nProceed to payment"},
        )
        # Similar to ss_017
        meta_paid = ScreenshotMetadata(
            screenshot_id="ss_017",
            filename="ss_017.jpg",
            image_uri="data/ss_017.jpg",
            category="Payment",
            tags=["Payment", "Receipt"],
            summary="Completed bank payment receipt",
            evidence={"text_snippet": "Payment successful\nRs 500 paid successfully to KIIT\nOrder ID: BILL-SAP"},
        )

        score_chk = _calculate_rerank_score(q, base_sim, "Review Boarding Details Proceed to payment", meta_chk)
        score_paid = _calculate_rerank_score(q, base_sim, "Payment successful Rs 500 paid successfully", meta_paid)

        self.assertGreater(score_chk, score_paid, "Pre-checkout screen must rank strictly above completed payment receipt")
        self.assertGreaterEqual(score_chk, 0.65, "Pre-checkout screen should receive checkout intent boost >= 0.65")
        self.assertLess(score_paid, 0.35, "Already-completed payments must be demoted below 0.35 cutoff for checkout queries")

    def test_7_completed_payment_query_ranks_completed_above_precheckout(self):
        """Payment query 'Find my payment successful screenshot' ranks completed payment above pre-checkout."""
        q = "Find my payment successful screenshot"
        base_sim = 0.45

        meta_paid = ScreenshotMetadata(
            screenshot_id="ss_017",
            filename="ss_017.jpg",
            image_uri="data/ss_017.jpg",
            category="Payment",
            tags=["Payment"],
            summary="Completed payment receipt",
            evidence={"text_snippet": "Payment successful\nRs 500 paid successfully\nTxn ID: 98124819"},
        )
        meta_chk = ScreenshotMetadata(
            screenshot_id="ss_016",
            filename="ss_016.jpg",
            image_uri="data/ss_016.jpg",
            category="Travel",
            tags=["Ticket"],
            summary="Train ticket review",
            evidence={"text_snippet": "Review\n15611 Express\nProceed to payment"},
        )

        score_paid = _calculate_rerank_score(q, base_sim, "Payment successful Rs 500 paid successfully Txn ID", meta_paid)
        score_chk = _calculate_rerank_score(q, base_sim, "Review 15611 Express Proceed to payment", meta_chk)

        self.assertGreater(score_paid, score_chk, "Completed payment must rank strictly above pre-checkout screen")
        self.assertGreaterEqual(score_paid, 0.70, "Completed payment confirmation should receive payment boost >= 0.70")
        self.assertLess(score_chk, 0.35, "Pre-checkout screen must be demoted below 0.35 threshold for payment query")

    def test_8_completed_order_with_noisy_ocr_is_boosted(self):
        """Completed order with imperfect OCR ('ordr #98214', '3 ttems delivered') is recognized and boosted."""
        q = "Find my grocery order"
        base_sim = 0.45

        meta_noisy_order = ScreenshotMetadata(
            screenshot_id="mock_noisy",
            filename="mock_noisy.jpg",
            image_uri="data/mock_noisy.jpg",
            category="Shopping",
            tags=["Shopping"],
            summary="Noisy grocery order screenshot",
            evidence={"text_snippet": "ordr #98214\n3 ttems delivered on 12 Aug\ntotl: rs 450"},
        )
        meta_noisy_disc = ScreenshotMetadata(
            screenshot_id="mock_noisy_disc",
            filename="mock_noisy_disc.jpg",
            image_uri="data/mock_noisy_disc.jpg",
            category="Shopping",
            tags=["Shopping"],
            summary="Noisy discovery screen",
            evidence={"text_snippet": "Search for dishes restaurants groceries biryani dineout"},
        )

        score_noisy_order = _calculate_rerank_score(q, base_sim, "ordr #98214 3 ttems delivered on 12 Aug", meta_noisy_order)
        score_noisy_disc = _calculate_rerank_score(q, base_sim, "Search for dishes restaurants groceries biryani dineout", meta_noisy_disc)

        self.assertGreater(score_noisy_order, score_noisy_disc, "Noisy completed order must rank above noisy discovery screen")
        self.assertGreaterEqual(score_noisy_order, 0.65, "Noisy completed order should still be boosted above 0.65")
        self.assertLess(score_noisy_disc, 0.35, "Discovery screen should be penalized and drop below 0.35 cutoff")

    def test_9_query_with_no_matching_order_or_transaction_evidence(self):
        """A query with no matching order or transaction evidence drops below threshold (honest no-match)."""
        q = "Find my grocery order"
        base_sim = 0.25

        meta_unrelated = ScreenshotMetadata(
            screenshot_id="mock_unrelated",
            filename="mock_unrelated.jpg",
            image_uri="data/mock_unrelated.jpg",
            category="General",
            tags=[],
            summary="Project sync meeting notes",
            evidence={"text_snippet": "Weekly team sprint planning and UI architecture review"},
        )

        score_unrelated = _calculate_rerank_score(q, base_sim, "Weekly team sprint planning and UI architecture review", meta_unrelated)
        self.assertLess(score_unrelated, 0.35, "Document with zero order or grocery evidence must drop below 0.35 threshold")

    def test_10_existing_regressions_preserved(self):
        """Preserves existing brand-conflict rejection, visiting card disambiguation, and travel ticket retrieval."""
        # 10a. Amazon vs Flipkart: Conflicting brand hard rejection
        meta_flipkart = ScreenshotMetadata(
            screenshot_id="ss_010",
            filename="ss_010.jpg",
            image_uri="data/ss_010.jpg",
            category="Shopping",
            tags=["Flipkart", "Order"],
            summary="Flipkart order",
            evidence={"text_snippet": "My Orders\nWith Flipkart SBI Credit Card\nDelivery by Aug 12"},
        )
        score_amazon_query = _calculate_rerank_score(
            "Find my Amazon order confirmation",
            0.60,
            "My Orders Flipkart SBI Credit Card Delivery by Aug 12",
            meta_flipkart,
        )
        self.assertEqual(score_amazon_query, -1.0, "Flipkart order must be rejected (-1.0) when Amazon was requested")

        # 10b. Visiting card vs Financial card
        meta_visiting = ScreenshotMetadata(
            screenshot_id="ss_478",
            filename="ss_478.jpg",
            image_uri="data/ss_478.jpg",
            category="Work & Career",
            tags=["Visiting Card"],
            summary="apnaCard visiting card",
            evidence={"text_snippet": "apnaCard\nEdit Card\nAahnik Dayama\nAllahabad Road, Kanpur"},
        )
        meta_financial = ScreenshotMetadata(
            screenshot_id="fin_card",
            filename="fin_card.jpg",
            image_uri="data/fin_card.jpg",
            category="Shopping",
            tags=["Credit Card"],
            summary="Credit card billing",
            evidence={"text_snippet": "Credit card payment Kanpur branch billing address"},
        )
        score_visiting = _calculate_rerank_score("Find my visiting card in Kanpur", 0.30, "apnaCard Edit Card Aahnik Kanpur", meta_visiting)
        score_financial = _calculate_rerank_score("Find my visiting card in Kanpur", 0.40, "Credit card payment Kanpur branch", meta_financial)

        self.assertGreaterEqual(score_visiting, 0.35, "Visiting card evidence must pass threshold")
        self.assertLess(score_financial, 0.35, "Financial credit card must drop below 0.35 threshold")

        # 10c. Positive travel query retrieves ticket despite 'Proceed to payment'
        meta_travel_ticket = ScreenshotMetadata(
            screenshot_id="ss_016",
            filename="ss_016.jpg",
            image_uri="data/ss_016.jpg",
            category="Travel",
            tags=["Ticket"],
            summary="Train ticket review",
            evidence={"text_snippet": "Review\n15611 Express\nYour Boarding Details\nProceed to payment"},
        )
        score_train_ticket = _calculate_rerank_score(
            "Find my train ticket",
            0.55,
            "Review 15611 Express Boarding Details Proceed to payment",
            meta_travel_ticket,
        )
        self.assertGreaterEqual(score_train_ticket, 0.50, "Travel train ticket must remain retrievable and not be demoted")


if __name__ == "__main__":
    unittest.main()
