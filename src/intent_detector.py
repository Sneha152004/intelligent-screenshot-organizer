"""
Deterministic Screenshot Intent Detector
=========================================
Infers the primary user intent / goal behind capturing a screenshot.
Canonical intent categories:
- "Shopping / Purchase"
- "Payment / Transaction / Bill"
- "Travel / Booking"
- "Event / Appointment"
- "Task / Reminder"
- "Communication / Contact"
- "Learning / Reference"
- "Coding / Technical Work"
- "Credential / Network Access"
- "General Info / None" (None when evidence is insufficient)

Provides confidence scoring and evidence tracking without requiring external ML models.
"""

from typing import Any, Dict, List, Optional, Tuple


# Standardized Canonical Intent Labels
INTENT_SHOPPING = "Shopping / Purchase"
INTENT_PAYMENT = "Payment / Transaction / Bill"
INTENT_TRAVEL = "Travel / Booking"
INTENT_EVENT = "Event / Appointment"
INTENT_TASK = "Task / Reminder"
INTENT_COMMUNICATION = "Communication / Contact"
INTENT_LEARNING = "Learning / Reference"
INTENT_CODING = "Coding / Technical Work"
INTENT_CREDENTIAL = "Credential / Network Access"
INTENT_GENERAL = "General Info / None"


class ScreenshotIntentDetector:
    """
    Deterministic, rule-based intent inference engine.
    Analyzes accumulated screenshot signals to establish the user's primary
    motivation for capturing the screenshot.
    """

    def __init__(self, confidence_threshold: float = 0.50):
        self.confidence_threshold = confidence_threshold

    def detect_intent(
        self,
        ocr_text: Optional[str],
        category: Optional[str] = "General",
        tags: Optional[List[str]] = None,
        content_type: Optional[str] = "unknown",
        financial_info: Optional[Dict[str, Any]] = None,
        dates: Optional[List[Dict[str, str]]] = None,
        entities: Optional[List[Dict[str, str]]] = None,
        sensitive_info: Optional[Dict[str, bool]] = None,
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """
        Determines user intent from combined multimodal evidence.

        Returns:
            Tuple of:
            - intent: Optional[str] (None if insufficient evidence)
            - confidence: float (0.0 to 1.0)
            - evidence: Dict[str, Any] (details supporting the inference)
        """
        text = str(ocr_text).strip() if ocr_text else ""
        if not text or len(text.split()) < 2:
            return None, 0.0, {"reason": "insufficient_text"}

        text_lower = text.lower()
        tags_lower = [t.lower() for t in (tags or [])]
        fin = financial_info or {}
        sens = sensitive_info or {}
        ents = entities or []

        # 1. Credential / Network Access
        if sens.get("contains_password") or "wifi password" in text_lower or "wi-fi password" in text_lower or (
            category == "Connectivity" and ("password" in text_lower or "ssid" in text_lower or "credentials" in text_lower)
        ):
            return INTENT_CREDENTIAL, 0.95, {
                "signal": "password_or_network_credentials",
                "category": category,
            }

        # 2. Shopping / Purchase (Order confirmations, carts, products)
        shopping_kws = [
            "add to cart", "buy now", "order placed", "order confirmation",
            "order #", "order id", "delivery expected", "delivery status",
            "tracking id", "item total", "shipping address", "items:",
        ]
        has_shop_kw = any(k in text_lower for k in shopping_kws)
        is_shop_tag = any(t in ["shopping", "order", "delivery", "ecommerce"] for t in tags_lower)
        has_merchant = any(e.get("type") == "merchant" or e.get("label") == "merchant" for e in ents)
        has_order_id = any(e.get("type") == "ORDER_ID" or e.get("label") == "ORDER_ID" for e in ents)

        if category == "Shopping" or content_type in ["shopping_order", "product_page", "delivery_tracking"] or (
            (has_shop_kw or has_order_id) and (has_merchant or is_shop_tag)
        ):
            conf = 0.95 if (has_order_id or "order confirmation" in text_lower) else 0.85
            return INTENT_SHOPPING, conf, {
                "signal": "product_or_order_confirmation",
                "merchant": [e["text"] for e in ents if e.get("type") == "merchant"],
            }

        # 3. Payment / Transaction / Bill
        has_fin_amounts = len(fin.get("amounts", [])) > 0 or fin.get("amount") is not None
        has_fin_tx = len(fin.get("transaction_ids", [])) > 0 or len(fin.get("upi_ids", [])) > 0 or fin.get("transaction_id") is not None
        has_pay_kw = fin.get("has_payment_keywords", False)
        is_pay_tag = any(t in ["payment", "receipt", "invoice", "upi", "bill"] for t in tags_lower)

        if category == "Payment" or content_type in ["payment_receipt", "transaction", "bill"] or (
            has_fin_amounts and (has_pay_kw or has_fin_tx or is_pay_tag)
        ):
            conf = 0.95 if (has_fin_amounts and has_fin_tx) else 0.85
            return INTENT_PAYMENT, conf, {
                "signal": "financial_transaction_or_bill",
                "amounts": fin.get("amounts", []),
                "payment_method": fin.get("payment_method"),
            }

        # 4. Travel / Booking
        travel_kws = ["boarding pass", "flight", "pnr", "departure", "arrival", "train ticket", "hotel booking", "check-in"]
        has_travel_kw = any(k in text_lower for k in travel_kws)
        is_travel_tag = any(t in ["travel", "flight", "ticket", "boarding_pass", "hotel"] for t in tags_lower)
        has_airline = any(e.get("type") == "airline" or e.get("label") == "airline" for e in ents)

        if category == "Travel" or content_type in ["flight_ticket", "train_ticket", "hotel_booking", "travel_itinerary"] or has_travel_kw or has_airline:
            conf = 0.95 if (has_airline or "boarding pass" in text_lower or "pnr" in text_lower) else 0.80
            return INTENT_TRAVEL, conf, {
                "signal": "travel_booking_or_itinerary",
                "airline": [e["text"] for e in ents if e.get("type") == "airline"],
            }

        # 5. Coding / Technical Work
        coding_kws = [
            "def ", "class ", "function ", "const ", "let ", "var ", "import ",
            "syntaxerror", "traceback", "connectionrefusederror", "public static void",
            "select * from", "git commit", "docker", "npm ", "pip install",
        ]
        has_code_kw = any(k in text_lower for k in coding_kws)
        is_coding_tag = any(t in ["coding", "programming", "software development", "python", "javascript", "developer"] for t in tags_lower)

        if category == "Work & Career" and (content_type in ["code_editor", "terminal", "error_log"] or has_code_kw or is_coding_tag):
            return INTENT_CODING, 0.90, {
                "signal": "code_snippet_or_tech_artifact",
                "tags": [t for t in tags_lower if t in ["coding", "programming", "developer"]],
            }
        elif is_coding_tag or content_type in ["code_editor", "terminal", "error_log"] or has_code_kw:
            return INTENT_CODING, 0.85, {
                "signal": "code_snippet_or_tech_artifact",
            }

        # 6. Event / Appointment
        event_kws = ["hackathon", "webinar", "conference", "appointment", "scheduled meeting", "zoom meeting", "meet.google", "venue:"]
        has_event_kw = any(k in text_lower for k in event_kws)
        if has_event_kw or content_type == "event":
            return INTENT_EVENT, 0.85, {
                "signal": "event_or_appointment",
                "keywords": [k for k in event_kws if k in text_lower],
            }

        # 7. Task / Reminder
        task_kws = ["due date", "pay before", "reminder", "todo", "to-do", "deadline", "urgent:"]
        has_task_kw = any(k in text_lower for k in task_kws)
        if has_task_kw:
            return INTENT_TASK, 0.80, {
                "signal": "task_or_reminder",
            }

        # 8. Communication / Contact
        if category == "Communication" or content_type in ["chat", "email", "message"] or (
            "whatsapp" in text_lower or "telegram" in text_lower or "unread messages" in text_lower
        ):
            return INTENT_COMMUNICATION, 0.80, {
                "signal": "communication_or_chat",
            }

        # 9. Learning / Reference
        if category == "Education" or content_type in ["course_material", "exam_schedule", "assignment", "notes"] or (
            "syllabus" in text_lower or "curriculum" in text_lower or "lecture" in text_lower
        ):
            return INTENT_LEARNING, 0.80, {
                "signal": "academic_or_educational_reference",
            }

        # Default fallback
        if len(text.split()) >= 5:
            return INTENT_GENERAL, 0.50, {
                "signal": "general_information_fallback",
            }

        return None, 0.0, {"reason": "insufficient_evidence"}
