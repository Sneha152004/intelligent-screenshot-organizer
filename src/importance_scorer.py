"""
Importance Scorer Module
========================
Provides deterministic, explainable importance scoring (1-5) for screenshots
based on content, extracted signals, intent, and usefulness to the user.

Score interpretation:
- 5: Critical or highly useful information (credentials, flight tickets, deadlines, payment proofs)
- 4: Important actionable information (appointments, bookings, bills, action items)
- 3: Useful reference information (educational material, receipts, work references, code)
- 2: General useful information (personal chats, entertainment, device settings)
- 1: Low-value or miscellaneous screenshots (blank, accidental, minimal utility)
"""

import re
from typing import Any, Dict, List, Optional, Tuple


class ImportanceScorer:
    """
    Deterministic, rule-based importance scoring engine.
    Calculates an explainable integer score between 1 and 5.
    """

    # High-confidence credentials & secrets regex
    CREDENTIALS_PATTERN = re.compile(
        r'\b(?:password|passwd|pwd|api[_\s]?key|secret[_\s]?key|private[_\s]?key|'
        r'access[_\s]?token|auth[_\s]?token|bearer\s+[a-zA-Z0-9_\-\.]{15,}|'
        r'recovery[_\s]?(?:phrase|words|key)|seed[_\s]?phrase|'
        r'client[_\s]?secret|ssh-rsa\s+[a-zA-Z0-9+/=]+|'
        r'your\s+(?:one[_\s-]?time\s+password|otp)\s+(?:is|code)|'
        r'verification\s+code\s*[:\s-]\s*\d{4,8})\b',
        re.I,
    )

    # Flight ticket & boarding pass regex
    FLIGHT_PATTERN = re.compile(
        r'\b(?:boarding\s*pass|flight\s*ticket|e-ticket|flight\s*booking|'
        r'flight\s*no|pnr\b|gate\s*[a-z0-9]|terminal\s*\d|seat\s*\d+[a-z]|'
        r'boarding\s*time|departure\s*date|airline|indigo|air\s*india|spicejet|'
        r'vistara|emirates|lufthansa|boarding\s*group)\b',
        re.I,
    )

    # Strict deadlines & due dates
    DEADLINE_PATTERN = re.compile(
        r'\b(?:submission\s*deadline|application\s*deadline|payment\s*due\s*date|'
        r'due\s*date\s*[:\s]|strictly\s*before|expires\s*(?:on|today|tomorrow)|'
        r'last\s*date\s*(?:for|to|of)|deadline\s*[:\s]|final\s*date\s*for\s*submission|'
        r'penalty\s*after|cutoff\s*date)\b',
        re.I,
    )

    # Completed transaction & payment verification
    PAYMENT_PROOF_PATTERN = re.compile(
        r'\b(?:payment\s*(?:successful|received|completed|done)|'
        r'paid\s*successfully|transaction\s*(?:successful|completed)|'
        r'debited\s*(?:by|from)|credited\s*(?:to|by)|transfer\s*successful|'
        r'order\s*placed\s*successfully|bill\s*paid|'
        r'upi\s*ref(?:erence)?(?:\s*no|\s*id)?|utr\s*(?:number|no)?\s*[:\s]?\d+|'
        r'transaction\s*id\s*[:\s]?[a-zA-Z0-9_\-]+)\b',
        re.I,
    )

    # Actionable bookings & appointments (non-flight)
    BOOKING_APPOINTMENT_PATTERN = re.compile(
        r'\b(?:appointment\s*(?:confirmed|booked|scheduled)|doctor\s*appointment|'
        r'consultation\s*confirmed|hotel\s*(?:booking|reservation|check-in)|'
        r'train\s*ticket|irctc|pnr\s*status|bus\s*ticket|cab\s*booking|'
        r'check-in\s*time|reservation\s*id|booking\s*id)\b',
        re.I,
    )

    # Actionable bills & invoices
    BILL_INVOICE_PATTERN = re.compile(
        r'\b(?:electricity\s*bill|water\s*bill|broadband\s*bill|wifi\s*bill|'
        r'utility\s*bill|credit\s*card\s*statement|amount\s*payable|'
        r'pay\s*by\s*date|total\s*due|minimum\s*amount\s*due|tax\s*invoice|'
        r'bill\s*amount\s*[:\s]?₹?\$?\d+)\b',
        re.I,
    )

    # Official identification & certificates
    OFFICIAL_DOC_PATTERN = re.compile(
        r'\b(?:aadhaar|pan\s*card|passport|driving\s*licen[sc]e|voter\s*id|'
        r'identity\s*card|birth\s*certificate|degree\s*certificate|'
        r'income\s*tax\s*return|itr-v|government\s*of\s*india)\b',
        re.I,
    )

    # Action items & calendar meetings
    ACTION_ITEM_PATTERN = re.compile(
        r'\b(?:action\s*items?|to-do|todo\s*list|meeting\s*agenda|calendar\s*invite|'
        r'reminder\s*[:\s]|zoom\.us/j|meet\.google\.com|teams\.microsoft\.com)\b',
        re.I,
    )

    # Educational notes & study material
    EDUCATION_PATTERN = re.compile(
        r'\b(?:lecture|syllabus|course\s*work|textbook|theorem|definition\s*:|'
        r'algorithm\s*:|proof\s*:|study\s*guide|chapter\s*\d+|homework|'
        r'exam\s*preparation|quiz|tutorial|formula)\b',
        re.I,
    )

    # Technical code snippets & error traces
    TECHNICAL_PATTERN = re.compile(
        r'\b(?:def\s+\w+\(|class\s+\w+[\(:]|public\s+static\s+void|'
        r'import\s+\w+|from\s+\w+\s+import|console\.log|'
        r'traceback\s*\(most\s*recent\s*call\s*last\)|'
        r'syntaxerror|typeerror|valueerror|nullpointerexception|'
        r'dockerfile|docker-compose|git\s+commit|kubectl|npm\s+run)\b',
        re.I,
    )

    def score_screenshot(
        self,
        ocr_text: Optional[str] = "",
        category: Optional[str] = None,
        intent: Optional[str] = None,
        content_type: Optional[str] = None,
        financial_info: Optional[Dict[str, Any]] = None,
        dates: Optional[List[Any]] = None,
        entities: Optional[List[Any]] = None,
        sensitive_info: Optional[Dict[str, Any]] = None,
        action_items: Optional[List[Any]] = None,
        urls: Optional[List[str]] = None,
        contacts: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, str]:
        """
        Evaluate extracted signals and return (importance_score, importance_reason).
        The score is strictly an integer from 1 to 5.
        """
        text = str(ocr_text or "").strip()
        text_clean = " ".join(text.split())
        c_type = str(content_type or "").lower().strip()
        cat = str(category or "").strip()
        int_str = str(intent or "").strip()
        fin = financial_info if isinstance(financial_info, dict) else {}
        sens = sensitive_info if isinstance(sensitive_info, dict) else {}
        act_items = action_items if isinstance(action_items, list) else []

        # ======================================================================
        # Check for empty / minimal OCR text first (boundary case)
        # ======================================================================
        if len(text_clean) < 10 and not any([fin.get("amounts"), fin.get("amount"), act_items]):
            return 1, "Low-value: Minimal or blank visual content with no actionable text"

        # ======================================================================
        # LEVEL 5: Critical or Highly Useful Information
        # ======================================================================

        # 1. Credentials / Sensitive Security Info
        if (
            sens.get("has_password")
            or sens.get("has_credentials")
            or sens.get("has_key")
            or sens.get("has_auth_token")
            or self.CREDENTIALS_PATTERN.search(text_clean)
        ):
            return 5, "Critical security credentials, password, or sensitive access keys detected"

        # 2. Flight Tickets & Boarding Passes
        if (
            c_type in ("flight_ticket", "boarding_pass")
            or (
                (cat == "Travel" or int_str in ("Travel / Navigation", "Booking / Reservation"))
                and self.FLIGHT_PATTERN.search(text_clean)
            )
            or (self.FLIGHT_PATTERN.search(text_clean) and re.search(r'\b(flight|airline|pnr|seat)\b', text_clean, re.I))
        ):
            return 5, "Critical travel document: Flight ticket or boarding pass with departure details"

        # 3. Strict Deadlines & Expirations
        if (
            self.DEADLINE_PATTERN.search(text_clean)
            or any(self.DEADLINE_PATTERN.search(str(a)) for a in act_items)
            or any(
                isinstance(d, dict) and d.get("date_type") in ("deadline", "due_date")
                for d in (dates or [])
            )
        ):
            return 5, "Critical upcoming deadline or strict time-sensitive expiration requirement"

        # 4. Important Verified Payment Proofs
        has_tx_proof = bool(fin.get("transaction_ids")) or bool(fin.get("upi_ids"))
        has_amount = fin.get("amount") is not None or bool(fin.get("amounts"))
        payment_proof_matched = bool(self.PAYMENT_PROOF_PATTERN.search(text_clean))

        if (
            (c_type in ("payment_receipt", "transaction_receipt") or cat == "Payment" or int_str == "Financial / Payment")
            and (payment_proof_matched or (has_tx_proof and has_amount))
        ):
            return 5, "Critical financial record: Verified payment receipt with transaction confirmation"

        if payment_proof_matched and (has_tx_proof or has_amount):
            return 5, "Critical financial record: Verified payment receipt with transaction confirmation"

        # ======================================================================
        # LEVEL 4: Important Actionable Information
        # ======================================================================

        # 1. Actionable Bookings & Appointments
        if (
            c_type in ("booking_confirmation", "hotel_booking", "train_ticket", "appointment")
            or self.BOOKING_APPOINTMENT_PATTERN.search(text_clean)
        ):
            return 4, "Important actionable booking or scheduled appointment confirmation"

        # 2. Actionable Bills & Invoices
        if (
            c_type in ("invoice", "utility_bill")
            or self.BILL_INVOICE_PATTERN.search(text_clean)
        ):
            return 4, "Important actionable billing statement or invoice"

        # 3. Action Items & Calendar Meetings
        if act_items or self.ACTION_ITEM_PATTERN.search(text_clean):
            return 4, "Important actionable task, meeting schedule, or to-do reminder"

        # 4. Official IDs & Verified Legal Documents
        if self.OFFICIAL_DOC_PATTERN.search(text_clean):
            return 4, "Important official identification or verified legal document"

        # ======================================================================
        # LEVEL 3: Useful Reference Information
        # ======================================================================

        # 1. Educational Material & Study References
        if (
            cat == "Education"
            or c_type in ("educational_notes", "study_material", "tutorial", "documentation")
            or int_str == "Learning / Education"
            or self.EDUCATION_PATTERN.search(text_clean)
        ):
            return 3, "Useful reference: Educational notes, tutorial, or study material"

        # 2. Shopping Orders & Purchases (Reference history)
        if (
            cat == "Shopping"
            or c_type == "shopping_order"
            or int_str == "Shopping / Purchase"
        ):
            return 3, "Useful reference: Shopping order or purchase history record"

        # 3. Work & Career Documentation
        if (
            cat == "Work & Career"
            or c_type in ("resume", "job_listing", "work_document")
            or re.search(r'\b(resume|curriculum\s*vitae|job\s*description|performance\s*review|project\s*spec)\b', text_clean, re.I)
        ):
            return 3, "Useful reference: Work documentation, career, or project record"

        # 4. Technical Code Snippets & Error Logs
        if (
            cat in ("Technical & Development", "Coding")
            or c_type in ("code_snippet", "error_log")
            or self.TECHNICAL_PATTERN.search(text_clean)
        ):
            return 3, "Useful reference: Technical code snippet, system log, or debugging trace"

        # 5. General Financial Mentions without full proof
        if cat == "Payment" or has_amount:
            return 3, "Useful reference: General financial summary or transaction record"

        # ======================================================================
        # LEVEL 2: General Useful Information
        # ======================================================================

        # 1. Communication & Social Chats
        if cat in ("Communication", "Social Media & Chats") or int_str in ("Social / Chat", "Communication / Chat"):
            return 2, "General information: Personal chat message or social conversation"

        # 2. Entertainment & Media
        if cat == "Entertainment" or int_str in ("Entertainment / Media"):
            return 2, "General information: Media recommendation, game, or entertainment reference"

        # 3. Connectivity & Device Settings
        if cat == "Connectivity" or re.search(r'\b(wi-fi|wifi|bluetooth|cellular|data\s*usage|battery\s*percentage|device\s*settings)\b', text_clean, re.I):
            return 2, "General information: Device settings or network connectivity record"

        # 4. General Travel Exploration
        if cat == "Travel":
            return 2, "General information: Travel browsing, destination, or map record"

        # ======================================================================
        # LEVEL 1: Low-Value / Miscellaneous
        # ======================================================================
        return 1, "Low-value: Miscellaneous or low-utility screenshot without specific reference value"
