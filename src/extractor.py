"""
Deterministic Structured Content Extractor
===========================================
Extracts structured metadata fields from OCR text in a non-destructive,
explainable, and strictly safe manner:
- entities (merchants, airlines, technologies, institutions, order IDs)
- dates (with ISO normalization and date typing)
- financial_info (amounts, currency, UPI IDs, transaction IDs, payment methods)
- contacts (emails, phone numbers)
- urls
- action_items (deadlines, reminders, follow-ups)
- sensitive_info (password, payment info, personal contact indicators)
- text_topics
- hierarchical content_type resolution

STRICT SAFETY:
Never extracts or stores passwords, tokens, or raw secrets in cleartext.
Passwords matched in OCR set `contains_password = True` in `sensitive_info`,
and any entity representation is redacted to `[REDACTED_PASSWORD]`.
"""

import re
from typing import Any, Dict, List, Optional, Set

# Month name to number mapping for normalization
MONTH_MAP: Dict[str, str] = {
    "jan": "01", "january": "01",
    "feb": "02", "february": "02",
    "mar": "03", "march": "03",
    "apr": "04", "april": "04",
    "may": "05",
    "jun": "06", "june": "06",
    "jul": "07", "july": "07",
    "aug": "08", "august": "08",
    "sep": "09", "september": "09", "sept": "09",
    "oct": "10", "october": "10",
    "nov": "11", "november": "11",
    "dec": "12", "december": "12",
}

# UPI handle suffixes that look like email addresses but are not personal emails
UPI_HANDLE_SUFFIXES = re.compile(
    r"@(okicici|okaxis|okhdfcbank|oksbi|pthdfc|ptaxis|ptsbi|ybl|paytm|axl|icici|apl|fbl|ibl)$",
    re.IGNORECASE,
)

# Known merchant and brand dictionary for deterministic entity extraction
KNOWN_MERCHANT_NAMES: List[str] = [
    "Amazon", "Flipkart", "Swiggy", "Zomato", "Instamart", "Zepto", "Blinkit",
    "Uber", "Ola", "Myntra", "Nykaa", "Netflix", "Spotify", "MakeMyTrip",
    "Tata Neu", "JioMart", "BookMyShow", "BigBasket", "Dominos", "Dunzo",
]

KNOWN_AIRLINES: List[str] = [
    "IndiGo", "Air India", "SpiceJet", "Vistara", "Akasa Air", "Emirates",
    "Qatar Airways", "Singapore Airlines", "Lufthansa", "British Airways",
]

KNOWN_TECH_TOOLS: List[str] = [
    "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Golang", "Rust",
    "React", "Express", "Node.js", "Django", "FastAPI", "VS Code", "GitHub", "GitLab",
    "Docker", "Kubernetes", "PostgreSQL", "MySQL", "MongoDB", "Redis",
]


class StructuredExtractor:
    """
    Extracts structured fields from raw OCR text in a robust, deterministic,
    explainable, and non-destructive manner.
    """

    def __init__(self):
        pass

    def extract_financial_info(self, ocr_text: Optional[str]) -> Dict[str, Any]:
        """
        Extracts currency amounts, payment keywords, UPI IDs, transaction IDs,
        and payment methods from OCR text.
        """
        if not ocr_text or not str(ocr_text).strip():
            return {}

        # 1. Match currency amounts (e.g. ₹689.00, ₹1,800.00, $50, RS 100, INR 450)
        amounts = re.findall(
            r"(?:₹|RS\.?|INR|\$|EUR|€|GBP|£)\s*[\d,]+(?:\.\d{2})?",
            ocr_text,
            re.IGNORECASE,
        )

        # 2. Match transaction / payment / booking / order IDs
        tx_matches = re.findall(
            r"(?:UPI\s*Transaction\s*ID|Transaction\s*ID|Txn\s*ID|Ref\s*(?:No|ID)?|Booking\s*ID|Order\s*(?:ID|#))[:\s]*([A-Za-z0-9-]+)",
            ocr_text,
            re.IGNORECASE,
        )
        tx_raw = re.findall(
            r"\b(?:pay_[A-Za-z0-9]+)\b",
            ocr_text,
            re.IGNORECASE,
        )
        tx_ids = list(dict.fromkeys(tx_raw + tx_matches))

        # 3. Match UPI handle strings (e.g. sinhasneha158-2@okicici, 9881732186@pthdfc)
        upi_ids = re.findall(r"\b[A-Za-z0-9._-]+@[a-zA-Z]{3,10}\b", ocr_text)
        valid_upi = [u for u in upi_ids if UPI_HANDLE_SUFFIXES.search(u)]

        ocr_lower = ocr_text.lower()
        payment_keywords = {
            "payment", "paid", "upi", "transaction", "receipt", "invoice", "bill",
            "paid successfully", "refund", "paytm", "gpay", "phonepe", "credited",
            "debited", "transfer", "bank", "due date", "subtotal", "grand total",
        }
        has_keywords = any(kw in ocr_lower for kw in payment_keywords)

        payment_method = None
        if "upi" in ocr_lower or len(valid_upi) > 0 or "google pay" in ocr_lower or "phonepe" in ocr_lower or "paytm" in ocr_lower:
            payment_method = "UPI"
        elif "netbanking" in ocr_lower or "neft" in ocr_lower or "rtgs" in ocr_lower or "imps" in ocr_lower:
            payment_method = "NetBanking"
        elif "credit card" in ocr_lower or "debit card" in ocr_lower or "mastercard" in ocr_lower or "visa" in ocr_lower:
            payment_method = "Card"
        elif "wallet" in ocr_lower:
            payment_method = "Wallet"

        # Determine primary parsed amount and currency
        parsed_amount = None
        currency = None
        if amounts:
            first_amt = amounts[0]
            if "₹" in first_amt or "rs" in first_amt.lower() or "inr" in first_amt.lower():
                currency = "INR"
            elif "$" in first_amt:
                currency = "USD"
            elif "€" in first_amt or "eur" in first_amt.lower():
                currency = "EUR"
            elif "£" in first_amt or "gbp" in first_amt.lower():
                currency = "GBP"

            cleaned_num = re.sub(r"[^\d.]", "", first_amt)
            try:
                parsed_amount = float(cleaned_num)
            except Exception:
                parsed_amount = None

        primary_upi = valid_upi[0] if valid_upi else None
        primary_tx = tx_ids[0] if tx_ids else None
        status = "success" if ("success" in ocr_lower or "paid" in ocr_lower or "completed" in ocr_lower) else None

        res: Dict[str, Any] = {
            "amounts": list(dict.fromkeys(amounts)),
            "transaction_ids": tx_ids,
            "upi_ids": list(dict.fromkeys(valid_upi)),
            "has_payment_keywords": has_keywords,
            "payment_method": payment_method,
        }
        if parsed_amount is not None:
            res["amount"] = parsed_amount
        if currency:
            res["currency"] = currency
        if primary_upi:
            res["upi_id"] = primary_upi
        if primary_tx:
            res["transaction_id"] = primary_tx
        if status:
            res["status"] = status

        return res

    def extract_dates(self, ocr_text: Optional[str]) -> List[Dict[str, str]]:
        """
        Extracts recognizable dates from OCR text with ISO normalization
        and semantic date typing.
        """
        if not ocr_text or not str(ocr_text).strip():
            return []

        # Common date regex patterns
        patterns = [
            # 15th Apr, 2026 or 15 Apr 2026 or 15 April 2026
            (r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)[,\s]+(\d{4})\b", "named_month_year"),
            # 2026-09-22 or 2026/09/22
            (r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", "iso"),
            # 22/09/2026 or 22-09-2026
            (r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", "dmy"),
        ]

        ocr_lower = ocr_text.lower()
        date_type = (
            "transaction_date"
            if any(k in ocr_lower for k in ["paid", "payment", "txn", "invoice", "receipt"])
            else (
                "travel_date"
                if any(k in ocr_lower for k in ["flight", "boarding", "departure", "train", "pnr"])
                else (
                    "due_date"
                    if any(k in ocr_lower for k in ["due date", "bill due", "pay before", "deadline"])
                    else "event_date"
                )
            )
        )

        dates_found: List[Dict[str, str]] = []
        seen: Set[str] = set()

        for pattern, p_type in patterns:
            for match in re.finditer(pattern, ocr_text, re.IGNORECASE):
                raw = match.group(0).strip()
                if raw.lower() in seen:
                    continue
                seen.add(raw.lower())

                # Attempt normalization
                normalized = None
                try:
                    if p_type == "named_month_year":
                        d_str, m_str, y_str = match.groups()
                        m_clean = m_str.lower()
                        if m_clean in MONTH_MAP:
                            m_num = MONTH_MAP[m_clean]
                            d_num = f"{int(d_str):02d}"
                            normalized = f"{y_str}-{m_num}-{d_num}"
                    elif p_type == "iso":
                        y_str, m_str, d_str = match.groups()
                        normalized = f"{int(y_str):04d}-{int(m_str):02d}-{int(d_str):02d}"
                    elif p_type == "dmy":
                        d_str, m_str, y_str = match.groups()
                        d_val, m_val = int(d_str), int(m_str)
                        if 1 <= d_val <= 31 and 1 <= m_val <= 12:
                            normalized = f"{int(y_str):04d}-{m_val:02d}-{d_val:02d}"
                except Exception:
                    normalized = None

                dates_found.append({
                    "raw_text": raw,
                    "text": normalized if normalized else raw,
                    "normalized": normalized if normalized else raw,
                    "type": date_type,
                })

        return dates_found

    def extract_contacts(self, ocr_text: Optional[str]) -> Dict[str, List[str]]:
        """
        Extracts email addresses and phone numbers.
        Carefully filters out UPI handles that look like email addresses (e.g. sinha@okaxis).
        """
        if not ocr_text or not str(ocr_text).strip():
            return {"emails": [], "phones": []}

        # Email regex
        raw_emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", ocr_text)
        # Filter out UPI handles
        emails = [e for e in raw_emails if not UPI_HANDLE_SUFFIXES.search(e)]

        # Phone regex (Indian/international standard: +91..., 10-digit mobile, 1800 toll free)
        phone_matches = re.findall(
            r"(?:\+91[\s-]?)?[6-9]\d{9}|1800[\s-]?\d{3}[\s-]?\d{3,4}",
            ocr_text,
        )

        return {
            "emails": list(dict.fromkeys(emails)),
            "phones": list(dict.fromkeys([p.strip() for p in phone_matches])),
        }

    def extract_urls(self, ocr_text: Optional[str]) -> List[str]:
        """Extracts web links and domain URLs."""
        if not ocr_text or not str(ocr_text).strip():
            return []

        urls = re.findall(
            r"\bhttps?://[^\s<>\"']+|www\.[^\s<>\"']+",
            ocr_text,
            re.IGNORECASE,
        )
        return list(dict.fromkeys([u.rstrip(".,;:)") for u in urls]))

    def detect_sensitive_info(self, ocr_text: Optional[str]) -> Dict[str, bool]:
        """
        Detects whether the screenshot contains sensitive information
        (passwords, payment card info, personal contacts).
        """
        if not ocr_text or not str(ocr_text).strip():
            return {
                "contains_password": False,
                "contains_payment_info": False,
                "contains_personal_contact": False,
            }

        ocr_lower = ocr_text.lower()

        # Password / secret indicators
        password_keywords = [
            "password", "show password", "wifi password", "wi-fi password",
            "passcode", "secret key", "api key", "pin: ", "otp: ",
        ]
        contains_password = any(kw in ocr_lower for kw in password_keywords) or bool(
            re.search(r"password\s*[:\n]\s*[^\n]+", ocr_text, re.IGNORECASE)
        )

        # Payment info indicators
        payment_keywords = [
            "paid successfully", "payment id:", "upi id", "card number",
            "cvv", "expiry date", "bank reference", "transaction id",
        ]
        fin = self.extract_financial_info(ocr_text)
        contains_payment_info = (
            any(kw in ocr_lower for kw in payment_keywords)
            or len(fin.get("amounts", [])) > 0
            or len(fin.get("upi_ids", [])) > 0
        )

        # Personal contact indicators
        contacts = self.extract_contacts(ocr_text)
        contains_personal_contact = len(contacts["emails"]) > 0 or len(contacts["phones"]) > 0

        return {
            "contains_password": contains_password,
            "contains_payment_info": contains_payment_info,
            "contains_personal_contact": contains_personal_contact,
        }

    def extract_entities(self, ocr_text: Optional[str], category: str = "General") -> List[Dict[str, str]]:
        """
        Extracts recognizable named entities (merchants, airlines, institutions,
        technologies, events, order IDs).
        STRICT SAFETY: NEVER extracts passwords, authentication tokens, or raw secrets
        as cleartext entities.
        """
        if not ocr_text or not str(ocr_text).strip():
            return []

        entities: List[Dict[str, str]] = []
        seen: Set[str] = set()

        def add_entity(text: str, ent_type: str):
            cleaned = text.strip()
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                entities.append({
                    "text": cleaned,
                    "value": cleaned,
                    "type": ent_type,
                    "label": ent_type,
                })

        ocr_lower = ocr_text.lower()

        # Strict safety check: if password detected, provide redacted placeholder only
        sens = self.detect_sensitive_info(ocr_text)
        if sens.get("contains_password"):
            add_entity("[REDACTED_PASSWORD]", "PASSWORD")

        # 1. Order IDs
        for match in re.finditer(r"(?:Order\s*(?:#|ID|Number)?[:\s]*)([A-Za-z0-9-]+)", ocr_text, re.IGNORECASE):
            val = match.group(1).strip()
            if val and len(val) >= 4 and not val.lower().startswith("id"):
                add_entity(val, "ORDER_ID")

        # 2. Merchants & Brands
        for merchant in KNOWN_MERCHANT_NAMES:
            if merchant.lower() in ocr_lower:
                add_entity(merchant, "merchant")

        # 3. Airlines
        for airline in KNOWN_AIRLINES:
            if airline.lower() in ocr_lower:
                add_entity(airline, "airline")

        # 4. Tech stack & developer tools
        for tool in KNOWN_TECH_TOOLS:
            pattern = rf"\b{re.escape(tool.lower())}\b"
            if re.search(pattern, ocr_lower):
                add_entity(tool, "technology")

        # 5. Academic / Institution entities
        if "kiit" in ocr_lower:
            add_entity("KIIT", "institution")
        if "iit" in ocr_lower:
            add_entity("IIT", "institution")
        if "deep learning" in ocr_lower:
            add_entity("Deep Learning", "subject")

        # 6. Messaging apps
        if "whatsapp" in ocr_lower:
            add_entity("WhatsApp", "application")
        elif "telegram" in ocr_lower:
            add_entity("Telegram", "application")

        # 7. Events
        if "hackathon" in ocr_lower:
            add_entity("Hackathon", "event")

        return entities

    def extract_events(
        self,
        ocr_text: Optional[str],
        financial_info: Optional[Dict[str, Any]] = None,
        dates: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Extracts structured event objects from accumulated evidence."""
        if not ocr_text or not str(ocr_text).strip():
            return []

        ocr_lower = ocr_text.lower()
        events: List[Dict[str, Any]] = []

        if financial_info and (financial_info.get("has_payment_keywords") or len(financial_info.get("amounts", [])) > 0):
            events.append({
                "type": "payment",
                "status": "successful" if "paid" in ocr_lower or "success" in ocr_lower else "pending",
                "details": "Financial transaction / payment receipt",
            })

        if any(w in ocr_lower for w in ["boarding pass", "flight", "terminal", "gate", "departure"]):
            events.append({
                "type": "travel",
                "status": "confirmed",
                "details": "Flight or transit travel booking",
            })

        if "hackathon" in ocr_lower or "webinar" in ocr_lower or "conference" in ocr_lower:
            events.append({
                "type": "event",
                "status": "scheduled",
                "details": "Conference, hackathon, or webinar event",
            })

        return events

    def extract_action_items(self, ocr_text: Optional[str]) -> List[Dict[str, str]]:
        """Identifies action items or pending steps implied by the screenshot."""
        if not ocr_text or not str(ocr_text).strip():
            return []

        ocr_lower = ocr_text.lower()
        action_items: List[Dict[str, str]] = []

        if "due date" in ocr_lower or "pay before" in ocr_lower:
            action_items.append({"text": "Pay due bill or invoice before deadline", "status": "pending"})
        elif "report to airport" in ocr_lower or "boarding gate" in ocr_lower or "boarding time" in ocr_lower:
            action_items.append({"text": "Report to airport boarding gate on time", "status": "pending"})
        elif "unread message" in ocr_lower or "reply" in ocr_lower:
            action_items.append({"text": "Reply to pending messages", "status": "pending"})
        elif "hackathon" in ocr_lower and ("register" in ocr_lower or "apply" in ocr_lower):
            action_items.append({"text": "Complete event registration before deadline", "status": "pending"})
        elif "assignment" in ocr_lower and "submission" in ocr_lower:
            action_items.append({"text": "Submit academic assignment on time", "status": "pending"})

        return action_items

    def extract_text_topics(self, ocr_text: Optional[str]) -> List[str]:
        """Extracts salient topic keywords from OCR text."""
        if not ocr_text or not str(ocr_text).strip():
            return []

        ocr_lower = ocr_text.lower()
        topics: List[str] = []

        topic_keywords = {
            "wifi": "connectivity",
            "wi-fi": "connectivity",
            "password": "credentials",
            "upi": "payment",
            "paid": "payment",
            "bill": "billing",
            "syllabus": "academics",
            "exam": "academics",
            "flight": "travel",
            "boarding": "travel",
            "express": "coding",
            "python": "coding",
            "javascript": "coding",
            "traceback": "error",
            "exception": "error",
            "error": "error",
            "git": "coding",
            "terminal": "terminal",
            "hackathon": "event",
            "whatsapp": "messaging",
            "order": "shopping",
            "cart": "shopping",
            "menu": "food",
        }

        for kw, topic in topic_keywords.items():
            if kw in ocr_lower and topic not in topics:
                topics.append(topic)

        return topics

    def resolve_content_type(
        self,
        category: str,
        ocr_text: str,
        financial_info: Dict[str, Any],
        events: List[Dict[str, Any]],
    ) -> str:
        """Derives specific content_type within the category hierarchy from accumulated evidence."""
        if not ocr_text or not str(ocr_text).strip():
            return "unknown"

        ocr_lower = ocr_text.lower()

        if category == "Connectivity":
            return "wifi_credentials" if ("password" in ocr_lower or "wifi" in ocr_lower or "ssid" in ocr_lower) else "network_settings"
        elif category == "Payment":
            if "bill" in ocr_lower or "electricity" in ocr_lower or "due date" in ocr_lower:
                return "bill"
            elif "receipt" in ocr_lower or "paid" in ocr_lower or "success" in ocr_lower:
                return "payment_receipt"
            return "transaction"
        elif category == "Education":
            if "exam" in ocr_lower or "schedule" in ocr_lower or "timetable" in ocr_lower:
                return "exam_schedule"
            elif "syllabus" in ocr_lower or "unit" in ocr_lower:
                return "course_material"
            return "notes"
        elif category == "Shopping":
            if "order" in ocr_lower or "invoice" in ocr_lower or "order #" in ocr_lower:
                return "shopping_order"
            elif "delivery" in ocr_lower or "track" in ocr_lower:
                return "delivery_tracking"
            return "product_page"
        elif category == "Travel":
            if "flight" in ocr_lower or "boarding" in ocr_lower or "airline" in ocr_lower:
                return "flight_ticket"
            elif "train" in ocr_lower or "irctc" in ocr_lower or "pnr" in ocr_lower:
                return "train_ticket"
            elif "hotel" in ocr_lower or "stay" in ocr_lower:
                return "hotel_booking"
            return "travel_itinerary"
        elif category == "Communication":
            return "chat" if ("whatsapp" in ocr_lower or "message" in ocr_lower or "chat" in ocr_lower) else "email"
        elif category == "Work & Career":
            if "traceback" in ocr_lower or "exception" in ocr_lower or "error" in ocr_lower:
                return "error_log"
            elif any(k in ocr_lower for k in ["def ", "class ", "function", "import ", "const ", "let "]):
                return "code_editor"
            elif "git " in ocr_lower or "terminal" in ocr_lower or "bash" in ocr_lower:
                return "terminal"
            elif "resume" in ocr_lower or "curriculum vitae" in ocr_lower:
                return "resume"
            elif "job" in ocr_lower or "hiring" in ocr_lower or "interview" in ocr_lower:
                return "job_posting"
            return "documentation"
        elif category == "Entertainment":
            return "movie" if "movie" in ocr_lower else "video"

        return "unknown"

    def extract_all(self, ocr_text: Optional[str], category: str = "General") -> Dict[str, Any]:
        """
        Executes the full suite of structured extractors and returns a clean dictionary
        ready for ScreenshotMetadata initialization.
        """
        if not ocr_text or not str(ocr_text).strip():
            return {
                "financial_info": {},
                "dates": [],
                "contacts": {"emails": [], "phones": []},
                "urls": [],
                "sensitive_info": {
                    "contains_password": False,
                    "contains_payment_info": False,
                    "contains_personal_contact": False,
                },
                "entities": [],
                "events": [],
                "action_items": [],
                "text_topics": [],
                "content_type": "unknown",
                "summary": "General screenshot with no detected text.",
            }

        financial_info = self.extract_financial_info(ocr_text)
        dates = self.extract_dates(ocr_text)
        contacts = self.extract_contacts(ocr_text)
        urls = self.extract_urls(ocr_text)
        sensitive_info = self.detect_sensitive_info(ocr_text)
        entities = self.extract_entities(ocr_text, category=category)
        events = self.extract_events(ocr_text, financial_info=financial_info, dates=dates)
        action_items = self.extract_action_items(ocr_text)
        text_topics = self.extract_text_topics(ocr_text)
        content_type = self.resolve_content_type(category, ocr_text, financial_info, events)

        # Generate a concise deterministic summary
        ent_names = [e["text"] for e in entities if e["text"] != "[REDACTED_PASSWORD]"]
        amt_str = f" of {financial_info['amounts'][0]}" if financial_info.get("amounts") else ""
        if category == "Payment" and financial_info.get("has_payment_keywords"):
            summary = f"Payment transaction{amt_str}."
        elif category == "Travel" and any(e.get("type") == "airline" for e in entities):
            airline = next(e["text"] for e in entities if e.get("type") == "airline")
            summary = f"{airline} travel booking confirmation."
        elif category == "Connectivity" and sensitive_info.get("contains_password"):
            summary = "Wi-Fi or network access credentials."
        elif ent_names:
            summary = f"{category} screenshot relating to {', '.join(ent_names[:3])}."
        else:
            summary = f"{category} screenshot."

        return {
            "financial_info": financial_info,
            "dates": dates,
            "contacts": contacts,
            "urls": urls,
            "sensitive_info": sensitive_info,
            "entities": entities,
            "events": events,
            "action_items": action_items,
            "text_topics": text_topics,
            "content_type": content_type,
            "summary": summary,
        }
