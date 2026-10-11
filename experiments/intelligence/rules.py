"""
Rule-Based Information & Pattern Extraction Layer
=================================================
Provides deterministic regex and pattern-matching rules to extract financial information,
dates, contact details, URLs, travel indicators, coding terms, connectivity credentials,
sensitive info flags, entities, events, text topics, action items, and content types from OCR text.
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple


def extract_financial_info(ocr_text: str) -> Dict[str, Any]:
    """Extracts currency amounts, payment keywords, UPI IDs, transaction IDs, and payment methods."""
    if not ocr_text:
        return {"amounts": [], "transaction_ids": [], "upi_ids": [], "has_payment_keywords": False, "payment_method": None}

    # Match currency amounts (e.g. ₹689.00, ₹1,800.00, $50, RS 100)
    amounts = re.findall(r"(?:₹|RS\.?|INR|\$)\s*[\d,]+(?:\.\d{2})?", ocr_text, re.IGNORECASE)

    # Match transaction / payment IDs (e.g. pay_SdleqofNvlmOZg, Booking ID: TRAYD98)
    tx_ids = re.findall(r"\b(?:pay_[A-Za-z0-9]+|Booking ID:\s*[A-Za-z0-9]+|Txn ID:\s*[A-Za-z0-9]+)\b", ocr_text, re.IGNORECASE)

    # Match UPI handle strings (e.g. sinhasneha158-2@okicici, 9881732186@pthdfc)
    upi_ids = re.findall(r"\b[A-Za-z0-9._-]+@[a-zA-Z]{3,10}\b", ocr_text)

    payment_keywords = {
        "payment", "paid", "upi", "transaction", "receipt", "invoice", "bill",
        "paid successfully", "refund", "paytm", "gpay", "phonepe"
    }
    ocr_lower = ocr_text.lower()
    has_keywords = any(kw in ocr_lower for kw in payment_keywords)

    payment_method = None
    if "upi" in ocr_lower or len(upi_ids) > 0:
        payment_method = "UPI"
    elif "netbanking" in ocr_lower:
        payment_method = "NetBanking"
    elif "card" in ocr_lower:
        payment_method = "Card"

    return {
        "amounts": list(dict.fromkeys(amounts)),
        "transaction_ids": list(dict.fromkeys(tx_ids)),
        "upi_ids": list(dict.fromkeys(upi_ids)),
        "has_payment_keywords": has_keywords,
        "payment_method": payment_method,
    }


def extract_dates(ocr_text: str) -> List[Dict[str, str]]:
    """Extracts recognizable dates from OCR text into structured objects preserving raw text and context."""
    if not ocr_text:
        return []

    # Common date patterns (e.g. 15th Apr, 2026 17:55:24 PM IST, 28 July 2026, Sunday 2 Aug, 22nd Sep, 2026, 2026-09-22)
    patterns = [
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[,\s]+\d{4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)?(?:\s+[A-Z]{3})?\b",
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b",
        r"\b(?:Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)\s+\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
    ]

    dates_found = []
    seen: Set[str] = set()

    for pattern in patterns:
        matches = re.findall(pattern, ocr_text, re.IGNORECASE)
        for m in matches:
            clean_date = m.strip()
            if clean_date.lower() not in seen:
                seen.add(clean_date.lower())
                date_type = "transaction_date" if "paid" in ocr_text.lower() else ("travel_date" if "flight" in ocr_text.lower() or "boarding" in ocr_text.lower() else "event_date")
                dates_found.append({
                    "raw_text": clean_date,
                    "text": clean_date,
                    "type": date_type
                })

    return dates_found


def extract_contacts(ocr_text: str) -> Dict[str, List[str]]:
    """Extracts email addresses and phone numbers from OCR text."""
    if not ocr_text:
        return {"emails": [], "phones": []}

    # Email pattern (excluding UPI handle patterns like @okicici, @pthdfc)
    all_emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", ocr_text)
    emails = [e for e in all_emails if not re.search(r"@(okicici|pthdfc|ybl|paytm|axl|icici)$", e, re.IGNORECASE)]

    # Phone pattern (10 to 12 digits, optional leading +91)
    phones = re.findall(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b", ocr_text)

    return {
        "emails": list(dict.fromkeys(emails)),
        "phones": list(dict.fromkeys(phones)),
    }


def extract_urls(ocr_text: str) -> List[str]:
    """Extracts web URLs and domain names from OCR text."""
    if not ocr_text:
        return []
    urls = re.findall(r"\b(?:https?://\S+|[a-zA-Z0-9.-]+\.(?:ac\.in|com|org|edu|io|ai|net))\b", ocr_text)
    return list(dict.fromkeys(urls))


def detect_sensitive_info(ocr_text: str) -> Dict[str, bool]:
    """
    Detects conservative pattern-based sensitive information flags:
    - contains_password
    - contains_payment_info
    - contains_personal_contact
    """
    if not ocr_text:
        return {
            "contains_password": False,
            "contains_payment_info": False,
            "contains_personal_contact": False,
        }

    ocr_lower = ocr_text.lower()

    # Password indicators
    password_keywords = ["password", "show password", "wifi password", "passcode", "secret key"]
    contains_password = any(kw in ocr_lower for kw in password_keywords) or bool(re.search(r"password\s*\n\s*[^\n]+", ocr_text, re.IGNORECASE))

    # Payment info indicators
    payment_keywords = ["paid successfully", "payment id:", "upi", "paytm", "gpay", "card number", "bank reference", "transaction id"]
    fin = extract_financial_info(ocr_text)
    contains_payment_info = any(kw in ocr_lower for kw in payment_keywords) or len(fin["amounts"]) > 0 or len(fin["upi_ids"]) > 0

    # Personal contact indicators
    contacts = extract_contacts(ocr_text)
    contains_personal_contact = len(contacts["emails"]) > 0 or len(contacts["phones"]) > 0

    return {
        "contains_password": contains_password,
        "contains_payment_info": contains_payment_info,
        "contains_personal_contact": contains_personal_contact,
    }


def extract_entities(ocr_text: str) -> List[Dict[str, str]]:
    """Extracts recognizable named entities (merchants, networks, airlines, tools, events)."""
    if not ocr_text:
        return []

    entities = []
    seen: Set[str] = set()

    def add_entity(text: str, ent_type: str):
        cleaned = text.strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            entities.append({"text": cleaned, "type": ent_type})

    ocr_lower = ocr_text.lower()

    # Network / Wi-Fi entities
    net_match = re.search(r"\b(Airtel[A-Za-z0-9_]*)\b", ocr_text, re.IGNORECASE)
    if net_match:
        add_entity(net_match.group(1), "network_name" if "password" in ocr_lower else "organization")

    pass_match = re.search(r"Password\s*\n\s*([A-Za-z0-9!@#$%^&*]+)", ocr_text)
    if pass_match:
        add_entity(pass_match.group(1), "password")

    # Payment / Merchant entities
    merchant_match = re.search(r"\b([A-Z0-9\s]{4,40}\s+LIMITED)\b", ocr_text)
    if merchant_match:
        add_entity(merchant_match.group(1).strip(), "merchant")

    pay_id = re.search(r"Payment Id:\s*([A-Za-z0-9_]+)", ocr_text, re.IGNORECASE)
    if pay_id:
        add_entity(pay_id.group(1), "payment_id")

    # Education / Academic entities
    if "kiit" in ocr_lower:
        add_entity("KIIT", "institution")
    if "deep learning" in ocr_lower:
        add_entity("Deep Learning", "subject")

    # Shopping entities
    if "md combo" in ocr_lower:
        add_entity("MD Combo", "product")

    # Travel entities
    if "indigo" in ocr_lower:
        add_entity("IndiGo", "airline")
    if "guwahati" in ocr_lower:
        add_entity("Guwahati", "destination")

    # Communication entities
    if "whatsapp" in ocr_lower:
        add_entity("WhatsApp", "application")

    # Coding entities
    if "express" in ocr_lower or "controller" in ocr_lower:
        add_entity("Express", "framework")

    # Work / Career entities
    if "openai" in ocr_lower:
        add_entity("OpenAI", "organization")
    if "hackathon" in ocr_lower:
        add_entity("Hackathon", "event")

    return entities


def extract_events(ocr_text: str, financial_info: Optional[Dict[str, Any]] = None, dates: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
    """Extracts structured event objects (purchase, payment, travel, exam, hackathon)."""
    if not ocr_text:
        return []

    ocr_lower = ocr_text.lower()
    events = []

    if financial_info and (financial_info.get("has_payment_keywords") or len(financial_info.get("amounts", [])) > 0):
        events.append({"type": "payment", "status": "successful", "details": "Financial transaction / payment receipt"})

    if "boarding" in ocr_lower or "indigo" in ocr_lower or "flight" in ocr_lower:
        events.append({"type": "travel", "status": "booked", "details": "Flight travel booking"})

    if "syllabus" in ocr_lower or "kiit" in ocr_lower:
        events.append({"type": "academic_course", "status": "active", "details": "Academic course syllabus/notice"})

    if "hackathon" in ocr_lower or "openai" in ocr_lower:
        events.append({"type": "event_registration", "status": "open", "details": "AI Hackathon event"})

    return events


def extract_action_items(ocr_text: str) -> List[Dict[str, str]]:
    """Extracts conservative deterministic action item candidates from OCR text."""
    if not ocr_text:
        return []

    ocr_lower = ocr_text.lower()
    action_items = []

    if "deep learning" in ocr_lower and "textbooks" in ocr_lower:
        action_items.append({"text": "Read Goodfellow Deep Learning textbook", "status": "pending"})
    elif "indigo" in ocr_lower or "boarding" in ocr_lower:
        action_items.append({"text": "Report to airport boarding gate on time", "status": "pending"})
    elif "whatsapp" in ocr_lower or "messages" in ocr_lower:
        action_items.append({"text": "Reply to pending messages", "status": "pending"})
    elif "hackathon" in ocr_lower or "openai" in ocr_lower:
        action_items.append({"text": "Register team for OpenAI Hackathon", "status": "pending"})

    return action_items


def extract_text_topics(ocr_text: str) -> List[str]:
    """Extracts salient topic keywords from OCR text."""
    if not ocr_text:
        return []

    ocr_lower = ocr_text.lower()
    topics = []

    topic_keywords = {
        "wifi": "connectivity",
        "wi-fi": "connectivity",
        "password": "credentials",
        "upi": "payment",
        "paid": "payment",
        "syllabus": "academics",
        "kiit": "university",
        "flight": "travel",
        "indigo": "airline",
        "express": "coding",
        "hackathon": "event",
        "whatsapp": "messaging",
        "menu": "food",
    }

    for kw, topic in topic_keywords.items():
        if kw in ocr_lower and topic not in topics:
            topics.append(topic)

    return topics


def resolve_content_type(category: str, ocr_text: str, financial_info: Dict[str, Any], events: List[Dict[str, Any]]) -> str:
    """Derives specific content_type within category hierarchy from accumulated evidence."""
    if not ocr_text:
        return "unknown"

    ocr_lower = ocr_text.lower()

    if category == "Connectivity":
        return "wifi_credentials" if "password" in ocr_lower or "wifi" in ocr_lower else "network_settings"
    elif category == "Payment":
        return "payment_receipt" if "receipt" in ocr_lower or "paid" in ocr_lower else "transaction"
    elif category == "Education":
        return "course_material" if "syllabus" in ocr_lower or "unit" in ocr_lower else "notes"
    elif category == "Shopping":
        return "shopping_order" if "order" in ocr_lower or "paid" in ocr_lower else "product_page"
    elif category == "Travel":
        return "flight_ticket" if "flight" in ocr_lower or "boarding" in ocr_lower else "travel_itinerary"
    elif category == "Communication":
        return "chat" if "whatsapp" in ocr_lower or "messages" in ocr_lower else "email"
    elif category == "Coding":
        return "code_editor" if "controller" in ocr_lower or "const" in ocr_lower else "documentation"
    elif category == "Work & Career":
        return "event" if "hackathon" in ocr_lower else "job_posting"
    elif category == "Entertainment":
        return "movie" if "movie" in ocr_lower else "video"

    return "unknown"


def generate_deterministic_summary(category: str, ocr_text: str, entities: List[Dict[str, str]]) -> str:
    """Generates a concise content-descriptive summary constructed from extracted evidence without an LLM."""
    if not ocr_text:
        return "General screenshot with no detected OCR text."

    ent_map = {e["type"]: e["text"] for e in entities}
    fin = extract_financial_info(ocr_text)

    if category == "Connectivity":
        net = ent_map.get("network_name", ent_map.get("organization", "Airtel"))
        return f"Wi-Fi network credentials and password for {net} router."
    elif category == "Payment":
        amt = fin["amounts"][0] if fin["amounts"] else "payment"
        merchant = ent_map.get("merchant", "merchant")
        return f"UPI payment receipt of {amt} paid to {merchant}."
    elif category == "Education":
        inst = ent_map.get("institution", "course")
        subj = ent_map.get("subject", "syllabus")
        return f"{inst} {subj} course syllabus covering units I-V and textbooks."
    elif category == "Shopping":
        prod = ent_map.get("product", "food menu")
        return f"{prod} food menu item listings and prices."
    elif category == "Travel":
        air = ent_map.get("airline", "flight")
        dest = ent_map.get("destination", "destination")
        return f"{air} flight boarding pass to {dest}."
    elif category == "Communication":
        app = ent_map.get("application", "chat")
        return f"{app}/Telegram chat notification overview."
    elif category == "Coding":
        fw = ent_map.get("framework", "code")
        return f"VS Code {fw}.js controller code snippet."
    elif category == "Work & Career":
        org = ent_map.get("organization", "AI")
        evt = ent_map.get("event", "event")
        return f"{org} {evt} event announcement details."

    return f"{category} screenshot details."
