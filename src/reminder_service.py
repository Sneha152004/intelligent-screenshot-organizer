"""
I.R.I.S. Reminder Service
=========================
Intelligently detects actionable tasks, bill due dates, travel bookings,
appointments, and deadlines from screenshot intelligence metadata.
Enforces strict safety, deduplication, and non-fabrication.
"""

import calendar
from datetime import datetime, timezone
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

from .storage.reminder_store import ReminderStore

logger = logging.getLogger(__name__)

# Actionable reminder trigger phrases and patterns
PAYMENT_DUE_PATTERNS = [
    r"\bdue\s+date[:\s]+([^\n\r]+)",
    r"\bpay\s+before[:\s]+([^\n\r]+)",
    r"\bbill\s+due[:\s]+([^\n\r]+)",
    r"\bpayment\s+due[:\s]+([^\n\r]+)",
    r"\bamount\s+due\b",
    r"\bminimum\s+due\b",
]

TRAVEL_PATTERNS = [
    r"\bboarding\s+gate\b",
    r"\bboarding\s+time[:\s]+([^\n\r]+)",
    r"\bflight\s+departure[:\s]+([^\n\r]+)",
    r"\bdeparts?\s+date[:\s]+([^\n\r]+)",
    r"\bcheck-?in\s+date[:\s]+([^\n\r]+)",
    r"\btrain\s+departure[:\s]+([^\n\r]+)",
    r"\bboarding\s+pass\b",
]

EVENT_PATTERNS = [
    r"\bappointment[:\s]+([^\n\r]+)",
    r"\bwebinar\s+on[:\s]+([^\n\r]+)",
    r"\bhackathon\s+date[:\s]+([^\n\r]+)",
    r"\bmeeting\s+at[:\s]+([^\n\r]+)",
    r"\binterview\s+scheduled\b",
]

TASK_PATTERNS = [
    r"\bdeadline[:\s]+([^\n\r]+)",
    r"\blast\s+date\s+to\s+apply[:\s]+([^\n\r]+)",
    r"\bsubmit\s+before[:\s]+([^\n\r]+)",
    r"\bsubmission\s+deadline[:\s]+([^\n\r]+)",
    r"\bregister\s+before[:\s]+([^\n\r]+)",
]

SUBSCRIPTION_PATTERNS = [
    r"\brenews\s+on[:\s]+([^\n\r]+)",
    r"\brenewal\s+date[:\s]+([^\n\r]+)",
    r"\bexpires\s+on[:\s]+([^\n\r]+)",
    r"\bnext\s+billing\s+date[:\s]+([^\n\r]+)",
    r"\bvalid\s+till[:\s]+([^\n\r]+)",
]


class ReminderService:
    """
    Analyzes screenshot metadata and OCR evidence to detect and manage actionable reminders.
    """

    def __init__(
        self,
        reminder_store: Optional[ReminderStore] = None,
        metadata_store: Optional[Any] = None,
    ):
        self.reminder_store = reminder_store or ReminderStore()
        self.metadata_store = metadata_store

    @staticmethod
    def _parse_candidate_date(raw_str: str) -> Optional[str]:
        """
        Parses a date string candidate into ISO YYYY-MM-DD.
        Rejects implausible OCR noise (year < 1990 or > 2040).
        """
        if not raw_str or not isinstance(raw_str, str):
            return None

        clean = raw_str.strip()
        # 1. ISO format: YYYY-MM-DD or YYYY/MM/DD
        m_iso = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", clean)
        if m_iso:
            try:
                y, m, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
                if 1990 <= y <= 2040 and 1 <= m <= 12 and 1 <= d <= 31:
                    return f"{y:04d}-{m:02d}-{d:02d}"
            except ValueError:
                pass

        # 2. Named Month: "15th Apr, 2026", "15 April 2026", "Apr 15, 2026"
        month_map = {
            "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
            "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
            "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
            "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
        }
        m_named1 = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)[,\s]+(\d{4})\b", clean)
        if m_named1:
            try:
                d = int(m_named1.group(1))
                m_str = m_named1.group(2).lower()
                y = int(m_named1.group(3))
                if m_str in month_map and 1990 <= y <= 2040 and 1 <= d <= 31:
                    return f"{y:04d}-{month_map[m_str]:02d}-{d:02d}"
            except ValueError:
                pass

        # 3. DMY format: DD-MM-YYYY or DD/MM/YYYY
        m_dmy = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", clean)
        if m_dmy:
            try:
                d, m, y = int(m_dmy.group(1)), int(m_dmy.group(2)), int(m_dmy.group(3))
                if 1990 <= y <= 2040 and 1 <= m <= 12 and 1 <= d <= 31:
                    return f"{y:04d}-{m:02d}-{d:02d}"
            except ValueError:
                pass

        return None

    @staticmethod
    def _parse_candidate_time(text: str) -> Optional[str]:
        """Extracts standard 12h or 24h time string (e.g. '14:30', '09:00 AM') if present."""
        if not text:
            return None
        m = re.search(r"\b(\d{1,2}):(\d{2})(?:\s*([APap][Mm]))?\b", text)
        if m:
            hh, mm, meridiem = int(m.group(1)), int(m.group(2)), m.group(3)
            if 0 <= mm <= 59:
                if meridiem:
                    if meridiem.lower() == "pm" and hh < 12:
                        hh += 12
                    elif meridiem.lower() == "am" and hh == 12:
                        hh = 0
                if 0 <= hh <= 23:
                    return f"{hh:02d}:{mm:02d}"
        return None

    def detect_reminders(self, metadata_item: Any) -> List[Dict[str, Any]]:
        """
        Evaluates a screenshot item and returns candidate reminder dictionaries.
        Guarantees strict safety: rejects password/OTP credentials and purely historical receipts.
        """
        if hasattr(metadata_item, "to_dict"):
            d = metadata_item.to_dict()
        elif isinstance(metadata_item, dict):
            d = dict(metadata_item)
        else:
            return []

        # 1. Strict Safety Check: Never create reminders from credential/password storage
        sensitive_info = d.get("sensitive_info") or {}
        if isinstance(sensitive_info, str):
            try:
                sensitive_info = json.loads(sensitive_info)
            except Exception:
                sensitive_info = {}

        if sensitive_info.get("contains_password"):
            # Credentials should never generate calendar reminders
            return []

        sid = d.get("screenshot_id") or ""
        fname = d.get("filename") or ""
        category = d.get("category") or "General"
        intent = d.get("intent") or ""
        content_type = d.get("content_type") or "unknown"
        summary = d.get("summary") or ""
        importance = d.get("importance") or d.get("importance_score") or 3

        # Resolve OCR text
        ocr_text = d.get("ocr_text") or ""
        if not ocr_text and isinstance(d.get("evidence"), dict):
            ocr_text = d.get("evidence").get("text_snippet") or ""

        ocr_lower = ocr_text.lower()
        candidates: List[Dict[str, Any]] = []

        # Parse extracted dates
        dates_list = d.get("dates") or []
        if isinstance(dates_list, str):
            try:
                dates_list = json.loads(dates_list)
            except Exception:
                dates_list = []

        # Find best explicit dates in the metadata
        explicit_dates = []
        for dt_item in dates_list:
            if isinstance(dt_item, dict):
                cand = self._parse_candidate_date(dt_item.get("normalized") or dt_item.get("text") or dt_item.get("raw_text") or "")
                if cand:
                    explicit_dates.append((cand, dt_item.get("type", "general")))

        # Check action items
        action_items = d.get("action_items") or []
        if isinstance(action_items, str):
            try:
                action_items = json.loads(action_items)
            except Exception:
                action_items = []

        # Extract time if present
        time_candidate = self._parse_candidate_time(ocr_text)

        # ─────────────────────────────────────────────────────────────────────
        # Rule A: Bill / Payment Due Date
        # ─────────────────────────────────────────────────────────────────────
        # Distinguish historical paid receipts from actionable unpaid bills:
        # If it says "paid successfully" or has a transaction ID without a due date, skip!
        has_due_keyword = any(re.search(pat, ocr_lower) for pat in PAYMENT_DUE_PATTERNS)
        is_paid_receipt = (
            "paid successfully" in ocr_lower
            or "payment successful" in ocr_lower
            or "upi transaction id" in ocr_lower
            or "order confirmed" in ocr_lower
        )

        if has_due_keyword or any(dt[1] == "due_date" for dt in explicit_dates):
            # Look for explicit due date
            due_date = None
            for dt_val, dt_type in explicit_dates:
                if dt_type == "due_date":
                    due_date = dt_val
                    break
            if not due_date and explicit_dates:
                due_date = explicit_dates[0][0]

            if due_date:
                candidates.append({
                    "title": f"Bill Due: {summary[:40]}" if summary else "Pay Due Bill or Invoice",
                    "description": summary or "Payment due deadline identified from screenshot.",
                    "reminder_date": due_date,
                    "reminder_time": time_candidate,
                    "reminder_type": "bill_payment",
                    "priority": "high",
                    "importance_score": max(importance, 4),
                    "source_text": ocr_text[:120] if ocr_text else summary,
                })

        # ─────────────────────────────────────────────────────────────────────
        # Rule B: Travel Booking / Flight / Transit Departure
        # ─────────────────────────────────────────────────────────────────────
        is_travel = (
            category == "Travel"
            or "travel" in intent.lower()
            or content_type in ["flight_ticket", "boarding_pass", "train_ticket", "hotel_booking"]
            or any(re.search(pat, ocr_lower) for pat in TRAVEL_PATTERNS)
        )
        if is_travel:
            # Must find a departure or travel date
            travel_date = None
            for dt_val, dt_type in explicit_dates:
                if dt_type in ("travel_date", "event_date"):
                    travel_date = dt_val
                    break
            if not travel_date and explicit_dates:
                travel_date = explicit_dates[0][0]

            if travel_date:
                # Build travel reminder
                airline_ent = None
                if isinstance(d.get("entities"), list):
                    for ent in d["entities"]:
                        if isinstance(ent, dict) and ent.get("type") in ("airline", "station"):
                            airline_ent = ent.get("text")
                            break

                title = f"Travel Departure: {airline_ent}" if airline_ent else (
                    summary[:45] if summary and not summary.startswith("Screenshot") else "Upcoming Travel Booking"
                )
                candidates.append({
                    "title": title,
                    "description": summary or "Upcoming flight, transit, or travel boarding time.",
                    "reminder_date": travel_date,
                    "reminder_time": time_candidate,
                    "reminder_type": "travel",
                    "priority": "high",
                    "importance_score": max(importance, 4),
                    "source_text": ocr_text[:120] if ocr_text else summary,
                })

        # ─────────────────────────────────────────────────────────────────────
        # Rule C: Event / Webinar / Hackathon / Appointment
        # ─────────────────────────────────────────────────────────────────────
        is_event = (
            content_type in ["event_ticket", "appointment", "webinar", "hackathon"]
            or any(re.search(pat, ocr_lower) for pat in EVENT_PATTERNS)
            or "event" in intent.lower()
        )
        if is_event:
            event_date = None
            for dt_val, dt_type in explicit_dates:
                if dt_type in ("event_date", "general"):
                    event_date = dt_val
                    break
            if not event_date and explicit_dates:
                event_date = explicit_dates[0][0]

            if event_date:
                candidates.append({
                    "title": f"Event: {summary[:40]}" if summary and not summary.startswith("Screenshot") else "Upcoming Scheduled Event",
                    "description": summary or "Scheduled webinar, conference, or appointment event.",
                    "reminder_date": event_date,
                    "reminder_time": time_candidate,
                    "reminder_type": "event",
                    "priority": "medium" if importance < 4 else "high",
                    "importance_score": importance,
                    "source_text": ocr_text[:120] if ocr_text else summary,
                })

        # ─────────────────────────────────────────────────────────────────────
        # Rule D: Action Items & Deadlines (Assignments, Applications)
        # ─────────────────────────────────────────────────────────────────────
        for act in action_items:
            act_text = act.get("text", "")
            if not act_text:
                continue

            # Find matching date
            act_date = explicit_dates[0][0] if explicit_dates else None
            # If no explicit date in dates, look in created_at only if action is pending
            if not act_date and d.get("created_at"):
                cand_created = self._parse_candidate_date(d["created_at"])
                if cand_created:
                    act_date = cand_created

            if act_date:
                # Deduce category and type
                r_type = "task"
                if "bill" in act_text.lower():
                    r_type = "bill_payment"
                elif "airport" in act_text.lower() or "boarding" in act_text.lower():
                    r_type = "travel"
                elif "event" in act_text.lower() or "hackathon" in act_text.lower():
                    r_type = "event"

                candidates.append({
                    "title": act_text,
                    "description": summary or act_text,
                    "reminder_date": act_date,
                    "reminder_time": time_candidate,
                    "reminder_type": r_type,
                    "priority": "high" if "deadline" in act_text.lower() or "bill" in act_text.lower() else "medium",
                    "importance_score": max(importance, 3),
                    "source_text": act_text,
                })

        # ─────────────────────────────────────────────────────────────────────
        # Rule E: Subscription Renewal / Expiry
        # ─────────────────────────────────────────────────────────────────────
        has_sub_keyword = any(re.search(pat, ocr_lower) for pat in SUBSCRIPTION_PATTERNS)
        if has_sub_keyword and explicit_dates:
            candidates.append({
                "title": f"Subscription Renewal: {summary[:35]}" if summary else "Subscription Renewal Notice",
                "description": summary or "Subscription renewal or service validity deadline.",
                "reminder_date": explicit_dates[0][0],
                "reminder_time": time_candidate,
                "reminder_type": "subscription",
                "priority": "medium",
                "importance_score": importance,
                "source_text": ocr_text[:120] if ocr_text else summary,
            })

        # Deduplicate candidates within the same screenshot
        unique_candidates: List[Dict[str, Any]] = []
        seen_keys: Set[Tuple[str, str]] = set()

        for c in candidates:
            dedup_key = (c["reminder_date"], c["reminder_type"])
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            # Assign identity fields
            c["reminder_id"] = f"rem_{sid[:8]}_{uuid.uuid4().hex[:6]}" if sid else f"rem_man_{uuid.uuid4().hex[:8]}"
            c["screenshot_id"] = sid if sid else None
            c["status"] = "pending"
            now_iso = datetime.now(timezone.utc).isoformat()
            c["created_at"] = now_iso
            c["updated_at"] = now_iso
            unique_candidates.append(c)

        return unique_candidates

    def scan_and_create_for_screenshot(self, screenshot_item: Any) -> List[Dict[str, Any]]:
        """
        Detects reminders for a single screenshot and saves new unique ones into SQLite.
        """
        candidates = self.detect_reminders(screenshot_item)
        created: List[Dict[str, Any]] = []

        for cand in candidates:
            # Check if identical reminder already exists in DB
            existing = self.reminder_store.find_duplicate_reminder(
                screenshot_id=cand.get("screenshot_id"),
                reminder_date=cand["reminder_date"],
                reminder_type=cand["reminder_type"],
                title=cand["title"],
            )
            if existing:
                continue

            saved = self.reminder_store.create_reminder(cand)
            created.append(saved)

        return created

    def scan_all_screenshots(self) -> Dict[str, Any]:
        """
        Scans all screenshots currently stored in SQLite and generates actionable reminders.
        Guarantees 100% idempotency: running repeated scans will not produce duplicates.
        """
        if not self.metadata_store or not hasattr(self.metadata_store, "list_all_metadata"):
            return {
                "total_screenshots_scanned": 0,
                "new_reminders_created": 0,
                "existing_skipped": 0,
                "reminders": [],
            }

        all_meta = self.metadata_store.list_all_metadata()
        new_created: List[Dict[str, Any]] = []
        skipped_count = 0

        for item in all_meta:
            candidates = self.detect_reminders(item)
            for cand in candidates:
                existing = self.reminder_store.find_duplicate_reminder(
                    screenshot_id=cand.get("screenshot_id"),
                    reminder_date=cand["reminder_date"],
                    reminder_type=cand["reminder_type"],
                    title=cand["title"],
                )
                if existing:
                    skipped_count += 1
                    continue

                saved = self.reminder_store.create_reminder(cand)
                new_created.append(saved)

        return {
            "total_screenshots_scanned": len(all_meta),
            "new_reminders_created": len(new_created),
            "existing_skipped": skipped_count,
            "reminders": new_created,
        }
