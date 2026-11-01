"""
I.R.I.S. Alert Service
======================
Generates and synchronizes actionable system alerts from reminders and high-importance screenshots.
Applies deterministic severity scoring, time horizons, safety filtering, and deduplication.
"""

from datetime import date, datetime, timezone
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import uuid

from .storage.alert_store import AlertStore
from .storage.reminder_store import ReminderStore

logger = logging.getLogger(__name__)


class AlertService:
    """
    Evaluates reminders and screenshot intelligence to generate, prioritize,
    and resolve system alerts.
    """

    def __init__(
        self,
        alert_store: Optional[AlertStore] = None,
        reminder_store: Optional[ReminderStore] = None,
        metadata_store: Optional[Any] = None,
    ):
        self.alert_store = alert_store or AlertStore()
        self.reminder_store = reminder_store or ReminderStore()
        self.metadata_store = metadata_store

    @staticmethod
    def _parse_iso_date(date_str: str) -> Optional[date]:
        """Parses YYYY-MM-DD string into date object safely."""
        if not date_str:
            return None
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", str(date_str).strip())
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass
        return None

    @staticmethod
    def sanitize_message(text: str) -> str:
        """Ensures passwords, OTPs, or authentication tokens are never exposed in alerts."""
        if not text:
            return ""
        redacted = re.sub(
            r"(?i)\b(password|passcode|secret|pin|otp)(?:\s+(?:is|was))?\s*[:=-]?\s*([^\s,.;]+)",
            r"\1: [REDACTED]",
            text,
        )
        return redacted

    def evaluate_reminder_for_alerts(
        self,
        reminder: Dict[str, Any],
        reference_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluates a reminder against time horizons to generate alert candidates.
        Excludes completed, dismissed, or sensitive items.
        """
        # Rule 1: Do not generate alerts for completed or dismissed reminders
        if reminder.get("status") in ("completed", "dismissed"):
            return []

        rem_date_str = reminder.get("reminder_date")
        rem_date = self._parse_iso_date(rem_date_str)
        if not rem_date:
            return []

        ref_date = reference_date or datetime.now().date()
        days_diff = (rem_date - ref_date).days

        rem_type = reminder.get("reminder_type", "general")
        importance = reminder.get("importance_score") or 3
        priority = reminder.get("priority", "medium")
        title = self.sanitize_message(reminder.get("title", "Upcoming Action"))
        rem_id = reminder.get("reminder_id")
        sid = reminder.get("screenshot_id")

        candidates: List[Dict[str, Any]] = []

        # ─────────────────────────────────────────────────────────────────────
        # 1. Overdue Alerts (days_diff < 0)
        # ─────────────────────────────────────────────────────────────────────
        if days_diff < 0:
            past_days = abs(days_diff)
            day_text = "yesterday" if past_days == 1 else f"{past_days} days ago"

            if rem_type == "bill_payment":
                alert_type = "overdue_payment"
                severity = "critical"
                msg = f"Payment deadline of {rem_date_str} has passed ({day_text}). Please resolve to avoid penalties."
            elif rem_type == "travel":
                # Only alert for recent travel (within 2 days), otherwise historical
                if past_days <= 2:
                    alert_type = "upcoming_travel"
                    severity = "high"
                    msg = f"Travel departure date {rem_date_str} was {day_text}. Confirm your itinerary."
                else:
                    return []
            else:
                alert_type = "approaching_deadline"
                severity = "critical" if (importance >= 4 or priority == "high") else "high"
                msg = f"Task deadline of {rem_date_str} is overdue by {past_days} day(s)."

            candidates.append({
                "reminder_id": rem_id,
                "screenshot_id": sid,
                "title": f"OVERDUE: {title}",
                "message": msg,
                "alert_type": alert_type,
                "severity": severity,
                "trigger_date": rem_date_str,
            })

        # ─────────────────────────────────────────────────────────────────────
        # 2. Due Today (days_diff == 0)
        # ─────────────────────────────────────────────────────────────────────
        elif days_diff == 0:
            if rem_type == "bill_payment":
                alert_type = "upcoming_payment"
                severity = "critical"
                msg = f"Payment is due TODAY ({rem_date_str}). Complete payment before end of day."
            elif rem_type == "travel":
                alert_type = "upcoming_travel"
                severity = "critical"
                time_info = f" at {reminder['reminder_time']}" if reminder.get("reminder_time") else ""
                msg = f"Travel departure is scheduled for TODAY ({rem_date_str}){time_info}."
            elif rem_type == "event":
                alert_type = "upcoming_event"
                severity = "high"
                msg = f"Event is happening TODAY ({rem_date_str})."
            else:
                alert_type = "approaching_deadline"
                severity = "high" if (importance >= 4 or priority == "high") else "medium"
                msg = f"Deadline is TODAY ({rem_date_str}). Action required."

            candidates.append({
                "reminder_id": rem_id,
                "screenshot_id": sid,
                "title": f"DUE TODAY: {title}",
                "message": msg,
                "alert_type": alert_type,
                "severity": severity,
                "trigger_date": rem_date_str,
            })

        # ─────────────────────────────────────────────────────────────────────
        # 3. Urgent (Due within 24 hours / tomorrow: days_diff == 1)
        # ─────────────────────────────────────────────────────────────────────
        elif days_diff == 1:
            if rem_type == "bill_payment":
                alert_type = "upcoming_payment"
                severity = "high"
                msg = f"Bill payment is due TOMORROW ({rem_date_str})."
            elif rem_type == "travel":
                alert_type = "upcoming_travel"
                severity = "critical"
                msg = f"Travel departs TOMORROW ({rem_date_str}). Prepare boarding documents."
            elif rem_type == "event":
                alert_type = "upcoming_event"
                severity = "medium"
                msg = f"Event scheduled for TOMORROW ({rem_date_str})."
            else:
                alert_type = "approaching_deadline"
                severity = "high" if importance >= 4 else "medium"
                msg = f"Approaching deadline tomorrow ({rem_date_str})."

            candidates.append({
                "reminder_id": rem_id,
                "screenshot_id": sid,
                "title": f"URGENT: {title}",
                "message": msg,
                "alert_type": alert_type,
                "severity": severity,
                "trigger_date": rem_date_str,
            })

        # ─────────────────────────────────────────────────────────────────────
        # 4. Upcoming (Due within 3 days: 2 <= days_diff <= 3)
        # ─────────────────────────────────────────────────────────────────────
        elif 2 <= days_diff <= 3:
            if rem_type == "bill_payment":
                alert_type = "upcoming_payment"
                severity = "medium"
                msg = f"Payment due in {days_diff} days ({rem_date_str})."
            elif rem_type == "travel":
                alert_type = "upcoming_travel"
                severity = "high"
                msg = f"Travel departure in {days_diff} days ({rem_date_str})."
            elif rem_type == "event":
                alert_type = "upcoming_event"
                severity = "medium"
                msg = f"Event taking place in {days_diff} days ({rem_date_str})."
            elif rem_type == "subscription":
                alert_type = "subscription_expiry"
                severity = "medium"
                msg = f"Subscription renewal in {days_diff} days ({rem_date_str})."
            else:
                alert_type = "approaching_deadline"
                severity = "medium" if importance >= 4 else "low"
                msg = f"Deadline in {days_diff} days ({rem_date_str})."

            candidates.append({
                "reminder_id": rem_id,
                "screenshot_id": sid,
                "title": f"UPCOMING: {title}",
                "message": msg,
                "alert_type": alert_type,
                "severity": severity,
                "trigger_date": rem_date_str,
            })

        # ─────────────────────────────────────────────────────────────────────
        # 5. Advance Notice (Due within 7 days: 4 <= days_diff <= 7)
        # ─────────────────────────────────────────────────────────────────────
        elif 4 <= days_diff <= 7:
            if rem_type in ("bill_payment", "travel") or importance >= 4:
                alert_type = "upcoming_payment" if rem_type == "bill_payment" else (
                    "upcoming_travel" if rem_type == "travel" else "approaching_deadline"
                )
                severity = "low"
                msg = f"Advance notice: {title} scheduled on {rem_date_str} (in {days_diff} days)."
                candidates.append({
                    "reminder_id": rem_id,
                    "screenshot_id": sid,
                    "title": f"NOTICE: {title}",
                    "message": msg,
                    "alert_type": alert_type,
                    "severity": severity,
                    "trigger_date": rem_date_str,
                })

        return candidates

    def evaluate_high_importance_screenshot(
        self,
        meta_item: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Creates a high_importance alert for critical screenshots (importance 5)
        that require user review (e.g. flight tickets, urgent bookings).
        """
        if hasattr(meta_item, "to_dict"):
            d = meta_item.to_dict()
        elif isinstance(meta_item, dict):
            d = dict(meta_item)
        else:
            return None

        # Exclude passwords/credentials from alert messages
        sensitive_info = d.get("sensitive_info") or {}
        if isinstance(sensitive_info, str):
            try:
                sensitive_info = json.loads(sensitive_info)
            except Exception:
                sensitive_info = {}

        if sensitive_info.get("contains_password"):
            return None

        importance = d.get("importance") or d.get("importance_score") or 0
        if importance < 5:
            return None

        sid = d.get("screenshot_id")
        fname = d.get("filename") or f"{sid}.jpg"
        category = d.get("category") or "General"
        reason = d.get("importance_reason") or "Critical screenshot content"
        created_at_str = d.get("created_at") or datetime.now().isoformat()
        trigger_date = created_at_str.split("T")[0]

        return {
            "reminder_id": None,
            "screenshot_id": sid,
            "title": f"HIGH IMPORTANCE: {category} ({fname})",
            "message": f"Critical screenshot requiring user attention: {reason}",
            "alert_type": "high_importance",
            "severity": "high",
            "trigger_date": trigger_date,
        }

    def sync_reminder_to_alerts(
        self,
        reminder: Dict[str, Any],
        reference_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """
        Synchronizes alerts for a single reminder.
        Resolves active alerts if reminder is completed/dismissed;
        otherwise generates or updates active alerts.
        """
        rem_id = reminder.get("reminder_id")
        rem_status = reminder.get("status", "pending")

        # If reminder is completed or dismissed, resolve linked active alerts
        if rem_status in ("completed", "dismissed"):
            new_status = "resolved" if rem_status == "completed" else "dismissed"
            self.alert_store.resolve_alerts_for_reminder(rem_id, new_status=new_status)
            return []

        candidates = self.evaluate_reminder_for_alerts(reminder, reference_date=reference_date)
        created_or_updated: List[Dict[str, Any]] = []

        for cand in candidates:
            existing = self.alert_store.find_active_alert(
                reminder_id=cand["reminder_id"],
                alert_type=cand["alert_type"],
            )
            if existing:
                # Update existing alert severity / message
                updated = self.alert_store.update_alert(
                    existing["alert_id"],
                    {
                        "severity": cand["severity"],
                        "message": cand["message"],
                        "title": cand["title"],
                        "trigger_date": cand["trigger_date"],
                    },
                )
                if updated:
                    created_or_updated.append(updated)
            else:
                cand["alert_id"] = f"alt_{cand['reminder_id'][:12]}_{uuid.uuid4().hex[:6]}"
                cand["status"] = "active"
                saved = self.alert_store.create_alert(cand)
                created_or_updated.append(saved)

        return created_or_updated

    def scan_all_for_alerts(self, reference_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Evaluates all reminders and high-importance screenshots to generate and synchronize alerts.
        100% idempotent: skips duplicates and updates existing active alerts cleanly.
        """
        reminders = self.reminder_store.list_reminders()
        new_count = 0
        updated_count = 0

        # 1. Sync reminders
        for rem in reminders:
            if rem.get("status") in ("completed", "dismissed"):
                self.alert_store.resolve_alerts_for_reminder(rem["reminder_id"], new_status="resolved")
                continue

            candidates = self.evaluate_reminder_for_alerts(rem, reference_date=reference_date)
            for cand in candidates:
                existing = self.alert_store.find_active_alert(
                    reminder_id=cand["reminder_id"],
                    alert_type=cand["alert_type"],
                )
                if existing:
                    self.alert_store.update_alert(
                        existing["alert_id"],
                        {
                            "severity": cand["severity"],
                            "message": cand["message"],
                            "title": cand["title"],
                            "trigger_date": cand["trigger_date"],
                        },
                    )
                    updated_count += 1
                else:
                    cand["alert_id"] = f"alt_{cand['reminder_id'][:12]}_{uuid.uuid4().hex[:6]}"
                    cand["status"] = "active"
                    self.alert_store.create_alert(cand)
                    new_count += 1

        # 2. Check high-importance screenshots if metadata_store is available
        if self.metadata_store and hasattr(self.metadata_store, "list_all_metadata"):
            all_meta = self.metadata_store.list_all_metadata()
            for meta_item in all_meta:
                cand = self.evaluate_high_importance_screenshot(meta_item)
                if cand and cand.get("screenshot_id"):
                    # Check if alert already exists for this screenshot_id
                    with self.alert_store._get_connection() as conn:
                        row = conn.execute(
                            "SELECT alert_id FROM screenshot_alerts WHERE screenshot_id = ? AND alert_type = 'high_importance' LIMIT 1",
                            (cand["screenshot_id"],),
                        ).fetchone()
                    if not row:
                        cand["alert_id"] = f"alt_hi_{cand['screenshot_id'][:8]}_{uuid.uuid4().hex[:6]}"
                        cand["status"] = "active"
                        self.alert_store.create_alert(cand)
                        new_count += 1

        summary = self.alert_store.get_summary_metrics()
        return {
            "new_alerts_created": new_count,
            "alerts_updated": updated_count,
            "summary": summary,
        }
