"""
I.R.I.S. Timeline Service
=========================
Constructs chronological timeline entries from screenshot metadata,
prioritizing extracted event dates with graceful fallback to creation dates.
Groups entries by year and month with multi-parameter filtering.
"""

import calendar
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Canonical month abbreviation/name mapping
MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _image_url_from_path(image_uri: Optional[str], filename: Optional[str]) -> str:
    """Derive relative static URL (/data/screenshots/...) for screenshot image."""
    if image_uri:
        if os.path.isabs(image_uri):
            try:
                rel = Path(image_uri).relative_to(Path(os.getcwd()))
                return "/" + str(rel).replace("\\", "/")
            except ValueError:
                pass
        elif image_uri.startswith("/data/") or image_uri.startswith("data/"):
            return "/" + image_uri.lstrip("/")

    fname = filename or "unknown.png"
    return f"/data/screenshots/{fname}"


class TimelineService:
    """
    Builds chronological timeline representations from screenshot metadata.
    Uses extracted event dates when reliable, falling back to file creation date.
    """

    def __init__(self, metadata_store: Optional[Any] = None):
        self.metadata_store = metadata_store

    @staticmethod
    def parse_event_date_candidate(raw_str: str) -> Optional[datetime]:
        """
        Safely attempts to parse a raw or normalized date string candidate.
        Ensures dates fall within plausible human event range (1990 - 2040).
        Rejects noisy OCR artifacts.
        """
        if not raw_str or not isinstance(raw_str, str):
            return None

        clean = raw_str.strip()
        if not clean:
            return None

        # 1. ISO format: YYYY-MM-DD or YYYY/MM/DD
        m_iso = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", clean)
        if m_iso:
            try:
                y, m, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
                if 1990 <= y <= 2040 and 1 <= m <= 12 and 1 <= d <= 31:
                    return datetime(y, m, d, tzinfo=timezone.utc)
            except ValueError:
                pass

        # 2. Named Month: "15th Apr, 2026", "15 April 2026", "Apr 15, 2026"
        m_named1 = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)[,\s]+(\d{4})\b", clean)
        if m_named1:
            try:
                d = int(m_named1.group(1))
                m_str = m_named1.group(2).lower()
                y = int(m_named1.group(3))
                if m_str in MONTH_MAP and 1990 <= y <= 2040 and 1 <= d <= 31:
                    return datetime(y, MONTH_MAP[m_str], d, tzinfo=timezone.utc)
            except ValueError:
                pass

        m_named2 = re.search(r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?[,\s]+(\d{4})\b", clean)
        if m_named2:
            try:
                m_str = m_named2.group(1).lower()
                d = int(m_named2.group(2))
                y = int(m_named2.group(3))
                if m_str in MONTH_MAP and 1990 <= y <= 2040 and 1 <= d <= 31:
                    return datetime(y, MONTH_MAP[m_str], d, tzinfo=timezone.utc)
            except ValueError:
                pass

        # 3. DMY format: DD-MM-YYYY or DD/MM/YYYY
        m_dmy = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$", clean)
        if m_dmy:
            try:
                d, m, y = int(m_dmy.group(1)), int(m_dmy.group(2)), int(m_dmy.group(3))
                if 1990 <= y <= 2040 and 1 <= m <= 12 and 1 <= d <= 31:
                    return datetime(y, m, d, tzinfo=timezone.utc)
            except ValueError:
                pass

        # 4. Full ISO 8601 string
        try:
            dt = datetime.fromisoformat(clean.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if 1990 <= dt.year <= 2040:
                return dt
        except Exception:
            pass

        return None

    def extract_reliable_event_date(self, dates_field: Any) -> Optional[datetime]:
        """
        Inspects extracted dates from metadata, validating and returning the first reliable event date.
        """
        if not dates_field:
            return None

        # Parse stringified JSON if needed
        items = dates_field
        if isinstance(dates_field, str):
            try:
                items = json.loads(dates_field)
            except Exception:
                return None

        if not isinstance(items, list):
            return None

        for item in items:
            if isinstance(item, dict):
                # Try normalized first, then text, then raw_text
                for key in ("normalized", "text", "raw_text"):
                    candidate = item.get(key)
                    if candidate:
                        parsed = self.parse_event_date_candidate(candidate)
                        if parsed:
                            return parsed
            elif isinstance(item, str):
                parsed = self.parse_event_date_candidate(item)
                if parsed:
                    return parsed

        return None

    @staticmethod
    def parse_creation_date(created_at_field: Any) -> Optional[datetime]:
        """Parses created_at timestamp safely into timezone-aware UTC datetime."""
        if not created_at_field:
            return None

        if isinstance(created_at_field, datetime):
            if created_at_field.tzinfo is None:
                return created_at_field.replace(tzinfo=timezone.utc)
            return created_at_field

        if isinstance(created_at_field, str):
            try:
                dt = datetime.fromisoformat(created_at_field.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                pass

            # Try basic date match
            m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})", created_at_field)
            if m:
                try:
                    return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
                except ValueError:
                    pass

        return None

    def resolve_date_for_screenshot(
        self,
        dates_field: Any,
        created_at_field: Any,
    ) -> Tuple[datetime, Optional[str], bool, str]:
        """
        Resolves the chronological date for a screenshot.
        Returns:
            (resolved_datetime, event_date_str_or_none, is_event_date, display_date_str)
        """
        # Step 1: Check reliable extracted event date
        event_dt = self.extract_reliable_event_date(dates_field)
        if event_dt is not None:
            event_date_str = event_dt.strftime("%Y-%m-%d")
            display_date = f"{calendar.month_abbr[event_dt.month]} {event_dt.day}, {event_dt.year}"
            return event_dt, event_date_str, True, display_date

        # Step 2: Fallback to creation date
        created_dt = self.parse_creation_date(created_at_field)
        if created_dt is not None:
            display_date = f"{calendar.month_abbr[created_dt.month]} {created_dt.day}, {created_dt.year}"
            return created_dt, None, False, display_date

        # Step 3: Graceful fallback for missing/invalid dates (epoch reference)
        fallback_dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
        return fallback_dt, None, False, "Undated"

    @staticmethod
    def derive_short_description(item_dict: Dict[str, Any]) -> str:
        """
        Extracts a concise, non-fabricated description from existing OCR and metadata.
        """
        summary = item_dict.get("summary")
        if summary and isinstance(summary, str):
            cleaned = summary.strip()
            # If summary contains useful content and is not generic placeholder
            if cleaned and not cleaned.startswith("Screenshot showing general app, utility"):
                return cleaned

        # Check evidence text snippet
        evidence = item_dict.get("evidence")
        if isinstance(evidence, dict) and evidence.get("text_snippet"):
            snip = evidence["text_snippet"].strip().replace("\n", " ")
            if snip:
                return snip[:120] + ("..." if len(snip) > 120 else "")

        # Check raw OCR text
        ocr_text = item_dict.get("ocr_text")
        if ocr_text and isinstance(ocr_text, str):
            clean_ocr = " ".join(ocr_text.split()).strip()
            if clean_ocr:
                return clean_ocr[:120] + ("..." if len(clean_ocr) > 120 else "")

        # Default fallback without fabrication
        fname = item_dict.get("filename") or item_dict.get("screenshot_id") or "item"
        return f"Screenshot {fname}"

    def build_entry(self, item: Any) -> Dict[str, Any]:
        """Converts a ScreenshotMetadata instance or dictionary into a normalized Timeline entry."""
        if hasattr(item, "to_dict"):
            d = item.to_dict()
        elif isinstance(item, dict):
            d = dict(item)
        else:
            d = {}

        sid = d.get("screenshot_id") or ""
        fname = d.get("filename") or f"{sid}.jpg"
        image_uri = d.get("image_uri")
        img_url = d.get("image_url") or _image_url_from_path(image_uri, fname)

        resolved_dt, event_date_str, is_event_date, display_date = self.resolve_date_for_screenshot(
            dates_field=d.get("dates"),
            created_at_field=d.get("created_at"),
        )

        category = d.get("category") or "General"
        intent = d.get("intent") or "General Information"
        content_type = d.get("content_type") or "unknown"
        importance = d.get("importance") or d.get("importance_score") or 2
        importance_reason = d.get("importance_reason") or ""
        short_desc = self.derive_short_description(d)
        tags = d.get("tags") or []

        return {
            "screenshot_id": sid,
            "filename": fname,
            "image_url": img_url,
            "thumbnail_url": img_url,
            "event_date": event_date_str,
            "is_event_date": is_event_date,
            "display_date": display_date,
            "timeline_date": resolved_dt.strftime("%Y-%m-%d"),
            "resolved_dt": resolved_dt,
            "created_at": d.get("created_at") or resolved_dt.isoformat(),
            "year": resolved_dt.year,
            "month": resolved_dt.month,
            "category": category,
            "intent": intent,
            "content_type": content_type,
            "importance": importance,
            "importance_score": importance,
            "importance_reason": importance_reason,
            "short_description": short_desc,
            "tags": tags,
        }

    def build_timeline(
        self,
        screenshots: Optional[List[Any]] = None,
        year: Optional[int] = None,
        month: Optional[int] = None,
        category: Optional[str] = None,
        intent: Optional[str] = None,
        sort_order: str = "desc",
    ) -> Dict[str, Any]:
        """
        Builds chronological timeline groups from screenshots.
        Supports filtering by year, month, category, and intent.
        Sorts newest first (desc) by default.
        """
        # If no screenshots provided directly, fetch from metadata_store if available
        if screenshots is None:
            if self.metadata_store and hasattr(self.metadata_store, "list_all_metadata"):
                screenshots = self.metadata_store.list_all_metadata()
            else:
                screenshots = []

        # Build raw entries
        all_entries: List[Dict[str, Any]] = []
        available_years: Set[int] = set()
        available_categories: Set[str] = set()
        available_intents: Set[str] = set()

        for item in screenshots:
            entry = self.build_entry(item)
            all_entries.append(entry)
            if entry["year"] > 1970:
                available_years.add(entry["year"])
            if entry["category"]:
                available_categories.add(entry["category"])
            if entry["intent"]:
                available_intents.add(entry["intent"])

        # Apply filters
        filtered: List[Dict[str, Any]] = []
        for entry in all_entries:
            if year is not None and entry["year"] != year:
                continue
            if month is not None and entry["month"] != month:
                continue
            if category and entry["category"].strip().lower() != category.strip().lower():
                continue
            if intent and intent.strip().lower() not in entry["intent"].strip().lower():
                continue
            filtered.append(entry)

        # Sort entries chronologically
        reverse = (sort_order.lower() != "asc")
        filtered.sort(
            key=lambda e: (e["resolved_dt"], e["screenshot_id"]),
            reverse=reverse,
        )

        # Group by year and month
        groups_dict: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for entry in filtered:
            group_key = (entry["year"], entry["month"])
            if group_key not in groups_dict:
                groups_dict[group_key] = []
            
            # Create public entry dict (clean out internal resolved_dt object)
            pub_entry = dict(entry)
            pub_entry.pop("resolved_dt", None)
            groups_dict[group_key].append(pub_entry)

        # Order groups
        group_keys = list(groups_dict.keys())
        group_keys.sort(reverse=reverse)

        groups_list = []
        for y, m in group_keys:
            if y == 1970 and m == 1:
                month_name = "Undated / Legacy"
            else:
                month_name = f"{calendar.month_name[m]} {y}"

            groups_list.append({
                "year": y,
                "month": m,
                "month_name": month_name,
                "count": len(groups_dict[(y, m)]),
                "entries": groups_dict[(y, m)],
            })

        return {
            "total_entries": len(filtered),
            "total_groups": len(groups_list),
            "filters_applied": {
                "year": year,
                "month": month,
                "category": category,
                "intent": intent,
                "sort_order": sort_order,
            },
            "available_years": sorted(list(available_years), reverse=True),
            "available_categories": sorted(list(available_categories)),
            "available_intents": sorted(list(available_intents)),
            "groups": groups_list,
        }
