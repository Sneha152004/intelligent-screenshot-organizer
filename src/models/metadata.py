"""
Screenshot Metadata Model & Contract Definition
================================================
Defines the finalized ScreenshotMetadata data contract for identity, system metadata,
and future VLM/LLM extractions (tags, intent, summary, importance, entities, dates,
action_items, sensitive_info).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional, Union

DEFAULT_SENSITIVE_INFO: Dict[str, bool] = {
    "contains_password": False,
    "contains_payment_info": False,
    "contains_personal_contact": False,
}

SUPPORTED_CATEGORIES: List[str] = [
    "Education",
    "Payment",
    "Shopping",
    "Travel",
    "Coding",
    "Entertainment",
    "Connectivity",
    "Work & Career",
    "Communication",
    "General",
]


@dataclass
class ScreenshotMetadata:
    """
    Structured Application Metadata Model for Screenshot Entities.

    Contract specification:
    - Identity/Storage: screenshot_id, filename, image_uri
    - System/OCR Metadata: category, ocr_available, created_at, indexed_at
    - Intelligence Fields (VLM ready): tags, intent, summary, importance, entities, dates, action_items, sensitive_info
    """

    screenshot_id: str
    filename: str
    image_uri: str
    category: str = "General"
    ocr_available: bool = True
    created_at: Optional[str] = None
    indexed_at: Optional[str] = None

    # Future VLM / Intelligence Fields (Nullable / Default Empty)
    tags: List[str] = field(default_factory=list)
    intent: Optional[str] = None
    summary: Optional[str] = None
    importance: Optional[int] = None
    entities: List[Dict[str, str]] = field(default_factory=list)
    dates: List[Dict[str, str]] = field(default_factory=list)
    action_items: List[Dict[str, str]] = field(default_factory=list)
    sensitive_info: Dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_SENSITIVE_INFO))

    def __post_init__(self):
        # Set ISO 8601 timestamps if not provided
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.indexed_at:
            self.indexed_at = datetime.now(timezone.utc).isoformat()

        self.validate()

    def validate(self) -> None:
        """Validates required identity fields, boundary constraints, and structural types."""
        if not self.screenshot_id or not isinstance(self.screenshot_id, str) or not self.screenshot_id.strip():
            raise ValueError("screenshot_id must be a non-empty string.")

        if not self.filename or not isinstance(self.filename, str) or not self.filename.strip():
            raise ValueError("filename must be a non-empty string.")

        if not self.image_uri or not isinstance(self.image_uri, str) or not self.image_uri.strip():
            raise ValueError("image_uri must be a non-empty string.")

        # Importance validation: integer 1 to 5 when present
        if self.importance is not None:
            if isinstance(self.importance, bool) or not isinstance(self.importance, int):
                raise ValueError(
                    f"importance must be an integer between 1 and 5, got {type(self.importance).__name__}."
                )
            if not (1 <= self.importance <= 5):
                raise ValueError(f"importance must be an integer between 1 and 5, got {self.importance}.")

        # Structural type validations
        if not isinstance(self.tags, list) or not all(isinstance(t, str) for t in self.tags):
            raise ValueError("tags must be a list of strings.")

        if not isinstance(self.entities, list):
            raise ValueError("entities must be a list of structured dictionaries.")

        if not isinstance(self.dates, list):
            raise ValueError("dates must be a list of structured dictionaries.")

        if not isinstance(self.action_items, list):
            raise ValueError("action_items must be a list of structured dictionaries.")

        if not isinstance(self.sensitive_info, dict):
            raise ValueError("sensitive_info must be a dictionary of boolean flags.")

    def to_dict(self) -> Dict[str, Any]:
        """Converts metadata object to clean dictionary representation with Python structures."""
        return {
            "screenshot_id": self.screenshot_id,
            "filename": self.filename,
            "image_uri": self.image_uri,
            "category": self.category,
            "ocr_available": self.ocr_available,
            "created_at": self.created_at,
            "indexed_at": self.indexed_at,
            "tags": self.tags,
            "intent": self.intent,
            "summary": self.summary,
            "importance": self.importance,
            "entities": self.entities,
            "dates": self.dates,
            "action_items": self.action_items,
            "sensitive_info": self.sensitive_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScreenshotMetadata":
        """Instantiates ScreenshotMetadata cleanly from dictionary, handling raw/JSON data safely."""
        importance_val = data.get("importance")
        if importance_val is not None:
            try:
                importance_val = int(importance_val)
            except (ValueError, TypeError):
                raise ValueError(f"importance must be an integer between 1 and 5, got {importance_val}.")

        return cls(
            screenshot_id=str(data["screenshot_id"]).strip() if data.get("screenshot_id") else "",
            filename=str(data["filename"]).strip() if data.get("filename") else "",
            image_uri=str(data["image_uri"]).strip() if data.get("image_uri") else "",
            category=str(data.get("category", "General")),
            ocr_available=bool(data.get("ocr_available", True)),
            created_at=data.get("created_at"),
            indexed_at=data.get("indexed_at"),
            tags=cls._parse_json_list(data.get("tags")),
            intent=data.get("intent") if data.get("intent") else None,
            summary=data.get("summary") if data.get("summary") else None,
            importance=importance_val,
            entities=cls._parse_json_list(data.get("entities")),
            dates=cls._parse_json_list(data.get("dates")),
            action_items=cls._parse_json_list(data.get("action_items")),
            sensitive_info=cls._parse_sensitive_info(data.get("sensitive_info")),
        )

    @classmethod
    def _parse_json_list(cls, val: Any) -> List[Any]:
        """Safely parses list or JSON stringified array into a Python list."""
        if val is None or val == "":
            return []
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        return []

    @classmethod
    def _parse_sensitive_info(cls, val: Any) -> Dict[str, bool]:
        """Safely parses sensitive_info input into structured boolean dict."""
        default_info = dict(DEFAULT_SENSITIVE_INFO)
        if val is None or val == "":
            return default_info
        if isinstance(val, dict):
            for k, v in val.items():
                default_info[k] = bool(v)
            return default_info
        if isinstance(val, (bool, int)):
            default_info["contains_password"] = bool(val)
            return default_info
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        default_info[k] = bool(v)
                    return default_info
                elif isinstance(parsed, bool):
                    default_info["contains_password"] = parsed
                    return default_info
            except Exception:
                pass
        return default_info
