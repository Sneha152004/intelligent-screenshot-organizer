"""
Screenshot Metadata Model & Contract Definition
================================================
Defines the ScreenshotMetadata data contract supporting identity, system metadata,
hierarchical taxonomy (category -> content_type), structured evidence representation
(entities, events, dates, financial_info, contacts, urls, action_items, sensitive_info, text_topics),
field-level system confidence scores, and optional user_intent.
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

# Hierarchical taxonomy: Category -> Content Types
CATEGORY_CONTENT_TYPES: Dict[str, List[str]] = {
    "Education": ["course_material", "exam_schedule", "assignment", "notes"],
    "Payment": ["payment_receipt", "bill", "transaction", "transfer"],
    "Shopping": ["product_page", "shopping_order", "delivery_tracking", "refund"],
    "Travel": ["flight_ticket", "train_ticket", "hotel_booking", "travel_itinerary"],
    "Coding": ["code_editor", "terminal", "error_log", "documentation"],
    "Communication": ["chat", "email", "message"],
    "Work & Career": ["job_posting", "interview", "resume", "professional_message"],
    "Entertainment": ["movie", "video", "music", "event"],
    "Connectivity": ["wifi", "bluetooth", "network_settings"],
    "General": ["unknown", "general"],
}


@dataclass
class ScreenshotMetadata:
    """
    Structured Application Metadata Model for Screenshot Entities.

    Contract specification:
    - Identity/Storage: screenshot_id, filename, image_uri
    - System/OCR Metadata: category, content_type, ocr_available, created_at, indexed_at
    - Evidence Fields: evidence, events, financial_info, contacts, urls, text_topics
    - Metadata & Safety: tags, intent, summary, importance, entities, dates, action_items, sensitive_info, confidence_scores
    """

    screenshot_id: str
    filename: str
    image_uri: str
    category: str = "General"
    content_type: Optional[str] = "unknown"
    ocr_available: bool = True
    created_at: Optional[str] = None
    indexed_at: Optional[str] = None

    # Intelligence & Evidence Fields (Nullable / Default Empty)
    tags: List[str] = field(default_factory=list)
    intent: Optional[str] = None  # Optional: None if evidence is insufficient
    summary: Optional[str] = None
    importance: Optional[int] = None

    # Structured Evidence & Signals
    evidence: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    financial_info: Dict[str, Any] = field(default_factory=dict)
    contacts: Dict[str, Any] = field(default_factory=dict)
    urls: List[str] = field(default_factory=list)
    text_topics: List[str] = field(default_factory=list)
    entities: List[Dict[str, str]] = field(default_factory=list)
    dates: List[Dict[str, str]] = field(default_factory=list)
    action_items: List[Dict[str, str]] = field(default_factory=list)
    sensitive_info: Dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_SENSITIVE_INFO))
    confidence_scores: Dict[str, float] = field(default_factory=dict)

    # Multi-Signal Classification & Ambiguity Fields
    category_candidates: List[Dict[str, Any]] = field(default_factory=list)
    category_scores: Dict[str, float] = field(default_factory=dict)
    classification_confidence: float = 0.0
    classification_ambiguous: bool = False
    classification_margin: float = 0.0
    classification_evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Set ISO 8601 timestamps if not provided
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.indexed_at:
            self.indexed_at = datetime.now(timezone.utc).isoformat()

        if self.content_type is None or not str(self.content_type).strip():
            self.content_type = "unknown"

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

        if not isinstance(self.evidence, dict):
            raise ValueError("evidence must be a dictionary.")

        if not isinstance(self.events, list):
            raise ValueError("events must be a list of structured dictionaries.")

        if not isinstance(self.financial_info, dict):
            raise ValueError("financial_info must be a dictionary.")

        if not isinstance(self.contacts, dict):
            raise ValueError("contacts must be a dictionary.")

        if not isinstance(self.urls, list):
            raise ValueError("urls must be a list of strings.")

        if not isinstance(self.text_topics, list):
            raise ValueError("text_topics must be a list of strings.")

        if not isinstance(self.confidence_scores, dict):
            raise ValueError("confidence_scores must be a dictionary of float values.")

    def to_dict(self) -> Dict[str, Any]:
        """Converts metadata object to clean dictionary representation with Python structures."""
        return {
            "screenshot_id": self.screenshot_id,
            "filename": self.filename,
            "image_uri": self.image_uri,
            "category": self.category,
            "content_type": self.content_type,
            "ocr_available": self.ocr_available,
            "created_at": self.created_at,
            "indexed_at": self.indexed_at,
            "tags": self.tags,
            "intent": self.intent,
            "summary": self.summary,
            "importance": self.importance,
            "evidence": self.evidence,
            "events": self.events,
            "financial_info": self.financial_info,
            "contacts": self.contacts,
            "urls": self.urls,
            "text_topics": self.text_topics,
            "entities": self.entities,
            "dates": self.dates,
            "action_items": self.action_items,
            "sensitive_info": self.sensitive_info,
            "confidence_scores": self.confidence_scores,
            "category_candidates": self.category_candidates,
            "category_scores": self.category_scores,
            "classification_confidence": self.classification_confidence,
            "classification_ambiguous": self.classification_ambiguous,
            "classification_margin": self.classification_margin,
            "classification_evidence": self.classification_evidence,
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

        intent_val = data.get("intent")
        if intent_val is not None and not str(intent_val).strip():
            intent_val = None

        return cls(
            screenshot_id=str(data["screenshot_id"]).strip() if data.get("screenshot_id") else "",
            filename=str(data["filename"]).strip() if data.get("filename") else "",
            image_uri=str(data["image_uri"]).strip() if data.get("image_uri") else "",
            category=str(data.get("category", "General")),
            content_type=str(data.get("content_type", "unknown")) if data.get("content_type") else "unknown",
            ocr_available=bool(data.get("ocr_available", True)),
            created_at=data.get("created_at"),
            indexed_at=data.get("indexed_at"),
            tags=cls._parse_json_list(data.get("tags")),
            intent=intent_val,
            summary=data.get("summary") if data.get("summary") else None,
            importance=importance_val,
            evidence=cls._parse_json_dict(data.get("evidence")),
            events=cls._parse_json_list(data.get("events")),
            financial_info=cls._parse_json_dict(data.get("financial_info")),
            contacts=cls._parse_json_dict(data.get("contacts")),
            urls=cls._parse_json_list(data.get("urls")),
            text_topics=cls._parse_json_list(data.get("text_topics")),
            entities=cls._parse_json_list(data.get("entities")),
            dates=cls._parse_json_list(data.get("dates")),
            action_items=cls._parse_json_list(data.get("action_items")),
            sensitive_info=cls._parse_sensitive_info(data.get("sensitive_info")),
            confidence_scores=cls._parse_json_dict_float(data.get("confidence_scores")),
            category_candidates=cls._parse_json_list(data.get("category_candidates")),
            category_scores=cls._parse_json_dict_float(data.get("category_scores")),
            classification_confidence=float(data.get("classification_confidence", 0.0)),
            classification_ambiguous=bool(data.get("classification_ambiguous", False)),
            classification_margin=float(data.get("classification_margin", 0.0)),
            classification_evidence=cls._parse_json_dict(data.get("classification_evidence")),
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
    def _parse_json_dict(cls, val: Any) -> Dict[str, Any]:
        """Safely parses dict or JSON stringified dict into a Python dict."""
        if val is None or val == "":
            return {}
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    @classmethod
    def _parse_json_dict_float(cls, val: Any) -> Dict[str, float]:
        """Safely parses confidence_scores dict or JSON string."""
        d = cls._parse_json_dict(val)
        res = {}
        for k, v in d.items():
            try:
                res[str(k)] = float(v)
            except (ValueError, TypeError):
                pass
        return res

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
