"""
VLM / Screenshot Understanding Module
=====================================
Abstract base class, deterministic mock engine, and factory function for Visual Language Model (VLM)
screenshot understanding, extracting structured metadata (category, tags, intent, summary,
importance, entities, dates, action_items, sensitive_info).
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import config


class BaseVLM(ABC):
    """
    Abstract Base Class for Visual Language Model (VLM) screenshot understanding engines.
    Real VLM implementations (e.g. Qwen2-VL, LLaVA, Gemini VLM) should subclass this interface.
    """

    @abstractmethod
    def analyze_image(self, image_path: str, ocr_text: str = "") -> Dict[str, Any]:
        """
        Analyze a screenshot image and optional pre-extracted OCR text to return structured understanding metadata.

        Args:
            image_path: Absolute or relative path to target image file.
            ocr_text: Optional pre-extracted OCR text string.

        Returns:
            Dictionary matching the VLM understanding schema:
            {
                "category": str,
                "tags": list[str],
                "intent": str | None,
                "summary": str | None,
                "importance": int | None,
                "entities": list[dict],
                "dates": list[dict],
                "action_items": list[dict],
                "sensitive_info": dict
            }

        Raises:
            FileNotFoundError: If image_path does not exist.
            ValueError: If returned analysis output violates structural schema constraints.
        """
        pass

    def validate_output(self, output: Dict[str, Any]) -> None:
        """
        Validates structure and types of the returned VLM analysis dictionary.

        Args:
            output: Analysis result dictionary to validate.

        Raises:
            ValueError: If any field violates structural schema constraints.
        """
        if not isinstance(output, dict):
            raise ValueError("VLM analysis output must be a dictionary.")

        category = output.get("category")
        if not isinstance(category, str):
            raise ValueError(f"category must be a string, got {type(category).__name__}.")

        tags = output.get("tags")
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise ValueError("tags must be a list of strings.")

        intent = output.get("intent")
        if intent is not None and not isinstance(intent, str):
            raise ValueError("intent must be None or a string.")

        summary = output.get("summary")
        if summary is not None and not isinstance(summary, str):
            raise ValueError("summary must be None or a string.")

        importance = output.get("importance")
        if importance is not None:
            if isinstance(importance, bool) or not isinstance(importance, int):
                raise ValueError(
                    f"importance must be an integer (1-5) or None, got {type(importance).__name__}."
                )
            if not (1 <= importance <= 5):
                raise ValueError(f"importance must be between 1 and 5, got {importance}.")

        for list_field in ("entities", "dates", "action_items"):
            val = output.get(list_field)
            if not isinstance(val, list):
                raise ValueError(f"{list_field} must be a list, got {type(val).__name__}.")

        sensitive_info = output.get("sensitive_info")
        if not isinstance(sensitive_info, dict):
            raise ValueError(
                f"sensitive_info must be a dictionary, got {type(sensitive_info).__name__}."
            )


class MockVLM(BaseVLM):
    """
    Deterministic Mock VLM engine providing simulated visual understanding outputs
    grounded in representative sample screenshots across functional categories.
    """

    DEFAULT_MOCK_VLM_MAPPINGS: Dict[str, Dict[str, Any]] = {
        "ss_001.jpg": {
            "category": "Connectivity",
            "tags": ["wifi", "password", "network", "airtel"],
            "intent": "Remember network credential",
            "summary": "Wi-Fi network credentials and password for Airtel router",
            "importance": 4,
            "entities": [
                {"text": "Airtel_runu_7550", "type": "network_name"},
                {"text": "Air@53054", "type": "password"},
            ],
            "dates": [],
            "action_items": [],
            "sensitive_info": {
                "contains_password": True,
                "contains_payment_info": False,
                "contains_personal_contact": False,
            },
        },
        "ss_002.jpg": {
            "category": "Payment",
            "tags": ["payment", "receipt", "upi", "braingrow"],
            "intent": "Keep payment record",
            "summary": "UPI payment receipt of ₹689 paid to BRAINGROW EDUSERV",
            "importance": 4,
            "entities": [
                {"text": "BRAINGROW EDUSERV PRIVATE LIMITED", "type": "merchant"},
                {"text": "pay_SdleqofNvlmOZg", "type": "payment_id"},
            ],
            "dates": [
                {"text": "15th Apr, 2026 17:55:24 PM IST", "type": "transaction_date"}
            ],
            "action_items": [],
            "sensitive_info": {
                "contains_password": False,
                "contains_payment_info": True,
                "contains_personal_contact": True,
            },
        },
        "ss_003.jpg": {
            "category": "Education",
            "tags": ["syllabus", "deep-learning", "kiit", "course"],
            "intent": "Study course material",
            "summary": "KIIT Deep Learning course syllabus covering units I-V and textbooks",
            "importance": 3,
            "entities": [
                {"text": "KIIT", "type": "institution"},
                {"text": "Deep Learning", "type": "subject"},
            ],
            "dates": [],
            "action_items": [
                {"text": "Read Goodfellow Deep Learning textbook", "status": "pending"}
            ],
            "sensitive_info": {
                "contains_password": False,
                "contains_payment_info": False,
                "contains_personal_contact": False,
            },
        },
        "ss_008.jpg": {
            "category": "Shopping",
            "tags": ["menu", "food", "prices", "combo"],
            "intent": "View food options and pricing",
            "summary": "MD Combo food menu item listings and prices",
            "importance": 2,
            "entities": [
                {"text": "MD Combo", "type": "product"}
            ],
            "dates": [],
            "action_items": [],
            "sensitive_info": {
                "contains_password": False,
                "contains_payment_info": False,
                "contains_personal_contact": False,
            },
        },
        "ss_0011.jpg": {
            "category": "Travel",
            "tags": ["flight", "indigo", "boarding-pass", "guwahati"],
            "intent": "Prepare for flight travel",
            "summary": "IndiGo flight boarding pass to Guwahati",
            "importance": 5,
            "entities": [
                {"text": "IndiGo", "type": "airline"},
                {"text": "Guwahati", "type": "destination"},
            ],
            "dates": [
                {"text": "22nd Sep, 2026", "type": "travel_date"}
            ],
            "action_items": [
                {"text": "Report to airport boarding gate on time", "status": "pending"}
            ],
            "sensitive_info": {
                "contains_password": False,
                "contains_payment_info": False,
                "contains_personal_contact": True,
            },
        },
        "ss_0021.jpg": {
            "category": "Communication",
            "tags": ["chat", "whatsapp", "messages", "notification"],
            "intent": "Check messaging conversation",
            "summary": "WhatsApp/Telegram chat notification overview",
            "importance": 3,
            "entities": [
                {"text": "WhatsApp", "type": "application"}
            ],
            "dates": [],
            "action_items": [
                {"text": "Reply to pending messages", "status": "pending"}
            ],
            "sensitive_info": {
                "contains_password": False,
                "contains_payment_info": False,
                "contains_personal_contact": True,
            },
        },
        "ss_0028.jpg": {
            "category": "Coding",
            "tags": ["express", "controller", "code", "javascript"],
            "intent": "Reference code implementation",
            "summary": "VS Code Express.js controller code snippet",
            "importance": 3,
            "entities": [
                {"text": "Express", "type": "framework"}
            ],
            "dates": [],
            "action_items": [],
            "sensitive_info": {
                "contains_password": False,
                "contains_payment_info": False,
                "contains_personal_contact": False,
            },
        },
        "ss_0030.jpg": {
            "category": "Work & Career",
            "tags": ["hackathon", "openai", "event", "career"],
            "intent": "Participate in AI competition",
            "summary": "OpenAI AI Hackathon event announcement details",
            "importance": 4,
            "entities": [
                {"text": "OpenAI", "type": "organization"},
                {"text": "Hackathon", "type": "event"},
            ],
            "dates": [],
            "action_items": [
                {"text": "Register team for OpenAI Hackathon", "status": "pending"}
            ],
            "sensitive_info": {
                "contains_password": False,
                "contains_payment_info": False,
                "contains_personal_contact": False,
            },
        },
    }

    FALLBACK_VLM_RESPONSE: Dict[str, Any] = {
        "category": "General",
        "tags": ["screenshot", "general"],
        "intent": "General screenshot capture",
        "summary": "General screenshot image with no domain-specific visual template match.",
        "importance": 1,
        "entities": [],
        "dates": [],
        "action_items": [],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": False,
            "contains_personal_contact": False,
        },
    }

    def __init__(self, mappings: Optional[Dict[str, Dict[str, Any]]] = None):
        self.mappings = mappings if mappings is not None else self.DEFAULT_MOCK_VLM_MAPPINGS

    def analyze_image(self, image_path: str, ocr_text: str = "") -> Dict[str, Any]:
        """
        Analyze screenshot image and optional pre-extracted OCR text using deterministic mappings.

        Args:
            image_path: Absolute or relative path to target image file.
            ocr_text: Optional pre-extracted OCR text string.

        Returns:
            Dictionary matching the VLM understanding schema.

        Raises:
            FileNotFoundError: If specified image_path does not exist.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Screenshot image file not found: {image_path}")

        filename = path.name
        raw_result = self.mappings.get(filename)
        if raw_result is None:
            result = json.loads(json.dumps(self.FALLBACK_VLM_RESPONSE))
        else:
            result = json.loads(json.dumps(raw_result))

        self.validate_output(result)
        return result


def get_vlm_engine(engine_type: str = "mock", **kwargs) -> BaseVLM:
    """
    Factory function to instantiate and return a swappable BaseVLM engine.

    Args:
        engine_type: Type identifier ('mock'). Default is 'mock'.
        **kwargs: Optional keyword arguments passed to the VLM engine constructor.

    Returns:
        Instance of BaseVLM (MockVLM).

    Raises:
        ValueError: If engine_type is unrecognized.
    """
    engine_type_clean = engine_type.strip().lower()
    if engine_type_clean == "mock":
        return MockVLM(**kwargs)
    else:
        raise ValueError(
            f"Unsupported VLM engine type: '{engine_type}'. Supported engine: 'mock'."
        )
