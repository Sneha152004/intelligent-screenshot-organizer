"""
Phase 7J: Visual Screen-Type Integration Contract & Schema Definition.

Defines the formal data contracts, validation rules, taxonomy boundaries,
and non-production Unknown/Unsupported class handling protocols.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# Full 13-class taxonomy
FULL_TAXONOMY_13_CLASSES: List[str] = [
    "product_listing",
    "product_detail",
    "cart_checkout_payment",
    "login_authentication",
    "messaging_chat",
    "email_view",
    "media_player_library",
    "code_editor_terminal",
    "document_article_view",
    "travel_booking",
    "dashboard_settings",
    "error_dialog_alert",
    "other_screen"
]

# Currently supported 5-class supervised benchmark model classes
SUPPORTED_5_CLASSES: List[str] = [
    "document_article_view",
    "travel_booking",
    "dashboard_settings",
    "media_player_library",
    "code_editor_terminal"
]


@dataclass
class RankedPrediction:
    """Represents a single ranked visual screen-type prediction alternative."""
    screen_type: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "screen_type": self.screen_type,
            "confidence": round(float(self.confidence), 4)
        }


@dataclass
class VisualUnderstandingResult:
    """
    Formal Data Contract for Visual Screen-Type Model Output.
    """
    screen_type: str
    confidence: float
    ranked_predictions: List[Dict[str, Any]]
    model_name: str = "yolov8n-cls"
    model_version: str = "1.0.0-phase7i"
    checkpoint_type: str = "FP32"
    inference_latency_ms: float = 0.0
    supported_classes: List[str] = field(default_factory=lambda: list(SUPPORTED_5_CLASSES))
    is_supported_class: bool = True
    evidence_status: str = "SUPPORTING_VISUAL_EVIDENCE"

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validates contract invariants and structural boundaries."""
        if not isinstance(self.screen_type, str) or not self.screen_type.strip():
            raise ValueError("screen_type must be a non-empty string.")

        if self.screen_type not in FULL_TAXONOMY_13_CLASSES:
            raise ValueError(f"screen_type '{self.screen_type}' is not in full 13-class taxonomy.")

        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}.")

        if not isinstance(self.ranked_predictions, list) or len(self.ranked_predictions) == 0:
            raise ValueError("ranked_predictions must be a non-empty list.")

        # Check ranking ordering
        prev_conf = 1.01
        for item in self.ranked_predictions:
            if "screen_type" not in item or "confidence" not in item:
                raise ValueError("Each ranked prediction item must contain 'screen_type' and 'confidence'.")
            conf = float(item["confidence"])
            if conf > prev_conf + 1e-6:
                raise ValueError(f"Ranked predictions are not sorted descending: {conf} > {prev_conf}")
            prev_conf = conf

        if self.model_name != "yolov8n-cls":
            raise ValueError(f"Invalid model_name: {self.model_name}, expected 'yolov8n-cls'.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "screen_type": self.screen_type,
            "confidence": round(float(self.confidence), 4),
            "ranked_predictions": self.ranked_predictions,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "checkpoint_type": self.checkpoint_type,
            "inference_latency_ms": round(float(self.inference_latency_ms), 2),
            "supported_classes": self.supported_classes,
            "is_supported_class": self.is_supported_class,
            "evidence_status": self.evidence_status
        }


def validate_visual_contract(result_dict: Dict[str, Any]) -> VisualUnderstandingResult:
    """
    Factory method validating raw dictionary output against the VisualUnderstandingResult contract.
    """
    return VisualUnderstandingResult(
        screen_type=str(result_dict.get("screen_type", "other_screen")),
        confidence=float(result_dict.get("confidence", 0.0)),
        ranked_predictions=list(result_dict.get("ranked_predictions", [])),
        model_name=str(result_dict.get("model_name", "yolov8n-cls")),
        model_version=str(result_dict.get("model_version", "1.0.0-phase7i")),
        checkpoint_type=str(result_dict.get("checkpoint_type", "FP32")),
        inference_latency_ms=float(result_dict.get("inference_latency_ms", 0.0)),
        supported_classes=list(result_dict.get("supported_classes", SUPPORTED_5_CLASSES)),
        is_supported_class=bool(result_dict.get("is_supported_class", True)),
        evidence_status=str(result_dict.get("evidence_status", "SUPPORTING_VISUAL_EVIDENCE"))
    )
