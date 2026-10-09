"""
Visual Intelligence Branch - Vision Encoders and Evidence Module
Provides abstract base interface, concrete implementations for visual encoders (TinyCLIP-8M, MobileNetV2),
and the VisualCategoryEvidenceEngine for 9-category zero-shot evidence generation.
"""

from src.visual.base_visual_encoder import (
    BaseVisualEncoder,
    VisualEncoderError,
    ModelLoadError,
    PreprocessingError,
    InferenceError,
)
from src.visual.tiny_clip_encoder import TinyCLIPEncoder
from src.visual.mobilenet_encoder import MobileNetV2Encoder
from src.visual.visual_evidence import (
    VisualCategoryEvidenceEngine,
    CANONICAL_CATEGORIES,
    DEFAULT_CATEGORY_PROMPTS,
)

__all__ = [
    "BaseVisualEncoder",
    "VisualEncoderError",
    "ModelLoadError",
    "PreprocessingError",
    "InferenceError",
    "TinyCLIPEncoder",
    "MobileNetV2Encoder",
    "VisualCategoryEvidenceEngine",
    "CANONICAL_CATEGORIES",
    "DEFAULT_CATEGORY_PROMPTS",
]
