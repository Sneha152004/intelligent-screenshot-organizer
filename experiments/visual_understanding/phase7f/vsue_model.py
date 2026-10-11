import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from .dataset_utils import (
    SUPPORTED_CLASSES,
    UNSUPPORTED_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
)


class VisualScreenTypeClassifier(nn.Module):
    """
    Supervised Visual Screen Understanding Engine (VSUE) Baseline.
    Uses MobileNetV3-Small backbone pretrained on ImageNet, replacing the classification head
    with a 5-class output layer for supported screen types.
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(VisualScreenTypeClassifier, self).__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)

        # Replace 1000-class classifier head with 5-class linear layer
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def format_vsue_payload(
        self,
        probs: np.ndarray,
        meta_dict: dict = None,
        model_version: str = "vsue_mobilenet_v3_small_v1"
    ) -> dict:
        """
        Formats raw model output probabilities into the structured VSUE output contract payload.
        """
        top_idx = int(np.argmax(probs))
        pred_class = IDX_TO_CLASS[top_idx]
        conf = float(probs[top_idx])

        class_probs = {IDX_TO_CLASS[i]: round(float(probs[i]), 4) for i in range(len(SUPPORTED_CLASSES))}

        payload = {
            "screen_type": pred_class,
            "screen_type_confidence": round(conf, 4),
            "screen_type_probabilities": class_probs,
            "unsupported_taxonomy_classes": UNSUPPORTED_CLASSES,
            "layout_features": {
                "aspect_ratio": meta_dict.get("aspect_ratio", 0.52) if meta_dict else 0.52,
                "orientation": meta_dict.get("orientation", "portrait") if meta_dict else "portrait",
                "orig_dimensions": meta_dict.get("orig_size", (0, 0)) if meta_dict else (0, 0),
            },
            "model_version": model_version
        }
        return payload
