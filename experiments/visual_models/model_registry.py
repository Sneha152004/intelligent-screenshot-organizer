"""
Model Registry for Visual Model Research.
Defines unified model creation, feature extraction, and 9-class head creation
for candidate vision backbones: MobileNetV3-Small, MobileNetV3-Large, EfficientNet-B0, ShuffleNetV2, MobileNetV2, TinyCLIP.
"""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
)

CANONICAL_CATEGORIES = [
    "Education",
    "Shopping",
    "Payment",
    "Travel",
    "Communication",
    "Entertainment",
    "Work & Career",
    "Connectivity",
    "General",
]


class ScreenshotClassifierHead(nn.Module):
    """Lightweight 9-class classifier head for frozen CNN feature representations."""

    def __init__(self, in_features: int, num_classes: int = 9, dropout_prob: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob / 2.0),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class CNNVisualClassifier(nn.Module):
    """Combines a frozen CNN backbone feature extractor with ScreenshotClassifierHead."""

    def __init__(self, backbone_name: str, num_classes: int = 9):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        if backbone_name == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT
            base = mobilenet_v3_small(weights=weights)
            in_features = base.classifier[0].in_features
            base.classifier = nn.Identity()
            self.transform = weights.transforms()

        elif backbone_name == "mobilenet_v3_large":
            weights = MobileNet_V3_Large_Weights.DEFAULT
            base = mobilenet_v3_large(weights=weights)
            in_features = base.classifier[0].in_features
            base.classifier = nn.Identity()
            self.transform = weights.transforms()

        elif backbone_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT
            base = efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.transform = weights.transforms()

        elif backbone_name == "shufflenet_v2":
            weights = ShuffleNet_V2_X1_0_Weights.DEFAULT
            base = shufflenet_v2_x1_0(weights=weights)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.transform = weights.transforms()

        elif backbone_name == "mobilenet_v2":
            weights = MobileNet_V2_Weights.DEFAULT
            base = mobilenet_v2(weights=weights)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.transform = weights.transforms()

        else:
            raise ValueError(f"Unsupported backbone_name: '{backbone_name}'")

        self.backbone = base
        self.in_features = in_features
        self.head = ScreenshotClassifierHead(in_features=in_features, num_classes=num_classes)

        # Freeze backbone weights initially
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        logits = self.head(feat)
        return logits

    def unfreeze_backbone_top_layers(self, unfreeze_last_blocks: int = 2):
        """Unfreezes top layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_model_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        size_mb = round(total_params * 4 / (1024 * 1024), 2)
        return {
            "backbone": self.backbone_name,
            "in_features": self.in_features,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "fp32_size_mb": size_mb,
        }


def get_candidate_model_registry() -> Dict[str, Dict[str, Any]]:
    """Metadata dictionary describing candidate visual architectures."""
    return {
        "mobilenet_v3_small": {
            "display_name": "MobileNetV3-Small",
            "type": "CNN",
            "weights_source": "torchvision.models.MobileNet_V3_Small_Weights.DEFAULT",
            "input_resolution": "224x224",
            "feature_dim": 576,
        },
        "mobilenet_v3_large": {
            "display_name": "MobileNetV3-Large",
            "type": "CNN",
            "weights_source": "torchvision.models.MobileNet_V3_Large_Weights.DEFAULT",
            "input_resolution": "224x224",
            "feature_dim": 960,
        },
        "efficientnet_b0": {
            "display_name": "EfficientNet-B0",
            "type": "CNN",
            "weights_source": "torchvision.models.EfficientNet_B0_Weights.DEFAULT",
            "input_resolution": "224x224",
            "feature_dim": 1280,
        },
        "shufflenet_v2": {
            "display_name": "ShuffleNetV2 (1.0x)",
            "type": "CNN",
            "weights_source": "torchvision.models.ShuffleNet_V2_X1_0_Weights.DEFAULT",
            "input_resolution": "224x224",
            "feature_dim": 1024,
        },
        "mobilenet_v2": {
            "display_name": "MobileNetV2 Baseline",
            "type": "CNN",
            "weights_source": "torchvision.models.MobileNet_V2_Weights.DEFAULT",
            "input_resolution": "224x224",
            "feature_dim": 1280,
        },
        "tiny_clip": {
            "display_name": "TinyCLIP-8M Baseline",
            "type": "Vision-Language (ViT)",
            "weights_source": "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
            "input_resolution": "224x224",
            "feature_dim": 512,
        },
        "mobile_clip_s0": {
            "display_name": "MobileCLIP-S0 (Incompatible)",
            "type": "Vision-Language",
            "weights_source": "apple/MobileCLIP-S0",
            "input_resolution": "224x224",
            "feature_dim": 512,
            "status": "Incompatible (Unrecognized model_type in HF transformers v4.57)",
        },
    }
