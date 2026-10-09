"""
Phase 7J: Visual Screen-Type Model Adapter for Ultralytics YOLOv8n-cls.

Provides a isolated, non-production wrapper around the verified Phase 7I YOLOv8n-cls model checkpoint.
Enforces aspect-ratio-preserving letterbox preprocessing, structured prediction dictionary outputs,
and explicit taxonomy boundaries.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import copy
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from ultralytics import YOLO

PHASE7J_DIR = Path(__file__).resolve().parent
VISUAL_UNDERSTANDING_DIR = PHASE7J_DIR.parent
PROJECT_ROOT = VISUAL_UNDERSTANDING_DIR.parent.parent
PHASE7I_DIR = VISUAL_UNDERSTANDING_DIR / "phase7i"

# Add Phase 7I to sys.path to access dataset_preprocessing
if str(PHASE7I_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE7I_DIR))

from dataset_preprocessing import letterbox_resize, IMAGENET_MEAN, IMAGENET_STD


class YOLOClassificationWrapper(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        yolo_wrapper = YOLO("yolov8n-cls.pt")
        self.backbone = copy.deepcopy(yolo_wrapper.model)
        # Adapt final classification head
        in_features = self.backbone.model[9].linear.in_features if hasattr(self.backbone.model[9], "linear") else 1280
        self.backbone.model[9].linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

# Standard 13-class taxonomy
FULL_TAXONOMY_13_CLASSES = [
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

# Primary 5-class benchmark taxonomy supported by current supervised YOLOv8n-cls model
SUPPORTED_5_CLASSES = [
    "document_article_view",
    "travel_booking",
    "dashboard_settings",
    "media_player_library",
    "code_editor_terminal"
]


class VisualScreenTypeAdapter:
    """
    Isolated adapter for visual screen-type inference using verified YOLOv8n-cls model weights.
    """
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        use_int8: bool = False,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.use_int8 = use_int8
        self.supported_classes = list(SUPPORTED_5_CLASSES)
        self.class_to_idx = {c: i for i, c in enumerate(self.supported_classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.supported_classes)}

        if checkpoint_path is None:
            ckpt_name = "visual_screen_type_yolo26n_cls_int8.pth" if use_int8 else "visual_screen_type_yolo26n_cls.pth"
            checkpoint_path = PHASE7I_DIR / "model_checkpoints" / ckpt_name

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Visual model checkpoint not found at {self.checkpoint_path}")

        self.model = self._load_model()
        self.model.eval()

        # ImageNet normalization transform parameters
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(self.device)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(self.device)

    def _load_model(self) -> nn.Module:
        """Instantiates YOLOv8n-cls architecture and loads checkpoint state dict."""
        base_model = YOLOClassificationWrapper(num_classes=len(self.supported_classes))

        if self.use_int8:
            int8_model = torch.ao.quantization.quantize_dynamic(
                base_model.to("cpu"),
                {nn.Linear},
                dtype=torch.qint8
            )
            state_dict = torch.load(self.checkpoint_path, map_location="cpu")
            int8_model.load_state_dict(state_dict)
            return int8_model.to(self.device)
        else:
            ckpt_data = torch.load(self.checkpoint_path, map_location=self.device)
            state_dict = ckpt_data.get("model_state_dict", ckpt_data)
            base_model.load_state_dict(state_dict)
            return base_model.to(self.device)

    def preprocess_image(self, image_input: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Preprocesses image using Phase 7I letterboxing (224x224), RGBA compositing, and ImageNet normalization.
        """
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Screenshot image file missing: {img_path}")
            with Image.open(img_path) as img:
                img.load()
                processed_img = letterbox_resize(img, target_size=(224, 224), fill_color=(128, 128, 128))
        elif isinstance(image_input, Image.Image):
            processed_img = letterbox_resize(image_input, target_size=(224, 224), fill_color=(128, 128, 128))
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        img_np = np.array(processed_img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(self.device)
        norm_tensor = (img_tensor - self.mean) / self.std
        return norm_tensor.unsqueeze(0)  # Shape (1, 3, 224, 224)

    def predict_image(self, image_input: Union[str, Path, Image.Image]) -> Dict:
        """
        Runs visual inference and returns structured visual prediction object.
        """
        t_start = time.perf_counter()

        tensor_input = self.preprocess_image(image_input)

        with torch.no_grad():
            logits = self.model(tensor_input)
            probs_tensor = torch.softmax(logits, dim=1).squeeze(0)
            probs = probs_tensor.cpu().numpy()

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        top_idx = int(np.argmax(probs))
        top_screen_type = self.idx_to_class[top_idx]
        top_conf = float(probs[top_idx])

        # Rank all predictions in descending confidence order
        sorted_indices = np.argsort(probs)[::-1]
        ranked_predictions = [
            {
                "screen_type": self.idx_to_class[i],
                "confidence": round(float(probs[i]), 4)
            }
            for i in sorted_indices
        ]

        return {
            "screen_type": top_screen_type,
            "confidence": round(top_conf, 4),
            "ranked_predictions": ranked_predictions,
            "model_name": "yolov8n-cls",
            "model_version": "1.0.0-phase7i",
            "checkpoint_type": "INT8" if self.use_int8 else "FP32",
            "inference_latency_ms": round(latency_ms, 2),
            "supported_classes": list(self.supported_classes),
            "is_supported_class": True  # All predictions mapped from the 5 supported classes
        }


if __name__ == "__main__":
    print("Testing VisualScreenTypeAdapter...")
    adapter = VisualScreenTypeAdapter(use_int8=False)
    dummy_img = Image.new("RGB", (800, 600), (100, 150, 200))
    res = adapter.predict_image(dummy_img)
    print("Adapter Output:")
    import json
    print(json.dumps(res, indent=2))
    print("VisualScreenTypeAdapter test passed.")
