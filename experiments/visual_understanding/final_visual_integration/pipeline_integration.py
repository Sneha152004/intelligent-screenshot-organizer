"""
Final Visual Integration Pipeline & Safe Fallback Module.

Implements the production-ready multimodal screenshot understanding architecture:
  Screenshot -> Semantic Classifier (FinalRefinedClassifier)
             -> Visual Classifier (YOLOv8n-cls 13-Class)
             -> Selective Fusion Layer
             -> Rich ScreenshotMetadata with visual_evidence & fallback protection.
"""

import os
import sys
import time
import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from typing import Dict, Any, Optional, Union
from ultralytics import YOLO

# Setup paths
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent.parent
MODEL_PATH = MODULE_DIR / "models" / "yolov8n_cls_13class.pth"
INT8_MODEL_PATH = MODULE_DIR / "models" / "yolov8n_cls_13class_int8.pth"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "intelligence" / "final_refinement"))

from final_refinement_experiment import FinalRefinedClassifier

logger = logging.getLogger(__name__)

VIS_13_TO_SEMANTIC = {
    "code_editor_terminal": "Coding",
    "travel_booking": "Travel",
    "media_player_library": "Entertainment",
    "document_article_view": "Education",
    "dashboard_settings": "Connectivity",
    "product_listing": "Shopping",
    "product_detail": "Shopping",
    "cart_checkout_payment": "Payment",
    "login_authentication": "Connectivity",
    "messaging_chat": "Communication",
    "email_view": "Communication",
    "error_dialog_alert": "General",
    "other_screen": "General"
}

def letterbox_resize(img: Image.Image, target_size: int = 224) -> Image.Image:
    if img.mode != 'RGB':
        img = img.convert('RGB')
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img_resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
    new_img = Image.new('RGB', (target_size, target_size), (114, 114, 114))
    new_img.paste(img_resized, ((target_size - nw) // 2, (target_size - nh) // 2))
    return new_img

transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class YOLO13ClassModelWrapper(nn.Module):
    def __init__(self, num_classes: int = 13):
        super().__init__()
        yolo_base = YOLO("yolov8n-cls.pt")
        self.backbone = yolo_base.model
        in_features = self.backbone.model[9].linear.in_features if hasattr(self.backbone.model[9], "linear") else 1280
        self.backbone.model[9].linear = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


class IntegratedMultimodalPipeline:
    """
    Unified Production-Ready Multimodal Screenshot Classifier.
    Includes automated visual evidence enrichment, selective fusion, and fallback protection.
    """
    def __init__(self, model_path: Optional[Path] = None, use_int8: bool = False):
        self.semantic_classifier = FinalRefinedClassifier()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_model = None
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.visual_model_loaded = False
        
        target_path = model_path if model_path else MODEL_PATH
        if use_int8 and INT8_MODEL_PATH.exists():
            target_path = INT8_MODEL_PATH
            
        try:
            if target_path.exists():
                ckpt = torch.load(target_path, map_location=self.device)
                self.class_to_idx = ckpt.get('class_to_idx', {})
                self.idx_to_class = ckpt.get('idx_to_class', {i: c for c, i in self.class_to_idx.items()})
                
                self.visual_model = YOLO13ClassModelWrapper(num_classes=len(self.class_to_idx)).to(self.device)
                self.visual_model.load_state_dict(ckpt['model_state_dict'])
                self.visual_model.eval()
                self.visual_model_loaded = True
                logger.info(f"Loaded 13-class visual model checkpoint from {target_path}")
            else:
                logger.warning(f"Visual model checkpoint not found at {target_path}. Pipeline running in semantic-only fallback mode.")
        except Exception as e:
            logger.error(f"Failed to load visual model ({e}). Pipeline running in semantic-only fallback mode.")
            self.visual_model_loaded = False

    def classify_screenshot(
        self,
        image_path: Union[str, Path],
        ocr_text: str = "",
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Main end-to-end processing pipeline.
        Always returns valid prediction dict, with safe fallback on visual failures.
        """
        img_p = Path(image_path)
        text_input = ocr_text if ocr_text else (notes if notes else img_p.name)
        
        # 1. Semantic Branch Prediction (Authoritative Baseline)
        sem_res = self.semantic_classifier.predict_final(raw_text=text_input, notes=notes, variant="C")
        sem_category = sem_res.get('prediction', 'General')
        sem_confidence = sem_res.get('confidence', 'Low')
        is_ambiguous = bool(sem_res.get('is_ambiguous', False))
        
        # Initialize Visual Evidence & Fusion Defaults
        visual_evidence = None
        final_category = sem_category
        fusion_applied = False
        fusion_reason = "Semantic classifier prediction authoritative"
        
        # 2. Visual Branch Prediction with Safe Fallback
        if self.visual_model_loaded and img_p.exists():
            try:
                with Image.open(img_p) as img:
                    img_lb = letterbox_resize(img, 224)
                    t_img = transform_pipeline(img_lb).unsqueeze(0).to(self.device)
                    
                with torch.no_grad():
                    logits = self.visual_model(t_img)
                    probs = torch.softmax(logits / 1.5, dim=1).cpu().numpy()[0]
                    vis_idx = int(np.argmax(probs))
                    vis_st = self.idx_to_class[vis_idx]
                    vis_conf = float(probs[vis_idx])
                    
                # Visual Evidence Dict
                visual_evidence = {
                    "screen_type": vis_st,
                    "confidence": round(vis_conf, 4),
                    "model_name": "YOLOv8n-cls",
                    "model_version": "13-class-v1",
                    "fusion_applied": False,
                    "fusion_reason": fusion_reason,
                    "original_semantic_category": sem_category
                }
                
                # 3. Multimodal Selective Fusion Override Rule
                if (sem_confidence == "Low" or is_ambiguous or sem_category == "General") and vis_conf >= 0.40:
                    if vis_st in VIS_13_TO_SEMANTIC:
                        override_target = VIS_13_TO_SEMANTIC[vis_st]
                        if override_target != sem_category:
                            final_category = override_target
                            fusion_applied = True
                            fusion_reason = f"Visual layout '{vis_st}' (conf {vis_conf:.2f}) resolved low-confidence/ambiguous semantic fallback '{sem_category}' -> '{final_category}'"
                            
                            visual_evidence["fusion_applied"] = True
                            visual_evidence["fusion_reason"] = fusion_reason
                            
            except Exception as e:
                logger.warning(f"Visual inference failed for {img_p.name} ({e}). Falling back safely to semantic prediction.")
                visual_evidence = None
                fusion_reason = f"Visual branch fallback due to error: {e}"
        else:
            if not self.visual_model_loaded:
                fusion_reason = "Visual branch fallback (model unavailable)"
            else:
                fusion_reason = "Visual branch fallback (image file not found)"
                
        return {
            "file": img_p.name,
            "category": final_category,
            "confidence": sem_confidence if not fusion_applied else "High",
            "semantic_category": sem_category,
            "fusion_applied": fusion_applied,
            "fusion_reason": fusion_reason,
            "evidence": {
                "text_snippet": text_input[:100],
                "visual_evidence": visual_evidence
            }
        }
