"""
Phase 7K: Multimodal Calibration & Fusion Harness.

Implements non-production experimental multimodal fusion strategies combining
the frozen semantic classifier (FinalRefinedClassifier) and the frozen visual classifier (YOLOv8n-cls).
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

PHASE7K_DIR = Path(__file__).resolve().parent
VISUAL_UNDERSTANDING_DIR = PHASE7K_DIR.parent
PROJECT_ROOT = VISUAL_UNDERSTANDING_DIR.parent.parent
PHASE7I_DIR = VISUAL_UNDERSTANDING_DIR / "phase7i"
PHASE7J_DIR = VISUAL_UNDERSTANDING_DIR / "phase7j"

sys.path.insert(0, str(PHASE7I_DIR))
sys.path.insert(0, str(PHASE7J_DIR))
sys.path.insert(0, str(PHASE7K_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "intelligence" / "final_refinement"))

from visual_screen_type_adapter import VisualScreenTypeAdapter, SUPPORTED_5_CLASSES
from final_refinement_experiment import FinalRefinedClassifier

SCREENSHOTS_DIR = PROJECT_ROOT / "data" / "screenshots"
PRIMARY_MANIFEST_PATH = VISUAL_UNDERSTANDING_DIR / "phase7hg" / "phase7hg_primary_manifest.csv"

VIS_TO_SEM_MAP = {
    "code_editor_terminal": "Coding",
    "travel_booking": "Travel",
    "media_player_library": "Entertainment",
    "document_article_view": "Education",
    "dashboard_settings": "Connectivity"
}

CANONICAL_CATEGORIES = [
    "Education", "Payment", "Shopping", "Travel", "Coding",
    "Entertainment", "Connectivity", "Work & Career", "Communication", "General"
]


class MultimodalFusionHarness:
    """
    Experimental non-production harness combining semantic and visual signals.
    """
    def __init__(self, temperature: float = 1.5, fusion_weight: float = 0.15, use_int8: bool = False):
        self.temperature = temperature
        self.fusion_weight = fusion_weight
        self.visual_adapter = VisualScreenTypeAdapter(use_int8=use_int8)
        self.semantic_classifier = FinalRefinedClassifier()

    def process_screenshot(
        self,
        screenshot_id: str,
        filename: str,
        ocr_text: str = "",
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Processes a screenshot through both semantic and visual branches and evaluates all 5 fusion strategies.
        """
        img_path = SCREENSHOTS_DIR / filename
        text_content = ocr_text if ocr_text else (notes if notes else filename)

        # 1. Semantic Branch Execution
        sem_res = self.semantic_classifier.predict_final(raw_text=text_content, notes=notes, variant="C")
        sem_cat = sem_res.get("prediction", "General")
        sem_conf = sem_res.get("confidence", "Low")
        is_ambig = bool(sem_res.get("is_ambiguous", False))
        sem_scores = sem_res.get("category_scores", {})

        # 2. Visual Branch Execution
        raw_vis_res = self.visual_adapter.predict_image(img_path)
        vis_st = raw_vis_res["screen_type"]
        vis_conf = raw_vis_res["confidence"]
        vis_mapped = VIS_TO_SEM_MAP.get(vis_st, "General")

        # 3. Strategy A: Semantic-Only Baseline
        strat_a_cat = sem_cat

        # 4. Strategy B: Visual-Only Mapped Baseline
        strat_b_cat = vis_mapped

        # 5. Strategy C: Rule-Based Agreement Fusion
        if is_ambig and vis_st in ["code_editor_terminal", "travel_booking", "media_player_library"]:
            strat_c_cat = vis_mapped
            reason_c = f"Visual layout '{vis_st}' resolved ambiguous semantic boundary"
        else:
            strat_c_cat = sem_cat
            reason_c = "Semantic prediction authoritative"

        # 6. Strategy D: Calibrated Weighted Score Fusion
        vis_vec = np.zeros(len(CANONICAL_CATEGORIES))
        for item in raw_vis_res["ranked_predictions"]:
            st = item["screen_type"]
            c_mapped = VIS_TO_SEM_MAP.get(st, "General")
            idx = CANONICAL_CATEGORIES.index(c_mapped)
            vis_vec[idx] += item["confidence"]

        sem_vec = np.array([sem_scores.get(c, 0.0) for c in CANONICAL_CATEGORIES])
        sem_norm = sem_vec / (np.sum(sem_vec) + 1e-8)
        vis_norm = vis_vec / (np.sum(vis_vec) + 1e-8)

        fused_vec = (1.0 - self.fusion_weight) * sem_norm + self.fusion_weight * vis_norm
        strat_d_cat = CANONICAL_CATEGORIES[int(np.argmax(fused_vec))]

        # 7. Strategy E: Selective Overriding (Winning Strategy)
        if (sem_conf == "Low" or is_ambig or sem_cat == "General") and vis_st == "code_editor_terminal":
            strat_e_cat = "Coding"
            reason_e = "Visual layout 'code_editor_terminal' overrode low-confidence/ambiguous semantic fallback 'General'"
        elif (sem_conf == "Low" or is_ambig or sem_cat == "General") and vis_st == "travel_booking":
            strat_e_cat = "Travel"
            reason_e = "Visual layout 'travel_booking' overrode low-confidence/ambiguous semantic fallback 'General'"
        else:
            strat_e_cat = sem_cat
            reason_e = "Semantic category preserved"

        return {
            "screenshot_id": screenshot_id,
            "filename": filename,
            "semantic_prediction": {
                "category": sem_cat,
                "confidence": sem_conf,
                "is_ambiguous": is_ambig,
                "margin": sem_res.get("margin", 0.0)
            },
            "visual_prediction": {
                "screen_type": vis_st,
                "confidence": vis_conf,
                "mapped_semantic_category": vis_mapped,
                "is_supported_class": True
            },
            "fusion_predictions": {
                "strategy_a_semantic_only": strat_a_cat,
                "strategy_b_visual_only_mapped": strat_b_cat,
                "strategy_c_rule_agreement": strat_c_cat,
                "strategy_d_weighted_score": strat_d_cat,
                "strategy_e_selective_override": strat_e_cat
            },
            "combined_prediction": {
                "category": strat_e_cat,
                "selected_strategy": "Strategy E (Selective Overriding)",
                "reason": reason_e
            }
        }


if __name__ == "__main__":
    print("Testing MultimodalFusionHarness...")
    harness = MultimodalFusionHarness()
    manifest_df = pd.read_csv(PRIMARY_MANIFEST_PATH)
    meta_df = pd.read_csv(PROJECT_ROOT / "data" / "metadata.csv")
    file_to_notes = dict(zip(meta_df["file"], meta_df["notes"].fillna("")))

    sample = manifest_df[manifest_df["partition"] == "validation"].iloc[0]
    ocr_text = file_to_notes.get(sample["filename"], sample["filename"])

    res = harness.process_screenshot(
        screenshot_id=str(sample["screenshot_id"]),
        filename=str(sample["filename"]),
        ocr_text=ocr_text
    )
    print("Harness Output:")
    print(json.dumps(res, indent=2))
    print("MultimodalFusionHarness test passed.")
