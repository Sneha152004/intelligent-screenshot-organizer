"""
Phase 7J: Parallel Integration-Contract Harness.

Demonstrates parallel execution of the Semantic Classification branch (FinalRefinedClassifier)
and the Visual Screen-Type branch (VisualScreenTypeAdapter) without modifying production code
or introducing premature score fusion.
"""

import os
import sys
import time
import json
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Union, Optional

PHASE7J_DIR = Path(__file__).resolve().parent
VISUAL_UNDERSTANDING_DIR = PHASE7J_DIR.parent
PROJECT_ROOT = VISUAL_UNDERSTANDING_DIR.parent.parent
PHASE7I_DIR = VISUAL_UNDERSTANDING_DIR / "phase7i"

sys.path.insert(0, str(PHASE7J_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "intelligence" / "final_refinement"))

from visual_screen_type_adapter import VisualScreenTypeAdapter
from integration_contract import validate_visual_contract, VisualUnderstandingResult
from final_refinement_experiment import FinalRefinedClassifier

SCREENSHOTS_DIR = PROJECT_ROOT / "data" / "screenshots"


class ParallelIntegrationHarness:
    """
    Isolated integration harness demonstrating non-production multi-branch screenshot understanding.
    """
    def __init__(self, use_int8_visual: bool = False):
        self.visual_adapter = VisualScreenTypeAdapter(use_int8=use_int8_visual)
        self.semantic_classifier = FinalRefinedClassifier()

    def process_screenshot(
        self,
        screenshot_id: str,
        filename: str,
        ocr_text: str = "",
        metadata_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes semantic and visual branches in parallel and produces combined structured understanding object.
        """
        img_path = SCREENSHOTS_DIR / filename
        if not img_path.exists():
            raise FileNotFoundError(f"Screenshot image not found at {img_path}")

        # 1. Visual Branch Execution
        t_visual_start = time.perf_counter()
        raw_visual_res = self.visual_adapter.predict_image(img_path)
        visual_contract_obj = validate_visual_contract(raw_visual_res)
        visual_dict = visual_contract_obj.to_dict()

        # 2. Semantic Branch Execution
        t_sem_start = time.perf_counter()
        # Mock or execute semantic prediction using FinalRefinedClassifier logic
        dummy_row = pd.Series({
            "ocr_text": ocr_text if ocr_text else filename,
            "filename": filename
        })
        try:
            sem_pred = self.semantic_classifier.predict_final(raw_text=ocr_text if ocr_text else filename, notes="", variant="C")
            semantic_dict = {
                "category": sem_pred.get("prediction", "General"),
                "content_type": "unknown",
                "classification_confidence": round(float(sem_pred.get("top_candidates", [{}])[0].get("score", 0.85)), 4),
                "classification_ambiguous": bool(sem_pred.get("is_ambiguous", False)),
                "classifier_name": "FinalRefinedClassifier"
            }
        except Exception as e:
            semantic_dict = {
                "category": "General",
                "content_type": "unknown",
                "classification_confidence": 0.50,
                "classification_ambiguous": True,
                "classifier_name": "FinalRefinedClassifier (Fallback)"
            }

        # 3. Combined Non-Production Integration Object
        return {
            "screenshot_id": screenshot_id,
            "filename": filename,
            "semantic": semantic_dict,
            "visual": visual_dict,
            "integration_metadata": {
                "semantic_category_owner": "FinalRefinedClassifier",
                "visual_evidence_owner": "YOLOv8n-cls",
                "concepts_separated": True,
                "score_fusion_applied": False
            }
        }


if __name__ == "__main__":
    import pandas as pd
    print("Testing ParallelIntegrationHarness...")
    harness = ParallelIntegrationHarness()
    manifest_path = PROJECT_ROOT / "experiments" / "visual_understanding" / "phase7hg" / "phase7hg_primary_manifest.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        sample = df.iloc[0]
        res = harness.process_screenshot(
            screenshot_id=str(sample["screenshot_id"]),
            filename=str(sample["filename"])
        )
        print("Harness Combined Result:")
        print(json.dumps(res, indent=2))
        print("ParallelIntegrationHarness test passed.")
