"""
Error Analysis Runner Script.
=============================
Inspects development & validation population errors across semantic, visual,
and multimodal components to identify root causes and confusion patterns.
Generates:
  - semantic_error_analysis.json
  - visual_error_analysis.json
  - fusion_error_analysis.json
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "intelligence" / "final_refinement"))

from final_refinement_experiment import FinalRefinedClassifier
from experiments.visual_understanding.phase7j.visual_screen_type_adapter import VisualScreenTypeAdapter
from experiments.intelligence.ocr_evidence_semantic.ocr_evidence_semantic import (
    get_ocr_text_for_filename,
    load_cached_ocr_texts
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "accuracy_improvement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_V2 = PROJECT_ROOT / "data" / "metadata_v2.csv"
PRIMARY_MANIFEST = PROJECT_ROOT / "experiments" / "visual_understanding" / "phase7hg" / "phase7hg_primary_manifest.csv"
SCREENSHOTS_DIR = PROJECT_ROOT / "data" / "screenshots"

VIS_5_TO_SEMANTIC_MAP = {
    "code_editor_terminal": "Coding",
    "travel_booking": "Travel",
    "media_player_library": "Entertainment",
    "document_article_view": "Education",
    "dashboard_settings": "Connectivity"
}


def run_semantic_error_analysis():
    print("Running Semantic Error Analysis on Validation Partition (125 imgs)...")
    classifier = FinalRefinedClassifier()
    cached_ocr = load_cached_ocr_texts()
    
    df_meta = pd.read_csv(METADATA_V2)
    df_meta = df_meta[~df_meta["file"].isin(["ss_441.png", "ss_443.png", "ss_560.png"])].copy()
    
    # Validation split: rows 250 to 375
    val_df = df_meta.iloc[250:375]
    
    errors = []
    confusion_pairs = {}
    
    for idx, row in val_df.iterrows():
        fname = row["file"]
        true_cat = row["category"]
        notes = str(row.get("notes", ""))
        
        ocr_text = get_ocr_text_for_filename(fname, cached_ocr)
        res = classifier.predict_final(raw_text=ocr_text, notes=notes, variant="C")
        pred_cat = res.get("prediction", "General")
        conf = res.get("confidence", "Low")
        
        if pred_cat != true_cat:
            pair = f"{true_cat} -> {pred_cat}"
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
            
            # Identify root cause error type
            ocr_len = len(ocr_text.strip())
            tokens = ocr_text.lower().split()
            
            if ocr_len < 20:
                error_type = "OCR Sparsity"
            elif any(word in ocr_text.lower() for word in ["unit", "course", "lecture", "faculty"]):
                error_type = "Domain-Specific Academic Vocabulary"
            elif pred_cat == "General":
                error_type = "Fallback to General (Low Evidence)"
            elif true_cat in ["Education", "Work & Career"] and pred_cat in ["Education", "Work & Career"]:
                error_type = "Category Boundary Ambiguity (Education vs Work & Career)"
            else:
                error_type = "Keyword Match Noise"
                
            errors.append({
                "screenshot_id": fname.split(".")[0],
                "filename": fname,
                "ground_truth": true_cat,
                "predicted_category": pred_cat,
                "confidence": conf,
                "ocr_length": ocr_len,
                "error_type": error_type,
                "ocr_snippet": ocr_text[:120].replace("\n", " ")
            })
            
    summary = {
        "total_validation_samples": len(val_df),
        "total_errors": len(errors),
        "accuracy": round((len(val_df) - len(errors)) / len(val_df), 4),
        "top_confusion_pairs": dict(sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)),
        "error_type_breakdown": pd.DataFrame(errors)["error_type"].value_counts().to_dict() if errors else {},
        "error_details": errors
    }
    
    with open(OUTPUT_DIR / "semantic_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Semantic error analysis completed. Total errors: {len(errors)}")


def run_visual_error_analysis():
    print("Running Visual Error Analysis on Validation Partition (25 imgs)...")
    adapter = VisualScreenTypeAdapter(use_int8=False)
    
    df_primary = pd.read_csv(PRIMARY_MANIFEST)
    val_df = df_primary[df_primary["partition"] == "validation"]
    
    errors = []
    confusion_pairs = {}
    
    for idx, row in val_df.iterrows():
        fname = row["filename"]
        true_st = row["screen_type"]
        img_path = SCREENSHOTS_DIR / fname
        
        if not img_path.exists():
            continue
            
        res = adapter.predict_image(str(img_path))
        pred_st = res["screen_type"]
        conf = res["confidence"]
        
        if pred_st != true_st:
            pair = f"{true_st} -> {pred_st}"
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
            
            errors.append({
                "screenshot_id": row["screenshot_id"],
                "filename": fname,
                "ground_truth_screen_type": true_st,
                "predicted_screen_type": pred_st,
                "confidence": conf,
                "ranked_predictions": res["ranked_predictions"][:3]
            })
            
    summary = {
        "total_validation_samples": len(val_df),
        "total_errors": len(errors),
        "accuracy": round((len(val_df) - len(errors)) / len(val_df), 4),
        "top_confusion_pairs": dict(sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)),
        "error_details": errors
    }
    
    with open(OUTPUT_DIR / "visual_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Visual error analysis completed. Total errors: {len(errors)}")


def run_fusion_error_analysis():
    print("Running Fusion Error Analysis on Validation & External Populations...")
    classifier = FinalRefinedClassifier()
    adapter = VisualScreenTypeAdapter(use_int8=False)
    cached_ocr = load_cached_ocr_texts()
    
    df_primary = pd.read_csv(PRIMARY_MANIFEST)
    val_df = df_primary[df_primary["partition"] == "validation"]
    
    opportunity_cases = []
    
    for idx, row in val_df.iterrows():
        fname = row["filename"]
        true_cat = row["original_semantic_category"]
        img_p = SCREENSHOTS_DIR / fname
        
        ocr_text = get_ocr_text_for_filename(fname, cached_ocr)
        sem_res = classifier.predict_final(raw_text=ocr_text, notes="", variant="C")
        sem_pred = sem_res.get("prediction", "General")
        sem_conf = sem_res.get("confidence", "Low")
        
        vis_res = adapter.predict_image(str(img_p)) if img_p.exists() else {}
        vis_st = vis_res.get("screen_type", "other_screen")
        vis_conf = vis_res.get("confidence", 0.0)
        vis_mapped = VIS_5_TO_SEMANTIC_MAP.get(vis_st, "General")
        
        # Check opportunity: semantic incorrect, but visual mapped is correct
        if sem_pred != true_cat and vis_mapped == true_cat:
            opportunity_cases.append({
                "filename": fname,
                "true_category": true_cat,
                "semantic_prediction": sem_pred,
                "semantic_confidence": sem_conf,
                "visual_screen_type": vis_st,
                "visual_confidence": vis_conf,
                "visual_mapped_category": vis_mapped,
                "opportunity_reason": "Visual model identified layout correctly, but fusion trigger was not met or override threshold was too conservative"
            })
            
    summary = {
        "validation_opportunity_cases_count": len(opportunity_cases),
        "opportunity_cases": opportunity_cases
    }
    
    with open(OUTPUT_DIR / "fusion_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Fusion error analysis completed. Total fusion opportunities identified: {len(opportunity_cases)}")


if __name__ == "__main__":
    run_semantic_error_analysis()
    run_visual_error_analysis()
    run_fusion_error_analysis()
