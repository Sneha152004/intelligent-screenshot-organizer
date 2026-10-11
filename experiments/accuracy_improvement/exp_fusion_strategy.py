"""
EXP-006: Candidate Multimodal System (Refined Semantic + Strategy E+ Selective Fusion).
========================================================================================
Evaluates Candidate Multimodal Pipeline on Validation Partition and External Benchmark Population.
"""

import sys
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "intelligence" / "final_refinement"))

from final_refinement_experiment import FinalRefinedClassifier
from exp_semantic_refinement import RefinedSemanticCandidate
from experiments.visual_understanding.phase7j.visual_screen_type_adapter import VisualScreenTypeAdapter
from experiments.intelligence.ocr_evidence_semantic.ocr_evidence_semantic import (
    get_ocr_text_for_filename,
    load_cached_ocr_texts
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

METADATA_V2 = PROJECT_ROOT / "data" / "metadata_v2.csv"
PRIMARY_MANIFEST = PROJECT_ROOT / "experiments" / "visual_understanding" / "phase7hg" / "phase7hg_primary_manifest.csv"
EXTERNAL_MANIFEST_V2 = PROJECT_ROOT / "experiments" / "intelligence" / "final_refinement" / "external_manifest_v2.csv"
SCREENSHOTS_DIR = PROJECT_ROOT / "data" / "screenshots"
EXT_IMAGES_DIR = PROJECT_ROOT / "experiments" / "intelligence" / "external" / "images"

VIS_5_TO_SEMANTIC_MAP = {
    "code_editor_terminal": "Coding",
    "travel_booking": "Travel",
    "media_player_library": "Entertainment",
    "document_article_view": "Education",
    "dashboard_settings": "Connectivity"
}


def evaluate_multimodal_system(semantic_cls, visual_adapter, items, cached_ocr, is_external=False, use_strategy_e_plus=False):
    """Evaluates multimodal pipeline on item list."""
    y_true = []
    y_pred = []
    corrections = 0
    harmed = 0
    
    for item in items:
        fname = item["filename"]
        true_cat = item["true_category"]
        notes = item.get("notes", "")
        img_p = (EXT_IMAGES_DIR / fname) if is_external else (SCREENSHOTS_DIR / fname)
        
        ocr_text = get_ocr_text_for_filename(fname, cached_ocr)
        sem_res = semantic_cls.predict_final(raw_text=ocr_text, notes=notes, variant="C")
        sem_pred = sem_res.get("prediction", "General")
        sem_conf = sem_res.get("confidence", "Low")
        is_ambig = bool(sem_res.get("is_ambiguous", False))
        
        vis_st = "other_screen"
        vis_conf = 0.0
        if visual_adapter is not None and img_p.exists():
            try:
                vis_res = visual_adapter.predict_image(str(img_p))
                vis_st = vis_res.get("screen_type", "other_screen")
                vis_conf = vis_res.get("confidence", 0.0)
            except Exception:
                pass
                
        fused_pred = sem_pred
        fusion_applied = False
        
        if use_strategy_e_plus:
            # Strategy E+ Threshold Logic: triggers on low confidence, ambiguity, General fallback, or visual code_editor_terminal activation
            if (sem_conf == "Low" or is_ambig or sem_pred in ["General", "Work & Career"]) and vis_conf >= 0.25:
                if vis_st in VIS_5_TO_SEMANTIC_MAP:
                    target = VIS_5_TO_SEMANTIC_MAP[vis_st]
                    if target != sem_pred and (sem_pred != "Coding" or vis_st == "code_editor_terminal"):
                        fused_pred = target
                        fusion_applied = True
        else:
            # Standard Strategy E
            if (sem_conf == "Low" or is_ambig or sem_pred == "General") and vis_conf >= 0.40:
                if vis_st in VIS_5_TO_SEMANTIC_MAP:
                    target = VIS_5_TO_SEMANTIC_MAP[vis_st]
                    if target != sem_pred:
                        fused_pred = target
                        fusion_applied = True
                        
        if fusion_applied:
            sem_correct = (sem_pred == true_cat)
            fused_correct = (fused_pred == true_cat)
            if not sem_correct and fused_correct:
                corrections += 1
            elif sem_correct and not fused_correct:
                harmed += 1
                
        y_true.append(true_cat)
        y_pred.append(fused_pred)
        
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    _, _, f1_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    
    return {
        "count": len(items),
        "top1_accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(f1_mac),
        "corrections": corrections,
        "harmed": harmed,
        "net_correction": corrections - harmed
    }


def main():
    print("Running EXP-006: Candidate Multimodal System Experiment...")
    base_sem = FinalRefinedClassifier()
    cand_sem = RefinedSemanticCandidate()
    vis_adapter = VisualScreenTypeAdapter(use_int8=False)
    cached_ocr = load_cached_ocr_texts()
    
    # Primary Validation Set (25 imgs)
    df_primary = pd.read_csv(PRIMARY_MANIFEST)
    val_df = df_primary[df_primary["partition"] == "validation"]
    val_items = [
        {"filename": row["filename"], "true_category": row["original_semantic_category"], "notes": ""}
        for _, row in val_df.iterrows()
    ]
    
    base_val_res = evaluate_multimodal_system(base_sem, vis_adapter, val_items, cached_ocr, is_external=False, use_strategy_e_plus=False)
    cand_val_res = evaluate_multimodal_system(cand_sem, vis_adapter, val_items, cached_ocr, is_external=False, use_strategy_e_plus=True)
    
    print("\n--- PRIMARY VALIDATION RESULTS (25 imgs) ---")
    print(f"Baseline Multimodal Top-1: {base_val_res['top1_accuracy']:.4f} | Candidate Top-1: {cand_val_res['top1_accuracy']:.4f} (Delta: {cand_val_res['top1_accuracy'] - base_val_res['top1_accuracy']:+.4f})")
    print(f"Baseline Macro F1:       {base_val_res['macro_f1']:.4f} | Candidate Macro F1: {cand_val_res['macro_f1']:.4f} (Delta: {cand_val_res['macro_f1'] - base_val_res['macro_f1']:+.4f})")
    print(f"Corrections: {cand_val_res['corrections']} | Harmed: {cand_val_res['harmed']} | Net Gain: {cand_val_res['net_correction']}")


if __name__ == "__main__":
    main()
