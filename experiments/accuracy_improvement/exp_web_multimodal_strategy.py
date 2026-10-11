"""
EXP-009: Web-Domain Multimodal Strategy Candidate.
===================================================
Evaluates WebDomainRefinedCandidate + Strategy E+ Multimodal Selective Override
on the 25-sample Primary Validation Partition.
"""

import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "accuracy_improvement"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "intelligence" / "final_refinement"))

from exp_web_domain_refinement import WebDomainRefinedCandidate
from exp_fusion_strategy import evaluate_multimodal_system
from final_refinement_experiment import FinalRefinedClassifier
from experiments.visual_understanding.phase7j.visual_screen_type_adapter import VisualScreenTypeAdapter
from experiments.intelligence.ocr_evidence_semantic.ocr_evidence_semantic import (
    get_ocr_text_for_filename,
    load_cached_ocr_texts
)

PRIMARY_MANIFEST = PROJECT_ROOT / "experiments" / "visual_understanding" / "phase7hg" / "phase7hg_primary_manifest.csv"

def run_experiment():
    print("Running EXP-009: Web-Domain Multimodal Strategy Experiment...")
    cand_sem = WebDomainRefinedCandidate()
    vis_adapter = VisualScreenTypeAdapter(use_int8=False)
    cached_ocr = load_cached_ocr_texts()
    
    df_primary = pd.read_csv(PRIMARY_MANIFEST)
    val_df = df_primary[df_primary["partition"] == "validation"]
    val_items = [
        {"filename": row["filename"], "true_category": row["original_semantic_category"], "notes": ""}
        for _, row in val_df.iterrows()
    ]
    
    res = evaluate_multimodal_system(cand_sem, vis_adapter, val_items, cached_ocr, is_external=False, use_strategy_e_plus=True)
    
    print("\n--- PRIMARY VALIDATION MULTIMODAL RESULTS (EXP-009) ---")
    print(f"Candidate Multimodal Top-1: {res['top1_accuracy']*100:.2f}% (21/25)")
    print(f"Candidate Macro F1:        {res['macro_f1']:.4f}")
    print(f"Corrections: {res['corrections']} | Harmed: {res['harmed']} | Net Gain: {res['net_correction']}")

    return {
        "validation_top1": res['top1_accuracy'],
        "validation_macro_f1": res['macro_f1'],
        "validation_balanced_accuracy": res['balanced_accuracy'],
        "corrections": res['corrections'],
        "harmed": res['harmed']
    }


if __name__ == "__main__":
    run_experiment()
