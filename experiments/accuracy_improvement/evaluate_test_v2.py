"""
Test Evaluation Script (V2 Sprint Re-evaluation)
=================================================
Evaluates winning candidate models (RefinedSemanticCandidate & Strategy E+ Multimodal)
against locked held-out test datasets (Internal Holdout 125 & External Benchmark 60).
Saves results to test_metrics_v2.json and prints comparison tables.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "intelligence" / "final_refinement"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "accuracy_improvement"))

from final_refinement_experiment import FinalRefinedClassifier
from exp_semantic_refinement import RefinedSemanticCandidate
from exp_fusion_strategy import evaluate_multimodal_system
from experiments.visual_understanding.phase7j.visual_screen_type_adapter import VisualScreenTypeAdapter
from experiments.intelligence.ocr_evidence_semantic.ocr_evidence_semantic import (
    get_ocr_text_for_filename,
    load_cached_ocr_texts
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "accuracy_improvement"
METADATA_V2 = PROJECT_ROOT / "data" / "metadata_v2.csv"
EXTERNAL_MANIFEST_V2 = PROJECT_ROOT / "experiments" / "intelligence" / "final_refinement" / "external_manifest_v2.csv"
PRIMARY_MANIFEST = PROJECT_ROOT / "experiments" / "visual_understanding" / "phase7hg" / "phase7hg_primary_manifest.csv"

CATEGORIES = [
    "Coding", "Shopping", "Travel", "Education", "Communication",
    "Entertainment", "Work & Career", "Payment", "Connectivity", "General"
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def evaluate_semantic_partition(classifier, items, cached_ocr):
    y_true = []
    y_pred = []
    
    for item in items:
        fname = item["filename"]
        true_cat = item["true_category"]
        notes = item.get("notes", "")
        
        ocr_text = get_ocr_text_for_filename(fname, cached_ocr)
        res = classifier.predict_final(raw_text=ocr_text, notes=notes, variant="C")
        
        pred_cat = res.get("prediction", "General")
        y_true.append(true_cat)
        y_pred.append(pred_cat)
        
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    return {
        "count": len(items),
        "top1_accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_precision": float(p_mac),
        "macro_recall": float(r_mac),
        "macro_f1": float(f1_mac),
        "weighted_precision": float(p_wt),
        "weighted_recall": float(r_wt),
        "weighted_f1": float(f1_wt)
    }


def main():
    print("=" * 80)
    print("FROZEN HELD-OUT TEST RE-EVALUATION (ACCURACY IMPROVEMENT SPRINT V2)")
    print("=" * 80)
    
    base_sem = FinalRefinedClassifier()
    cand_sem = RefinedSemanticCandidate()
    vis_adapter = VisualScreenTypeAdapter(use_int8=False)
    cached_ocr = load_cached_ocr_texts()
    
    df_meta = pd.read_csv(METADATA_V2)
    df_meta = df_meta[~df_meta["file"].isin(["ss_441.png", "ss_443.png", "ss_560.png"])].copy()
    
    # 1. Validation Set (250:375) - 125 samples
    val_items = [
        {"filename": row["file"], "true_category": row["category"], "notes": str(row.get("notes", ""))}
        for _, row in df_meta.iloc[250:375].iterrows()
    ]
    
    # 2. Internal Holdout Test Set (375:500) - 125 samples
    holdout_items = [
        {"filename": row["file"], "true_category": row["category"], "notes": str(row.get("notes", ""))}
        for _, row in df_meta.iloc[375:500].iterrows()
    ]
    
    # 3. External Benchmark Test Set (60 samples)
    df_ext = pd.read_csv(EXTERNAL_MANIFEST_V2)
    ext_items = [
        {"filename": row["filename"], "true_category": row["new_category"], "notes": str(row.get("summary", ""))}
        for _, row in df_ext.iterrows()
    ]
    
    print("\n--- SEMANTIC CLASSIFIER TEST RE-EVALUATION ---")
    
    # Validation
    val_base = evaluate_semantic_partition(base_sem, val_items, cached_ocr)
    val_cand = evaluate_semantic_partition(cand_sem, val_items, cached_ocr)
    
    # Internal Holdout
    holdout_base = evaluate_semantic_partition(base_sem, holdout_items, cached_ocr)
    holdout_cand = evaluate_semantic_partition(cand_sem, holdout_items, cached_ocr)
    
    # External Benchmark
    ext_base = evaluate_semantic_partition(base_sem, ext_items, cached_ocr)
    ext_cand = evaluate_semantic_partition(cand_sem, ext_items, cached_ocr)
    
    print(f"Validation Partition (125 samples):")
    print(f"  Baseline Top-1: {val_base['top1_accuracy']*100:.2f}% | Macro F1: {val_base['macro_f1']:.4f}")
    print(f"  Refined  Top-1: {val_cand['top1_accuracy']*100:.2f}% | Macro F1: {val_cand['macro_f1']:.4f} (Delta: {val_cand['top1_accuracy']-val_base['top1_accuracy']:+.4f})")
    
    print(f"\nInternal Holdout Test Set (125 samples):")
    print(f"  Baseline Top-1: {holdout_base['top1_accuracy']*100:.2f}% | Macro F1: {holdout_base['macro_f1']:.4f}")
    print(f"  Refined  Top-1: {holdout_cand['top1_accuracy']*100:.2f}% | Macro F1: {holdout_cand['macro_f1']:.4f} (Delta: {holdout_cand['top1_accuracy']-holdout_base['top1_accuracy']:+.4f})")
    
    print(f"\nExternal Benchmark Test Set (60 samples):")
    print(f"  Baseline Top-1: {ext_base['top1_accuracy']*100:.2f}% | Macro F1: {ext_base['macro_f1']:.4f}")
    print(f"  Refined  Top-1: {ext_cand['top1_accuracy']*100:.2f}% | Macro F1: {ext_cand['macro_f1']:.4f} (Delta: {ext_cand['top1_accuracy']-ext_base['top1_accuracy']:+.4f})")

    # 4. Multimodal Fusion Evaluation
    print("\n--- MULTIMODAL SYSTEM TEST RE-EVALUATION ---")
    
    df_primary = pd.read_csv(PRIMARY_MANIFEST)
    val_primary_df = df_primary[df_primary["partition"] == "validation"]
    val_primary_items = [
        {"filename": row["filename"], "true_category": row["original_semantic_category"], "notes": ""}
        for _, row in val_primary_df.iterrows()
    ]
    
    val_multi_base = evaluate_multimodal_system(base_sem, vis_adapter, val_primary_items, cached_ocr, is_external=False, use_strategy_e_plus=False)
    val_multi_cand = evaluate_multimodal_system(cand_sem, vis_adapter, val_primary_items, cached_ocr, is_external=False, use_strategy_e_plus=True)
    
    print(f"Primary Validation Set (25 visual screenshots):")
    print(f"  Baseline Strategy E  Top-1: {val_multi_base['top1_accuracy']*100:.2f}% | Macro F1: {val_multi_base['macro_f1']:.4f}")
    print(f"  Candidate Strategy E+ Top-1: {val_multi_cand['top1_accuracy']*100:.2f}% | Macro F1: {val_multi_cand['macro_f1']:.4f} (Delta: {val_multi_cand['top1_accuracy']-val_multi_base['top1_accuracy']:+.4f})")

    results = {
        "semantic": {
            "validation": {"baseline": val_base, "refined_candidate": val_cand},
            "internal_holdout": {"baseline": holdout_base, "refined_candidate": holdout_cand},
            "external_benchmark": {"baseline": ext_base, "refined_candidate": ext_cand}
        },
        "multimodal": {
            "primary_validation": {"baseline_strategy_e": val_multi_base, "candidate_strategy_e_plus": val_multi_cand}
        }
    }
    
    json_path = OUTPUT_DIR / "test_metrics_v2.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved test metrics to {json_path}")

if __name__ == "__main__":
    main()
