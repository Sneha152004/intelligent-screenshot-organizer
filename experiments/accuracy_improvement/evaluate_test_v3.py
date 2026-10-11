"""
Test Evaluation Script (V3 Final External Investigation)
=========================================================
Evaluates WebDomainRefinedCandidate against locked test datasets:
- Validation Set (125 samples)
- Internal Holdout Test Set (125 samples)
- External Benchmark Test Set (60 samples)
- Primary Visual Validation Set (25 samples)
Saves results to test_metrics_v3.json and prints comparison tables.
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

from exp_semantic_refinement import RefinedSemanticCandidate
from exp_web_domain_refinement import WebDomainRefinedCandidate
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
    
    # Per-class metrics
    p_cls, r_cls, f1_cls, supp_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=CATEGORIES, average=None, zero_division=0
    )
    per_class = {
        cat: {
            "precision": float(p_cls[i]),
            "recall": float(r_cls[i]),
            "f1": float(f1_cls[i]),
            "support": int(supp_cls[i])
        }
        for i, cat in enumerate(CATEGORIES)
    }

    return {
        "count": len(items),
        "correct_count": int(np.sum(np.array(y_true) == np.array(y_pred))),
        "top1_accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_precision": float(p_mac),
        "macro_recall": float(r_mac),
        "macro_f1": float(f1_mac),
        "weighted_precision": float(p_wt),
        "weighted_recall": float(r_wt),
        "weighted_f1": float(f1_wt),
        "per_class": per_class
    }


def main():
    print("=" * 80)
    print("FINAL EXTERNAL INVESTIGATION: FROZEN HELD-OUT TEST RE-EVALUATION (V3)")
    print("=" * 80)
    
    v2_sem = RefinedSemanticCandidate()
    v3_sem = WebDomainRefinedCandidate()
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
    
    val_v2 = evaluate_semantic_partition(v2_sem, val_items, cached_ocr)
    val_v3 = evaluate_semantic_partition(v3_sem, val_items, cached_ocr)
    
    holdout_v2 = evaluate_semantic_partition(v2_sem, holdout_items, cached_ocr)
    holdout_v3 = evaluate_semantic_partition(v3_sem, holdout_items, cached_ocr)
    
    ext_v2 = evaluate_semantic_partition(v2_sem, ext_items, cached_ocr)
    ext_v3 = evaluate_semantic_partition(v3_sem, ext_items, cached_ocr)
    
    print(f"Validation Partition (125 samples):")
    print(f"  V2 Refined Top-1: {val_v2['top1_accuracy']*100:.2f}% | Macro F1: {val_v2['macro_f1']:.4f}")
    print(f"  V3 WebRef  Top-1: {val_v3['top1_accuracy']*100:.2f}% | Macro F1: {val_v3['macro_f1']:.4f} (Delta: {val_v3['top1_accuracy']-val_v2['top1_accuracy']:+.4f})")
    
    print(f"\nInternal Holdout Test Set (125 samples):")
    print(f"  V2 Refined Top-1: {holdout_v2['top1_accuracy']*100:.2f}% | Macro F1: {holdout_v2['macro_f1']:.4f}")
    print(f"  V3 WebRef  Top-1: {holdout_v3['top1_accuracy']*100:.2f}% | Macro F1: {holdout_v3['macro_f1']:.4f} (Delta: {holdout_v3['top1_accuracy']-holdout_v2['top1_accuracy']:+.4f})")
    
    print(f"\nExternal Benchmark Test Set (60 samples):")
    print(f"  V2 Refined Top-1: {ext_v2['top1_accuracy']*100:.2f}% ({ext_v2['correct_count']}/60) | Macro F1: {ext_v2['macro_f1']:.4f}")
    print(f"  V3 WebRef  Top-1: {ext_v3['top1_accuracy']*100:.2f}% ({ext_v3['correct_count']}/60) | Macro F1: {ext_v3['macro_f1']:.4f} (Delta: {ext_v3['top1_accuracy']-ext_v2['top1_accuracy']:+.4f})")

    results = {
        "semantic": {
            "validation": {"v2_baseline": val_v2, "v3_web_candidate": val_v3},
            "internal_holdout": {"v2_baseline": holdout_v2, "v3_web_candidate": holdout_v3},
            "external_benchmark": {"v2_baseline": ext_v2, "v3_web_candidate": ext_v3}
        }
    }
    
    json_path = OUTPUT_DIR / "test_metrics_v3.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved test metrics V3 to {json_path}")


if __name__ == "__main__":
    main()
