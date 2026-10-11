"""
Semantic Classifier Standardization Evaluation Script.
======================================================
Evaluates FinalRefinedClassifier on Train, Validation, Internal Holdout,
and External Benchmark partitions.
Generates semantic_metrics.json and confusion_matrix_semantic.png.
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

from final_refinement_experiment import FinalRefinedClassifier
from experiments.intelligence.ocr_evidence_semantic.ocr_evidence_semantic import (
    get_ocr_text_for_filename,
    load_cached_ocr_texts
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "final_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_V2 = PROJECT_ROOT / "data" / "metadata_v2.csv"
EXTERNAL_MANIFEST_V2 = PROJECT_ROOT / "experiments" / "intelligence" / "final_refinement" / "external_manifest_v2.csv"

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


def evaluate_partition(classifier, items, cached_ocr):
    """Evaluates classifier on list of dict items: [{filename, true_category, notes}]."""
    y_true = []
    y_pred = []
    conf_scores = []
    
    for item in items:
        fname = item["filename"]
        true_cat = item["true_category"]
        notes = item.get("notes", "")
        
        ocr_text = get_ocr_text_for_filename(fname, cached_ocr)
        res = classifier.predict_final(raw_text=ocr_text, notes=notes, variant="C")
        
        pred_cat = res.get("prediction", "General")
        y_true.append(true_cat)
        y_pred.append(pred_cat)
        conf_scores.append(res.get("confidence", "Low"))
        
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    # Per-class metrics
    p_cls, r_cls, f1_cls, supp_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=CATEGORIES, average=None, zero_division=0
    )
    
    per_class = {}
    for i, cat in enumerate(CATEGORIES):
        per_class[cat] = {
            "precision": float(p_cls[i]),
            "recall": float(r_cls[i]),
            "f1": float(f1_cls[i]),
            "support": int(supp_cls[i])
        }
        
    cm = confusion_matrix(y_true, y_pred, labels=CATEGORIES)
    
    return {
        "count": len(items),
        "top1_accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_precision": float(p_mac),
        "macro_recall": float(r_mac),
        "macro_f1": float(f1_mac),
        "weighted_precision": float(p_wt),
        "weighted_recall": float(r_wt),
        "weighted_f1": float(f1_wt),
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }


def main():
    print("Initializing FinalRefinedClassifier...")
    classifier = FinalRefinedClassifier()
    cached_ocr = load_cached_ocr_texts()
    
    df_meta = pd.read_csv(METADATA_V2)
    # Remove corrupted files
    df_meta = df_meta[~df_meta["file"].isin(["ss_441.png", "ss_443.png", "ss_560.png"])].copy()
    
    # Dev-Train: 1 to 250
    train_items = [
        {"filename": row["file"], "true_category": row["category"], "notes": str(row.get("notes", ""))}
        for _, row in df_meta.iloc[:250].iterrows()
    ]
    
    # Validation: 251 to 375
    val_items = [
        {"filename": row["file"], "true_category": row["category"], "notes": str(row.get("notes", ""))}
        for _, row in df_meta.iloc[250:375].iterrows()
    ]
    
    # Internal Holdout: 376 to 500
    holdout_items = [
        {"filename": row["file"], "true_category": row["category"], "notes": str(row.get("notes", ""))}
        for _, row in df_meta.iloc[375:500].iterrows()
    ]
    
    # External Benchmark
    df_ext = pd.read_csv(EXTERNAL_MANIFEST_V2)
    ext_items = [
        {"filename": row["filename"], "true_category": row["new_category"], "notes": str(row.get("summary", ""))}
        for _, row in df_ext.iterrows()
    ]
    
    print("Evaluating Train set...")
    train_metrics = evaluate_partition(classifier, train_items, cached_ocr)
    
    print("Evaluating Validation set...")
    val_metrics = evaluate_partition(classifier, val_items, cached_ocr)
    
    print("Evaluating Internal Holdout set...")
    holdout_metrics = evaluate_partition(classifier, holdout_items, cached_ocr)
    
    print("Evaluating External Benchmark set...")
    ext_metrics = evaluate_partition(classifier, ext_items, cached_ocr)
    
    results = {
        "model_name": "FinalRefinedClassifier",
        "train": train_metrics,
        "validation": val_metrics,
        "internal_holdout": holdout_metrics,
        "external_test": ext_metrics,
        "generalization_gap": {
            "train_to_val_top1_gap": train_metrics["top1_accuracy"] - val_metrics["top1_accuracy"],
            "val_to_holdout_top1_gap": val_metrics["top1_accuracy"] - holdout_metrics["top1_accuracy"],
            "internal_to_external_top1_gap": holdout_metrics["top1_accuracy"] - ext_metrics["top1_accuracy"]
        }
    }
    
    metrics_json_path = OUTPUT_DIR / "semantic_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"Saved semantic metrics to {metrics_json_path}")
    
    # Plot Confusion Matrix for Internal Holdout
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_array = np.array(holdout_metrics["confusion_matrix"])
    cax = ax.matshow(cm_array, cmap="Blues")
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_yticks(range(len(CATEGORIES)))
    ax.set_xticklabels(CATEGORIES, rotation=45, ha="left")
    ax.set_yticklabels(CATEGORIES)
    
    for i in range(len(CATEGORIES)):
        for j in range(len(CATEGORIES)):
            ax.text(j, i, str(cm_array[i, j]), ha="center", va="center", color="black" if cm_array[i, j] < cm_array.max()/2 else "white")
            
    plt.title("Semantic Classifier Confusion Matrix (Internal Holdout Test)", pad=20)
    plt.xlabel("Predicted Category")
    plt.ylabel("True Category")
    plt.tight_layout()
    
    plot_path = OUTPUT_DIR / "confusion_matrix_semantic.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {plot_path}")


if __name__ == "__main__":
    main()
