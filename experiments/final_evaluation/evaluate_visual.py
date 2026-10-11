"""
Visual Classifier Standardization Evaluation Script.
====================================================
Evaluates 5-Class YOLOv8n-cls model on Train, Calibration, and Validation partitions.
Generates visual_metrics.json and confusion_matrix_visual.png.
"""

import sys
import json
import time
import torch
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

from experiments.visual_understanding.phase7j.visual_screen_type_adapter import VisualScreenTypeAdapter

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "final_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_MANIFEST = PROJECT_ROOT / "experiments" / "visual_understanding" / "phase7hg" / "phase7hg_primary_manifest.csv"
SCREENSHOTS_DIR = PROJECT_ROOT / "data" / "screenshots"

VISUAL_5_CLASSES = [
    "code_editor_terminal", "travel_booking", "media_player_library",
    "document_article_view", "dashboard_settings"
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


def evaluate_partition(adapter, df_partition):
    """Evaluates visual adapter on dataframe partition."""
    y_true = []
    y_pred = []
    top2_correct = 0
    top3_correct = 0
    latencies = []
    
    for _, row in df_partition.iterrows():
        fname = row["filename"]
        true_st = row["screen_type"]
        img_path = SCREENSHOTS_DIR / fname
        
        if not img_path.exists():
            print(f"Warning: Image file not found {img_path}")
            continue
            
        t0 = time.perf_counter()
        res = adapter.predict_image(str(img_path))
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)
        
        pred_st = res["screen_type"]
        y_true.append(true_st)
        y_pred.append(pred_st)
        
        probs = {item["screen_type"]: item["confidence"] for item in res["ranked_predictions"]}
        sorted_classes = [pair[0] for pair in sorted(probs.items(), key=lambda x: x[1], reverse=True)]
        
        if true_st in sorted_classes[:2]:
            top2_correct += 1
        if true_st in sorted_classes[:3]:
            top3_correct += 1
            
    n_samples = len(y_true)
    if n_samples == 0:
        return {}
        
    acc = accuracy_score(y_true, y_pred)
    top2_acc = top2_correct / n_samples
    top3_acc = top3_correct / n_samples
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    # Per-class metrics
    p_cls, r_cls, f1_cls, supp_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=VISUAL_5_CLASSES, average=None, zero_division=0
    )
    
    per_class = {}
    for i, cat in enumerate(VISUAL_5_CLASSES):
        per_class[cat] = {
            "precision": float(p_cls[i]),
            "recall": float(r_cls[i]),
            "f1": float(f1_cls[i]),
            "support": int(supp_cls[i])
        }
        
    cm = confusion_matrix(y_true, y_pred, labels=VISUAL_5_CLASSES)
    
    med_lat = float(np.median(latencies)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
    
    return {
        "count": n_samples,
        "top1_accuracy": float(acc),
        "top2_accuracy": float(top2_acc),
        "top3_accuracy": float(top3_acc),
        "balanced_accuracy": float(bal_acc),
        "macro_precision": float(p_mac),
        "macro_recall": float(r_mac),
        "macro_f1": float(f1_mac),
        "weighted_precision": float(p_wt),
        "weighted_recall": float(r_wt),
        "weighted_f1": float(f1_wt),
        "median_latency_ms": med_lat,
        "p95_latency_ms": p95_lat,
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }


def main():
    print("Loading 5-Class VisualScreenTypeAdapter...")
    adapter = VisualScreenTypeAdapter(use_int8=False)
    
    df_primary = pd.read_csv(PRIMARY_MANIFEST)
    train_df = df_primary[df_primary["partition"] == "train"]
    calib_df = df_primary[df_primary["partition"] == "calibration"]
    val_df = df_primary[df_primary["partition"] == "validation"]
    
    print("Evaluating Train partition...")
    train_metrics = evaluate_partition(adapter, train_df)
    
    print("Evaluating Calibration partition...")
    calib_metrics = evaluate_partition(adapter, calib_df)
    
    print("Evaluating Validation partition...")
    val_metrics = evaluate_partition(adapter, val_df)
    
    results = {
        "model_name": "Ultralytics YOLOv8n-cls (5-Class)",
        "parameter_count": 1444693,
        "fp32_size_mb": 5.58,
        "int8_size_mb": 5.58,
        "train": train_metrics,
        "calibration": calib_metrics,
        "validation": val_metrics,
        "generalization_gap": {
            "train_to_val_top1_gap": train_metrics["top1_accuracy"] - val_metrics["top1_accuracy"]
        }
    }
    
    metrics_json_path = OUTPUT_DIR / "visual_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"Saved visual metrics to {metrics_json_path}")
    
    # Plot Confusion Matrix for Visual Validation
    fig, ax = plt.subplots(figsize=(9, 7))
    cm_array = np.array(val_metrics["confusion_matrix"])
    cax = ax.matshow(cm_array, cmap="Greens")
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(VISUAL_5_CLASSES)))
    ax.set_yticks(range(len(VISUAL_5_CLASSES)))
    ax.set_xticklabels(VISUAL_5_CLASSES, rotation=45, ha="left")
    ax.set_yticklabels(VISUAL_5_CLASSES)
    
    for i in range(len(VISUAL_5_CLASSES)):
        for j in range(len(VISUAL_5_CLASSES)):
            ax.text(j, i, str(cm_array[i, j]), ha="center", va="center", color="black" if cm_array[i, j] < cm_array.max()/2 else "white")
            
    plt.title("5-Class YOLOv8n-cls Visual Confusion Matrix (Validation Partition)", pad=20)
    plt.xlabel("Predicted Visual Screen Type")
    plt.ylabel("True Visual Screen Type")
    plt.tight_layout()
    
    plot_path = OUTPUT_DIR / "confusion_matrix_visual.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {plot_path}")


if __name__ == "__main__":
    main()
