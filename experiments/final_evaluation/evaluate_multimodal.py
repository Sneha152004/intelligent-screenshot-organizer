"""
Multimodal Pipeline Standardization Evaluation Script.
======================================================
Evaluates the complete production multimodal system:
FinalRefinedClassifier + 5-Class YOLOv8n-cls + Selective Fusion Strategy E.
Compares Semantic-Only vs Visual-Only vs Multimodal Fusion across validation and test sets.
Generates multimodal_metrics.json and confusion_matrix_multimodal.png.
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
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "intelligence" / "final_refinement"))

from final_refinement_experiment import FinalRefinedClassifier
from experiments.visual_understanding.phase7j.visual_screen_type_adapter import VisualScreenTypeAdapter
from experiments.intelligence.ocr_evidence_semantic.ocr_evidence_semantic import (
    get_ocr_text_for_filename,
    load_cached_ocr_texts
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "final_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_MANIFEST = PROJECT_ROOT / "experiments" / "visual_understanding" / "phase7hg" / "phase7hg_primary_manifest.csv"
EXTERNAL_MANIFEST_V2 = PROJECT_ROOT / "experiments" / "intelligence" / "final_refinement" / "external_manifest_v2.csv"
SCREENSHOTS_DIR = PROJECT_ROOT / "data" / "screenshots"
EXT_IMAGES_DIR = PROJECT_ROOT / "experiments" / "intelligence" / "external" / "images"

CATEGORIES = [
    "Coding", "Shopping", "Travel", "Education", "Communication",
    "Entertainment", "Work & Career", "Payment", "Connectivity", "General"
]

VIS_5_TO_SEMANTIC_MAP = {
    "code_editor_terminal": "Coding",
    "travel_booking": "Travel",
    "media_player_library": "Entertainment",
    "document_article_view": "Education",
    "dashboard_settings": "Connectivity"
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def evaluate_multimodal_population(semantic_cls, visual_adapter, items, cached_ocr, is_external=False):
    """Evaluates semantic, visual, and multimodal fusion on an item list."""
    y_true = []
    y_sem = []
    y_vis = []
    y_fused = []
    
    fusion_safety_counts = {
        "agreement_cases": 0,
        "high_confidence_disagreement": 0,
        "low_confidence_override": 0,
        "visual_unavailable_fallback": 0,
        "predictions_changed": 0,
        "errors_corrected": 0,
        "errors_harmed": 0,
        "net_correction": 0
    }
    
    latencies = []
    
    for item in items:
        fname = item["filename"]
        true_cat = item["true_category"]
        notes = item.get("notes", "")
        
        img_p = (EXT_IMAGES_DIR / fname) if is_external else (SCREENSHOTS_DIR / fname)
        
        t0 = time.perf_counter()
        
        # 1. Semantic Branch
        ocr_text = get_ocr_text_for_filename(fname, cached_ocr)
        sem_res = semantic_cls.predict_final(raw_text=ocr_text, notes=notes, variant="C")
        sem_pred = sem_res.get("prediction", "General")
        sem_conf = sem_res.get("confidence", "Low")
        is_ambig = bool(sem_res.get("is_ambiguous", False))
        
        # 2. Visual Branch
        vis_pred_mapped = "General"
        vis_conf = 0.0
        vis_st = "other_screen"
        
        if visual_adapter is not None and img_p.exists():
            try:
                vis_res = visual_adapter.predict_image(str(img_p))
                vis_st = vis_res["screen_type"]
                vis_conf = vis_res["confidence"]
                vis_pred_mapped = VIS_5_TO_SEMANTIC_MAP.get(vis_st, "General")
            except Exception as e:
                fusion_safety_counts["visual_unavailable_fallback"] += 1
        else:
            fusion_safety_counts["visual_unavailable_fallback"] += 1
            
        # 3. Multimodal Selective Fusion (Strategy E)
        fused_pred = sem_pred
        fusion_applied = False
        
        if (sem_conf == "Low" or is_ambig or sem_pred == "General") and vis_conf >= 0.40:
            if vis_st in VIS_5_TO_SEMANTIC_MAP:
                override_target = VIS_5_TO_SEMANTIC_MAP[vis_st]
                if override_target != sem_pred:
                    fused_pred = override_target
                    fusion_applied = True
                    fusion_safety_counts["low_confidence_override"] += 1
                    
        # Track agreement / disagreement
        if sem_pred == vis_pred_mapped:
            fusion_safety_counts["agreement_cases"] += 1
        elif sem_conf in ["High", "Medium"] and not is_ambig:
            fusion_safety_counts["high_confidence_disagreement"] += 1
            
        # Track accuracy changes
        if fusion_applied:
            fusion_safety_counts["predictions_changed"] += 1
            sem_correct = (sem_pred == true_cat)
            fused_correct = (fused_pred == true_cat)
            
            if not sem_correct and fused_correct:
                fusion_safety_counts["errors_corrected"] += 1
            elif sem_correct and not fused_correct:
                fusion_safety_counts["errors_harmed"] += 1
                
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)
        
        y_true.append(true_cat)
        y_sem.append(sem_pred)
        y_vis.append(vis_pred_mapped)
        y_fused.append(fused_pred)
        
    fusion_safety_counts["net_correction"] = (
        fusion_safety_counts["errors_corrected"] - fusion_safety_counts["errors_harmed"]
    )
    
    # Calculate metrics for all 3 configurations
    def calc_metrics(y_t, y_p):
        acc = accuracy_score(y_t, y_p)
        bal_acc = balanced_accuracy_score(y_t, y_p)
        p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_t, y_p, average="macro", zero_division=0)
        p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_t, y_p, average="weighted", zero_division=0)
        cm = confusion_matrix(y_t, y_p, labels=CATEGORIES)
        return {
            "top1_accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_precision": float(p_mac),
            "macro_recall": float(r_mac),
            "macro_f1": float(f1_mac),
            "weighted_precision": float(p_wt),
            "weighted_recall": float(r_wt),
            "weighted_f1": float(f1_wt),
            "confusion_matrix": cm.tolist()
        }
        
    sem_metrics = calc_metrics(y_true, y_sem)
    vis_metrics = calc_metrics(y_true, y_vis)
    fused_metrics = calc_metrics(y_true, y_fused)
    
    return {
        "count": len(items),
        "median_latency_ms": float(np.median(latencies)) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "semantic_only": sem_metrics,
        "visual_only": vis_metrics,
        "multimodal_fusion": fused_metrics,
        "fusion_safety": fusion_safety_counts
    }


def main():
    print("Initializing Semantic & Visual Classifier models...")
    semantic_cls = FinalRefinedClassifier()
    visual_adapter = VisualScreenTypeAdapter(use_int8=False)
    cached_ocr = load_cached_ocr_texts()
    
    # 1. Primary Validation Population (25 screenshots)
    df_primary = pd.read_csv(PRIMARY_MANIFEST)
    val_df = df_primary[df_primary["partition"] == "validation"]
    
    val_items = [
        {"filename": row["filename"], "true_category": row["original_semantic_category"], "notes": ""}
        for _, row in val_df.iterrows()
    ]
    
    # 2. External Benchmark Population (60 screenshots)
    df_ext = pd.read_csv(EXTERNAL_MANIFEST_V2)
    ext_items = [
        {"filename": row["filename"], "true_category": row["new_category"], "notes": str(row.get("summary", ""))}
        for _, row in df_ext.iterrows()
    ]
    
    print("Evaluating Multimodal Pipeline on Primary Validation Population (25 imgs)...")
    primary_val_metrics = evaluate_multimodal_population(semantic_cls, visual_adapter, val_items, cached_ocr, is_external=False)
    
    print("Evaluating Multimodal Pipeline on External Benchmark Population (60 imgs)...")
    ext_test_metrics = evaluate_multimodal_population(semantic_cls, visual_adapter, ext_items, cached_ocr, is_external=True)
    
    results = {
        "pipeline_name": "IntegratedMultimodalPipeline (Strategy E)",
        "primary_validation_benchmark": primary_val_metrics,
        "external_test_benchmark": ext_test_metrics
    }
    
    metrics_json_path = OUTPUT_DIR / "multimodal_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"Saved multimodal metrics to {metrics_json_path}")
    
    # Plot Confusion Matrix for Multimodal Fusion (Primary Validation)
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_array = np.array(primary_val_metrics["multimodal_fusion"]["confusion_matrix"])
    cax = ax.matshow(cm_array, cmap="Purples")
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_yticks(range(len(CATEGORIES)))
    ax.set_xticklabels(CATEGORIES, rotation=45, ha="left")
    ax.set_yticklabels(CATEGORIES)
    
    for i in range(len(CATEGORIES)):
        for j in range(len(CATEGORIES)):
            ax.text(j, i, str(cm_array[i, j]), ha="center", va="center", color="black" if cm_array[i, j] < cm_array.max()/2 else "white")
            
    plt.title("Multimodal Fusion System Confusion Matrix (Primary Validation Set)", pad=20)
    plt.xlabel("Predicted Category")
    plt.ylabel("True Category")
    plt.tight_layout()
    
    plot_path = OUTPUT_DIR / "confusion_matrix_multimodal.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {plot_path}")


if __name__ == "__main__":
    main()
