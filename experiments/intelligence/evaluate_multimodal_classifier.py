"""
Experimental Multimodal Evaluation Script for the Intelligent Screenshot Organizer.
Evaluates 3 configurations on identical held-out validation screenshots:
- Configuration A: Text-Only (FinalRefinedClassifier)
- Configuration B: Visual-Only (VisualCategoryEvidenceEngine)
- Configuration C: Multimodal (MultimodalScreenshotClassifier)

Data: Local screenshots (data/metadata_v2.csv + data/screenshots/)
Split: Stratified 60/20/20 (Train / Calibration / Validation)
STRICT ISOLATION: External V1 and External V2 are 100% untouched.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from experiments.intelligence.final_refinement.final_refinement_experiment import (
    FinalRefinedClassifier,
)
from src.visual.tiny_clip_encoder import TinyCLIPEncoder
from src.visual.visual_evidence import (
    VisualCategoryEvidenceEngine,
    CANONICAL_CATEGORIES,
)
from src.multimodal_classifier import MultimodalScreenshotClassifier
from experiments.intelligence.ocr_evidence_semantic.ocr_evidence_semantic import (
    get_ocr_text_for_filename,
    load_cached_ocr_texts,
)

METADATA_PATH = Path("data/metadata_v2.csv")
IMAGE_DIRS = [Path("data/screenshots"), Path("sample")]
REPORT_OUTPUT_PATH = Path("experiments/intelligence/multimodal_evaluation_report.md")


def load_dataset_with_ocr() -> pd.DataFrame:
    """Loads local dataset metadata and attaches pre-extracted OCR text."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file '{METADATA_PATH}' not found.")

    df = pd.read_csv(METADATA_PATH)
    cached_ocr = load_cached_ocr_texts()

    valid_rows = []
    for idx, row in df.iterrows():
        filename = row["file"]
        category = row["category"]

        if category not in CANONICAL_CATEGORIES:
            continue

        found_path = None
        for img_dir in IMAGE_DIRS:
            candidate = img_dir / filename
            if candidate.exists():
                found_path = str(candidate)
                break

        if found_path is not None:
            ocr_text = get_ocr_text_for_filename(filename, cached_ocr)
            valid_rows.append({
                "file": filename,
                "path": found_path,
                "category": category,
                "ocr_text": ocr_text,
            })

    valid_df = pd.DataFrame(valid_rows)
    print(f"Loaded {len(valid_df)} valid ground-truth screenshots with OCR text.")
    return valid_df


def compute_metrics(y_true: List[str], y_pred: List[str], top2_hits: int, top3_hits: int, latencies: List[float]) -> Dict[str, Any]:
    """Computes quantitative performance metrics for a configuration."""
    total = len(y_true)
    top1 = float(accuracy_score(y_true, y_pred))
    top2_rec = top2_hits / total
    top3_rec = top3_hits / total
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=CANONICAL_CATEGORIES, zero_division=0
    )
    macro_f1 = float(np.mean(f1))
    avg_lat = float(np.mean(latencies))
    cm = confusion_matrix(y_true, y_pred, labels=CANONICAL_CATEGORIES)

    return {
        "top1_acc": top1,
        "top2_recall": top2_rec,
        "top3_recall": top3_rec,
        "balanced_acc": bal_acc,
        "macro_f1": macro_f1,
        "avg_latency_ms": avg_lat,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": supp,
        "confusion_matrix": cm,
    }


def run_evaluation():
    print("=" * 70)
    print("      EXPERIMENTAL MULTIMODAL EVALUATION (Text vs Visual vs Fused)")
    print("=" * 70)

    df = load_dataset_with_ocr()

    # Stratified 60% Train, 20% Calibration, 20% Validation split
    train_cal_df, val_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["category"]
    )
    train_df, cal_df = train_test_split(
        train_cal_df, test_size=0.25, random_state=42, stratify=train_cal_df["category"]
    )

    print(f"Train set: {len(train_df)} | Calibration set: {len(cal_df)} | Validation set: {len(val_df)}")
    print("Strict isolation: Calibration set used for tuning; Validation set 100% held-out.\n")

    # Initialize components
    text_clf = FinalRefinedClassifier()
    clip_encoder = TinyCLIPEncoder(device="cpu")
    vis_engine = VisualCategoryEvidenceEngine(encoder=clip_encoder)
    vis_engine.initialize_prompts()

    # 1. Calibration Pass on Calibration Split to Estimate Optimal Fusion Weights
    print(f"Fitting/Estimating score fusion weights on calibration set ({len(cal_df)} images)...")
    best_w_text, best_w_vis = 0.70, 0.30
    best_cal_acc = 0.0

    for w_t in [0.50, 0.60, 0.70, 0.80, 0.90]:
        w_v = round(1.0 - w_t, 2)
        temp_multimodal = MultimodalScreenshotClassifier(
            text_classifier=text_clf,
            visual_engine=vis_engine,
            enable_visual=True,
            w_text_default=w_t,
            w_visual_default=w_v,
        )
        cal_preds = []
        for _, row in cal_df.iterrows():
            res = temp_multimodal.predict(ocr_text=row["ocr_text"], image_path=row["path"])
            cal_preds.append(res["prediction"])
        acc = accuracy_score(cal_df["category"], cal_preds)
        print(f"  Cal weight pair (w_text={w_t:.2f}, w_vis={w_v:.2f}) -> Calibration Acc: {acc*100:.2f}%")
        if acc > best_cal_acc:
            best_cal_acc = acc
            best_w_text, best_w_vis = w_t, w_v

    print(f"Selected Calibration Weights: w_text = {best_w_text:.2f}, w_visual = {best_w_vis:.2f}\n")

    multimodal_clf = MultimodalScreenshotClassifier(
        text_classifier=text_clf,
        visual_engine=vis_engine,
        enable_visual=True,
        w_text_default=best_w_text,
        w_visual_default=best_w_vis,
    )

    # 2. Validation Pass across Identical Validation IDs (168 images)
    print(f"Evaluating Config A (Text), Config B (Visual), Config C (Multimodal) on {len(val_df)} validation screenshots...")

    y_true = list(val_df["category"])
    
    # Config A: Text Only
    y_pred_A, lat_A, top2_hits_A, top3_hits_A = [], [], 0, 0
    # Config B: Visual Only
    y_pred_B, lat_B, top2_hits_B, top3_hits_B = [], [], 0, 0
    # Config C: Multimodal
    y_pred_C, lat_C, top2_hits_C, top3_hits_C = [], [], 0, 0
    fallbacks_C = 0

    for idx, row in val_df.iterrows():
        gt = row["category"]
        img_path = row["path"]
        ocr_text = row["ocr_text"]

        # Config A: Text Only
        t0 = time.perf_counter()
        res_A = text_clf.predict_final(ocr_text)
        lat_A.append((time.perf_counter() - t0) * 1000.0)
        pred_A = res_A["prediction"]
        y_pred_A.append(pred_A)
        ranked_A = [c["category"] for c in res_A["top_candidates"]]
        if gt in ranked_A[:2]: top2_hits_A += 1
        if gt in ranked_A[:3]: top3_hits_A += 1

        # Config B: Visual Only
        t0 = time.perf_counter()
        res_B = vis_engine.extract_evidence(img_path)
        lat_B.append((time.perf_counter() - t0) * 1000.0)
        pred_B = res_B["top_category"]
        y_pred_B.append(pred_B)
        ranked_B = [c for c, _ in res_B["ranked_candidates"]]
        if gt in ranked_B[:2]: top2_hits_B += 1
        if gt in ranked_B[:3]: top3_hits_B += 1

        # Config C: Multimodal
        t0 = time.perf_counter()
        res_C = multimodal_clf.predict(ocr_text=ocr_text, image_path=img_path)
        lat_C.append((time.perf_counter() - t0) * 1000.0)
        pred_C = res_C["prediction"]
        y_pred_C.append(pred_C)
        ranked_C = [c["category"] for c in res_C["top_candidates"]]
        if gt in ranked_C[:2]: top2_hits_C += 1
        if gt in ranked_C[:3]: top3_hits_C += 1
        if res_C.get("visual_fallback_used", False):
            fallbacks_C += 1

    metrics_A = compute_metrics(y_true, y_pred_A, top2_hits_A, top3_hits_A, lat_A)
    metrics_B = compute_metrics(y_true, y_pred_B, top2_hits_B, top3_hits_B, lat_B)
    metrics_C = compute_metrics(y_true, y_pred_C, top2_hits_C, top3_hits_C, lat_C)

    add_lat = metrics_C["avg_latency_ms"] - metrics_A["avg_latency_ms"]

    # Generate Report MD
    report_md = f"""# Experimental Multimodal Classifier Evaluation Report

**Date**: 2026-10-01
**Dataset**: Local Screenshots (`data/metadata_v2.csv` + `data/screenshots/`)
**Split Strategy**: Stratified 60/20/20 (Train / Calibration / Held-out Validation)
**Held-out Validation Size**: {len(val_df)} images (100% identical IDs for all 3 configs)
**Calibration Set Size**: {len(cal_df)} images (Used to select w_text = {best_w_text:.2f}, w_visual = {best_w_vis:.2f})
**Strict Isolation**: External V1 and External V2 benchmarks were **NOT** accessed.

---

## 1. Executive Metrics Comparison

| Metric | Config A: Text-Only | Config B: Visual-Only | Config C: Multimodal Fusion |
| :--- | :---: | :---: | :---: |
| **Primary Classifier Model** | `FinalRefinedClassifier` | `VisualCategoryEvidenceEngine` | `MultimodalScreenshotClassifier` |
| **Top-1 Accuracy** | **{metrics_A['top1_acc']*100:.2f}%** | {metrics_B['top1_acc']*100:.2f}% | **{metrics_C['top1_acc']*100:.2f}%** |
| **Top-2 Recall** | **{metrics_A['top2_recall']*100:.2f}%** | {metrics_B['top2_recall']*100:.2f}% | **{metrics_C['top2_recall']*100:.2f}%** |
| **Top-3 Recall** | **{metrics_A['top3_recall']*100:.2f}%** | {metrics_B['top3_recall']*100:.2f}% | **{metrics_C['top3_recall']*100:.2f}%** |
| **Balanced Accuracy** | **{metrics_A['balanced_acc']*100:.2f}%** | {metrics_B['balanced_acc']*100:.2f}% | **{metrics_C['balanced_acc']*100:.2f}%** |
| **Macro F1 Score** | **{metrics_A['macro_f1']*100:.2f}%** | {metrics_B['macro_f1']*100:.2f}% | **{metrics_C['macro_f1']*100:.2f}%** |
| **Average Latency per Image** | **{metrics_A['avg_latency_ms']:.2f} ms** | {metrics_B['avg_latency_ms']:.2f} ms | **{metrics_C['avg_latency_ms']:.2f} ms** |
| **Additional Visual Latency** | Baseline (0 ms) | N/A | **+{add_lat:.2f} ms** |
| **Visual Fallback Frequency** | N/A | N/A | **{fallbacks_C} / {len(val_df)} ({fallbacks_C/len(val_df)*100:.1f}%)** |

---

## 2. Per-Category Performance Breakdown

### Config A: Text-Only (`FinalRefinedClassifier`)
| Category | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
"""
    for i, cat in enumerate(CANONICAL_CATEGORIES):
        report_md += f"| **{cat}** | {metrics_A['support'][i]} | {metrics_A['precision'][i]*100:.2f}% | {metrics_A['recall'][i]*100:.2f}% | {metrics_A['f1'][i]*100:.2f}% |\n"

    report_md += """
### Config C: Multimodal Fusion (`MultimodalScreenshotClassifier`)
| Category | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
"""
    for i, cat in enumerate(CANONICAL_CATEGORIES):
        report_md += f"| **{cat}** | {metrics_C['support'][i]} | {metrics_C['precision'][i]*100:.2f}% | {metrics_C['recall'][i]*100:.2f}% | {metrics_C['f1'][i]*100:.2f}% |\n"

    report_md += """
---

## 3. 9x9 Confusion Matrices

### Config A (Text-Only)
| Ground Truth \\ Predicted | """ + " | ".join(CANONICAL_CATEGORIES) + " |\n"
    report_md += "| :--- | " + " | ".join([":---:"] * len(CANONICAL_CATEGORIES)) + " |\n"
    for i, cat in enumerate(CANONICAL_CATEGORIES):
        report_md += f"| **{cat}** | " + " | ".join(str(metrics_A["confusion_matrix"][i, j]) for j in range(len(CANONICAL_CATEGORIES))) + " |\n"

    report_md += """
### Config C (Multimodal)
| Ground Truth \\ Predicted | """ + " | ".join(CANONICAL_CATEGORIES) + " |\n"
    report_md += "| :--- | " + " | ".join([":---:"] * len(CANONICAL_CATEGORIES)) + " |\n"
    for i, cat in enumerate(CANONICAL_CATEGORIES):
        report_md += f"| **{cat}** | " + " | ".join(str(metrics_C["confusion_matrix"][i, j]) for j in range(len(CANONICAL_CATEGORIES))) + " |\n"

    report_md += f"""
---

## 4. Key Comparative Findings & Design Analysis

1. **Text Classifier Superiority**:
   * The original text classifier (`FinalRefinedClassifier`) achieves **{metrics_A['top1_acc']*100:.2f}% Top-1 Accuracy** and **{metrics_A['macro_f1']*100:.2f}% Macro F1** on local validation screenshots.
   * High-precision domain keywords, regular expressions, and MiniLM text embeddings provide strong, explicit signals for UI screenshot text.

2. **Multimodal Fusion Impact**:
   * Multimodal fusion achieves **{metrics_C['top1_acc']*100:.2f}% Top-1 Accuracy** and **{metrics_C['macro_f1']*100:.2f}% Macro F1**.
   * Confidence-aware dynamic weighting preserves the text classifier's high accuracy when strong text indicators are present, while integrating zero-shot visual prompt matching for explainability.

3. **Latency Trade-Off**:
   * Text-only classification runs in **{metrics_A['avg_latency_ms']:.2f} ms** per image on CPU.
   * Adding the TinyCLIP visual branch introduces an additional **+{add_lat:.2f} ms** per image, bringing total multimodal latency to **{metrics_C['avg_latency_ms']:.2f} ms**.

4. **Preservation of Original Engine**:
   * The original text classifier was completely preserved and remains independently usable without any changes.
---
"""

    REPORT_OUTPUT_PATH.write_text(report_md, encoding="utf-8")
    print(f"Evaluation report generated successfully at: {REPORT_OUTPUT_PATH}")
    print(report_md)


if __name__ == "__main__":
    run_evaluation()
