"""
Generates the Multimodal Validation Predictions Audit CSV.
Evaluates the 168 held-out validation screenshots and outputs detailed prediction, margin, weight, and fused score records.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.intelligence.evaluate_multimodal_classifier import load_dataset_with_ocr
from experiments.intelligence.final_refinement.final_refinement_experiment import (
    FinalRefinedClassifier,
)
from src.visual.tiny_clip_encoder import TinyCLIPEncoder
from src.visual.visual_evidence import (
    VisualCategoryEvidenceEngine,
    CANONICAL_CATEGORIES,
)
from src.multimodal_classifier import MultimodalScreenshotClassifier

AUDIT_CSV_PATH = Path("experiments/intelligence/multimodal_validation_predictions_audit.csv")


def generate_audit_csv():
    df = load_dataset_with_ocr()
    train_cal_df, val_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["category"]
    )

    text_clf = FinalRefinedClassifier()
    clip_encoder = TinyCLIPEncoder(device="cpu")
    vis_engine = VisualCategoryEvidenceEngine(encoder=clip_encoder)
    vis_engine.initialize_prompts()

    multimodal_clf = MultimodalScreenshotClassifier(
        text_classifier=text_clf,
        visual_engine=vis_engine,
        enable_visual=True,
        w_text_default=0.70,
        w_visual_default=0.30,
    )

    rows = []
    for idx, row in val_df.iterrows():
        filename = row["file"]
        gt = row["category"]
        img_path = row["path"]
        ocr_text = row["ocr_text"]

        # Text branch
        res_text = text_clf.predict_final(ocr_text)
        text_pred = res_text["prediction"]
        text_margin = res_text["margin"]
        text_raw = res_text["category_scores"]

        # Visual branch
        res_vis = vis_engine.extract_evidence(img_path)
        vis_pred = res_vis["top_category"]
        vis_raw = res_vis["category_raw_scores"]
        sorted_vis = sorted(vis_raw.values(), reverse=True)
        vis_margin = round(sorted_vis[0] - sorted_vis[1], 4) if len(sorted_vis) > 1 else 0.0

        # Multimodal fused
        res_multi = multimodal_clf.predict(ocr_text=ocr_text, image_path=img_path)
        multi_pred = res_multi["prediction"]
        w_text = res_multi["text_weight"]
        w_vis = res_multi["visual_weight"]
        fallback = res_multi["visual_fallback_used"]

        # Scaled dicts
        text_cal = multimodal_clf._min_max_scale_dict(text_raw)
        vis_cal = multimodal_clf._min_max_scale_dict(vis_raw)
        fused_scores = res_multi["category_scores"]

        rows.append({
            "screenshot_id": filename,
            "ground_truth_category": gt,
            "text_prediction": text_pred,
            "visual_prediction": vis_pred,
            "multimodal_prediction": multi_pred,
            "text_margin": text_margin,
            "visual_margin": vis_margin,
            "selected_text_weight": w_text,
            "selected_visual_weight": w_vis,
            "normalized_text_scores": json.dumps({k: round(v, 4) for k, v in text_cal.items()}),
            "normalized_visual_scores": json.dumps({k: round(v, 4) for k, v in vis_cal.items()}),
            "final_fused_scores": json.dumps(fused_scores),
            "fallback_status": fallback,
        })

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(AUDIT_CSV_PATH, index=False)
    print(f"Audit CSV successfully generated at {AUDIT_CSV_PATH} with {len(audit_df)} rows.")


if __name__ == "__main__":
    generate_audit_csv()
