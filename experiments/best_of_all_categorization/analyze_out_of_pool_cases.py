"""
Forensic Failure Analysis of External V2 Out-of-Pool Screenshots
================================================================
Identifies and traces all 18 External V2 screenshots where the ground-truth category
was missing from the final candidate pool in the frozen Best-of-All evaluation.

Generates:
- experiments/best_of_all_categorization/out_of_pool_cases.json
- experiments/best_of_all_categorization/out_of_pool_analysis.md
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "best_of_all_categorization"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "candidate_reranking"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "accuracy_improvement"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "final_external_v2_validation"))

from src.embedder import SentenceTransformerEmbedder
from expanded_candidate_pool import ExpandedCandidatePoolBuilder
from best_of_all_reranker import BestOfAllCandidateReranker, VIS_5_TO_SEMANTIC_MAP
from experiments.intelligence.external_v2.evaluation_targets import load_evaluation_targets
from experiments.visual_understanding.phase7j.visual_screen_type_adapter import VisualScreenTypeAdapter

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "best_of_all_categorization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def classify_ocr_density(ocr_text: str) -> str:
    text = ocr_text.strip()
    words = text.split()
    n_words = len(words)
    if n_words == 0:
        return "near_zero_text"
    elif n_words <= 5:
        return "sparse_text"
    elif n_words <= 20:
        return "moderate_text"
    else:
        return "rich_text"


def main():
    print("=" * 80)
    print("FORENSIC FAILURE ANALYSIS: 18 EXTERNAL V2 OUT-OF-POOL SCREENSHOTS")
    print("=" * 80)

    embedder = SentenceTransformerEmbedder()
    pool_builder = ExpandedCandidatePoolBuilder(embedder=embedder)
    reranker = BestOfAllCandidateReranker(pool_builder=pool_builder)
    vis_adapter = VisualScreenTypeAdapter()

    raw_text_v2_path = PROJECT_ROOT / "scratch" / "raw_text_v2.json"
    raw_ocr_map = {}
    if raw_text_v2_path.exists():
        with open(raw_text_v2_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            for k, v in raw_data.items():
                raw_ocr_map[k] = v["text"] if isinstance(v, dict) and "text" in v else str(v)

    targets = load_evaluation_targets(allow_empty_categories=False)

    out_of_pool_cases = []
    
    for idx, target in enumerate(targets):
        rec_id = str(getattr(target, "id", f"ext2_{idx+1:03d}"))
        fname = str(getattr(target, "filename", getattr(target, "file", "")))
        gt_cat = str(target.category)
        label_reason = str(getattr(target, "label_reason", getattr(target, "notes", "")))
        img_path = str(getattr(target, "image_path", ""))

        ocr_text = raw_ocr_map.get(rec_id, raw_ocr_map.get(fname, raw_ocr_map.get(fname.split(".")[0], label_reason)))
        if not ocr_text.strip():
            ocr_text = label_reason if label_reason.strip() else fname

        # Run Candidate Pool Generation & Reranking
        pool_res = pool_builder.build_candidate_pool(raw_text=ocr_text, notes=label_reason)
        clf_top3 = pool_res["classifier_top3"]
        minilm_top5 = pool_res["minilm_top5"]
        pool_records = pool_res["candidate_pool"]
        pool_cats = [c["category"] for c in pool_records]

        # Get visual prediction if image exists
        vis_res = {"screen_type": "other_screen", "confidence": 0.0}
        if img_path and Path(img_path).exists():
            try:
                v_pred = vis_adapter.predict(img_path)
                vis_res = {
                    "screen_type": v_pred.get("screen_type", "other_screen"),
                    "confidence": float(v_pred.get("confidence", 0.0))
                }
            except Exception:
                pass

        # Check if GT category is in pool
        in_pool = (gt_cat in pool_cats)

        if not in_pool:
            res = reranker.rerank(raw_text=ocr_text, notes=label_reason, vis_screen_type=vis_res["screen_type"], vis_conf=vis_res["confidence"])
            
            # Check intermediate candidates
            sem_full = pool_builder.base_classifier.predict_final(raw_text=ocr_text, notes=label_reason, variant="C")
            clf_all_cands = [c["category"] for c in sem_full.get("top_candidates", [])]
            minilm_all_cands = pool_res["minilm_top5"]

            in_clf_top3 = gt_cat in clf_top3
            in_clf_all = gt_cat in clf_all_cands
            in_minilm_top5 = gt_cat in minilm_top5

            if not in_clf_all and not in_minilm_top5:
                failure_stage = "both_miss"
            elif not in_clf_all and in_minilm_top5:
                failure_stage = "classifier_miss"
            elif in_clf_all and not in_minilm_top5:
                failure_stage = "minilm_miss"
            else:
                failure_stage = "candidate_pool_construction_issue"

            ocr_density = classify_ocr_density(ocr_text)

            case_item = {
                "screenshot_id": rec_id,
                "file": fname,
                "image_path": img_path,
                "ground_truth_category": gt_cat,
                "label_reason": label_reason,
                "classifier_top1": clf_top3[0] if clf_top3 else "General",
                "classifier_top3": clf_top3,
                "classifier_top_candidates": clf_all_cands,
                "minilm_candidates": minilm_top5,
                "final_candidate_pool": pool_cats,
                "yolo_visual_prediction": vis_res,
                "ocr_text": ocr_text,
                "ocr_density": ocr_density,
                "word_count": len(ocr_text.strip().split()),
                "in_intermediate_sources": {
                    "in_clf_top3": in_clf_top3,
                    "in_clf_all": in_clf_all,
                    "in_minilm_top5": in_minilm_top5
                },
                "final_best_of_all_prediction": res["final_category"],
                "failure_stage": failure_stage
            }
            out_of_pool_cases.append(case_item)

    print(f"Verified Out-of-Pool Cases Count: {len(out_of_pool_cases)}")
    
    # Save raw json for manual annotation filling
    (OUTPUT_DIR / "out_of_pool_cases_raw.json").write_text(json.dumps(out_of_pool_cases, indent=2, cls=NumpyEncoder), encoding="utf-8")
    print("Saved out_of_pool_cases_raw.json successfully.")


if __name__ == "__main__":
    main()
