"""
Failure Taxonomy Analysis Script
================================
Analyzes categorization errors and breaks them down into:
- Group A1: Correct candidate exists in pool, but Top-1 prediction is wrong (Reranking Failure)
- Group A2: Correct candidate exists in pool, and Top-1 prediction is correct (Positive Control)
- Group B1: Correct category is outside candidate pool (Candidate Generation Failure)
"""

import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "categorization_recovery"))

from reranker_recovery_engine import CategorizationRecoveryReranker
from experiments.intelligence.external_v2.evaluation_targets import load_evaluation_targets
from experiments.intelligence.ocr_evidence_semantic.ocr_evidence_semantic import load_cached_ocr_texts

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "categorization_recovery"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_failure_taxonomy():
    reranker = CategorizationRecoveryReranker()
    records = load_evaluation_targets(allow_empty_categories=False)
    
    raw_text_v2_path = PROJECT_ROOT / "scratch" / "raw_text_v2.json"
    raw_ocr_map = {}
    if raw_text_v2_path.exists():
        with open(raw_text_v2_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            for k, v in raw_data.items():
                if isinstance(v, dict) and "text" in v:
                    raw_ocr_map[k] = v["text"]
                elif isinstance(v, str):
                    raw_ocr_map[k] = v

    group_a1 = []  # In pool, wrong top-1
    group_a2 = []  # In pool, correct top-1
    group_b1 = []  # Out of pool

    for rec in records:
        rec_id = rec.id
        fname = rec.filename
        gt_cat = rec.category
        notes = rec.label_reason
        ocr_text = raw_ocr_map.get(rec_id, raw_ocr_map.get(fname, ""))

        res = reranker.rerank(raw_text=ocr_text, notes=notes)
        pred_cat = res["final_category"]
        pool_cats = [c["category"] for c in res["candidate_pool"]]

        in_pool = (gt_cat in pool_cats)
        is_corr = (pred_cat == gt_cat)

        rec_info = {
            "id": rec_id,
            "filename": fname,
            "ground_truth": gt_cat,
            "prediction": pred_cat,
            "in_pool": in_pool,
            "is_correct": is_corr,
            "ocr_length": len(ocr_text),
            "ocr_snippet": ocr_text[:100],
            "pool": pool_cats,
            "reranked": res["reranked"],
            "reason": res["reranking_reason"]
        }

        if is_corr:
            group_a2.append(rec_info)
        elif in_pool:
            group_a1.append(rec_info)
        else:
            group_b1.append(rec_info)

    taxonomy_summary = {
        "total_records": len(records),
        "group_a2_positive_control_count": len(group_a2),
        "group_a1_reranking_failures_count": len(group_a1),
        "group_b1_candidate_gen_failures_count": len(group_b1),
        "group_a1_by_category": pd.DataFrame(group_a1)["ground_truth"].value_counts().to_dict() if group_a1 else {},
        "group_b1_by_category": pd.DataFrame(group_b1)["ground_truth"].value_counts().to_dict() if group_b1 else {},
        "group_a1_failures": group_a1,
        "group_b1_failures": group_b1
    }

    out_json = OUTPUT_DIR / "failure_taxonomy_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(taxonomy_summary, f, indent=2)
    print(f"Saved failure taxonomy summary JSON to {out_json}")


if __name__ == "__main__":
    generate_failure_taxonomy()
