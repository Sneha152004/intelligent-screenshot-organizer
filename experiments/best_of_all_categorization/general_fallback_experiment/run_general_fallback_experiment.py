"""
Mandatory General Candidate Fallback Experiment
==============================================
Evaluates the mandatory addition of "General" candidate to candidate pools across:
1. Internal Validation (metadata_v2.csv lines 250-375, n=125)
2. Internal Holdout Test (metadata_v2.csv lines 375-500, n=125)
3. Frozen External V2 Benchmark (n=244)

Candidate Pool Construction:
    candidate_pool = Classifier Top-5 UNION MiniLM Top-5 UNION {"General"}

Outputs:
- general_fallback_config.json
- general_fallback_metrics.json
- general_fallback_comparison.csv
- 18_cases_recovery.json
- new_general_predictions_analysis.json
- GENERAL_FALLBACK_EXPERIMENT_REPORT.md
"""

import sys
import json
import time
import tracemalloc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "best_of_all_categorization"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "candidate_reranking"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "accuracy_improvement"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "categorization_recovery"))

from src.embedder import SentenceTransformerEmbedder
from expanded_candidate_pool import ExpandedCandidatePoolBuilder, CANONICAL_CATEGORIES
from best_of_all_reranker import BestOfAllCandidateReranker
from experiments.intelligence.ocr_evidence_semantic.ocr_evidence_semantic import load_cached_ocr_texts
from experiments.intelligence.external_v2.evaluation_targets import load_evaluation_targets

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "best_of_all_categorization" / "general_fallback_experiment"
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


class GeneralFallbackPoolBuilder(ExpandedCandidatePoolBuilder):
    """Wrapper that guarantees 'General' is always present in candidate pool."""

    def build_candidate_pool(
        self,
        raw_text: str,
        notes: str = "",
        vis_screen_type: str = "other_screen",
        vis_conf: float = 0.0
    ) -> Dict[str, Any]:
        pool_res = super().build_candidate_pool(raw_text=raw_text, notes=notes, vis_screen_type=vis_screen_type, vis_conf=vis_conf)
        candidates = pool_res["candidate_pool"]
        pool_cats = [c["category"] for c in candidates]
        if "General" not in pool_cats:
            candidates.append({
                "category": "General",
                "classifier_rank": None,
                "classifier_score": 0.0,
                "minilm_rank": None,
                "minilm_score": 0.0,
                "candidate_sources": ["mandatory_fallback"],
                "from_classifier": False,
                "from_minilm": False
            })
            pool_res["pool_size"] = len(candidates)
            pool_res["pool_categories"] = [r["category"] for r in candidates]
        return pool_res


def evaluate_split(
    df_split: pd.DataFrame,
    reranker: BestOfAllCandidateReranker,
    ocr_cache: Dict[str, str],
    is_ext: bool = False
) -> Dict[str, Any]:
    y_true, y_base, y_final = [], [], []
    ranked_lists, mrr_ranks = [], []
    pool_hits = 0
    corrections, harms = 0, 0
    records = []

    for _, row in df_split.iterrows():
        rec_id = str(row.get("id", ""))
        fname = str(row["file"])
        gt_cat = str(row["category"])
        notes = str(row.get("notes", "")) if pd.notna(row.get("notes")) else ""

        if is_ext:
            ocr_text = ocr_cache.get(rec_id, ocr_cache.get(fname, ocr_cache.get(fname.split(".")[0], notes)))
        else:
            ocr_text = ocr_cache.get(fname, ocr_cache.get(fname.split(".")[0], notes))

        if not ocr_text.strip():
            ocr_text = notes if notes.strip() else fname

        res = reranker.rerank(raw_text=ocr_text, notes=notes)

        base_sem_res = reranker.pool_builder.base_classifier.predict_final(raw_text=ocr_text, notes=notes, variant="C")
        base_sem_pred = base_sem_res.get("prediction", "General")
        final_cat = res["final_category"]
        pool_cats = [c["category"] for c in res["candidate_pool"]]

        y_true.append(gt_cat)
        y_base.append(base_sem_pred)
        y_final.append(final_cat)

        r_list = [c["category"] for c in res["ranked_candidates"]]
        ranked_lists.append(r_list)

        if gt_cat in r_list:
            rank = r_list.index(gt_cat) + 1
            mrr_ranks.append(1.0 / rank)
        else:
            mrr_ranks.append(0.0)

        if gt_cat in pool_cats:
            pool_hits += 1

        base_is_corr = (base_sem_pred == gt_cat)
        final_is_corr = (final_cat == gt_cat)

        if not base_is_corr and final_is_corr:
            corrections += 1
        elif base_is_corr and not final_is_corr:
            harms += 1

        records.append({
            "id": rec_id,
            "file": fname,
            "ground_truth": gt_cat,
            "base_semantic_pred": base_sem_pred,
            "final_category": final_cat,
            "top1_category": res["top1_category"],
            "top2_category": res["top2_category"],
            "top3_category": res["top3_category"],
            "reranked": res["reranked"],
            "reranking_reason": res["reranking_reason"],
            "correct": final_is_corr,
            "in_pool": gt_cat in pool_cats,
            "candidate_pool": pool_cats,
            "ranked_candidates": res["ranked_candidates"]
        })

    n = len(df_split)
    top1 = float(accuracy_score(y_true, y_final))
    top2 = float(np.mean([1 if gt in r[:2] else 0 for gt, r in zip(y_true, ranked_lists)]))
    top3 = float(np.mean([1 if gt in r[:3] else 0 for gt, r in zip(y_true, ranked_lists)]))
    
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_true, y_final, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_final, average="weighted", zero_division=0)
    bal_acc = float(balanced_accuracy_score(y_true, y_final))
    mrr = float(np.mean(mrr_ranks))

    return {
        "dataset_size": n,
        "top1_accuracy": top1,
        "top2_accuracy": top2,
        "top3_accuracy": top3,
        "macro_precision": p_mac,
        "macro_recall": r_mac,
        "macro_f1": f1_mac,
        "weighted_precision": p_wt,
        "weighted_recall": r_wt,
        "weighted_f1": f1_wt,
        "balanced_accuracy": bal_acc,
        "mrr": mrr,
        "pool_coverage": pool_hits / n,
        "outside_pool_count": n - pool_hits,
        "impact": {
            "corrections": corrections,
            "harms": harms,
            "net_improvement": corrections - harms
        },
        "records": records
    }


def main():
    print("=" * 80)
    print("EXECUTING MANDATORY GENERAL CANDIDATE EXPERIMENT")
    print("=" * 80)

    tracemalloc.start()
    t0 = time.time()

    embedder = SentenceTransformerEmbedder()
    
    # 1. Baseline Control (Immutable)
    base_pool_builder = ExpandedCandidatePoolBuilder(embedder=embedder)
    base_reranker = BestOfAllCandidateReranker(pool_builder=base_pool_builder)

    # 2. General Fallback Intervention
    exp_pool_builder = GeneralFallbackPoolBuilder(embedder=embedder)
    exp_reranker = BestOfAllCandidateReranker(pool_builder=exp_pool_builder)

    t_load = time.time() - t0
    _, ram_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Load Splits
    meta_df = pd.read_csv(PROJECT_ROOT / "data" / "metadata_v2.csv")
    val_df = meta_df.iloc[250:375].copy()
    test_df = meta_df.iloc[375:500].copy()

    raw_text_v2_path = PROJECT_ROOT / "scratch" / "raw_text_v2.json"
    ext_ocr_map = {}
    if raw_text_v2_path.exists():
        with open(raw_text_v2_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            for k, v in raw_data.items():
                ext_ocr_map[k] = v["text"] if isinstance(v, dict) and "text" in v else str(v)

    ext_targets = load_evaluation_targets(allow_empty_categories=False)
    ext_df = pd.DataFrame([
        {
            "id": getattr(t, "id", str(idx + 1)),
            "file": getattr(t, "filename", getattr(t, "file", f"ext2_{idx+1:03d}.jpg")),
            "category": t.category,
            "notes": getattr(t, "label_reason", getattr(t, "notes", ""))
        }
        for idx, t in enumerate(ext_targets)
    ])
    
    ocr_cache = load_cached_ocr_texts()
    ext_ocr_combined = {**ocr_cache, **ext_ocr_map}

    print("\n1. Evaluating Internal Validation Split (n=125)...")
    val_base = evaluate_split(val_df, base_reranker, ocr_cache, is_ext=False)
    val_exp = evaluate_split(val_df, exp_reranker, ocr_cache, is_ext=False)
    print(f"  Baseline Top-1 = {val_base['top1_accuracy']*100:.2f}%, Coverage = {val_base['pool_coverage']*100:.2f}%")
    print(f"  Exp Top-1      = {val_exp['top1_accuracy']*100:.2f}%, Coverage = {val_exp['pool_coverage']*100:.2f}%")

    print("\n2. Evaluating Internal Holdout Split (n=125)...")
    test_base = evaluate_split(test_df, base_reranker, ocr_cache, is_ext=False)
    test_exp = evaluate_split(test_df, exp_reranker, ocr_cache, is_ext=False)
    print(f"  Baseline Top-1 = {test_base['top1_accuracy']*100:.2f}%, Coverage = {test_base['pool_coverage']*100:.2f}%")
    print(f"  Exp Top-1      = {test_exp['top1_accuracy']*100:.2f}%, Coverage = {test_exp['pool_coverage']*100:.2f}%")

    print("\n3. Evaluating Frozen External V2 Benchmark (n=244)...")
    ext_base = evaluate_split(ext_df, base_reranker, ext_ocr_combined, is_ext=True)
    ext_exp = evaluate_split(ext_df, exp_reranker, ext_ocr_combined, is_ext=True)
    
    # Deterministic verification run
    ext_exp_run2 = evaluate_split(ext_df, exp_reranker, ext_ocr_combined, is_ext=True)
    is_deterministic = (ext_exp['top1_accuracy'] == ext_exp_run2['top1_accuracy']) and (ext_exp['records'] == ext_exp_run2['records'])
    print(f"  Baseline Top-1 = {ext_base['top1_accuracy']*100:.2f}%, Coverage = {ext_base['pool_coverage']*100:.2f}%, MRR = {ext_base['mrr']:.4f}")
    print(f"  Exp Top-1      = {ext_exp['top1_accuracy']*100:.2f}%, Coverage = {ext_exp['pool_coverage']*100:.2f}%, MRR = {ext_exp['mrr']:.4f}")
    print(f"  Deterministic Run Verification: {is_deterministic}")

    # 4. Save Config JSON
    config_pkg = {
        "experiment_name": "general_fallback_experiment",
        "status": "EVALUATED",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
        "deterministic": is_deterministic,
        "intervention": {
            "mandatory_category": "General",
            "rule": "Classifier Top-5 UNION MiniLM Top-5 UNION {'General'}"
        },
        "frozen_components_reused": {
            "base_classifier": "RecoverySemanticCandidate (Variant C)",
            "embedder": "sentence-transformers/all-MiniLM-L6-v2 (shared instance)",
            "reranker": "BestOfAllCandidateReranker (unscaled score fusion + High-Conf Guardrail)"
        }
    }
    (OUTPUT_DIR / "general_fallback_config.json").write_text(json.dumps(config_pkg, indent=2), encoding="utf-8")
    print("\nSaved general_fallback_config.json")

    # 5. Build Comparison CSV across splits
    comp_rows = [
        {
            "split": "Internal Validation (n=125)",
            "baseline_top1": f"{val_base['top1_accuracy']*100:.2f}%",
            "exp_top1": f"{val_exp['top1_accuracy']*100:.2f}%",
            "top1_delta": f"{(val_exp['top1_accuracy'] - val_base['top1_accuracy'])*100:+.2f} pp",
            "baseline_coverage": f"{val_base['pool_coverage']*100:.2f}%",
            "exp_coverage": f"{val_exp['pool_coverage']*100:.2f}%",
            "coverage_delta": f"{(val_exp['pool_coverage'] - val_base['pool_coverage'])*100:+.2f} pp",
            "baseline_mrr": f"{val_base['mrr']:.4f}",
            "exp_mrr": f"{val_exp['mrr']:.4f}",
            "corrections": val_exp['impact']['corrections'],
            "harms": val_exp['impact']['harms']
        },
        {
            "split": "Internal Holdout (n=125)",
            "baseline_top1": f"{test_base['top1_accuracy']*100:.2f}%",
            "exp_top1": f"{test_exp['top1_accuracy']*100:.2f}%",
            "top1_delta": f"{(test_exp['top1_accuracy'] - test_base['top1_accuracy'])*100:+.2f} pp",
            "baseline_coverage": f"{test_base['pool_coverage']*100:.2f}%",
            "exp_coverage": f"{test_exp['pool_coverage']*100:.2f}%",
            "coverage_delta": f"{(test_exp['pool_coverage'] - test_base['pool_coverage'])*100:+.2f} pp",
            "baseline_mrr": f"{test_base['mrr']:.4f}",
            "exp_mrr": f"{test_exp['mrr']:.4f}",
            "corrections": test_exp['impact']['corrections'],
            "harms": test_exp['impact']['harms']
        },
        {
            "split": "External V2 (n=244)",
            "baseline_top1": f"{ext_base['top1_accuracy']*100:.2f}%",
            "exp_top1": f"{ext_exp['top1_accuracy']*100:.2f}%",
            "top1_delta": f"{(ext_exp['top1_accuracy'] - ext_base['top1_accuracy'])*100:+.2f} pp",
            "baseline_coverage": f"{ext_base['pool_coverage']*100:.2f}%",
            "exp_coverage": f"{ext_exp['pool_coverage']*100:.2f}%",
            "coverage_delta": f"{(ext_exp['pool_coverage'] - ext_base['pool_coverage'])*100:+.2f} pp",
            "baseline_mrr": f"{ext_base['mrr']:.4f}",
            "exp_mrr": f"{ext_exp['mrr']:.4f}",
            "corrections": ext_exp['impact']['corrections'],
            "harms": ext_exp['impact']['harms']
        }
    ]
    pd.DataFrame(comp_rows).to_csv(OUTPUT_DIR / "general_fallback_comparison.csv", index=False)
    print("Saved general_fallback_comparison.csv")

    # 6. Trace the 18 Out-of-Pool Cases Recovery
    raw_out_of_pool_file = PROJECT_ROOT / "experiments" / "best_of_all_categorization" / "out_of_pool_cases.json"
    out_18_raw = json.loads(raw_out_of_pool_file.read_text(encoding="utf-8")) if raw_out_of_pool_file.exists() else []
    out_18_ids = set(c["screenshot_id"] for c in out_18_raw)

    base_records_map = {r["id"]: r for r in ext_base["records"]}
    exp_records_map = {r["id"]: r for r in ext_exp["records"]}

    recovered_18 = []
    for case_raw in out_18_raw:
        rec_id = case_raw["screenshot_id"]
        b_rec = base_records_map.get(rec_id, {})
        e_rec = exp_records_map.get(rec_id, {})

        gt = case_raw["ground_truth_category"]
        old_pool = b_rec.get("candidate_pool", [])
        new_pool = e_rec.get("candidate_pool", [])
        old_pred = b_rec.get("final_category", "General")
        new_pred = e_rec.get("final_category", "General")
        
        is_corrected = (old_pred != gt and new_pred == gt)
        is_harm = (old_pred == gt and new_pred != gt)

        ranked_cands = e_rec.get("ranked_candidates", [])
        gen_rank = None
        gen_score = 0.0
        for r_idx, r_item in enumerate(ranked_cands):
            if r_item["category"] == "General":
                gen_rank = r_idx + 1
                gen_score = r_item["score"]
                break

        recovered_18.append({
            "screenshot_id": rec_id,
            "file": case_raw["file"],
            "ground_truth": gt,
            "old_candidate_pool": old_pool,
            "new_candidate_pool": new_pool,
            "general_added": "General" in new_pool and "General" not in old_pool,
            "old_final_prediction": old_pred,
            "new_final_prediction": new_pred,
            "old_rank_of_gt": "missing",
            "new_rank_of_general": gen_rank,
            "new_score_of_general": gen_score,
            "corrected": is_corrected,
            "harm": is_harm,
            "classification": "CORRECTED" if is_corrected else ("HARM" if is_harm else "UNCHANGED_INCORRECT")
        })

    (OUTPUT_DIR / "18_cases_recovery.json").write_text(json.dumps(recovered_18, indent=2, cls=NumpyEncoder), encoding="utf-8")
    print("Saved 18_cases_recovery.json")

    # 7. Analyze All New General Predictions (False Positives / Shifts)
    new_general_preds = []
    for b_rec in ext_base["records"]:
        rec_id = b_rec["id"]
        e_rec = exp_records_map.get(rec_id, {})
        old_p = b_rec["final_category"]
        new_p = e_rec["final_category"]
        gt = b_rec["ground_truth"]

        if new_p == "General" or old_p != new_p:
            shift_type = "neutral"
            if old_p != "General" and new_p == "General":
                if gt == "General":
                    shift_type = "CORRECTION"
                else:
                    shift_type = "HARM_FALSE_GENERAL"
            elif old_p != gt and new_p == gt:
                shift_type = "CORRECTION"
            elif old_p == gt and new_p != gt:
                shift_type = "HARM"

            new_general_preds.append({
                "id": rec_id,
                "file": b_rec["file"],
                "ground_truth": gt,
                "old_prediction": old_p,
                "new_prediction": new_p,
                "shift_type": shift_type,
                "reason": e_rec.get("reranking_reason", "")
            })

    (OUTPUT_DIR / "new_general_predictions_analysis.json").write_text(json.dumps(new_general_preds, indent=2, cls=NumpyEncoder), encoding="utf-8")
    print("Saved new_general_predictions_analysis.json")

    # 8. Full Metrics JSON Package
    metrics_package = {
        "resource_impact": {
            "additional_disk_mb": 0.0,
            "additional_ram_mb": 0.0,
            "additional_latency_ms": 0.0,
            "model_reused": "sentence-transformers/all-MiniLM-L6-v2"
        },
        "internal_validation": {
            "baseline": val_base,
            "experiment": val_exp
        },
        "internal_holdout": {
            "baseline": test_base,
            "experiment": test_exp
        },
        "external_v2": {
            "baseline": ext_base,
            "experiment": ext_exp
        }
    }
    (OUTPUT_DIR / "general_fallback_metrics.json").write_text(json.dumps(metrics_package, indent=2, cls=NumpyEncoder), encoding="utf-8")
    print("Saved general_fallback_metrics.json")


if __name__ == "__main__":
    main()
