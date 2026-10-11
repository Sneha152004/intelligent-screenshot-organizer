"""
Phase 2 & Phase 3: Baseline Reproduction & True Baseline Benchmark Script
==========================================================================
Reproduces Baseline A (Previous Best) and Baseline B (Recovery V2) on identical
frozen evaluation splits using a standardized, bug-free OCR lookup protocol.

Splits:
1. Internal Validation (metadata_v2.csv lines 250-375, n=125)
2. Internal Holdout Test (metadata_v2.csv lines 375-500, n=125)
3. External V2 Benchmark (n=244)
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "best_of_all_categorization"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "candidate_reranking"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "categorization_recovery"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "accuracy_improvement"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "final_external_v2_validation"))

from src.embedder import SentenceTransformerEmbedder
from expanded_candidate_pool import ExpandedCandidatePoolBuilder
from reranker_experiments import MultiSignalCandidateReranker
from expanded_candidate_generator_v2 import CandidateGeneratorV2
from reranker_recovery_engine import CategorizationRecoveryReranker

from experiments.intelligence.ocr_evidence_semantic.ocr_evidence_semantic import load_cached_ocr_texts
from experiments.intelligence.external_v2.evaluation_targets import load_evaluation_targets

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
        return super().default(obj)


def get_ocr_text(row: pd.Series, ocr_cache: Dict[str, str], is_ext: bool) -> str:
    """Robust, bug-free OCR text lookup handling integer IDs and filenames."""
    if is_ext:
        rec_id_str = str(row["id"])
        fname = str(row["file"])
        text = ocr_cache.get(rec_id_str, ocr_cache.get(fname, ocr_cache.get(fname.split(".")[0], "")))
        if not text.strip():
            notes = str(row.get("notes", "")) if pd.notna(row.get("notes")) else ""
            text = notes if notes.strip() else fname
        return text
    else:
        fname = str(row["file"])
        notes = str(row.get("notes", "")) if pd.notna(row.get("notes")) else ""
        text = ocr_cache.get(fname, ocr_cache.get(fname.split(".")[0], notes))
        if not text.strip():
            text = notes if notes.strip() else fname
        return text


def evaluate_baseline_a(df_split: pd.DataFrame, reranker_a: MultiSignalCandidateReranker, ocr_cache: Dict[str, str], is_ext: bool) -> Dict[str, Any]:
    """Evaluates Baseline A (Previous Best) under bug-free protocol."""
    y_true, y_base, y_final = [], [], []
    ranked_lists, mrr_ranks = [], []
    pool_hits = 0
    corrections, harms = 0, 0
    records = []

    for _, row in df_split.iterrows():
        gt_cat = str(row["category"])
        notes = str(row.get("notes", "")) if pd.notna(row.get("notes")) else ""
        raw_text = get_ocr_text(row, ocr_cache, is_ext)

        res = reranker_a.rerank(raw_text=raw_text, notes=notes, strategy="E")
        base_sem_pred = reranker_a.pool_builder.base_classifier.predict_final(raw_text=raw_text, notes=notes, variant="C").get("prediction", "General")
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
            "file": str(row.get("file", "")),
            "ground_truth": gt_cat,
            "base_pred": base_sem_pred,
            "final_pred": final_cat,
            "in_pool": gt_cat in pool_cats,
            "correct": final_is_corr
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
        "n": n,
        "top1": top1,
        "top2": top2,
        "top3": top3,
        "pool_coverage": pool_hits / n,
        "outside_pool_count": n - pool_hits,
        "macro_precision": p_mac,
        "macro_recall": r_mac,
        "macro_f1": f1_mac,
        "weighted_precision": p_wt,
        "weighted_recall": r_wt,
        "weighted_f1": f1_wt,
        "balanced_accuracy": bal_acc,
        "mrr": mrr,
        "corrections": corrections,
        "harms": harms,
        "records": records
    }


def evaluate_baseline_b(df_split: pd.DataFrame, reranker_b: CategorizationRecoveryReranker, ocr_cache: Dict[str, str], is_ext: bool) -> Dict[str, Any]:
    """Evaluates Baseline B (Recovery V2) under bug-free protocol."""
    y_true, y_base, y_final = [], [], []
    ranked_lists, mrr_ranks = [], []
    pool_hits = 0
    corrections, harms = 0, 0
    records = []

    for _, row in df_split.iterrows():
        gt_cat = str(row["category"])
        notes = str(row.get("notes", "")) if pd.notna(row.get("notes")) else ""
        raw_text = get_ocr_text(row, ocr_cache, is_ext)

        res = reranker_b.rerank(raw_text=raw_text, notes=notes)
        base_sem_pred = reranker_b.generator.base_classifier.predict_final(raw_text=raw_text, notes=notes, variant="C").get("prediction", "General")
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
            "file": str(row.get("file", "")),
            "ground_truth": gt_cat,
            "base_pred": base_sem_pred,
            "final_pred": final_cat,
            "in_pool": gt_cat in pool_cats,
            "correct": final_is_corr
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
        "n": n,
        "top1": top1,
        "top2": top2,
        "top3": top3,
        "pool_coverage": pool_hits / n,
        "outside_pool_count": n - pool_hits,
        "macro_precision": p_mac,
        "macro_recall": r_mac,
        "macro_f1": f1_mac,
        "weighted_precision": p_wt,
        "weighted_recall": r_wt,
        "weighted_f1": f1_wt,
        "balanced_accuracy": bal_acc,
        "mrr": mrr,
        "corrections": corrections,
        "harms": harms,
        "records": records
    }


def main():
    print("=" * 80)
    print("PHASE 2 & 3: REPRODUCING BASELINES UNDER STANDARDIZED PROTOCOL")
    print("=" * 80)

    embedder = SentenceTransformerEmbedder()
    
    # Initialize Baseline A & Baseline B
    reranker_a = MultiSignalCandidateReranker(pool_builder=ExpandedCandidatePoolBuilder(embedder=embedder))
    reranker_b = CategorizationRecoveryReranker(candidate_generator=CandidateGeneratorV2(embedder=embedder))

    # Load Splits
    meta_df = pd.read_csv(PROJECT_ROOT / "data" / "metadata_v2.csv")
    val_df = meta_df.iloc[250:375].copy()
    test_df = meta_df.iloc[375:500].copy()

    # Load External V2 OCR Map from scratch/raw_text_v2.json
    raw_text_v2_path = PROJECT_ROOT / "scratch" / "raw_text_v2.json"
    ext_ocr_map = {}
    if raw_text_v2_path.exists():
        with open(raw_text_v2_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            for k, v in raw_data.items():
                if isinstance(v, dict) and "text" in v:
                    ext_ocr_map[k] = v["text"]
                elif isinstance(v, str):
                    ext_ocr_map[k] = v

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
    
    # Combined OCR lookup for External V2 vs Internal
    ext_ocr_combined = {**ocr_cache, **ext_ocr_map}

    splits = [
        ("Internal Validation (n=125)", val_df, ocr_cache, False),
        ("Internal Holdout (n=125)", test_df, ocr_cache, False),
        ("External V2 (n=244)", ext_df, ext_ocr_combined, True)
    ]

    results = {"Baseline_A": {}, "Baseline_B": {}}

    for name, df_split, cache, is_ext in splits:
        res_a = evaluate_baseline_a(df_split, reranker_a, cache, is_ext)
        res_b = evaluate_baseline_b(df_split, reranker_b, cache, is_ext)

        results["Baseline_A"][name] = res_a
        results["Baseline_B"][name] = res_b

        print(f"\n--- {name} ---")
        print(f"Baseline A (Prev Best): Top-1 = {res_a['top1']*100:.2f}%, Coverage = {res_a['pool_coverage']*100:.2f}%, Top-3 = {res_a['top3']*100:.2f}%, MRR = {res_a['mrr']:.4f}")
        print(f"Baseline B (Recovery V2): Top-1 = {res_b['top1']*100:.2f}%, Coverage = {res_b['pool_coverage']*100:.2f}%, Top-3 = {res_b['top3']*100:.2f}%, MRR = {res_b['mrr']:.4f}")

    (OUTPUT_DIR / "reproduced_baselines.json").write_text(json.dumps(results, indent=2, cls=NumpyEncoder), encoding="utf-8")
    print("\nSaved reproduced_baselines.json successfully.")


if __name__ == "__main__":
    main()
