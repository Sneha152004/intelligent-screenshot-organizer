"""
Phase 5 Ablations & Phase 6 Error-Driven Group Analysis
======================================================
Performs controlled component ablations and 4-group error analysis strictly on
internal validation data (metadata_v2.csv lines 250-375) and External V2 benchmark.

Group Taxonomy:
- Group 1: Both Baseline A and Baseline B correct
- Group 2: Baseline A (Prev Best) correct, Baseline B (Recovery V2) wrong
- Group 3: Baseline B (Recovery V2) correct, Baseline A (Prev Best) wrong
- Group 4: Both wrong
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
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "categorization_recovery"))

from src.embedder import SentenceTransformerEmbedder
from best_of_all_reranker import BestOfAllCandidateReranker, HybridCandidateGenerator
from expanded_candidate_pool import ExpandedCandidatePoolBuilder
from reranker_experiments import MultiSignalCandidateReranker
from expanded_candidate_generator_v2 import CandidateGeneratorV2
from reranker_recovery_engine import CategorizationRecoveryReranker

from experiments.intelligence.ocr_evidence_semantic.ocr_evidence_semantic import load_cached_ocr_texts
from experiments.intelligence.external_v2.evaluation_targets import load_evaluation_targets

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "best_of_all_categorization"


def run_ablations(df_val: pd.DataFrame, ocr_cache: Dict[str, str], embedder: SentenceTransformerEmbedder) -> Dict[str, Any]:
    gen = HybridCandidateGenerator(embedder=embedder)
    n = len(df_val)

    ablation_configs = [
        ("Classifier Base Only", False, False, 0.0, 0.0, False, False),
        ("+ Unscaled MiniLM", True, False, 0.0, 0.0, False, False),
        ("+ Rank Agreement", True, True, 0.0, 0.0, False, False),
        ("+ OCR Phrase Evidence", True, True, 0.20, 0.0, False, False),
        ("+ YOLO Visual Support", True, True, 0.20, 0.15, False, False),
        ("+ High-Conf Guardrail & Sparse Awareness", True, True, 0.20, 0.15, True, True)
    ]

    ablation_results = []
    for label, use_m, use_ag, w_ocr, w_vis, use_guard, use_sparse in ablation_configs:
        reranker = BestOfAllCandidateReranker(
            candidate_generator=gen,
            w_clf=0.55 if use_m else 1.0,
            w_minilm=0.25 if use_m else 0.0,
            w_agree=0.15 if use_ag else 0.0,
            w_ocr=w_ocr,
            w_vis=w_vis,
            enable_guardrail=use_guard,
            enable_sparse_text_handling=use_sparse
        )

        corrects, pool_hits = 0, 0
        for _, row in df_val.iterrows():
            gt_cat = str(row["category"])
            fn = str(row["file"])
            notes = str(row.get("notes", "")) if pd.notna(row.get("notes")) else ""
            raw_text = ocr_cache.get(fn, ocr_cache.get(fn.split(".")[0], notes))

            res = reranker.rerank(raw_text=raw_text, notes=notes)
            if res["final_category"] == gt_cat:
                corrects += 1
            if gt_cat in [c["category"] for c in res["candidate_pool"]]:
                pool_hits += 1

        ablation_results.append({
            "step": label,
            "top1_acc": round(corrects / n, 4),
            "coverage": round(pool_hits / n, 4)
        })

    return ablation_results


def run_group_analysis(ext_df: pd.DataFrame, ext_ocr_map: Dict[str, str], embedder: SentenceTransformerEmbedder) -> Dict[str, Any]:
    reranker_a = MultiSignalCandidateReranker(pool_builder=ExpandedCandidatePoolBuilder(embedder=embedder))
    reranker_b = CategorizationRecoveryReranker(candidate_generator=CandidateGeneratorV2(embedder=embedder))
    reranker_boa = BestOfAllCandidateReranker(candidate_generator=HybridCandidateGenerator(embedder=embedder))

    group1, group2, group3, group4 = [], [], [], []

    for _, row in ext_df.iterrows():
        rec_id = str(row["id"])
        fn = str(row["file"])
        gt_cat = str(row["category"])
        notes = str(row.get("notes", ""))
        ocr_text = ext_ocr_map.get(rec_id, ext_ocr_map.get(fn, ext_ocr_map.get(fn.split(".")[0], notes)))
        if not ocr_text.strip():
            ocr_text = notes if notes.strip() else fn

        res_a = reranker_a.rerank(raw_text=ocr_text, notes=notes, strategy="E")
        res_b = reranker_b.rerank(raw_text=ocr_text, notes=notes)
        res_boa = reranker_boa.rerank(raw_text=ocr_text, notes=notes)

        pred_a = res_a["final_category"]
        pred_b = res_b["final_category"]
        pred_boa = res_boa["final_category"]

        corr_a = (pred_a == gt_cat)
        corr_b = (pred_b == gt_cat)
        corr_boa = (pred_boa == gt_cat)

        item_info = {
            "id": rec_id,
            "file": fn,
            "ground_truth": gt_cat,
            "pred_a": pred_a,
            "pred_b": pred_b,
            "pred_boa": pred_boa,
            "corr_a": corr_a,
            "corr_b": corr_b,
            "corr_boa": corr_boa,
            "ocr_text_snippet": ocr_text[:120]
        }

        if corr_a and corr_b:
            group1.append(item_info)
        elif corr_a and not corr_b:
            group2.append(item_info)
        elif not corr_a and corr_b:
            group3.append(item_info)
        else:
            group4.append(item_info)

    n_tot = len(ext_df)
    return {
        "dataset_size": n_tot,
        "group1_both_correct": {"count": len(group1), "pct": round(len(group1)/n_tot, 4)},
        "group2_prev_correct_rec_wrong": {"count": len(group2), "pct": round(len(group2)/n_tot, 4), "items": group2},
        "group3_rec_correct_prev_wrong": {"count": len(group3), "pct": round(len(group3)/n_tot, 4), "items": group3},
        "group4_both_wrong": {"count": len(group4), "pct": round(len(group4)/n_tot, 4), "items": group4}
    }


def main():
    print("=" * 80)
    print("PHASE 5 & 6: ABLATION STUDY & ERROR-DRIVEN GROUP ANALYSIS")
    print("=" * 80)

    embedder = SentenceTransformerEmbedder()

    meta_df = pd.read_csv(PROJECT_ROOT / "data" / "metadata_v2.csv")
    val_df = meta_df.iloc[250:375].copy()
    ocr_cache = load_cached_ocr_texts()

    print("\n1. Running Controlled Component Ablations on Internal Validation...")
    ablations = run_ablations(val_df, ocr_cache, embedder)
    for step in ablations:
        print(f"Step: {step['step']:45s} -> Top-1 = {step['top1_acc']*100:.2f}%, Coverage = {step['coverage']*100:.2f}%")

    print("\n2. Running 4-Group Error Analysis on External V2...")
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

    group_analysis = run_group_analysis(ext_df, ext_ocr_map, embedder)
    print(f"\nGroup 1 (Both Correct): {group_analysis['group1_both_correct']['count']} screens ({group_analysis['group1_both_correct']['pct']*100:.2f}%)")
    print(f"Group 2 (Baseline A Correct, B Wrong): {group_analysis['group2_prev_correct_rec_wrong']['count']} screens ({group_analysis['group2_prev_correct_rec_wrong']['pct']*100:.2f}%)")
    print(f"Group 3 (Baseline B Correct, A Wrong): {group_analysis['group3_rec_correct_prev_wrong']['count']} screens ({group_analysis['group3_rec_correct_prev_wrong']['pct']*100:.2f}%)")
    print(f"Group 4 (Both Wrong): {group_analysis['group4_both_wrong']['count']} screens ({group_analysis['group4_both_wrong']['pct']*100:.2f}%)")

    output_pkg = {
        "internal_ablations": ablations,
        "group_analysis": group_analysis
    }
    (OUTPUT_DIR / "ablations_and_group_analysis.json").write_text(json.dumps(output_pkg, indent=2), encoding="utf-8")
    print("\nSaved ablations_and_group_analysis.json successfully.")


if __name__ == "__main__":
    main()
