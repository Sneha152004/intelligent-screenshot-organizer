"""
Evidence-Based Candidate Reranker Experiments
=============================================
Evaluates multi-signal reranking strategies on the expanded candidate pool:
- Strategy A: Baseline Production Classifier
- Strategy B: Classifier + MiniLM Score Fusion
- Strategy C: Classifier + MiniLM + OCR Phrase Evidence
- Strategy D: Classifier + MiniLM + Validated Visual Evidence
- Strategy E: Full Multi-Signal Evidence Reranker

All tuning and weight selection are performed strictly on internal validation data (lines 250-375).
External V2 remains 100% frozen.
"""

import sys
import json
import time
import tracemalloc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "candidate_reranking"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "accuracy_improvement"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "final_external_v2_validation"))

from expanded_candidate_pool import ExpandedCandidatePoolBuilder, CANONICAL_CATEGORIES
from candidate_reranker import TwoStageEvidenceReranker, CATEGORY_EVIDENCE_MAP
from experiments.visual_understanding.phase7j.visual_screen_type_adapter import VisualScreenTypeAdapter

# Validated Visual Screen Type to Semantic Category Mapping
VIS_5_TO_SEMANTIC_MAP = {
    "code_editor_terminal": "Coding",
    "travel_booking": "Travel",
    "media_player_library": "Entertainment",
    "document_article_view": "Education",
    "dashboard_settings": "Connectivity"
}


class MultiSignalCandidateReranker:
    """Multi-Signal Evidence Reranker for Expanded Candidate Pools."""

    def __init__(
        self,
        pool_builder: Optional[ExpandedCandidatePoolBuilder] = None,
        w_clf: float = 0.55,
        w_minilm: float = 0.25,
        w_ocr: float = 0.20,
        w_vis: float = 0.15
    ):
        self.pool_builder = pool_builder or ExpandedCandidatePoolBuilder()
        self.w_clf = w_clf
        self.w_minilm = w_minilm
        self.w_ocr = w_ocr
        self.w_vis = w_vis

    def rerank(
        self,
        raw_text: str,
        notes: str = "",
        vis_screen_type: str = "other_screen",
        vis_conf: float = 0.0,
        strategy: str = "E"
    ) -> Dict[str, Any]:
        """
        Executes candidate pool generation and evidence-based reranking.
        """
        # 1. Build Expanded Candidate Pool
        pool_res = self.pool_builder.build_candidate_pool(
            raw_text=raw_text,
            notes=notes,
            vis_screen_type=vis_screen_type,
            vis_conf=vis_conf
        )
        
        clf_top3 = pool_res["classifier_top3"]
        minilm_top5 = pool_res["minilm_top5"]
        candidates = pool_res["candidate_pool"]
        
        text_lower = raw_text.lower()

        # Strategy A: Baseline Production Reranker
        if strategy == "A":
            base_rr = TwoStageEvidenceReranker(base_classifier=self.pool_builder.base_classifier)
            res = base_rr.predict_and_rerank(raw_text=raw_text, notes=notes, vis_screen_type=vis_screen_type, vis_conf=vis_conf)
            
            top1_c = res["top1_category"]
            top2_c = res["top2_category"]
            top3_c = res["top3_category"]
            fin_c = res["final_category"]
            
            # Format candidate pool output
            return {
                "top1_category": top1_c,
                "top1_score": res["top1_score"],
                "top2_category": top2_c,
                "top2_score": res["top2_score"],
                "top3_category": top3_c,
                "top3_score": res["top3_score"],
                "final_category": fin_c,
                "confidence_str": res["confidence_str"],
                "reranked": res["reranked"],
                "reranking_reason": res["reranking_reason"],
                "candidate_pool": candidates,
                "ranked_candidates": [
                    {"category": fin_c, "score": res["top1_score"]},
                    {"category": top1_c if fin_c != top1_c else top2_c, "score": res["top2_score"]}
                ]
            }

        # Calculate evidence scores for candidates in pool
        ranked_scores = []
        for cand in candidates:
            cat = cand["category"]
            
            # 1. Normalized Classifier Score
            s_clf = cand["classifier_score"]
            
            # 2. Normalized MiniLM Score
            s_minilm = cand["minilm_score"]
            
            # 3. OCR Phrase Evidence Score
            matches = [phrase for phrase in CATEGORY_EVIDENCE_MAP.get(cat, []) if phrase in text_lower]
            n_matches = len(matches)
            s_ocr = 0.0
            if n_matches >= 2:
                s_ocr = 0.35
            elif n_matches == 1:
                s_ocr = 0.20

            # 4. YOLO Visual Evidence Score
            s_vis = 0.0
            if vis_conf >= 0.40 and vis_screen_type in VIS_5_TO_SEMANTIC_MAP:
                mapped_cat = VIS_5_TO_SEMANTIC_MAP[vis_screen_type]
                if cat == mapped_cat:
                    s_vis = min(1.0, vis_conf)

            # Strategy Selection Score Computation
            if strategy == "B":
                # Classifier + MiniLM Score Fusion
                final_score = (self.w_clf * s_clf) + (self.w_minilm * s_minilm)
            elif strategy == "C":
                # Classifier + MiniLM + OCR Evidence
                final_score = (self.w_clf * s_clf) + (self.w_minilm * s_minilm) + (self.w_ocr * s_ocr)
            elif strategy == "D":
                # Classifier + MiniLM + Visual Evidence
                final_score = (self.w_clf * s_clf) + (self.w_minilm * s_minilm) + (self.w_vis * s_vis)
            else:  # Strategy E: Full Multi-Signal Reranker
                final_score = (self.w_clf * s_clf) + (self.w_minilm * s_minilm) + (self.w_ocr * s_ocr) + (self.w_vis * s_vis)

            # Extra rule: General penalty when specific phrase evidence is present for another candidate
            if cat == "General" and any(len([p for p in CATEGORY_EVIDENCE_MAP.get(other["category"], []) if p in text_lower]) >= 1 for other in candidates if other["category"] != "General"):
                final_score *= 0.50

            ranked_scores.append({
                "category": cat,
                "score": round(float(final_score), 4),
                "s_clf": s_clf,
                "s_minilm": s_minilm,
                "s_ocr": s_ocr,
                "s_vis": s_vis,
                "sources": cand["candidate_sources"]
            })

        # Sort candidate pool descending by final score
        ranked_scores.sort(key=lambda x: x["score"], reverse=True)

        top1_cat = ranked_scores[0]["category"] if len(ranked_scores) > 0 else "General"
        top1_score = ranked_scores[0]["score"] if len(ranked_scores) > 0 else 0.0

        top2_cat = ranked_scores[1]["category"] if len(ranked_scores) > 1 else ""
        top2_score = ranked_scores[1]["score"] if len(ranked_scores) > 1 else 0.0

        top3_cat = ranked_scores[2]["category"] if len(ranked_scores) > 2 else ""
        top3_score = ranked_scores[2]["score"] if len(ranked_scores) > 2 else 0.0

        base_top1 = clf_top3[0]
        reranked = (top1_cat != base_top1)

        if reranked:
            rerank_reason = f"Promoted '{top1_cat}' (score={top1_score:.4f}) over base Top-1 '{base_top1}' via Multi-Signal Fusion"
        else:
            rerank_reason = f"Base Top-1 '{base_top1}' retained (score={top1_score:.4f})"

        confidence_str = pool_res["classifier_confidence_str"]

        return {
            "top1_category": top1_cat,
            "top1_score": top1_score,
            "top2_category": top2_cat,
            "top2_score": top2_score,
            "top3_category": top3_cat,
            "top3_score": top3_score,
            "final_category": top1_cat,
            "confidence_str": confidence_str,
            "reranked": reranked,
            "reranking_reason": rerank_reason,
            "candidate_pool": candidates,
            "ranked_candidates": ranked_scores
        }
