"""
Best-of-All Multi-Signal Candidate Reranker Engine
=================================================
Consolidates the optimal Candidate Generator (Classifier Top-K + Trained MiniLM Strategy C Top-5)
with unscaled weighted score fusion, domain OCR phrase evidence, validated YOLO screen-type evidence,
and High-Confidence Protection Guardrail.

All hyperparameter choices and weight selections are performed strictly on internal validation data (lines 250-375).
External V2 remains 100% frozen.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "best_of_all_categorization"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "candidate_reranking"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "accuracy_improvement"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "final_external_v2_validation"))

from src.embedder import SentenceTransformerEmbedder
from expanded_candidate_pool import ExpandedCandidatePoolBuilder, CANONICAL_CATEGORIES
from candidate_reranker import CATEGORY_EVIDENCE_MAP

# Validated Visual Screen Type to Semantic Category Mapping
VIS_5_TO_SEMANTIC_MAP = {
    "code_editor_terminal": "Coding",
    "travel_booking": "Travel",
    "media_player_library": "Entertainment",
    "document_article_view": "Education",
    "dashboard_settings": "Connectivity"
}


class BestOfAllCandidateReranker:
    """Best-of-All Evidence-Based Candidate Reranker."""

    def __init__(
        self,
        pool_builder: Optional[ExpandedCandidatePoolBuilder] = None,
        w_clf: float = 0.55,
        w_minilm: float = 0.25,
        w_ocr: float = 0.20,
        w_vis: float = 0.15,
        enable_guardrail: bool = True
    ):
        self.pool_builder = pool_builder or ExpandedCandidatePoolBuilder()
        self.w_clf = w_clf
        self.w_minilm = w_minilm
        self.w_ocr = w_ocr
        self.w_vis = w_vis
        self.enable_guardrail = enable_guardrail

    def rerank(
        self,
        raw_text: str,
        notes: str = "",
        vis_screen_type: str = "other_screen",
        vis_conf: float = 0.0
    ) -> Dict[str, Any]:
        """Executes candidate pool generation and multi-signal evidence reranking."""
        pool_res = self.pool_builder.build_candidate_pool(
            raw_text=raw_text,
            notes=notes,
            vis_screen_type=vis_screen_type,
            vis_conf=vis_conf
        )

        clf_top3 = pool_res["classifier_top3"]
        minilm_top5 = pool_res["minilm_top5"]
        candidates = pool_res["candidate_pool"]
        sem_conf_str = pool_res["classifier_confidence_str"]
        base_top1 = clf_top3[0] if clf_top3 else "General"

        text_lower = raw_text.lower().strip()

        ranked_scores = []
        for cand in candidates:
            cat = cand["category"]

            # 1. Unscaled Classifier Score
            s_clf = cand["classifier_score"]

            # 2. Unscaled MiniLM Score
            s_minilm = cand["minilm_score"]

            # 3. Domain OCR Phrase Evidence
            matches = [phrase for phrase in CATEGORY_EVIDENCE_MAP.get(cat, []) if phrase in text_lower]
            n_matches = len(matches)
            s_ocr = 0.0
            if n_matches >= 2:
                s_ocr = 0.35
            elif n_matches == 1:
                s_ocr = 0.20

            # 4. Validated YOLO Visual Evidence
            s_vis = 0.0
            if vis_conf >= 0.40 and vis_screen_type in VIS_5_TO_SEMANTIC_MAP:
                mapped_cat = VIS_5_TO_SEMANTIC_MAP[vis_screen_type]
                if cat == mapped_cat:
                    s_vis = min(1.0, vis_conf)

            # Combined Multi-Signal Score
            final_score = (self.w_clf * s_clf) + (self.w_minilm * s_minilm) + (self.w_ocr * s_ocr) + (self.w_vis * s_vis)

            # General penalty if specific phrase evidence is present for another category
            if cat == "General" and any(len([p for p in CATEGORY_EVIDENCE_MAP.get(other["category"], []) if p in text_lower]) >= 1 for other in candidates if other["category"] != "General"):
                final_score *= 0.50

            ranked_scores.append({
                "category": cat,
                "score": round(float(final_score), 4),
                "s_clf": round(float(s_clf), 4),
                "s_minilm": round(float(s_minilm), 4),
                "s_ocr": round(float(s_ocr), 4),
                "s_vis": round(float(s_vis), 4),
                "sources": cand["candidate_sources"],
                "n_ocr_matches": n_matches
            })

        ranked_scores.sort(key=lambda x: x["score"], reverse=True)

        top1_cand = ranked_scores[0] if ranked_scores else {"category": "General", "score": 0.0}
        top1_cat = top1_cand["category"]
        top1_score = top1_cand["score"]

        reranked = (top1_cat != base_top1)

        # High-Confidence Protection Guardrail
        if reranked and self.enable_guardrail and sem_conf_str == "High":
            cand_matches = top1_cand.get("n_ocr_matches", 0)
            if cand_matches < 2:
                top1_cat = base_top1
                top1_score = ranked_scores[[c["category"] for c in ranked_scores].index(base_top1)]["score"] if base_top1 in [c["category"] for c in ranked_scores] else top1_score
                reranked = False
                rerank_reason = f"Base Top-1 '{base_top1}' retained via High-Confidence Guardrail"
            else:
                rerank_reason = f"Promoted '{top1_cat}' over High-Confidence base Top-1 via strong phrase evidence ({cand_matches} matches)"
        elif reranked:
            rerank_reason = f"Promoted '{top1_cat}' (score={top1_score:.4f}) over base Top-1 '{base_top1}' via Multi-Signal Evidence"
        else:
            rerank_reason = f"Base Top-1 '{base_top1}' retained (score={top1_score:.4f})"

        # Sort ranked candidates list
        top1_c = top1_cat
        top2_c = ranked_scores[1]["category"] if len(ranked_scores) > 1 and ranked_scores[1]["category"] != top1_c else (ranked_scores[0]["category"] if len(ranked_scores) > 0 and ranked_scores[0]["category"] != top1_c else "General")
        top3_c = ranked_scores[2]["category"] if len(ranked_scores) > 2 and ranked_scores[2]["category"] not in (top1_c, top2_c) else "General"

        return {
            "top1_category": top1_c,
            "top1_score": top1_score,
            "top2_category": top2_c,
            "top2_score": ranked_scores[1]["score"] if len(ranked_scores) > 1 else 0.0,
            "top3_category": top3_c,
            "top3_score": ranked_scores[2]["score"] if len(ranked_scores) > 2 else 0.0,
            "final_category": top1_c,
            "confidence_str": sem_conf_str,
            "reranked": reranked,
            "reranking_reason": rerank_reason,
            "candidate_pool": candidates,
            "ranked_candidates": ranked_scores
        }
