"""
Categorization Recovery Reranker Engine
======================================
Principled Multi-Signal Evidence Reranker featuring:
1. Scale-Normalized Classifier & MiniLM Scores
2. Rank-Aware Convergence Agreement Bonus (rewards candidates present in both sources)
3. Domain OCR Phrase Evidence Booster
4. Validated YOLO Visual Screen-Type Support
5. High-Confidence Top-1 Protection Guardrail
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "categorization_recovery"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "final_external_v2_validation"))

from expanded_candidate_generator_v2 import CandidateGeneratorV2, CANONICAL_CATEGORIES
from candidate_reranker import CATEGORY_EVIDENCE_MAP

VIS_5_TO_SEMANTIC_MAP = {
    "code_editor_terminal": "Coding",
    "travel_booking": "Travel",
    "media_player_library": "Entertainment",
    "document_article_view": "Education",
    "dashboard_settings": "Connectivity"
}


class CategorizationRecoveryReranker:
    """Multi-Signal Evidence Reranker with Scale Normalization & Convergence Agreement."""

    def __init__(
        self,
        candidate_generator: Optional[CandidateGeneratorV2] = None,
        w_clf: float = 0.45,
        w_minilm: float = 0.25,
        w_agree: float = 0.15,
        w_ocr: float = 0.20,
        w_vis: float = 0.15
    ):
        self.generator = candidate_generator or CandidateGeneratorV2()
        self.w_clf = w_clf
        self.w_minilm = w_minilm
        self.w_agree = w_agree
        self.w_ocr = w_ocr
        self.w_vis = w_vis

    def rerank(
        self,
        raw_text: str,
        notes: str = "",
        vis_screen_type: str = "other_screen",
        vis_conf: float = 0.0,
        enable_high_conf_guardrail: bool = True
    ) -> Dict[str, Any]:
        """
        Executes candidate pool generation and multi-signal evidence reranking.
        """
        # 1. Build Expanded Candidate Pool
        pool_res = self.generator.build_expanded_candidate_pool(raw_text=raw_text, notes=notes)
        candidates = pool_res["candidate_pool"]
        clf_top3 = pool_res["classifier_top_k"][:3]
        sem_conf_str = pool_res["classifier_confidence_str"]

        text_lower = raw_text.lower()
        word_count = len(text_lower.split())

        # Extract raw scores for normalization
        raw_clf_scores = [c["classifier_score"] for c in candidates]
        raw_m_scores = [c["minilm_score"] for c in candidates]

        max_clf = max(raw_clf_scores) if max(raw_clf_scores) > 0 else 1.0
        min_clf = min(raw_clf_scores)

        max_m = max(raw_m_scores) if max(raw_m_scores) > 0 else 1.0
        min_m = min(raw_m_scores)

        ranked_scores = []
        for cand in candidates:
            cat = cand["category"]

            # Normalized Classifier Score [0, 1]
            norm_clf = (cand["classifier_score"] - min_clf) / (max_clf - min_clf) if max_clf > min_clf else (1.0 if cand["classifier_score"] > 0 else 0.0)

            # Normalized MiniLM Score [0, 1]
            norm_m = (cand["minilm_score"] - min_m) / (max_m - min_m) if max_m > min_m else (1.0 if cand["minilm_score"] > 0 else 0.0)

            # Rank-Aware Convergence Agreement Bonus
            s_agree = 0.0
            if cand["both_sources"]:
                if cat != "General" and cand["classifier_rank"] <= 2 and cand["minilm_rank"] <= 2:
                    s_agree = 1.0
                elif cat != "General":
                    s_agree = 0.50
                else:
                    s_agree = 0.20
            elif cand["from_classifier"] and cand["classifier_rank"] == 1:
                s_agree = 0.60 if cat != "General" else 0.30

            # OCR Phrase Evidence Score
            matches = [phrase for phrase in CATEGORY_EVIDENCE_MAP.get(cat, []) if phrase in text_lower]
            n_matches = len(matches)
            s_ocr = 0.0
            if n_matches >= 3:
                s_ocr = 1.0
            elif n_matches == 2:
                s_ocr = 0.75
            elif n_matches == 1:
                s_ocr = 0.40

            # YOLO Visual Evidence Support
            s_vis = 0.0
            if vis_conf >= 0.40 and vis_screen_type in VIS_5_TO_SEMANTIC_MAP:
                mapped_cat = VIS_5_TO_SEMANTIC_MAP[vis_screen_type]
                if cat == mapped_cat:
                    s_vis = min(1.0, vis_conf)

            # Composite Reranking Score
            total_score = (
                (self.w_clf * norm_clf)
                + (self.w_minilm * norm_m)
                + (self.w_agree * s_agree)
                + (self.w_ocr * s_ocr)
                + (self.w_vis * s_vis)
            )

            # Special General penalty if competing candidate has strong OCR evidence
            if cat == "General" and any(len([p for p in CATEGORY_EVIDENCE_MAP.get(other["category"], []) if p in text_lower]) >= 1 for other in candidates if other["category"] != "General"):
                total_score *= 0.40

            ranked_scores.append({
                "category": cat,
                "score": round(float(total_score), 4),
                "norm_clf": round(norm_clf, 4),
                "norm_m": round(norm_m, 4),
                "s_agree": s_agree,
                "s_ocr": s_ocr,
                "s_vis": s_vis,
                "sources": cand["candidate_sources"],
                "clf_rank": cand["classifier_rank"],
                "minilm_rank": cand["minilm_rank"]
            })

        # Sort candidate pool descending by composite reranking score
        ranked_scores.sort(key=lambda x: x["score"], reverse=True)

        top1_cat = ranked_scores[0]["category"] if len(ranked_scores) > 0 else "General"
        top1_score = ranked_scores[0]["score"] if len(ranked_scores) > 0 else 0.0

        top2_cat = ranked_scores[1]["category"] if len(ranked_scores) > 1 else ""
        top2_score = ranked_scores[1]["score"] if len(ranked_scores) > 1 else 0.0

        top3_cat = ranked_scores[2]["category"] if len(ranked_scores) > 2 else ""
        top3_score = ranked_scores[2]["score"] if len(ranked_scores) > 2 else 0.0

        base_top1 = clf_top3[0]

        # High-Confidence Protection Guardrail
        # If classifier has High confidence on base_top1 and no candidate has >= 2 phrase matches, protect base_top1
        reranked = False
        rerank_reason = "Base Top-1 retained"

        if enable_high_conf_guardrail and sem_conf_str == "High" and base_top1 != top1_cat:
            has_strong_ocr = any(r["s_ocr"] >= 0.75 for r in ranked_scores if r["category"] != base_top1)
            if not has_strong_ocr:
                # Retain base_top1
                top1_cat = base_top1
                reranked = False
                rerank_reason = f"High-confidence base Top-1 '{base_top1}' retained (protected from unevidenced override)"
            else:
                reranked = True
                rerank_reason = f"Promoted '{top1_cat}' over High-confidence base '{base_top1}' via strong phrase evidence"
        elif base_top1 != top1_cat:
            reranked = True
            rerank_reason = f"Promoted '{top1_cat}' (score={top1_score:.4f}) over base Top-1 '{base_top1}'"

        return {
            "top1_category": top1_cat,
            "top1_score": top1_score,
            "top2_category": top2_cat,
            "top2_score": top2_score,
            "top3_category": top3_cat,
            "top3_score": top3_score,
            "final_category": top1_cat,
            "confidence_str": sem_conf_str,
            "reranked": reranked,
            "reranking_reason": rerank_reason,
            "candidate_pool": candidates,
            "ranked_candidates": ranked_scores
        }
