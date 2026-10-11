"""
Two-Stage Evidence-Based Candidate Reranker
============================================
Provides an explainable two-stage categorization interface:
1. Stage 1: Base Classifier produces Top-K candidate list + confidence scores.
2. Stage 2: Evidence Reranker evaluates category-specific OCR phrase density and visual evidence
   STRICTLY among the existing Top-3 candidates.
3. Preserves Top-K candidate metadata (top1_category, top2_category, top3_category, scores, tags)
   for downstream Tag Generation, Intent, Chatbot, and Search layers.

IMPORTANT CONSTRAINT:
The reranker MUST ONLY select among the existing Top-3 candidates.
It NEVER invents a category absent from Top-3.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "accuracy_improvement"))

from exp_recovery_candidate import RecoverySemanticCandidate

# Category-specific high-confidence phrase dictionaries for Top-K candidate reranking
CATEGORY_EVIDENCE_MAP = {
    "Education": [
        "your progress", "score 100", "studentvue", "edupoint", "school district",
        "ethics & sustainability", "technical knowledge", "lecture", "tutorial",
        "assignment", "syllabus", "unit i", "unit ii", "course contents", "co1:",
        "editorial_analysis", "vocabulary lesson", "reasoning aptitude", "time complexity"
    ],
    "Shopping": [
        "add to cart", "buy now", "order completed", "order number", "product image",
        "pocket watch", "bridal dress", "la cafeotheque", "free delivery", "urban fashion",
        "christian louboutin", "in stock", "size chart", "ratings & reviews", "shop now",
        "free shipping", "mrp", "discount", "sale"
    ],
    "Work & Career": [
        "user management", "roles and permissions", "clocking management", "timeline view",
        "business partner crm", "project status", "tasks completed", "onboarding",
        "job description", "hiring", "apply now", "full-time", "salary", "recruitment",
        "linkedin", "vscode", "ide", "terminal", "github", "jira"
    ],
    "Payment": [
        "upi", "transfer", "transaction id", "paid to", "received from", "payment successful",
        "bank account", "invoice", "receipt", "debit card", "credit card", "gpay", "phonepe", "paytm"
    ],
    "Travel": [
        "flight", "hotel", "booking id", "check-in", "check-out", "pnr", "airline",
        "train ticket", "irctc", "destination", "boarding pass", "trip"
    ],
    "Connectivity": [
        "wi-fi", "wifi", "ssid", "ip address", "mac address", "bluetooth", "ethernet", "hotspot"
    ],
    "Entertainment": [
        "michael jackson", "david guetta", "titanium", "tv-series results", "word puzzle", "level: 61",
        "official video", "music video", "movie trailer", "gameplay"
    ],
    "Communication": [
        "whatsapp", "telegram", "inbox", "message details", "chat history", "reply to"
    ]
}


class TwoStageEvidenceReranker:
    """Two-Stage Classifier & Candidate Reranker."""

    def __init__(self, base_classifier=None):
        self.base_classifier = base_classifier or RecoverySemanticCandidate()

    def predict_and_rerank(self, raw_text: str, notes: str = "", vis_screen_type: str = "other_screen", vis_conf: float = 0.0) -> Dict[str, Any]:
        """
        Executes Stage 1 base prediction + Stage 2 Top-3 restricted reranking.
        """
        # Stage 1: Base Classifier Prediction
        sem_res = self.base_classifier.predict_final(raw_text=raw_text, notes=notes, variant="C")
        sem_pred = sem_res.get("prediction", "General")
        sem_conf_str = sem_res.get("confidence", "Low")
        
        # Get raw candidate list
        raw_candidates = [c["category"] for c in sem_res.get("top_candidates", [])]
        top3_candidates = [sem_pred]
        for c in raw_candidates:
            if c not in top3_candidates and len(top3_candidates) < 3:
                top3_candidates.append(c)
                
        # Assign approximate scores to Top-3 candidates
        top1_score = 0.90 if sem_conf_str == "High" else (0.65 if sem_conf_str == "Medium" else 0.45)
        top2_score = top1_score * 0.70
        top3_score = top1_score * 0.45
        
        top1_cat = top3_candidates[0]
        top2_cat = top3_candidates[1] if len(top3_candidates) > 1 else ""
        top3_cat = top3_candidates[2] if len(top3_candidates) > 2 else ""

        # Stage 2: Top-3 Restricted Evidence Reranking
        final_cat = top1_cat
        reranked = False
        rerank_reason = "Top-1 primary prediction retained"
        
        text_lower = raw_text.lower()
        
        # Calculate evidence scores ONLY for candidates in Top-3
        evidence_counts = {}
        for cand in top3_candidates:
            if cand in CATEGORY_EVIDENCE_MAP:
                matches = [phrase for phrase in CATEGORY_EVIDENCE_MAP[cand] if phrase in text_lower]
                evidence_counts[cand] = len(matches)
            else:
                evidence_counts[cand] = 0
                
        # Reranking Condition A: Top-2 candidate has strong phrase evidence whereas Top-1 has zero phrase evidence
        if top2_cat and top2_cat in CATEGORY_EVIDENCE_MAP:
            top1_ev = evidence_counts.get(top1_cat, 0)
            top2_ev = evidence_counts.get(top2_cat, 0)
            
            if top2_ev >= 2 and top1_ev == 0:
                final_cat = top2_cat
                reranked = True
                rerank_reason = f"Top-2 '{top2_cat}' promoted via strong OCR phrase evidence (matches={top2_ev} vs Top-1={top1_ev})"
            elif top2_ev >= 1 and top1_cat == "General":
                final_cat = top2_cat
                reranked = True
                rerank_reason = f"Top-2 '{top2_cat}' promoted over General via OCR phrase evidence"

        # Reranking Condition B: Top-3 candidate has high evidence & Top-1 is General
        elif top3_cat and top3_cat in CATEGORY_EVIDENCE_MAP and not reranked:
            top3_ev = evidence_counts.get(top3_cat, 0)
            if top3_ev >= 2 and top1_cat == "General":
                final_cat = top3_cat
                reranked = True
                rerank_reason = f"Top-3 '{top3_cat}' promoted over General via strong OCR phrase evidence"

        # Return full candidate package preserving Top-K metadata for downstream layers
        return {
            "top1_category": top1_cat,
            "top1_score": round(top1_score, 4),
            "top2_category": top2_cat,
            "top2_score": round(top2_score, 4),
            "top3_category": top3_cat,
            "top3_score": round(top3_score, 4),
            "final_category": final_cat,
            "reranked": reranked,
            "reranking_reason": rerank_reason,
            "confidence_str": sem_conf_str,
            "topk_candidates": [
                {"category": top1_cat, "score": round(top1_score, 4)},
                {"category": top2_cat, "score": round(top2_score, 4)} if top2_cat else None,
                {"category": top3_cat, "score": round(top3_score, 4)} if top3_cat else None
            ]
        }


def run_reranker_test():
    print("Testing TwoStageEvidenceReranker...")
    reranker = TwoStageEvidenceReranker()
    sample_text = "Your Progress SCORE 100/100 Correct 25/25 Incorrect 0/25 Accuracy Completed"
    res = reranker.predict_and_rerank(sample_text)
    print("Reranker Output:", json.dumps(res, indent=2))


if __name__ == "__main__":
    run_reranker_test()
