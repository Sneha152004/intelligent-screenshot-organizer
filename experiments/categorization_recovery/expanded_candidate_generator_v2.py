"""
Expanded Candidate Generator V2 & Prototype Refinement
======================================================
Implements candidate pool expansion and refined MiniLM category prototypes:
- Classifier Top-5 + MiniLM Strategy C+ Top-5 candidate union
- Refined category text descriptions including UI element indicators and domain intents
- Sparse-text handling and token normalization
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "accuracy_improvement"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "final_external_v2_validation"))

from src.embedder import SentenceTransformerEmbedder
from exp_recovery_candidate import RecoverySemanticCandidate

CANONICAL_CATEGORIES = [
    "Coding", "Communication", "Connectivity", "Education",
    "Entertainment", "General", "Payment", "Shopping", "Travel", "Work & Career"
]

# Refined Category Text Prototypes (Strategy C+) with UI elements and domain intents
REFINED_CATEGORY_PROTOTYPES = {
    "Coding": "source code programming IDE editor terminal github git python java c++ debugging stackoverflow vscode compiler error function class import script repository pull request commit branch html css javascript react sql database query",
    "Communication": "chat messaging inbox text message whatsapp telegram conversation reply email notification sender recipient chatroom group chat call log direct message message details contact",
    "Connectivity": "wifi network bluetooth ip address mac address ethernet router settings hotspot signal connection wireless network setup router admin bandwidth speed test cellular data airplane mode paired devices",
    "Education": "quiz student course lecture tutorial exam assignment study grade syllabus university school progress online course score correct incorrect quiz summary editorial analysis vocabulary lesson reasoning aptitude time complexity question answer textbook solution e-learning LMS",
    "Entertainment": "music video movie stream player playlist youtube spotify track song episode game show gameplay trailer audio playback pause play artist album gaming level score high score stream live",
    "General": "general screenshot mobile phone screen app display home screen system info settings wallpaper UI status bar icons widget camera gallery desktop lockscreen screenshot image photo generic interface smartphone controls",
    "Payment": "upi bank transaction transfer invoice receipt debit credit card payment successful account balance money gpay phonepe paytm bill pay merchant total amount paid received transaction id reference number bank statement",
    "Shopping": "cart product order buy online price discount shipping amazon store item retail delivery MRP in stock size chart ratings reviews add to cart checkout purchase price tag e-commerce deal offer coupon vendor",
    "Travel": "flight hotel booking PNR train ticket IRCTC boarding pass airline trip travel destination reservation check-in check-out passenger itinerary luggage gate seat location map stay resort flight status ticket details",
    "Work & Career": "workplace business CRM project management task hiring career resume salary clocking dashboard admin user management roles permissions clocking management timeline view business partner crm project status tasks completed onboarding job description recruitment linkedin jira workspace employee schedule"
}


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


class CandidateGeneratorV2:
    """Candidate Generator V2 with Refined Strategy C+ MiniLM Prototypes."""

    def __init__(self, embedder: Optional[SentenceTransformerEmbedder] = None, base_classifier: Optional[RecoverySemanticCandidate] = None):
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.base_classifier = base_classifier or RecoverySemanticCandidate()
        
        # Build MiniLM Strategy C+ prototypes
        self.text_prototypes: Dict[str, np.ndarray] = {}
        for cat in CANONICAL_CATEGORIES:
            desc_text = REFINED_CATEGORY_PROTOTYPES.get(cat, cat)
            vec = self.embedder.embed([desc_text])[0]
            self.text_prototypes[cat] = normalize_vector(np.array(vec, dtype=np.float32))

    def predict_minilm_candidates(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not query_text.strip():
            query_text = "general screenshot mobile phone app display"

        q_vec = normalize_vector(np.array(self.embedder.embed([query_text])[0], dtype=np.float32))
        scores = []
        for cat in CANONICAL_CATEGORIES:
            p_vec = self.text_prototypes[cat]
            sim = float(np.dot(q_vec, p_vec))
            scores.append((cat, round(sim, 4)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def build_expanded_candidate_pool(
        self,
        raw_text: str,
        notes: str = "",
        top_k_clf: int = 5,
        top_k_minilm: int = 5
    ) -> Dict[str, Any]:
        """
        Generates expanded candidate pool combining Classifier Top-K + MiniLM Top-K.
        """
        # 1. Base Classifier Prediction & Top Candidates
        sem_res = self.base_classifier.predict_final(raw_text=raw_text, notes=notes, variant="C")
        sem_pred = sem_res.get("prediction", "General")
        sem_conf_str = sem_res.get("confidence", "Low")

        raw_cands = [c["category"] for c in sem_res.get("top_candidates", [])]
        clf_candidates = [sem_pred]
        for c in raw_cands:
            if c not in clf_candidates and len(clf_candidates) < top_k_clf:
                clf_candidates.append(c)

        top1_score = 0.90 if sem_conf_str == "High" else (0.65 if sem_conf_str == "Medium" else 0.45)
        clf_scores = {clf_candidates[0]: top1_score}
        if len(clf_candidates) > 1:
            clf_scores[clf_candidates[1]] = round(top1_score * 0.75, 4)
        if len(clf_candidates) > 2:
            clf_scores[clf_candidates[2]] = round(top1_score * 0.50, 4)
        if len(clf_candidates) > 3:
            clf_scores[clf_candidates[3]] = round(top1_score * 0.35, 4)
        if len(clf_candidates) > 4:
            clf_scores[clf_candidates[4]] = round(top1_score * 0.25, 4)

        # 2. MiniLM Strategy C+ Top-K Candidates
        minilm_ranked = self.predict_minilm_candidates(raw_text, top_k=top_k_minilm)
        minilm_cats = [item[0] for item in minilm_ranked]
        minilm_scores = {item[0]: item[1] for item in minilm_ranked}

        # 3. Unique Candidate Union
        union_categories = list(clf_candidates)
        for cat in minilm_cats:
            if cat not in union_categories:
                union_categories.append(cat)

        candidate_records = []
        for cat in union_categories:
            sources = []
            c_rank = clf_candidates.index(cat) + 1 if cat in clf_candidates else None
            c_score = clf_scores.get(cat, 0.0)
            if c_rank is not None:
                sources.append("classifier")

            m_rank = minilm_cats.index(cat) + 1 if cat in minilm_cats else None
            m_score = minilm_scores.get(cat, 0.0)
            if m_rank is not None:
                sources.append("minilm")

            candidate_records.append({
                "category": cat,
                "classifier_rank": c_rank,
                "classifier_score": c_score,
                "minilm_rank": m_rank,
                "minilm_score": m_score,
                "candidate_sources": sources,
                "both_sources": len(sources) == 2,
                "from_classifier": "classifier" in sources,
                "from_minilm": "minilm" in sources
            })

        return {
            "classifier_top_k": clf_candidates,
            "classifier_confidence_str": sem_conf_str,
            "minilm_top_k": minilm_cats,
            "candidate_pool": candidate_records,
            "pool_size": len(candidate_records),
            "pool_categories": [r["category"] for r in candidate_records]
        }
