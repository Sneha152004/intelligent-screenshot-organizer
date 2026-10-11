"""
Expanded Candidate Pool Generator & Provenance Manager
======================================================
Builds an expanded candidate pool combining:
1. Base Semantic Classifier Top-3 candidates + confidence scores
2. Existing MiniLM Strategy C Top-5 candidates + similarity scores

Preserves provenance metadata:
- category
- classifier_rank
- classifier_score
- minilm_rank
- minilm_score
- candidate_sources (classifier, minilm, or both)
- ocr_evidence
- visual_evidence
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
from minilm_candidate_experiment import MiniLMCandidateGenerator

CANONICAL_CATEGORIES = [
    "Coding", "Communication", "Connectivity", "Education",
    "Entertainment", "General", "Payment", "Shopping", "Travel", "Work & Career"
]


class ExpandedCandidatePoolBuilder:
    """Builds unified candidate pools from Semantic Classifier Top-3 + MiniLM Top-5."""

    def __init__(self, embedder: Optional[SentenceTransformerEmbedder] = None, base_classifier: Optional[RecoverySemanticCandidate] = None):
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.base_classifier = base_classifier or RecoverySemanticCandidate()
        
        # Fit MiniLM prototypes on internal training data
        meta_df = pd.read_csv(PROJECT_ROOT / "data" / "metadata_v2.csv")
        train_df = meta_df.iloc[0:250]
        
        from experiments.intelligence.ocr_evidence_semantic.ocr_evidence_semantic import load_cached_ocr_texts
        ocr_cache = load_cached_ocr_texts()
        
        self.minilm_generator = MiniLMCandidateGenerator(self.embedder)
        self.minilm_generator.fit_prototypes(train_df, ocr_cache)

    def build_candidate_pool(
        self,
        raw_text: str,
        notes: str = "",
        vis_screen_type: str = "other_screen",
        vis_conf: float = 0.0
    ) -> Dict[str, Any]:
        """
        Generates expanded candidate pool for a screenshot.
        """
        # 1. Candidate Source A: Semantic Classifier Top-3
        sem_res = self.base_classifier.predict_final(raw_text=raw_text, notes=notes, variant="C")
        sem_pred = sem_res.get("prediction", "General")
        sem_conf_str = sem_res.get("confidence", "Low")
        
        raw_cands = [c["category"] for c in sem_res.get("top_candidates", [])]
        clf_top3 = [sem_pred]
        for c in raw_cands:
            if c not in clf_top3 and len(clf_top3) < 3:
                clf_top3.append(c)

        top1_score = 0.90 if sem_conf_str == "High" else (0.65 if sem_conf_str == "Medium" else 0.45)
        clf_scores = {clf_top3[0]: top1_score}
        if len(clf_top3) > 1:
            clf_scores[clf_top3[1]] = round(top1_score * 0.70, 4)
        if len(clf_top3) > 2:
            clf_scores[clf_top3[2]] = round(top1_score * 0.45, 4)

        # 2. Candidate Source B: MiniLM Strategy C Top-5
        minilm_ranked = self.minilm_generator.predict_similarity(raw_text, strategy="C")
        minilm_top5_cats = [item[0] for item in minilm_ranked[:5]]
        minilm_scores = {item[0]: item[1] for item in minilm_ranked}

        # 3. Unified Candidate Pool Generation
        unique_categories = list(clf_top3)
        for cat in minilm_top5_cats:
            if cat not in unique_categories:
                unique_categories.append(cat)

        candidate_records = []
        for cat in unique_categories:
            sources = []
            c_rank = clf_top3.index(cat) + 1 if cat in clf_top3 else None
            c_score = clf_scores.get(cat, 0.0)
            if c_rank is not None:
                sources.append("classifier")

            m_rank = minilm_cats_rank = minilm_top5_cats.index(cat) + 1 if cat in minilm_top5_cats else None
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
                "from_classifier": "classifier" in sources,
                "from_minilm": "minilm" in sources
            })

        return {
            "classifier_top3": clf_top3,
            "classifier_confidence_str": sem_conf_str,
            "minilm_top5": minilm_top5_cats,
            "candidate_pool": candidate_records,
            "pool_size": len(candidate_records),
            "pool_categories": [r["category"] for r in candidate_records]
        }
