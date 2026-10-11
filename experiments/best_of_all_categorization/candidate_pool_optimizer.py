"""
Phase 4: Candidate Pool Generator Optimizer
===========================================
Evaluates controlled candidate generation options strictly on internal validation
data (metadata_v2.csv lines 250-375) to identify the strongest candidate generator.

Measures:
- Candidate Recall@1, Recall@3, Recall@5
- Total Candidate Pool Coverage (%)
- Average Candidate Pool Size
- Category-level Candidate Coverage
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "best_of_all_categorization"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "candidate_reranking"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "accuracy_improvement"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "categorization_recovery"))

from src.embedder import SentenceTransformerEmbedder
from exp_recovery_candidate import RecoverySemanticCandidate
from expanded_candidate_pool import CANONICAL_CATEGORIES
from expanded_candidate_generator_v2 import REFINED_CATEGORY_PROTOTYPES, normalize_vector
from experiments.intelligence.ocr_evidence_semantic.ocr_evidence_semantic import load_cached_ocr_texts

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "best_of_all_categorization"


class HybridCandidateGenerator:
    """Optimized Candidate Generator supporting dataset centroids, description vectors, and hybrid prototypes."""

    def __init__(self, embedder: Optional[SentenceTransformerEmbedder] = None, base_classifier: Optional[RecoverySemanticCandidate] = None):
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.base_classifier = base_classifier or RecoverySemanticCandidate()
        
        meta_df = pd.read_csv(PROJECT_ROOT / "data" / "metadata_v2.csv")
        train_df = meta_df.iloc[0:250]
        ocr_cache = load_cached_ocr_texts()
        
        # 1. Dataset Centroids (Strategy C)
        self.centroid_prototypes: Dict[str, np.ndarray] = {}
        for cat in CANONICAL_CATEGORIES:
            cat_rows = train_df[train_df["category"] == cat]
            vecs = []
            for _, r in cat_rows.iterrows():
                fn = str(r["file"])
                txt = ocr_cache.get(fn, str(r.get("notes", "")))
                if txt.strip():
                    v = self.embedder.embed([txt])[0]
                    vecs.append(normalize_vector(np.array(v, dtype=np.float32)))
            if vecs:
                mean_vec = np.mean(vecs, axis=0)
                self.centroid_prototypes[cat] = normalize_vector(mean_vec)
            else:
                self.centroid_prototypes[cat] = np.zeros(384, dtype=np.float32)

        # 2. Refined Description Prototypes (Strategy C+)
        self.desc_prototypes: Dict[str, np.ndarray] = {}
        for cat in CANONICAL_CATEGORIES:
            desc_text = REFINED_CATEGORY_PROTOTYPES.get(cat, cat)
            vec = self.embedder.embed([desc_text])[0]
            self.desc_prototypes[cat] = normalize_vector(np.array(vec, dtype=np.float32))

        # 3. Hybrid Prototypes (Average of Centroid + Description)
        self.hybrid_prototypes: Dict[str, np.ndarray] = {}
        for cat in CANONICAL_CATEGORIES:
            h_vec = (self.centroid_prototypes[cat] * 0.5) + (self.desc_prototypes[cat] * 0.5)
            self.hybrid_prototypes[cat] = normalize_vector(h_vec)

    def predict_minilm_top_k(self, query_text: str, prototype_mode: str = "hybrid", top_k: int = 5) -> List[Tuple[str, float]]:
        if not query_text.strip():
            query_text = "general screenshot mobile phone app display"

        q_vec = normalize_vector(np.array(self.embedder.embed([query_text])[0], dtype=np.float32))
        
        if prototype_mode == "centroid":
            proto_dict = self.centroid_prototypes
        elif prototype_mode == "description":
            proto_dict = self.desc_prototypes
        else:
            proto_dict = self.hybrid_prototypes

        scores = []
        for cat in CANONICAL_CATEGORIES:
            p_vec = proto_dict[cat]
            sim = float(np.dot(q_vec, p_vec))
            scores.append((cat, round(sim, 4)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def build_candidate_pool(
        self,
        raw_text: str,
        notes: str = "",
        top_k_clf: int = 5,
        top_k_minilm: int = 5,
        prototype_mode: str = "hybrid"
    ) -> Dict[str, Any]:
        sem_res = self.base_classifier.predict_final(raw_text=raw_text, notes=notes, variant="C")
        sem_pred = sem_res.get("prediction", "General")
        sem_conf_str = sem_res.get("confidence", "Low")
        
        raw_cands = [c["category"] for c in sem_res.get("top_candidates", [])]
        clf_candidates = [sem_pred]
        for c in raw_cands:
            if c not in clf_candidates and len(clf_candidates) < top_k_clf:
                clf_candidates.append(c)

        top1_score = 0.90 if sem_conf_str == "High" else (0.65 if sem_conf_str == "Medium" else 0.45)
        clf_scores = {}
        for rank_idx, cat in enumerate(clf_candidates):
            score_factor = 1.0 if rank_idx == 0 else (0.70 if rank_idx == 1 else (0.50 if rank_idx == 2 else (0.35 if rank_idx == 3 else 0.25)))
            clf_scores[cat] = round(top1_score * score_factor, 4)

        minilm_ranked = self.predict_minilm_top_k(raw_text, prototype_mode=prototype_mode, top_k=top_k_minilm)
        minilm_cats = [item[0] for item in minilm_ranked]
        minilm_scores = {item[0]: item[1] for item in minilm_ranked}

        unique_categories = list(clf_candidates)
        for cat in minilm_cats:
            if cat not in unique_categories:
                unique_categories.append(cat)

        candidate_records = []
        for cat in unique_categories:
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
                "both_sources": ("classifier" in sources and "minilm" in sources),
                "from_classifier": "classifier" in sources,
                "from_minilm": "minilm" in sources
            })

        return {
            "classifier_top_k": clf_candidates,
            "classifier_confidence_str": sem_conf_str,
            "minilm_top_k": minilm_cats,
            "candidate_pool": candidate_records,
            "pool_size": len(candidate_records)
        }


def evaluate_generator_config(
    df_val: pd.DataFrame,
    generator: HybridCandidateGenerator,
    ocr_cache: Dict[str, str],
    top_k_clf: int,
    top_k_minilm: int,
    prototype_mode: str
) -> Dict[str, Any]:
    n = len(df_val)
    r1_hits, r3_hits, r5_hits, pool_hits = 0, 0, 0, 0
    pool_sizes = []

    for _, row in df_val.iterrows():
        gt_cat = str(row["category"])
        fn = str(row["file"])
        notes = str(row.get("notes", "")) if pd.notna(row.get("notes")) else ""
        raw_text = ocr_cache.get(fn, ocr_cache.get(fn.split(".")[0], notes))

        res = generator.build_candidate_pool(
            raw_text=raw_text,
            notes=notes,
            top_k_clf=top_k_clf,
            top_k_minilm=top_k_minilm,
            prototype_mode=prototype_mode
        )
        pool = res["candidate_pool"]
        pool_cats = [c["category"] for c in pool]
        pool_sizes.append(len(pool))

        if len(pool_cats) >= 1 and pool_cats[0] == gt_cat:
            r1_hits += 1
        if gt_cat in pool_cats[:3]:
            r3_hits += 1
        if gt_cat in pool_cats[:5]:
            r5_hits += 1
        if gt_cat in pool_cats:
            pool_hits += 1

    return {
        "top_k_clf": top_k_clf,
        "top_k_minilm": top_k_minilm,
        "prototype_mode": prototype_mode,
        "recall_1": round(r1_hits / n, 4),
        "recall_3": round(r3_hits / n, 4),
        "recall_5": round(r5_hits / n, 4),
        "coverage": round(pool_hits / n, 4),
        "avg_pool_size": round(float(np.mean(pool_sizes)), 2)
    }


def main():
    print("=" * 80)
    print("PHASE 4: CANDIDATE POOL GENERATOR OPTIMIZATION")
    print("=" * 80)

    embedder = SentenceTransformerEmbedder()
    generator = HybridCandidateGenerator(embedder=embedder)

    meta_df = pd.read_csv(PROJECT_ROOT / "data" / "metadata_v2.csv")
    val_df = meta_df.iloc[250:375].copy()
    ocr_cache = load_cached_ocr_texts()

    configs = [
        (3, 5, "centroid"),
        (3, 5, "description"),
        (3, 5, "hybrid"),
        (5, 5, "centroid"),
        (5, 5, "description"),
        (5, 5, "hybrid"),
        (5, 3, "hybrid")
    ]

    results = []
    for top_k_clf, top_k_m, p_mode in configs:
        res = evaluate_generator_config(val_df, generator, ocr_cache, top_k_clf, top_k_m, p_mode)
        results.append(res)
        print(f"Clf K={top_k_clf} | MiniLM K={top_k_m} | Proto={p_mode:11s} -> Coverage = {res['coverage']*100:.2f}%, Recall@1 = {res['recall_1']*100:.2f}%, Recall@3 = {res['recall_3']*100:.2f}%, Avg Pool Size = {res['avg_pool_size']}")

    (OUTPUT_DIR / "candidate_pool_optimization.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved candidate_pool_optimization.json successfully.")


if __name__ == "__main__":
    main()
