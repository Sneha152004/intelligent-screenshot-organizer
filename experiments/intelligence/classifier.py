"""
Multi-Signal Hierarchical Screenshot Classifier
================================================
Combines rule/evidence signals, semantic text similarity, prototype vector embeddings,
and content-type taxonomy affinity to classify screenshot content into top-k candidates,
calculate confidence, detect ambiguity, and provide full explainability.

Target Flow:
Screenshot -> OCR -> Evidence extraction -> Content type ->
[Rule/evidence | Semantic | Embedding | Content-type] -> Score fusion ->
Top-k candidates -> Confidence + ambiguity -> Category
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.embedder import BaseEmbedder, SentenceTransformerEmbedder
from src.models.metadata import CATEGORY_CONTENT_TYPES
from experiments.intelligence.rules import (
    extract_financial_info,
    extract_contacts,
    extract_urls,
    detect_sensitive_info,
)

# 1. Prototype Anchor Strings for Embedding Similarity
CATEGORY_PROTOTYPES: Dict[str, str] = {
    "Connectivity": "wifi network password ssid router internet connection security key configuration settings airtel",
    "Payment": "upi payment receipt transaction successful paid amount bank reference transfer invoice billing account",
    "Education": "course syllabus curriculum textbook exam lecture assignment notes university college academic study learning",
    "Shopping": "shopping menu order item price food restaurant cart product deal purchase list online store ecommerce",
    "Travel": "flight boarding pass airline ticket seat gate departure arrival journey travel booking pnr hotel itinerary",
    "Communication": "chat message whatsapp telegram conversation reply inbox notification text talk email",
    "Coding": "source code programming function variable controller express backend code snippet github script typescript python javascript",
    "Work & Career": "hackathon event registration job career work company project announcement workshop team prize interview resume",
    "Entertainment": "movie video stream music show song watch playlist game play cinema podcast media player",
    "General": "general document screenshot image note text webpage dashboard screen summary",
}

# 2. Centralized Semantic Descriptions for Semantic Similarity Matching
SEMANTIC_DESCRIPTIONS: Dict[str, str] = {
    "Education": "Academic course syllabus, educational materials, lecture notes, textbook chapters, university study, exam schedules, and learning resources.",
    "Payment": "Financial payment receipts, UPI money transfers, transaction confirmations, bank reference numbers, paid invoices, and account balances.",
    "Shopping": "E-commerce product pages, online shopping cart, food menu listings, order receipts, restaurant items, and price displays.",
    "Travel": "Flight tickets, boarding passes, airline booking confirmations, gate departure information, hotel itineraries, and travel passes.",
    "Coding": "Software source code, code editor windows, terminal outputs, API controllers, stack traces, programming language documentation, and developer tools.",
    "Communication": "Messaging chat threads, WhatsApp conversations, email threads, social media messages, and contact notifications.",
    "Work & Career": "Job postings, recruitment drive announcements, hackathon event registrations, resume documents, career opportunities, and workplace updates.",
    "Entertainment": "Movie trailers, music streaming playlists, video media players, entertainment shows, cinema bookings, and gaming content.",
    "Connectivity": "Wi-Fi network access credentials, router passwords, network configuration settings, mobile hotspot details, and internet connectivity setup.",
    "General": "Unclassified screenshots, personal notes, generic web browsing pages, system settings, and document summaries.",
}

# 3. Rule Keyword Dictionary for keyword density scoring
RULE_KEYWORDS: Dict[str, List[str]] = {
    "Connectivity": ["wifi", "password", "ssid", "airtel", "network", "router", "wpa2", "wpa3", "connect"],
    "Payment": ["paid", "payment", "upi", "transaction", "receipt", "gpay", "paytm", "phonepe", "successful", "ref no", "bank"],
    "Education": ["syllabus", "course", "kiit", "deep learning", "textbook", "prerequisite", "credits", "evaluation", "lecture", "unit"],
    "Shopping": ["combo", "menu", "rs", "price", "serves", "food", "order", "item", "cart", "dish", "delivery"],
    "Travel": ["indigo", "boarding", "gate", "seat", "flight", "pnr", "departure", "guwahati", "airport", "terminal", "passenger"],
    "Communication": ["whatsapp", "messages", "chat", "unread", "telegram", "typing", "online", "received", "sender"],
    "Coding": ["const", "express", "require", "module.exports", "function", "controller", "await", "async", "router", "import", "class", "return"],
    "Work & Career": ["hackathon", "openai", "register", "event", "deadline", "prize", "careers", "hiring", "applicant", "submit"],
    "Entertainment": ["movie", "trailer", "spotify", "netflix", "youtube", "music", "episode", "playlist", "stream"],
}

# Default Score Fusion Weights (must sum to 1.0)
DEFAULT_SIGNAL_WEIGHTS: Dict[str, float] = {
    "rule_weight": 0.35,
    "semantic_weight": 0.30,
    "embedding_weight": 0.15,
    "content_type_weight": 0.20,
}


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two numeric vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    arr1 = np.array(v1, dtype=float)
    arr2 = np.array(v2, dtype=float)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(arr1, arr2) / (norm1 * norm2))


class HybridScreenshotClassifier:
    """
    Multi-Signal Hierarchical Screenshot Classifier.
    Integrates Rule/Evidence, Semantic Description Similarity, Prototype Vector Embedding,
    and Content-Type Taxonomy Affinity into a score fusion engine with Top-K candidate ranking,
    confidence calculation, and ambiguity detection.
    """

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        weights: Optional[Dict[str, float]] = None,
        ambiguity_threshold: float = 0.10,
    ):
        """
        Initialize classifier with an optional embedder, fusion weights, and ambiguity threshold.
        """
        self.embedder = embedder or SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        self.weights = dict(DEFAULT_SIGNAL_WEIGHTS)
        if weights:
            self.weights.update(weights)

        self.ambiguity_threshold = ambiguity_threshold
        self._prototype_embeddings: Optional[Dict[str, List[float]]] = None
        self._semantic_embeddings: Optional[Dict[str, List[float]]] = None

    def _get_prototype_embeddings(self) -> Dict[str, List[float]]:
        """Lazily generates and caches embedding vectors for category prototype texts."""
        if self._prototype_embeddings is None:
            categories = list(CATEGORY_PROTOTYPES.keys())
            texts = [CATEGORY_PROTOTYPES[c] for c in categories]
            embeddings = self.embedder.embed(texts)
            self._prototype_embeddings = {c: embeddings[i] for i, c in enumerate(categories)}
        return self._prototype_embeddings

    def _get_semantic_embeddings(self) -> Dict[str, List[float]]:
        """Lazily generates and caches embedding vectors for category semantic descriptions."""
        if self._semantic_embeddings is None:
            categories = list(SEMANTIC_DESCRIPTIONS.keys())
            texts = [SEMANTIC_DESCRIPTIONS[c] for c in categories]
            embeddings = self.embedder.embed(texts)
            self._semantic_embeddings = {c: embeddings[i] for i, c in enumerate(categories)}
        return self._semantic_embeddings

    def compute_rule_scores(
        self, ocr_text: str, evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculates normalized rule-based keyword density and structural pattern scores for each category.
        """
        rule_scores: Dict[str, float] = {cat: 0.0 for cat in CATEGORY_PROTOTYPES}
        if not ocr_text or not ocr_text.strip():
            return rule_scores

        ocr_lower = ocr_text.lower()

        # 1. Keyword density match
        for cat, keywords in RULE_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in ocr_lower)
            if matches > 0:
                rule_scores[cat] += min(1.0, (matches * 2.0) / len(keywords))

        # 2. Structural pattern boosts
        fin = evidence.get("financial_info") if evidence else extract_financial_info(ocr_text)
        if fin and (fin.get("has_payment_keywords") or len(fin.get("amounts", [])) > 0 or len(fin.get("upi_ids", [])) > 0):
            rule_scores["Payment"] += 0.6

        if "password" in ocr_lower and ("wifi" in ocr_lower or "airtel" in ocr_lower or "ssid" in ocr_lower or "wi-fi" in ocr_lower):
            rule_scores["Connectivity"] += 0.7

        if "syllabus" in ocr_lower or ("course" in ocr_lower and "learning" in ocr_lower):
            rule_scores["Education"] += 0.7

        if "boarding" in ocr_lower or ("indigo" in ocr_lower and "gate" in ocr_lower):
            rule_scores["Travel"] += 0.7

        if "express" in ocr_lower and ("controller" in ocr_lower or "const" in ocr_lower or "require" in ocr_lower):
            rule_scores["Coding"] += 0.7

        if "hackathon" in ocr_lower or ("openai" in ocr_lower and "register" in ocr_lower):
            rule_scores["Work & Career"] += 0.7

        if "whatsapp" in ocr_lower or "unread messages" in ocr_lower:
            rule_scores["Communication"] += 0.7

        if ("menu" in ocr_lower or "combo" in ocr_lower or "cart" in ocr_lower) and ("rs." in ocr_lower or "price" in ocr_lower or "order" in ocr_lower):
            rule_scores["Shopping"] += 0.7

        # Normalize rule scores to [0.0, 1.0]
        max_score = max(rule_scores.values()) if rule_scores else 0.0
        if max_score > 0:
            for cat in rule_scores:
                rule_scores[cat] = min(1.0, rule_scores[cat] / max_score)

        return rule_scores

    def compute_semantic_scores(self, ocr_text: str) -> Dict[str, float]:
        """Calculates vector embedding cosine similarity against semantic category descriptions."""
        if not ocr_text or not ocr_text.strip():
            return {cat: 0.0 for cat in CATEGORY_PROTOTYPES}

        text_vec = self.embedder.embed([ocr_text])[0]
        sem_vecs = self._get_semantic_embeddings()

        sem_scores: Dict[str, float] = {}
        for cat, sem_vec in sem_vecs.items():
            sim = cosine_similarity(text_vec, sem_vec)
            sem_scores[cat] = max(0.0, float(sim))

        # Normalize semantic scores
        max_s = max(sem_scores.values()) if sem_scores else 0.0
        if max_s > 0:
            for cat in sem_scores:
                sem_scores[cat] = min(1.0, sem_scores[cat] / max_s)

        return sem_scores

    def compute_embedding_scores(self, ocr_text: str) -> Dict[str, float]:
        """Calculates vector embedding cosine similarity scores against category prototype anchors."""
        if not ocr_text or not ocr_text.strip():
            return {cat: 0.0 for cat in CATEGORY_PROTOTYPES}

        text_vec = self.embedder.embed([ocr_text])[0]
        proto_vecs = self._get_prototype_embeddings()

        emb_scores: Dict[str, float] = {}
        for cat, proto_vec in proto_vecs.items():
            sim = cosine_similarity(text_vec, proto_vec)
            emb_scores[cat] = max(0.0, float(sim))

        return emb_scores

    def compute_content_type_scores(self, content_type: Optional[str]) -> Dict[str, float]:
        """
        Calculates content-type hierarchy score for each category.
        Returns 1.0 if content_type is present in category's taxonomy, 0.0 otherwise.
        """
        ct_scores: Dict[str, float] = {cat: 0.0 for cat in CATEGORY_PROTOTYPES}
        if not content_type or content_type == "unknown":
            return ct_scores

        for cat, valid_types in CATEGORY_CONTENT_TYPES.items():
            if content_type in valid_types:
                ct_scores[cat] = 1.0

        return ct_scores

    def classify_multi_signal(
        self,
        ocr_text: str,
        evidence: Optional[Dict[str, Any]] = None,
        content_type: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 3,
        ambiguity_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Performs multi-signal hierarchical score fusion, ranks top-k candidate categories,
        calculates system confidence, and evaluates classification ambiguity.

        Returns a structured result object with full explainability.
        """
        w = dict(self.weights)
        if weights:
            w.update(weights)

        thresh = ambiguity_threshold if ambiguity_threshold is not None else self.ambiguity_threshold

        if not ocr_text or len(ocr_text.strip()) < 3:
            default_candidates = [
                {
                    "category": cat,
                    "final_score": 0.0,
                    "rule_score": 0.0,
                    "semantic_score": 0.0,
                    "embedding_score": 0.0,
                    "content_type_score": 0.0,
                }
                for cat in CATEGORY_PROTOTYPES
            ]
            return {
                "selected_category": "General",
                "confidence": 0.0,
                "ambiguous": True,
                "score_margin": 0.0,
                "content_type": content_type or "unknown",
                "candidates": default_candidates[:top_k],
                "category_scores": {cat: 0.0 for cat in CATEGORY_PROTOTYPES},
                "signal_scores": {
                    "rule_scores": {cat: 0.0 for cat in CATEGORY_PROTOTYPES},
                    "semantic_scores": {cat: 0.0 for cat in CATEGORY_PROTOTYPES},
                    "embedding_scores": {cat: 0.0 for cat in CATEGORY_PROTOTYPES},
                    "content_type_scores": {cat: 0.0 for cat in CATEGORY_PROTOTYPES},
                },
                "evidence_summary": ["Empty OCR text"],
            }

        # 1. Compute Individual Signals
        rule_scores = self.compute_rule_scores(ocr_text, evidence=evidence)
        semantic_scores = self.compute_semantic_scores(ocr_text)
        embedding_scores = self.compute_embedding_scores(ocr_text)
        content_type_scores = self.compute_content_type_scores(content_type)

        # 2. Score Fusion
        w_rule = w.get("rule_weight", 0.35)
        w_sem = w.get("semantic_weight", 0.30)
        w_emb = w.get("embedding_weight", 0.15)
        w_ct = w.get("content_type_weight", 0.20)
        total_w = w_rule + w_sem + w_emb + w_ct

        hybrid_scores: Dict[str, float] = {}
        candidate_details: List[Dict[str, Any]] = []

        for cat in CATEGORY_PROTOTYPES:
            r_s = rule_scores.get(cat, 0.0)
            s_s = semantic_scores.get(cat, 0.0)
            e_s = embedding_scores.get(cat, 0.0)
            c_s = content_type_scores.get(cat, 0.0)

            f_score = (w_rule * r_s + w_sem * s_s + w_emb * e_s + w_ct * c_s) / total_w
            hybrid_scores[cat] = float(f_score)

            candidate_details.append({
                "category": cat,
                "final_score": round(float(f_score), 4),
                "rule_score": round(float(r_s), 4),
                "semantic_score": round(float(s_s), 4),
                "embedding_score": round(float(e_s), 4),
                "content_type_score": round(float(c_s), 4),
            })

        # Sort candidates descending by final_score
        candidate_details.sort(key=lambda x: x["final_score"], reverse=True)

        top1 = candidate_details[0]
        top2 = candidate_details[1] if len(candidate_details) > 1 else candidate_details[0]

        selected_category = top1["category"]
        score_margin = top1["final_score"] - top2["final_score"]

        # Ambiguity Evaluation
        is_ambiguous = (score_margin < thresh) or (
            top1["rule_score"] == 0.0 and top1["content_type_score"] == 0.0 and score_margin < (thresh * 1.5)
        )

        # Confidence Calculation taking signal agreement & margin into account
        signal_top_cats = [
            max(rule_scores, key=rule_scores.get) if max(rule_scores.values()) > 0 else "General",
            max(semantic_scores, key=semantic_scores.get) if max(semantic_scores.values()) > 0 else "General",
            max(embedding_scores, key=embedding_scores.get) if max(embedding_scores.values()) > 0 else "General",
        ]
        if content_type and content_type != "unknown":
            ct_top = max(content_type_scores, key=content_type_scores.get) if max(content_type_scores.values()) > 0 else "General"
            signal_top_cats.append(ct_top)

        agreement_count = sum(1 for c in signal_top_cats if c == selected_category)
        agreement_ratio = agreement_count / len(signal_top_cats)

        confidence = (top1["final_score"] * 0.5) + (score_margin * 0.3) + (agreement_ratio * 0.2)
        confidence = max(0.0, min(1.0, float(confidence)))

        # Fallback to General if confidence is extremely low
        if confidence < 0.12 and selected_category != "General":
            selected_category = "General"
            is_ambiguous = True

        return {
            "selected_category": selected_category,
            "confidence": round(confidence, 4),
            "ambiguous": is_ambiguous,
            "score_margin": round(score_margin, 4),
            "content_type": content_type or "unknown",
            "candidates": candidate_details[:top_k],
            "category_scores": hybrid_scores,
            "signal_scores": {
                "rule_scores": rule_scores,
                "semantic_scores": semantic_scores,
                "embedding_scores": embedding_scores,
                "content_type_scores": content_type_scores,
            },
            "evidence_summary": [
                f"Selected '{selected_category}' with margin {round(score_margin, 4)}",
                f"Signal Agreement: {agreement_count}/{len(signal_top_cats)}",
            ],
        }

    def classify(self, ocr_text: str, alpha: float = 0.5, beta: float = 0.5) -> Tuple[str, float, Dict[str, float]]:
        """
        Backward-compatible interface for hybrid classification.
        Delegates to multi-signal scoring while preserving contract signature.
        """
        res = self.classify_multi_signal(ocr_text)
        return res["selected_category"], res["confidence"], res["category_scores"]

    def infer_intent(self, category: str, ocr_text: str) -> str:
        """Infers a concise, actionable user intent string for a screenshot based on category & OCR text."""
        if not ocr_text or not ocr_text.strip():
            return "Store general screenshot note for future reference"

        ocr_lower = ocr_text.lower()

        if category == "Connectivity":
            return "Save Wi-Fi network credentials and password for local connection"
        elif category == "Payment":
            return "Verify successful UPI transaction receipt and payment status"
        elif category == "Education":
            return "Reference academic course syllabus and reading list"
        elif category == "Shopping":
            return "Browse menu item options and prices for ordering"
        elif category == "Travel":
            return "Track flight schedule and boarding pass details"
        elif category == "Communication":
            return "View chat notifications and unread messages"
        elif category == "Coding":
            return "Review backend API controller code implementation"
        elif category == "Work & Career":
            return "Check hackathon event details and registration deadline"
        elif category == "Entertainment":
            return "View media playback and entertainment options"

        return "Store general screenshot note for future reference"
