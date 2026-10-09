"""
Refactored 9-Category Screenshot Classifier Engine
==================================================
Implements an explainable multi-signal classification engine for the refactored 9-category taxonomy:
1. Communication
2. Connectivity
3. Education
4. Entertainment
5. General
6. Payment
7. Shopping
8. Travel
9. Work & Career

Key Structural Changes:
- Coding is REMOVED as a top-level category.
- Coding topics contribute to Education (for learning/courses/tutorials) or Work & Career (for software projects/repos).
- Deterministic Coding Tags are extracted separately alongside category prediction.
- General is implemented as a true evidence-based fallback.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.intelligence.taxonomy_refactor.coding_tag_extractor import (
    extract_deterministic_tags,
)

from experiments.intelligence.ocr_evidence_semantic.ocr_evidence_semantic import (
    content_focused_ocr_text,
    get_ocr_text_for_filename,
    normalize_ocr_text,
)

from src.embedder import SentenceTransformerEmbedder

CANONICAL_9_CATEGORIES = [
    "Communication",
    "Connectivity",
    "Education",
    "Entertainment",
    "General",
    "Payment",
    "Shopping",
    "Travel",
    "Work & Career",
]

SPECIFIC_8_CATEGORIES = [c for c in CANONICAL_9_CATEGORIES if c != "General"]

PROTOTYPE_DESCRIPTIONS_9: Dict[str, str] = {
    "Communication": "Instant messaging, chat conversations, emails, inbox, social media messages, WhatsApp, Telegram, Gmail, SMS, unread messages.",
    "Connectivity": "Wi-Fi network settings, wireless status, SSID, IP address, MAC address, Bluetooth pairing, hotspot settings, network connectivity.",
    "Education": "Educational courses, study material, lecture notes, tutorials, university assignments, exams, quizzes, LeetCode algorithms, computer science learning.",
    "Entertainment": "Streaming video, music players, movies, YouTube, Spotify, video games, comics, entertainment media, soundtracks, anime.",
    "General": "Generic system screen, launcher wallpaper, clock, simple calculator, status display without specific category evidence.",
    "Payment": "Payment confirmation, UPI transaction receipt, money transferred, Google Pay, Paytm, PhonePe, bank invoice total, paid successfully.",
    "Shopping": "E-commerce product listing, online store catalog, add to cart, checkout page, product price, Amazon, Flipkart, Myntra, order summary.",
    "Travel": "Flight booking confirmation, boarding pass, train ticket, IRCTC, hotel reservation, travel itinerary, flight PNR, departure destination.",
    "Work & Career": "Professional work, job applications, recruitment, resume, LinkedIn, company meetings, software development workspace, VS Code projects, GitHub repositories, corporate documents.",
}

# -----------------------------------------------------------------------------
# 1. 9-Category Evidence Rules
# -----------------------------------------------------------------------------
HIGH_CONFIDENCE_KEYWORDS_9: Dict[str, List[str]] = {
    "Education": ["assignment", "syllabus", "quiz", "homework", "classroom", "coursera", "udemy", "kiit", "mid-semester", "leetcode", "kaggle", "vocabulary", "idioms", "phrasal verbs", "mathematics"],
    "Shopping": ["add to cart", "checkout", "wishlist", "buy now", "amazon", "flipkart", "netmeds", "myntra", "souled store", "dress", "clothing"],
    "Payment": ["gpay", "phonepe", "paytm", "upi id", "ref no", "paid to", "debited", "credited", "transaction successful", "payment successful", "invoice total"],
    "Travel": ["boarding pass", "pnr", "flight number", "irctc", "indigo", "hotel booking", "travel itinerary", "departure", "destination", "railway"],
    "Communication": ["whatsapp", "telegram", "inbox", "unread", "subject:", "dear ", "thanks,", "regards,", "reply to"],
    "Entertainment": ["spotify", "youtube", "deviantart", "marvel", "superhero", "soundtrack", "episode", "season", "anime", "word game"],
    "Work & Career": ["vs code", "vscode", "github", "gitlab", "linkedin", "resume", "applicant", "recruitment", "interview", "salary", "internship", "deloitte", "zycus", "campus drive", "placement", "mcp-rag", "stock-predictor"],
    "Connectivity": ["wi-fi", "ssid", "ip address", "mac address", "bluetooth", "wpa", "hotspot", "ethernet", "show password"],
}

MEDIUM_CONFIDENCE_KEYWORDS_9: Dict[str, List[str]] = {
    "Education": ["lecture", "course", "university", "notes", "exam", "study", "student", "lesson", "algorithm", "learn", "class", "school", "degree", "chapter", "tutorial", "practice", "lab"],
    "Shopping": ["cart", "product", "price", "buy", "order", "shipping", "item", "store", "size", "rupees", "₹", "deal", "discount", "shoes", "fashion", "seller", "delivery"],
    "Payment": ["paid", "payment", "upi", "transaction", "successful", "receipt", "transfer", "bank", "amount", "billing", "invoice", "total paid", "account", "card"],
    "Travel": ["flight", "booking", "airline", "hotel", "train", "seat", "itinerary", "trip", "fare", "gate", "terminal", "passenger", "ticket"],
    "Communication": ["chat", "message", "email", "sent", "reply", "conversation", "notification", "mail", "sender", "recipient"],
    "Entertainment": ["music", "video", "movie", "play", "song", "game", "artist", "reel", "hero", "media", "streaming"],
    "Work & Career": ["job", "career", "hiring", "consultant", "role", "offer", "candidate", "position", "entry pass", "repository", "git", "project", "codebase", "developer", "ide"],
    "Connectivity": ["wifi", "network", "signal", "connected", "discovering", "settings", "wireless", "device name"],
}

HIGH_CONFIDENCE_PHRASES_9: Dict[str, List[str]] = {
    "Education": ["online course", "practice questions", "lecture notes", "submit assignment", "study material", "leetcode problem"],
    "Shopping": ["add to cart", "buy now", "free shipping", "customer reviews", "product details", "order summary"],
    "Payment": ["payment successful", "transaction successful", "money transferred", "invoice total", "paid successfully"],
    "Travel": ["travel itinerary", "boarding pass", "flight booking", "check-in", "gate number"],
    "Communication": ["new message", "reply to", "send message", "inbox messages", "chat conversation"],
    "Entertainment": ["watch now", "play episode", "listen now", "now playing", "official soundtrack"],
    "Work & Career": ["job application", "meeting scheduled", "work experience", "pull request", "git commit", "source code", "vs code"],
    "Connectivity": ["connected to wi-fi", "network settings", "bluetooth devices", "available networks", "hotspot settings"],
}

CONTEXTUAL_COMBINATIONS_9: List[Dict[str, Any]] = [
    # Programming + Education
    {"words": ["python", "tutorial"], "category": "Education", "weight": 5.0},
    {"words": ["python", "course"], "category": "Education", "weight": 5.0},
    {"words": ["python", "lecture"], "category": "Education", "weight": 5.0},
    {"words": ["learn", "python"], "category": "Education", "weight": 5.0},
    {"words": ["leetcode", "python"], "category": "Education", "weight": 5.0},
    {"words": ["kaggle", "notebook"], "category": "Education", "weight": 5.0},
    {"words": ["ailab", "assignment"], "category": "Education", "weight": 5.0},

    # Programming + Work & Career
    {"words": ["python", "vs code"], "category": "Work & Career", "weight": 5.0},
    {"words": ["git", "commit"], "category": "Work & Career", "weight": 5.0},
    {"words": ["github", "repository"], "category": "Work & Career", "weight": 5.0},
    {"words": ["pull", "request"], "category": "Work & Career", "weight": 5.0},
    {"words": ["mcp", "project"], "category": "Work & Career", "weight": 5.0},
]


def extract_evidence_9(text: str) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """Extracts raw category evidence scores for the 9 categories."""
    text_lower = text.lower()
    scores = {c: 0.0 for c in CANONICAL_9_CATEGORIES}
    matches = {c: [] for c in CANONICAL_9_CATEGORIES}

    # High Confidence Keywords
    for cat, kw_list in HIGH_CONFIDENCE_KEYWORDS_9.items():
        for kw in kw_list:
            if kw in text_lower:
                scores[cat] += 4.0
                matches[cat].append(f"High-KW: '{kw}'")

    # Medium Confidence Keywords
    for cat, kw_list in MEDIUM_CONFIDENCE_KEYWORDS_9.items():
        for kw in kw_list:
            if kw in text_lower:
                scores[cat] += 1.5
                matches[cat].append(f"Med-KW: '{kw}'")

    # High Confidence Phrases
    for cat, phrase_list in HIGH_CONFIDENCE_PHRASES_9.items():
        for phrase in phrase_list:
            if phrase in text_lower:
                scores[cat] += 3.5
                matches[cat].append(f"Phrase: '{phrase}'")

    # Contextual Combinations
    for rule in CONTEXTUAL_COMBINATIONS_9:
        words = rule["words"]
        cat = rule["category"]
        w = rule["weight"]
        if all(w_item in text_lower for w_item in words):
            scores[cat] += w
            matches[cat].append(f"Context Combo: {'+'.join(words)}")

    return scores, matches


# -----------------------------------------------------------------------------
# 2. Refactored 9-Category Classifier
# -----------------------------------------------------------------------------
class TaxonomyRefactorClassifier:
    """Explainable Multi-Signal Classifier for the 9-Category Taxonomy."""

    def __init__(
        self,
        categories: List[str] = CANONICAL_9_CATEGORIES,
        weights: Tuple[float, float, float] = (0.45, 0.40, 0.15),
        mode: str = "general_fallback_tags",
        embedder: SentenceTransformerEmbedder = None,
    ):
        self.categories = list(categories)
        self.w_ev, self.w_sem, self.w_ct = weights
        self.mode = mode
        self.embedder = embedder if embedder is not None else SentenceTransformerEmbedder("all-MiniLM-L6-v2")

        # Prototype Embeddings for 9 Categories
        proto_texts = [PROTOTYPE_DESCRIPTIONS_9.get(c, c) for c in self.categories]
        proto_vecs = self.embedder.embed(proto_texts)
        self.proto_vecs = np.array(proto_vecs, dtype=float)
        norms = np.linalg.norm(self.proto_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.proto_vecs = self.proto_vecs / norms

    def predict_single_explainable(self, raw_text: str, notes: str = "") -> Dict[str, Any]:
        norm_text = normalize_ocr_text(raw_text)
        content_text = content_focused_ocr_text(raw_text)

        tokens = norm_text.split()
        token_count = len(tokens)
        char_count = len(norm_text)
        is_sparse = (token_count < 5 or char_count < 20)

        # 1. Evidence Extraction (9 Categories)
        ev_scores_raw, ev_matches = extract_evidence_9(norm_text)

        # Content Type Scores (Adapted to 9 Categories)
        ct_scores_raw = {c: 0.0 for c in self.categories}
        if any(pat in norm_text.lower() for pat in ["upi", "gpay", "phonepe", "transaction successful", "payment successful"]):
            ct_scores_raw["Payment"] += 3.0
        if any(pat in norm_text.lower() for pat in ["add to cart", "buy now", "free shipping", "checkout"]):
            ct_scores_raw["Shopping"] += 3.0
        if any(pat in norm_text.lower() for pat in ["boarding pass", "pnr", "flight", "irctc", "itinerary"]):
            ct_scores_raw["Travel"] += 3.0
        if any(pat in norm_text.lower() for pat in ["show password", "ip address", "mac address", "ssid"]):
            ct_scores_raw["Connectivity"] += 3.0

        # 2. MiniLM Semantic Similarity (9 Categories)
        if not content_text.strip():
            sem_sims = {c: 0.0 for c in self.categories}
        else:
            emb = np.array(self.embedder.embed([content_text])[0], dtype=float)
            norm = np.linalg.norm(emb)
            emb_norm = (emb / norm) if norm > 0 else emb
            sims = np.dot(self.proto_vecs, emb_norm)
            sem_sims = {c: float(sims[i]) for i, c in enumerate(self.categories)}

        # 3. Deterministic Coding Tag Extraction
        extracted_tags = extract_deterministic_tags(norm_text, notes)

        # 4. Score Scaling & Fusion
        max_ev = max(ev_scores_raw.values()) if max(ev_scores_raw.values()) > 0 else 1.0
        max_ct = max(ct_scores_raw.values()) if max(ct_scores_raw.values()) > 0 else 1.0

        ev_scaled = {c: ev_scores_raw[c] / max_ev for c in self.categories}
        ct_scaled = {c: ct_scores_raw[c] / max_ct for c in self.categories}

        combined_scores = {}
        for c in self.categories:
            comb = (self.w_ev * ev_scaled[c]) + (self.w_sem * max(0.0, sem_sims[c])) + (self.w_ct * ct_scaled[c])
            combined_scores[c] = float(comb)

        # 5. General True Fallback Architecture
        spec_ev_max = max([ev_scores_raw[c] for c in SPECIFIC_8_CATEGORIES])
        spec_sem_max = max([sem_sims[c] for c in SPECIFIC_8_CATEGORIES])

        if self.mode in ["explicit_general_fallback", "general_fallback_tags"]:
            # General is assigned ONLY when specific category evidence is insufficient
            if spec_ev_max == 0 and spec_sem_max < 0.38:
                combined_scores["General"] += 0.45
            elif spec_ev_max < 1.2 and spec_sem_max < 0.30:
                combined_scores["General"] += 0.35

        # Rank Top Candidates
        sorted_cats = sorted(self.categories, key=lambda c: combined_scores[c], reverse=True)
        top1_cat = sorted_cats[0]
        top2_cat = sorted_cats[1] if len(sorted_cats) > 1 else sorted_cats[0]
        top3_cat = sorted_cats[2] if len(sorted_cats) > 2 else sorted_cats[0]

        top1_score = combined_scores[top1_cat]
        top2_score = combined_scores[top2_cat]
        margin = top1_score - top2_score
        is_ambiguous = (margin < 0.10)

        if top1_score > 0.55 and margin >= 0.15 and not is_sparse:
            confidence = "High"
        elif top1_score > 0.35 and margin >= 0.08:
            confidence = "Medium"
        else:
            confidence = "Low"

        return {
            "prediction": top1_cat,
            "top_candidates": [
                {"category": top1_cat, "score": round(top1_score, 4)},
                {"category": top2_cat, "score": round(top2_score, 4)},
                {"category": top3_cat, "score": round(combined_scores[top3_cat], 4)},
            ],
            "category_scores": {c: round(combined_scores[c], 4) for c in self.categories},
            "tags": extracted_tags,
            "confidence": confidence,
            "is_ambiguous": is_ambiguous,
            "is_sparse": is_sparse,
            "margin": round(margin, 4),
            "evidence_matches": ev_matches.get(top1_cat, []),
            "semantic_sim": round(sem_sims.get(top1_cat, 0.0), 4),
        }
