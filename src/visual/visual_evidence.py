"""
Visual Category Evidence Engine.
Uses TinyCLIP joint vision-language embeddings to calculate category similarity scores,
rank candidates, and generate explainable prompt-level evidence across 9 canonical categories.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch

from src.visual.base_visual_encoder import (
    BaseVisualEncoder,
    PreprocessingError,
    InferenceError,
)
from src.visual.tiny_clip_encoder import TinyCLIPEncoder, DEFAULT_TINYCLIP_CHECKPOINT

# Canonical 9 Categories (Coding remains a tag, not a top-level category)
CANONICAL_CATEGORIES = [
    "Education",
    "Shopping",
    "Payment",
    "Travel",
    "Communication",
    "Entertainment",
    "Work & Career",
    "Connectivity",
    "General",
]

# Descriptive multi-prompt definitions per category
DEFAULT_CATEGORY_PROMPTS: Dict[str, List[str]] = {
    "Education": [
        "a screenshot of an online course lecture, textbook page, or educational app",
        "a screenshot of study notes, slides, flashcards, or academic research",
        "a screenshot of an e-learning platform like Coursera, Udemy, or Khan Academy",
        "a screenshot of quiz questions, exam material, or educational diagrams",
    ],
    "Shopping": [
        "a screenshot of an e-commerce product catalog, shopping cart, or online store",
        "a screenshot of Amazon, Flipkart, or retail shopping items with prices",
        "a screenshot of product listings, customer reviews, or a wishlist",
        "a screenshot of an online clothing or electronics shopping app interface",
    ],
    "Payment": [
        "a screenshot of a digital payment confirmation, transaction receipt, or bank transfer",
        "a screenshot of a mobile wallet like Google Pay, Paytm, PhonePe, or PayPal",
        "a screenshot of an invoice, billing breakdown, or successful payment status screen",
        "a screenshot showing payment details, credit card checkout, or bank account balance",
    ],
    "Travel": [
        "a screenshot of a flight ticket, airline booking, or boarding pass",
        "a screenshot of a hotel reservation, Airbnb booking, or travel itinerary",
        "a screenshot of a train, bus, or cab booking app like Uber or Ola",
        "a screenshot of a map route, navigation directions, or travel destination guide",
    ],
    "Communication": [
        "a screenshot of a chat messaging app like WhatsApp, Telegram, or iMessage",
        "a screenshot of an email client inbox, Gmail, or email thread",
        "a screenshot of a social media comment thread, direct messages, or chat conversation",
        "a screenshot of a video call, voice call interface, or messaging list",
    ],
    "Entertainment": [
        "a screenshot of a video streaming app like YouTube, Netflix, or Prime Video",
        "a screenshot of a music player, Spotify playlist, or podcast app",
        "a screenshot of a mobile game, gaming leaderboard, or Twitch stream",
        "a screenshot of movie trailers, social media reels, or entertainment media content",
    ],
    "Work & Career": [
        "a screenshot of a professional dashboard, spreadsheet, project board, or Jira ticket",
        "a screenshot of LinkedIn profile, job posting, resume, or career portal",
        "a screenshot of a business document, Slack workplace channel, or presentation deck",
        "a screenshot of code editor, developer terminal, or technical documentation for work",
    ],
    "Connectivity": [
        "a screenshot of Wi-Fi settings, network connection status, or signal strength",
        "a screenshot of Bluetooth device pairing, hotspot settings, or VPN status",
        "a screenshot of mobile network settings, data usage, or connectivity status",
        "a screenshot of internet speed test, network configuration, or router settings",
    ],
    "General": [
        "a screenshot of a smartphone home screen, app launcher, or device wallpaper",
        "a screenshot of system settings, phone specs, clock, calculator, or system UI",
        "a screenshot of a generic web browser page, search engine, or miscellaneous site",
        "a screenshot of a file manager, device storage page, or generic system menu",
    ],
}


class VisualCategoryEvidenceEngine:
    """
    Visual Evidence Engine for Screenshot Classification.

    Computes zero-shot similarity between screenshot visual embeddings and
    pre-encoded text prompt embeddings across the 9 canonical categories.
    """

    def __init__(
        self,
        encoder: Optional[TinyCLIPEncoder] = None,
        category_prompts: Optional[Dict[str, List[str]]] = None,
        aggregation_method: str = "top_k_mean",
        top_k: int = 2,
    ):
        """
        Initializes the visual evidence engine.

        Args:
            encoder: Pre-configured TinyCLIPEncoder instance (or creates default lazily).
            category_prompts: Custom category prompt dictionary (defaults to DEFAULT_CATEGORY_PROMPTS).
            aggregation_method: Method for combining prompt similarities ('top_k_mean', 'mean', 'max').
            top_k: K value for 'top_k_mean' aggregation.
        """
        self.encoder = encoder or TinyCLIPEncoder(model_name=DEFAULT_TINYCLIP_CHECKPOINT, device="cpu")
        self.category_prompts = category_prompts or DEFAULT_CATEGORY_PROMPTS
        self.aggregation_method = aggregation_method
        self.top_k = top_k

        # Validate categories
        for cat in CANONICAL_CATEGORIES:
            if cat not in self.category_prompts:
                raise ValueError(f"Category prompts missing for canonical category: '{cat}'")

        # Caching structures for prompt text embeddings
        self._prompt_embeddings: Optional[Dict[str, np.ndarray]] = None
        self._flat_prompts: List[str] = []
        self._prompt_category_mapping: List[str] = []
        self._prompt_matrix: Optional[np.ndarray] = None
        self._is_initialized: bool = False

    def initialize_prompts(self) -> None:
        """
        Encodes all category descriptive text prompts into normalized 512-dim vectors.
        Caches the resultant matrix to avoid re-encoding text prompts during inference.
        """
        if self._is_initialized and self._prompt_matrix is not None:
            return

        if not self.encoder.is_loaded:
            self.encoder.load_model()

        flat_prompts = []
        prompt_category_mapping = []

        for category in CANONICAL_CATEGORIES:
            prompts = self.category_prompts[category]
            for p in prompts:
                flat_prompts.append(p)
                prompt_category_mapping.append(category)

        # Compute batch text embeddings using TinyCLIP text encoder
        matrix = self.encoder.get_text_embeddings(flat_prompts, normalize=True)  # (M, 512)

        self._flat_prompts = flat_prompts
        self._prompt_category_mapping = prompt_category_mapping
        self._prompt_matrix = matrix

        # Organize by category for quick indexing
        prompt_embeddings = {}
        idx = 0
        for category in CANONICAL_CATEGORIES:
            count = len(self.category_prompts[category])
            prompt_embeddings[category] = matrix[idx : idx + count]
            idx += count

        self._prompt_embeddings = prompt_embeddings
        self._is_initialized = True

    def _aggregate_similarities(self, similarities: np.ndarray) -> float:
        """Aggregates a 1D array of prompt similarities for a category into a single score."""
        if similarities.size == 0:
            return 0.0

        if self.aggregation_method == "max":
            return float(np.max(similarities))
        elif self.aggregation_method == "mean":
            return float(np.mean(similarities))
        elif self.aggregation_method == "top_k_mean":
            k = min(self.top_k, similarities.size)
            top_k_vals = np.partition(similarities, -k)[-k:]
            return float(np.mean(top_k_vals))
        else:
            raise ValueError(f"Unknown aggregation method: '{self.aggregation_method}'")

    def _softmax(self, x: np.ndarray, temperature: float = 10.0) -> np.ndarray:
        """Computes softmax probabilities with temperature scaling."""
        e_x = np.exp((x - np.max(x)) * temperature)
        return e_x / (e_x.sum(axis=0) + 1e-12)

    def extract_evidence(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        temperature: float = 10.0
    ) -> Dict[str, Any]:
        """
        Extracts visual evidence scores and prompt matches for a screenshot.

        Returns:
            Dict containing top_category, top_score, ranked_candidates,
            category_raw_scores, category_norm_scores, and prompt_evidence breakdown.
        """
        if not self._is_initialized:
            self.initialize_prompts()

        # Extract 512-dim visual image embedding
        image_embedding = self.encoder.get_image_embedding(image_input, normalize=True)  # (512,)

        # Compute cosine similarity across all prompts in a single matrix multiplication
        # _prompt_matrix is (M, 512), image_embedding is (512,) -> all_sims is (M,)
        all_sims = np.dot(self._prompt_matrix, image_embedding)  # Cosine similarities

        # Group by category and compute raw/explainable evidence
        category_raw_scores: Dict[str, float] = {}
        prompt_evidence: Dict[str, Dict[str, Any]] = {}

        idx = 0
        for category in CANONICAL_CATEGORIES:
            prompts = self.category_prompts[category]
            count = len(prompts)
            cat_sims = all_sims[idx : idx + count]
            idx += count

            raw_score = self._aggregate_similarities(cat_sims)
            category_raw_scores[category] = raw_score

            best_prompt_idx = int(np.argmax(cat_sims))
            prompt_evidence[category] = {
                "best_prompt": prompts[best_prompt_idx],
                "best_similarity": float(cat_sims[best_prompt_idx]),
                "all_prompt_similarities": [float(s) for s in cat_sims],
            }

        # Compute softmax normalized category probabilities
        raw_vector = np.array([category_raw_scores[c] for c in CANONICAL_CATEGORIES], dtype=np.float32)
        norm_vector = self._softmax(raw_vector, temperature=temperature)

        category_norm_scores: Dict[str, float] = {
            c: float(norm_vector[i]) for i, c in enumerate(CANONICAL_CATEGORIES)
        }

        # Rank categories by raw score descending
        ranked_candidates = sorted(
            category_raw_scores.items(), key=lambda item: item[1], reverse=True
        )

        top_category, top_score = ranked_candidates[0]

        return {
            "top_category": top_category,
            "top_score": top_score,
            "ranked_candidates": ranked_candidates,
            "category_raw_scores": category_raw_scores,
            "category_norm_scores": category_norm_scores,
            "prompt_evidence": prompt_evidence,
        }
