"""
Embedder Module
===============
Provides an abstract base class for text embedding models and a concrete implementation
using SentenceTransformers for generating dense vector representations of text chunks.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """
    Abstract Base Class for text embedder implementations.
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of text strings into numerical vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, where each vector is a list of floats.
        """
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Local text embedder based on the sentence-transformers library.
    Loads the underlying model once during initialization.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize SentenceTransformerEmbedder.

        Args:
            model_name: Name of the SentenceTransformer model to load.
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embedding vectors for a list of text strings.

        Args:
            texts: List of text strings.

        Returns:
            List of embedding vectors (lists of floats).

        Raises:
            TypeError: If input is not a list or if elements are not strings.
        """
        if not isinstance(texts, list):
            raise TypeError(f"Input to embed() must be a list, got {type(texts).__name__}")

        if len(texts) == 0:
            return []

        for idx, item in enumerate(texts):
            if not isinstance(item, str):
                raise TypeError(
                    f"Element at index {idx} must be a string, got {type(item).__name__}"
                )

        # Generate embeddings deterministically
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()
