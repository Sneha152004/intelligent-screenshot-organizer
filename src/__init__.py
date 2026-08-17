"""
Mock OCR Pipeline & Embedding Package
=====================================
A modular pipeline for screenshot ingestion, mock OCR extraction, text chunking, and embedding.
"""

from .loader import load_dataset
from .mock_ocr import BaseOCREngine, MockOCREngine
from .chunker import ParagraphChunker, chunk_text
from .pipeline import OCRPipeline
from .embedder import BaseEmbedder, SentenceTransformerEmbedder

__all__ = [
    "load_dataset",
    "BaseOCREngine",
    "MockOCREngine",
    "ParagraphChunker",
    "chunk_text",
    "OCRPipeline",
    "BaseEmbedder",
    "SentenceTransformerEmbedder",
]
