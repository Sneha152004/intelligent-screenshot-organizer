"""
Intelligent Screenshot Organizer Package
=========================================
A modular 3-Layer Storage Architecture pipeline for screenshot ingestion, OCR,
text chunking, SentenceTransformer vector embeddings, and persistent ChromaDB retrieval.
"""

from .config import AppConfig, config
from .embedder import BaseEmbedder, SentenceTransformerEmbedder
from .loader import load_dataset
from .mock_ocr import BaseOCREngine, MockOCREngine
from .chunker import ParagraphChunker, chunk_text
from .pipeline import OCRPipeline
from .vector_store import VectorStoreManager
from .models import Screenshot, ScreenshotMetadata, ScreenshotChunk
from .storage import BaseFileStore, LocalFileStore, BaseMetadataStore, SQLiteMetadataStore

__all__ = [
    "AppConfig",
    "config",
    "BaseEmbedder",
    "SentenceTransformerEmbedder",
    "load_dataset",
    "BaseOCREngine",
    "MockOCREngine",
    "ParagraphChunker",
    "chunk_text",
    "OCRPipeline",
    "VectorStoreManager",
    "Screenshot",
    "ScreenshotMetadata",
    "ScreenshotChunk",
    "BaseFileStore",
    "LocalFileStore",
    "BaseMetadataStore",
    "SQLiteMetadataStore",
]
