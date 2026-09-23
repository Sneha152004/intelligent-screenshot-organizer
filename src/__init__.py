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
from .easy_ocr import EasyOCREngine, get_ocr_engine
from .vlm import BaseVLM, MockVLM, get_vlm_engine
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
    "EasyOCREngine",
    "get_ocr_engine",
    "BaseVLM",
    "MockVLM",
    "get_vlm_engine",
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
