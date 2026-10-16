"""
Intelligent Screenshot Organizer Package
=========================================
A modular 3-Layer Storage Architecture pipeline for screenshot ingestion, OCR,
text chunking, SentenceTransformer vector embeddings, and persistent ChromaDB retrieval.
"""

from .config import AppConfig, config
from .embedder import BaseEmbedder, SentenceTransformerEmbedder
from .mock_ocr import BaseOCREngine, MockOCREngine
from .chunker import ParagraphChunker, chunk_text
from .pipeline import OCRPipeline
from .organizer import ScreenshotOrganizer, organize_folder
from .models import Screenshot, ScreenshotMetadata, ScreenshotChunk
from .storage import BaseFileStore, LocalFileStore, BaseMetadataStore, SQLiteMetadataStore

try:
    from .loader import load_dataset, discover_screenshots
except Exception:
    load_dataset, discover_screenshots = None, None

try:
    from .easy_ocr import EasyOCREngine, get_ocr_engine
except Exception:
    EasyOCREngine, get_ocr_engine = None, None

try:
    from .vlm import BaseVLM, MockVLM, get_vlm_engine
except Exception:
    BaseVLM, MockVLM, get_vlm_engine = None, None, None

try:
    from .vector_store import VectorStoreManager
except Exception:
    VectorStoreManager = None

__all__ = [
    "AppConfig",
    "config",
    "BaseEmbedder",
    "SentenceTransformerEmbedder",
    "load_dataset",
    "discover_screenshots",
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
    "ScreenshotOrganizer",
    "organize_folder",
    "VectorStoreManager",
    "Screenshot",
    "ScreenshotMetadata",
    "ScreenshotChunk",
    "BaseFileStore",
    "LocalFileStore",
    "BaseMetadataStore",
    "SQLiteMetadataStore",
]
