"""
Shared singleton accessors for storage, pipeline, embedder, and vector store.
Heavy ML models (embedder, pipeline) are loaded LAZILY on first request
so that API startup, dashboard rendering, category listing, and screenshot browsing
respond instantly with zero cold-start delay.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Organizer root on sys.path ───────────────────────────────────────────────
_ORGANIZER_ROOT = Path(__file__).resolve().parents[3]
if str(_ORGANIZER_ROOT) not in sys.path:
    sys.path.insert(0, str(_ORGANIZER_ROOT))

# ── Relative data paths resolve from organizer root ───────────────────────────
os.chdir(str(_ORGANIZER_ROOT))

from src.config import config                                   # noqa: E402
from src.storage.file_store import LocalFileStore               # noqa: E402
from src.storage.metadata_store import SQLiteMetadataStore      # noqa: E402

# ── Layer 1: File Storage (instant) ──────────────────────────────────────────
file_store = LocalFileStore(storage_dir=config.screenshots_dir)

# ── Layer 2: Metadata Storage (instant) ──────────────────────────────────────
metadata_store = SQLiteMetadataStore(db_path=config.metadata_db_path)

# ── Layer 3: Vector Store (lazy) ─────────────────────────────────────────────
_vector_store = None


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        try:
            from src.vector_store import VectorStoreManager
            _vector_store = VectorStoreManager(
                db_dir=config.chroma_db_dir,
                collection_name=config.collection_name,
            )
            logger.info("ChromaDB vector store initialised.")
        except Exception as e:
            logger.warning("ChromaDB unavailable (%s).", e)
            _vector_store = None
    return _vector_store


# ── Embedder (lazy) ──────────────────────────────────────────────────────────
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from src.embedder import SentenceTransformerEmbedder
            _embedder = SentenceTransformerEmbedder(model_name=config.embedding_model)
            logger.info("Embedder loaded: %s", config.embedding_model)
        except Exception as e:
            logger.warning("Embedder unavailable (%s).", e)
            _embedder = None
    return _embedder


# ── OCR Pipeline (lazy) ──────────────────────────────────────────────────────
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from src.pipeline import OCRPipeline
            _pipeline = OCRPipeline(
                file_store=file_store,
                metadata_store=metadata_store,
            )
            logger.info("OCR Pipeline ready.")
        except Exception as e:
            logger.warning("Pipeline unavailable (%s).", e)
            _pipeline = None
    return _pipeline
