"""
Configuration Module
====================
Centralized configuration settings cleanly separating the three storage layers:
1. File Storage (Image Objects)
2. Structured Metadata Storage (Application Data & Metadata)
3. Vector Storage (ChromaDB Index & Retrieval Layer)
4. Embeddings & Retrieval Parameters
"""

import os
from dataclasses import dataclass


@dataclass
class AppConfig:
    """Application configuration parameters for 3-layer storage architecture."""

    # 1. File Storage Settings
    screenshots_dir: str = "data/screenshots"

    # 2. Structured Metadata Storage Settings
    metadata_db_path: str = "data/metadata.sqlite"

    # 3. Vector Storage Settings (ChromaDB)
    chroma_db_dir: str = "chroma_db"
    collection_name: str = "screenshot_chunks"

    # 4. Embedding & Retrieval Settings
    embedding_model: str = "all-MiniLM-L6-v2"
    default_top_k: int = 5

    # 5. OCR Engine Settings
    ocr_engine_type: str = "mock"  # Options: "mock", "easyocr"

    # 6. VLM Engine Settings
    vlm_engine_type: str = "mock"  # Options: "mock"

    # Benchmark Dataset Settings
    sample_metadata_path: str = "sample/metadata.csv"
    sample_images_dir: str = "sample"


# Global configuration instance
config = AppConfig()
