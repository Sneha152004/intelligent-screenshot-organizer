"""
Mock OCR Pipeline Package
=========================
A modular pipeline for screenshot ingestion, mock OCR extraction, and text chunking.
"""

from .loader import load_dataset
from .mock_ocr import BaseOCREngine, MockOCREngine
from .chunker import ParagraphChunker, chunk_text
from .pipeline import OCRPipeline

__all__ = [
    "load_dataset",
    "BaseOCREngine",
    "MockOCREngine",
    "ParagraphChunker",
    "chunk_text",
    "OCRPipeline",
]
