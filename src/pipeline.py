"""
Pipeline Module
===============
Coordinates dataset loading, OCR text extraction (mock or real), paragraph chunking,
and metadata attachment into structured processed document dicts.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .chunker import ParagraphChunker
from .loader import load_dataset
from .mock_ocr import BaseOCREngine, MockOCREngine


class OCRPipeline:
    """
    Modular OCR pipeline for processing screenshot images into indexed documents.
    """

    def __init__(
        self,
        ocr_engine: Optional[BaseOCREngine] = None,
        chunker: Optional[ParagraphChunker] = None,
    ):
        """
        Initialize the pipeline with an OCR engine and text chunker.

        Args:
            ocr_engine: OCR engine instance (defaults to MockOCREngine).
            chunker: Text chunker instance (defaults to ParagraphChunker).
        """
        self.ocr_engine: BaseOCREngine = ocr_engine if ocr_engine else MockOCREngine()
        self.chunker: ParagraphChunker = chunker if chunker else ParagraphChunker()

    def process(
        self,
        metadata_path: str = "sample/metadata.csv",
        images_dir: Optional[str] = None,
        image_col: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs the complete mock OCR pipeline:
        1. Load & validate metadata and image paths.
        2. Perform OCR text extraction (mock or real).
        3. Chunk extracted text into paragraphs.
        4. Attach original metadata and return processed documents.

        Args:
            metadata_path: Path to CSV metadata file.
            images_dir: Optional directory override for screenshot images.
            image_col: Optional explicit column name for image filenames.

        Returns:
            List of processed document dictionaries containing image_path,
            metadata, ocr_text, and chunks.
        """
        # Step 1: Load metadata
        print("Loading metadata...")
        documents = load_dataset(
            metadata_path=metadata_path,
            images_dir=images_dir,
            image_col=image_col,
        )
        print(f"Loaded {len(documents)} images.\n")

        # Step 2 & 3: Run OCR and chunk text
        print("Running mock OCR...")
        processed_documents: List[Dict[str, Any]] = []

        for doc in documents:
            img_path = doc["image_path"]
            img_filename = Path(img_path).name

            # Run OCR extraction
            ocr_text = self.ocr_engine.extract_text(img_path)
            print(f"Processed {img_filename}")

            # Chunk extracted OCR text
            chunks = self.chunker.chunk(ocr_text)

            # Assemble output document
            processed_doc = {
                "image_path": img_path,
                "metadata": doc["metadata"],
                "ocr_text": ocr_text,
                "chunks": chunks,
            }
            processed_documents.append(processed_doc)

        print("\nChunking text...")
        print("Done.")

        return processed_documents
