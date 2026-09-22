"""
Pipeline Module
===============
Coordinates the 3-Layer Storage Architecture:
Layer 1: File / Object Storage (LocalFileStore)
Layer 2: Structured Metadata Storage (SQLiteMetadataStore)
Layer 3: ChromaDB Vector Index (VectorStoreManager)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .chunker import ParagraphChunker
from .config import config
from .easy_ocr import get_ocr_engine
from .embedder import BaseEmbedder, SentenceTransformerEmbedder
from .loader import load_dataset
from .mock_ocr import BaseOCREngine, MockOCREngine
from .models.metadata import ScreenshotMetadata
from .storage.file_store import BaseFileStore, LocalFileStore
from .storage.metadata_store import BaseMetadataStore, SQLiteMetadataStore
from .vector_store import VectorStoreManager


class OCRPipeline:
    """
    Modular OCR & indexing pipeline coordinating file storage, structured metadata,
    OCR extraction, text chunking, vector embedding, and ChromaDB retrieval.
    """

    def __init__(
        self,
        ocr_engine: Optional[BaseOCREngine] = None,
        chunker: Optional[ParagraphChunker] = None,
        file_store: Optional[BaseFileStore] = None,
        metadata_store: Optional[BaseMetadataStore] = None,
        ocr_engine_type: Optional[str] = None,
    ):
        """
        Initialize pipeline components and storage managers.

        Args:
            ocr_engine: Explicit OCR engine instance (takes precedence over configuration).
            chunker: Text chunker instance (defaults to ParagraphChunker).
            file_store: Object storage manager (defaults to LocalFileStore).
            metadata_store: Structured metadata manager (defaults to SQLiteMetadataStore).
            ocr_engine_type: Optional string identifier ('mock' or 'easyocr') overriding config setting.
        """
        if ocr_engine is not None:
            self.ocr_engine: BaseOCREngine = ocr_engine
        else:
            target_engine_type = ocr_engine_type if ocr_engine_type is not None else config.ocr_engine_type
            self.ocr_engine: BaseOCREngine = get_ocr_engine(target_engine_type)

        self.chunker: ParagraphChunker = chunker if chunker else ParagraphChunker()
        self.file_store: BaseFileStore = file_store if file_store else LocalFileStore()
        self.metadata_store: BaseMetadataStore = metadata_store if metadata_store else SQLiteMetadataStore()

    def process(
        self,
        metadata_path: str = "sample/metadata.csv",
        images_dir: Optional[str] = None,
        image_col: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Processes dataset through storage registration, OCR extraction, and text chunking:
        1. Reads raw metadata CSV & validates images.
        2. Registers screenshot files in Layer 1 File Storage.
        3. Saves structured application metadata in Layer 2 Metadata Storage.
        4. Extracts OCR text and partitions into paragraph chunks.

        Args:
            metadata_path: Path to CSV metadata file.
            images_dir: Directory override for screenshot images.
            image_col: Explicit column name for image filenames.

        Returns:
            List of processed document dictionaries containing image_uri, metadata, ocr_text, and chunks.
        """
        print("Loading metadata...")
        documents = load_dataset(
            metadata_path=metadata_path,
            images_dir=images_dir,
            image_col=image_col,
        )
        print(f"Loaded {len(documents)} images.\n")

        print(f"Running OCR extraction ({self.ocr_engine.__class__.__name__})...")
        processed_documents: List[Dict[str, Any]] = []

        for doc in documents:
            src_path = doc["image_path"]
            raw_meta = doc["metadata"]
            img_filename = Path(src_path).name

            screenshot_id = str(raw_meta.get("Screenshot ID", raw_meta.get("screenshot_id", img_filename.split(".")[0]))).strip()
            category = str(raw_meta.get("Actual Category", raw_meta.get("category", "General"))).strip()

            # Layer 1: Store screenshot in file storage and get image_uri
            image_uri = self.file_store.store_image(source_path=src_path, screenshot_id=screenshot_id)

            # Layer 2: Create & persist structured metadata object
            metadata_obj = ScreenshotMetadata(
                screenshot_id=screenshot_id,
                filename=img_filename,
                image_uri=image_uri,
                category=category,
                ocr_available=True,
            )
            self.metadata_store.save_metadata(metadata_obj)

            # Run OCR extraction
            ocr_text = self.ocr_engine.extract_text(src_path)
            print(f"Processed {img_filename}")

            # Chunk extracted OCR text
            chunks = self.chunker.chunk(ocr_text)

            processed_doc = {
                "screenshot_id": screenshot_id,
                "image_path": src_path,
                "image_uri": image_uri,
                "metadata": metadata_obj.to_dict(),
                "ocr_text": ocr_text,
                "chunks": chunks,
            }
            processed_documents.append(processed_doc)

        print("\nChunking text...")
        print("Done.")

        return processed_documents

    def search_and_enrich(
        self,
        query_text: str,
        embedder: BaseEmbedder,
        vector_store: VectorStoreManager,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Performs 2-stage retrieval:
        1. Query ChromaDB vector store (Layer 3) to retrieve lightweight vector matches.
        2. Resolve screenshot_id against Metadata Storage (Layer 2) and File Storage (Layer 1) to enrich results.

        Args:
            query_text: Natural language search string.
            embedder: Embedder model instance.
            vector_store: ChromaDB vector store instance.
            top_k: Top-K search parameter.

        Returns:
            List of enriched search result dictionaries.
        """
        query_vec = embedder.embed([query_text])[0]
        raw_results = vector_store.similarity_search(query_embedding=query_vec, top_k=top_k)

        enriched_results = []
        for res in raw_results:
            sid = res.get("screenshot_id")
            meta_obj = self.metadata_store.get_metadata(sid) if sid else None
            image_uri = self.file_store.get_image_uri(sid) if sid else None

            enriched_results.append({
                "screenshot_id": sid,
                "chunk_index": res.get("chunk_index", 0),
                "text": res.get("document", ""),
                "similarity_score": res.get("similarity_score", 0.0),
                "distance": res.get("distance", 0.0),
                "filename": meta_obj.filename if meta_obj else res.get("metadata", {}).get("filename", ""),
                "category": meta_obj.category if meta_obj else res.get("category", "General"),
                "image_uri": image_uri if image_uri else meta_obj.image_uri if meta_obj else "",
                "metadata": meta_obj.to_dict() if meta_obj else res.get("metadata", {}),
            })

        return enriched_results
