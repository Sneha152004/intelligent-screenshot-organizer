"""
Production Multimodal Pipeline Module
=====================================
Coordinates the 3-Layer Storage Architecture & Multimodal Classification Pipeline:
Layer 1: File / Object Storage (LocalFileStore)
Layer 2: Structured Application Metadata Storage (SQLiteMetadataStore)
Layer 3: Vector Retrieval Index (VectorStoreManager)

End-to-End Processing Flow:
Screenshot -> EasyOCR Engine -> Text Chunking -> FinalRefinedClassifier (Semantic)
           -> YOLOv8n-cls (5-Class Visual Layout) -> Selective Fusion
           -> ScreenshotMetadata (Layer 2 SQLite) -> MiniLM Embeddings -> ChromaDB (Layer 3)
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

logger = logging.getLogger(__name__)

# Visual Screen-Type to Semantic Category Map for Validated 5-Class Model
VIS_5_TO_SEMANTIC_MAP = {
    "code_editor_terminal": "Coding",
    "travel_booking": "Travel",
    "media_player_library": "Entertainment",
    "document_article_view": "Education",
    "dashboard_settings": "Connectivity"
}


class OCRPipeline:
    """
    Production Multimodal Pipeline coordinating file storage, structured metadata,
    OCR extraction, semantic text classification, visual screen-type analysis,
    multimodal selective fusion, vector embedding, and ChromaDB retrieval.
    """

    def __init__(
        self,
        ocr_engine: Optional[BaseOCREngine] = None,
        chunker: Optional[ParagraphChunker] = None,
        file_store: Optional[BaseFileStore] = None,
        metadata_store: Optional[BaseMetadataStore] = None,
        semantic_classifier: Optional[Any] = None,
        visual_adapter: Optional[Any] = None,
        enable_visual: bool = True,
        ocr_engine_type: Optional[str] = None,
    ):
        """
        Initialize pipeline components and storage managers.
        """
        if ocr_engine is not None:
            self.ocr_engine: BaseOCREngine = ocr_engine
        else:
            target_engine_type = ocr_engine_type if ocr_engine_type is not None else config.ocr_engine_type
            self.ocr_engine: BaseOCREngine = get_ocr_engine(target_engine_type)

        self.chunker: ParagraphChunker = chunker if chunker else ParagraphChunker()
        self.file_store: BaseFileStore = file_store if file_store else LocalFileStore()
        self.metadata_store: BaseMetadataStore = metadata_store if metadata_store else SQLiteMetadataStore()
        self.enable_visual = enable_visual

        # Lazy load semantic classifier
        self._semantic_classifier = semantic_classifier
        # Lazy load 5-class visual adapter
        self._visual_adapter = visual_adapter

    @property
    def semantic_classifier(self) -> Any:
        if self._semantic_classifier is None:
            try:
                from experiments.intelligence.final_refinement.final_refinement_experiment import (
                    FinalRefinedClassifier,
                )
                self._semantic_classifier = FinalRefinedClassifier()
            except Exception as e:
                logger.warning(f"Could not load FinalRefinedClassifier ({e}). Running basic semantic fallback.")
                self._semantic_classifier = None
        return self._semantic_classifier

    @property
    def visual_adapter(self) -> Any:
        if self._visual_adapter is None and self.enable_visual:
            try:
                from experiments.visual_understanding.phase7j.visual_screen_type_adapter import (
                    VisualScreenTypeAdapter,
                )
                self._visual_adapter = VisualScreenTypeAdapter(use_int8=False)
            except Exception as e:
                logger.warning(f"Could not load 5-class VisualScreenTypeAdapter ({e}). Visual branch disabled.")
                self._visual_adapter = None
        return self._visual_adapter

    def process(
        self,
        metadata_path: str = "sample/metadata.csv",
        images_dir: Optional[str] = None,
        image_col: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Processes dataset through the 3-Layer architecture & multimodal pipeline:
        1. Reads raw metadata & validates screenshot images.
        2. Registers screenshot files in Layer 1 File Storage.
        3. Extracts OCR text & runs semantic classification.
        4. Runs visual screen-type analysis (5-class YOLO) with safe fallback.
        5. Performs selective multimodal fusion to resolve ambiguity.
        6. Saves rich ScreenshotMetadata in Layer 2 SQLite Metadata Storage.
        7. Partitions OCR text into paragraph chunks.
        """
        print("Loading metadata...")
        documents = load_dataset(
            metadata_path=metadata_path,
            images_dir=images_dir,
            image_col=image_col,
        )
        print(f"Loaded {len(documents)} images.\n")
        return self.process_documents(documents)

    def process_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Processes a list of document dictionaries through the multimodal classification pipeline.
        """
        print(f"Running OCR & Multimodal Classification ({self.ocr_engine.__class__.__name__})...")
        processed_documents: List[Dict[str, Any]] = []

        for doc in documents:
            src_path = doc["image_path"]
            raw_meta = doc["metadata"]
            img_filename = Path(src_path).name

            screenshot_id = str(raw_meta.get("Screenshot ID", raw_meta.get("screenshot_id", img_filename.split(".")[0]))).strip()
            dataset_category = str(raw_meta.get("Actual Category", raw_meta.get("category", "General"))).strip()
            notes_text = str(raw_meta.get("notes", raw_meta.get("Notes", ""))).strip()

            # Layer 1: Store screenshot in file storage and get image_uri
            image_uri = self.file_store.store_image(source_path=src_path, screenshot_id=screenshot_id)

            # OCR Extraction
            ocr_text = self.ocr_engine.extract_text(src_path)
            text_input = ocr_text if ocr_text else (notes_text if notes_text else img_filename)

            # 1. Semantic Classification Branch
            sem_category = dataset_category
            sem_confidence = "Low"
            is_ambiguous = False

            if self.semantic_classifier is not None:
                try:
                    sem_res = self.semantic_classifier.predict_final(raw_text=text_input, notes=notes_text, variant="C")
                    sem_category = sem_res.get("prediction", dataset_category)
                    sem_confidence = sem_res.get("confidence", "Low")
                    is_ambiguous = bool(sem_res.get("is_ambiguous", False))
                except Exception as e:
                    logger.warning(f"Semantic prediction error for {img_filename} ({e}). Using dataset fallback.")

            # 2. Visual Screen-Type Branch (Safe Fallback)
            visual_evidence = None
            final_category = sem_category
            fusion_applied = False
            fusion_reason = "Semantic classifier prediction authoritative"

            if self.enable_visual and self.visual_adapter is not None and os.path.exists(src_path):
                try:
                    vis_res = self.visual_adapter.predict_image(src_path)
                    vis_st = vis_res["screen_type"]
                    vis_conf = vis_res["confidence"]

                    visual_evidence = {
                        "screen_type": vis_st,
                        "confidence": vis_conf,
                        "model_name": "YOLOv8n-cls",
                        "model_version": "1.0.0-phase7j",
                        "fusion_applied": False,
                        "fusion_reason": fusion_reason,
                        "original_semantic_category": sem_category
                    }

                    # 3. Multimodal Selective Fusion Rule (Strategy E)
                    if (sem_confidence == "Low" or is_ambiguous or sem_category == "General") and vis_conf >= 0.40:
                        if vis_st in VIS_5_TO_SEMANTIC_MAP:
                            override_target = VIS_5_TO_SEMANTIC_MAP[vis_st]
                            if override_target != sem_category:
                                final_category = override_target
                                fusion_applied = True
                                fusion_reason = f"Visual layout '{vis_st}' (conf {vis_conf:.2f}) overrode low-confidence/ambiguous semantic fallback '{sem_category}' -> '{final_category}'"
                                visual_evidence["fusion_applied"] = True
                                visual_evidence["fusion_reason"] = fusion_reason
                except Exception as e:
                    logger.warning(f"Visual inference failed for {img_filename} ({e}). Falling back safely to semantic prediction.")
                    visual_evidence = None

            # Layer 2: Create & persist rich structured metadata object
            metadata_obj = ScreenshotMetadata(
                screenshot_id=screenshot_id,
                filename=img_filename,
                image_uri=image_uri,
                category=final_category,
                tags=list(raw_meta.get("tags", [])),
                ocr_available=True,
                summary=notes_text if notes_text else f"Screenshot {img_filename}",
                evidence={
                    "text_snippet": ocr_text[:100],
                    "visual_evidence": visual_evidence,
                    "fusion_applied": fusion_applied,
                    "fusion_reason": fusion_reason,
                    "original_semantic_category": sem_category
                }
            )
            self.metadata_store.save_metadata(metadata_obj)
            print(f"Processed {img_filename} -> Category: {final_category} (Fused: {fusion_applied})")

            # Chunk extracted OCR text
            chunks = self.chunker.chunk(ocr_text)

            processed_doc = {
                "screenshot_id": screenshot_id,
                "image_path": src_path,
                "image_uri": image_uri,
                "metadata": metadata_obj.to_dict(),
                "ocr_text": ocr_text,
                "chunks": chunks,
                "visual_evidence": visual_evidence,
                "fusion_applied": fusion_applied
            }
            processed_documents.append(processed_doc)

        print("\nChunking text...")
        print("Done.")

        return processed_documents

    def organize_folder(
        self,
        input_dir: str,
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Organizes screenshots from input_dir into category subdirectories within output_dir.
        """
        from .organizer import ScreenshotOrganizer
        organizer = ScreenshotOrganizer(pipeline=self)
        return organizer.organize_folder(input_dir=input_dir, output_dir=output_dir)


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
