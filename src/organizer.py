"""
Automatic Category Folder Organizer Module
==========================================
Processes an input directory of screenshots through the production OCR & multimodal classification pipeline,
copies screenshots into canonical category subfolders in an output library directory,
and persists organized metadata in SQLite (Layer 2) and ChromaDB (Layer 3).
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

from .loader import discover_screenshots
from .models.metadata import ScreenshotMetadata
from .pipeline import OCRPipeline
from .storage.file_store import LocalFileStore
from .storage.metadata_store import SQLiteMetadataStore

logger = logging.getLogger(__name__)


class ScreenshotOrganizer:
    """
    Automatic Screenshot Category Folder Organizer.
    Organizes unorganized screenshot folders into structured category subdirectories.
    """

    def __init__(
        self,
        pipeline: Optional[OCRPipeline] = None,
        file_store: Optional[LocalFileStore] = None,
        metadata_store: Optional[SQLiteMetadataStore] = None,
    ):
        self.pipeline = pipeline if pipeline is not None else OCRPipeline()
        self.file_store = file_store
        self.metadata_store = metadata_store

    def organize_folder(
        self,
        input_dir: str,
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Organizes screenshots from input_dir into category subdirectories within output_dir.

        Args:
            input_dir: Path to directory containing screenshots to organize.
            output_dir: Target directory for organized category folders.

        Returns:
            Concise processing summary dictionary:
            {
                "processed": int,
                "organized": int,
                "skipped": int,
                "failed": int,
                "category_counts": Dict[str, int],
                "failed_files": List[Dict[str, str]],
                "output_dir": str,
            }
        """
        in_path = Path(input_dir).resolve()
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        target_file_store = (
            self.file_store
            if self.file_store is not None
            else LocalFileStore(storage_dir=str(out_path))
        )
        target_metadata_store = (
            self.metadata_store
            if self.metadata_store is not None
            else self.pipeline.metadata_store
        )

        # Temporarily adapt pipeline file_store and metadata_store for target library
        orig_file_store = self.pipeline.file_store
        orig_meta_store = self.pipeline.metadata_store
        self.pipeline.file_store = target_file_store
        self.pipeline.metadata_store = target_metadata_store

        processed = 0
        organized = 0
        skipped = 0
        failed = 0
        category_counts: Dict[str, int] = {}
        failed_files: List[Dict[str, str]] = []

        try:
            # Discover screenshots
            documents = discover_screenshots(str(in_path))

            for doc in documents:
                src_path = doc["image_path"]
                img_filename = Path(src_path).name

                # Pre-validate image readability & file size
                try:
                    if not os.path.exists(src_path) or os.path.getsize(src_path) == 0:
                        raise ValueError("Zero-byte or non-existent file.")
                    with Image.open(src_path) as img:
                        img.verify()
                except Exception as val_err:
                    failed += 1
                    failed_files.append({
                        "file": img_filename,
                        "path": src_path,
                        "error": str(val_err),
                    })
                    logger.warning(f"Skipping corrupted/unreadable image {img_filename}: {val_err}")
                    continue

                # Process single document through pipeline
                try:
                    processed_docs = self.pipeline.process_documents([doc])
                    if not processed_docs:
                        failed += 1
                        failed_files.append({
                            "file": img_filename,
                            "path": src_path,
                            "error": "Processing returned empty document.",
                        })
                        continue

                    p_doc = processed_docs[0]
                    category = p_doc["metadata"]["category"]
                    sid = p_doc["screenshot_id"]

                    # Copy into category subfolder in output library (idempotent & collision-safe)
                    target_path, is_new = target_file_store.store_image_in_category(
                        source_path=src_path,
                        screenshot_id=sid,
                        category=category,
                        filename=img_filename,
                    )

                    # Remove intermediate root file if stored at output_dir root
                    temp_root_path = p_doc.get("image_uri")
                    if temp_root_path and os.path.exists(temp_root_path):
                        temp_p = Path(temp_root_path).resolve()
                        targ_p = Path(target_path).resolve()
                        if temp_p != targ_p and temp_p.parent == out_path:
                            try:
                                os.remove(temp_p)
                            except Exception:
                                pass

                    # Update metadata image_uri with organized path
                    p_doc["metadata"]["image_uri"] = target_path
                    p_doc["image_uri"] = target_path

                    # Save updated metadata to SQLite
                    meta_obj = ScreenshotMetadata.from_dict(p_doc["metadata"])
                    target_metadata_store.save_metadata(meta_obj)

                    processed += 1
                    if is_new:
                        organized += 1
                    else:
                        skipped += 1

                    category_counts[category] = category_counts.get(category, 0) + 1

                except Exception as proc_err:
                    failed += 1
                    failed_files.append({
                        "file": img_filename,
                        "path": src_path,
                        "error": str(proc_err),
                    })
                    logger.warning(f"Failed processing {img_filename}: {proc_err}")

        finally:
            # Restore original pipeline stores
            self.pipeline.file_store = orig_file_store
            self.pipeline.metadata_store = orig_meta_store

        summary = {
            "processed": processed,
            "organized": organized,
            "skipped": skipped,
            "failed": failed,
            "category_counts": category_counts,
            "failed_files": failed_files,
            "output_dir": str(out_path),
        }
        return summary


def organize_folder(
    input_dir: str,
    output_dir: str,
    pipeline: Optional[OCRPipeline] = None,
) -> Dict[str, Any]:
    """
    Convenience function to organize screenshots from input_dir into output_dir.
    """
    organizer = ScreenshotOrganizer(pipeline=pipeline)
    return organizer.organize_folder(input_dir=input_dir, output_dir=output_dir)
