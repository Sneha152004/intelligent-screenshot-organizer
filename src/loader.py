"""
Dataset Loader Module
=====================
Loads screenshot metadata from a CSV file, validates the existence of referenced
image files on disk, and returns structured document dictionaries.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


def load_dataset(
    metadata_path: str = "sample/metadata.csv",
    images_dir: Optional[str] = None,
    image_col: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Reads metadata CSV file, validates image file existence, and returns document dictionaries.

    Args:
        metadata_path: Path to the CSV metadata file.
        images_dir: Optional directory containing images. Defaults to directory of metadata_path.
        image_col: Optional explicit CSV column name containing image filenames or IDs.

    Returns:
        List of dictionaries, each structured as:
        {
            "image_path": "path/to/image.jpg",
            "metadata": { ... }
        }

    Raises:
        FileNotFoundError: If metadata CSV or any referenced image file does not exist.
        ValueError: If no valid image column can be identified in the CSV.
    """
    meta_file = Path(metadata_path)
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

    # Default images directory to the parent directory of metadata.csv
    base_dir = Path(images_dir) if images_dir else meta_file.parent

    # Read CSV metadata
    df = pd.read_csv(meta_file)

    # Determine which column holds image reference / filename
    target_col = image_col
    if target_col is None:
        candidate_cols = [
            "filename",
            "file_name",
            "image_path",
            "image_name",
            "image",
            "Screenshot ID",
            "id",
        ]
        for col in candidate_cols:
            if col in df.columns:
                target_col = col
                break

    if target_col is None or target_col not in df.columns:
        raise ValueError(
            f"Could not automatically detect an image column in CSV. Available columns: {list(df.columns)}"
        )

    documents: List[Dict[str, Any]] = []

    for index, row in df.iterrows():
        img_ref = str(row[target_col]).strip()

        # Build candidate paths (direct, extension appended, zero-padded variants)
        candidate_paths = [
            base_dir / img_ref,
            base_dir / f"{img_ref}.jpg",
            base_dir / f"{img_ref}.png",
            base_dir / f"{img_ref}.jpeg",
        ]

        # Handle potential zero-padding mismatch (e.g. ss_010 -> ss_0010.jpg)
        if img_ref.startswith("ss_"):
            num_part = img_ref.replace("ss_", "").split(".")[0]
            if num_part.isdigit():
                padded_num = f"{int(num_part):04d}"  # 4-digit padding
                candidate_paths.extend([
                    base_dir / f"ss_{padded_num}.jpg",
                    base_dir / f"ss_{padded_num}.png",
                ])

        resolved_path: Optional[Path] = None
        for p in candidate_paths:
            if p.exists() and p.is_file():
                resolved_path = p.resolve()
                break

        if resolved_path is None:
            raise FileNotFoundError(
                f"Row {index}: Image reference '{img_ref}' could not be resolved to an existing file in '{base_dir}'."
            )

        # Build clean metadata dictionary
        metadata_dict = row.to_dict()

        documents.append(
            {
                "image_path": str(resolved_path),
                "metadata": metadata_dict,
            }
        )

    return documents
