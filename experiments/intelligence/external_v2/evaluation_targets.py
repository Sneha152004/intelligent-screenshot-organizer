"""
External V2 Benchmark Evaluation Targets & Validation Module
=============================================================
Provides a clean, machine-readable interface and schema validator
for the frozen External V2 evaluation benchmark.

IMPORTANT BENCHMARK RULES:
1. This module must NOT depend on any classifier or model code.
2. Coding is NOT accepted as a top-level category (Coding is a tag).
3. The top-level category taxonomy contains exactly 9 canonical categories.
4. Empty category/tag labels are permitted during initial benchmark setup.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

# Top-level Canonical Taxonomy (9 Categories)
CANONICAL_9_CATEGORIES: Sequence[str] = (
    "Education",
    "Shopping",
    "Payment",
    "Travel",
    "Communication",
    "Entertainment",
    "Work & Career",
    "Connectivity",
    "General",
)

# Rejected top-level categories
EXPLICITLY_REJECTED_CATEGORIES: Sequence[str] = ("Coding",)

# Required manifest CSV schema header
REQUIRED_SCHEMA_FIELDS: Sequence[str] = (
    "id",
    "filename",
    "image_path",
    "category",
    "tags",
    "label_confidence",
    "label_reason",
    "source",
)

EXPECTED_TOTAL_RECORDS: int = 244
BENCHMARK_DIR: Path = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH: Path = BENCHMARK_DIR / "manifest.csv"
DEFAULT_IMAGES_DIR: Path = BENCHMARK_DIR / "images"


@dataclass(frozen=True)
class ExternalV2Record:
    """Represents a single record in the External V2 benchmark manifest."""

    id: str
    filename: str
    image_path: Path
    category: str
    tags: List[str] = field(default_factory=list)
    label_confidence: str = ""
    label_reason: str = ""
    source: str = "UNKNOWN"

    def to_dict(self) -> Dict[str, Union[str, List[str]]]:
        """Converts record to dictionary representation."""
        return {
            "id": self.id,
            "filename": self.filename,
            "image_path": str(self.image_path),
            "category": self.category,
            "tags": list(self.tags),
            "label_confidence": self.label_confidence,
            "label_reason": self.label_reason,
            "source": self.source,
        }


def validate_manifest(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    allow_empty_categories: bool = True,
    expected_count: int = EXPECTED_TOTAL_RECORDS,
) -> List[ExternalV2Record]:
    """
    Reads and validates the External V2 benchmark manifest against benchmark integrity rules.

    Validation checks performed:
    1. Manifest file existence.
    2. Exact match of expected record count (default 244).
    3. Sequential IDs from ext2_001 through ext2_244.
    4. Schema header completeness (required fields).
    5. Filename uniqueness and existence in images directory.
    6. Category validation:
       - 'Coding' is strictly REJECTED as a top-level category.
       - Populated categories MUST belong to the 9 canonical categories.
       - Unpopulated categories raise ValueError if allow_empty_categories is False.
    7. Tags are parsed as comma-separated list distinct from category.

    Args:
        manifest_path: Path to manifest.csv.
        allow_empty_categories: If True, empty category fields are permitted (setup mode).
        expected_count: Total expected record count (default 244).

    Returns:
        List of validated ExternalV2Record objects.

    Raises:
        FileNotFoundError: If manifest or image file is missing.
        ValueError: If schema, counts, IDs, categories, or paths violate rules.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    records: List[ExternalV2Record] = []
    seen_ids = set()
    seen_filenames = set()

    with open(manifest_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        
        # Verify schema header
        if not reader.fieldnames or not set(REQUIRED_SCHEMA_FIELDS).issubset(set(reader.fieldnames)):
            missing = set(REQUIRED_SCHEMA_FIELDS) - set(reader.fieldnames or [])
            raise ValueError(f"Manifest header is missing required fields: {missing}")

        for idx, row in enumerate(reader, 1):
            rec_id = row["id"].strip()
            filename = row["filename"].strip()
            rel_img_path = row["image_path"].strip()
            category = row["category"].strip()
            tags_raw = row["tags"].strip()
            label_conf = row["label_confidence"].strip()
            label_reason = row["label_reason"].strip()
            source = row["source"].strip() or "UNKNOWN"

            # Check ID uniqueness & format
            if rec_id in seen_ids:
                raise ValueError(f"Duplicate ID found in manifest: '{rec_id}'")
            seen_ids.add(rec_id)

            expected_id = f"ext2_{idx:03d}"
            if rec_id != expected_id:
                raise ValueError(f"ID mismatch at index {idx}: expected '{expected_id}', found '{rec_id}'")

            # Check filename uniqueness
            if filename in seen_filenames:
                raise ValueError(f"Duplicate filename found in manifest: '{filename}'")
            seen_filenames.add(filename)

            # Check image existence
            abs_img_path = (manifest_path.parent / rel_img_path).resolve()
            if not abs_img_path.is_file():
                raise FileNotFoundError(
                    f"Image file referenced by ID '{rec_id}' does not exist on disk: {abs_img_path}"
                )

            # Category validation
            if category in EXPLICITLY_REJECTED_CATEGORIES:
                raise ValueError(
                    f"Invalid category '{category}' for ID '{rec_id}'. "
                    f"Coding is NOT a category (Coding must be represented as a tag)."
                )

            if category:
                if category not in CANONICAL_9_CATEGORIES:
                    raise ValueError(
                        f"Unknown category '{category}' for ID '{rec_id}'. "
                        f"Category must be one of: {list(CANONICAL_9_CATEGORIES)}"
                    )
            elif not allow_empty_categories:
                raise ValueError(f"Empty category field for ID '{rec_id}' when allow_empty_categories=False.")

            # Parse tags
            tags_list = [t.strip() for t in tags_raw.split(",") if t.strip()]

            records.append(
                ExternalV2Record(
                    id=rec_id,
                    filename=filename,
                    image_path=abs_img_path,
                    category=category,
                    tags=tags_list,
                    label_confidence=label_conf,
                    label_reason=label_reason,
                    source=source,
                )
            )

    if len(records) != expected_count:
        raise ValueError(
            f"Record count mismatch in manifest: expected {expected_count}, found {len(records)}"
        )

    return records


def load_evaluation_targets(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    allow_empty_categories: bool = True,
) -> List[ExternalV2Record]:
    """
    Loads and returns benchmark evaluation targets from manifest.csv.

    Args:
        manifest_path: Path to external_v2 manifest.csv.
        allow_empty_categories: Permit unpopulated category labels during setup.

    Returns:
        List of ExternalV2Record objects representing frozen benchmark targets.
    """
    return validate_manifest(
        manifest_path=manifest_path,
        allow_empty_categories=allow_empty_categories,
        expected_count=EXPECTED_TOTAL_RECORDS,
    )
