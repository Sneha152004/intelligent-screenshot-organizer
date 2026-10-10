"""
Unit test suite for Dataset Transition Audit (30 -> 836 Screenshot Dataset)
=============================================================================
Verifies:
1. Audit JSON and Markdown artifacts exist and follow strict protocol schemas.
2. Dataset size metrics (836 images, 836 CSV metadata rows, 1-to-1 correspondence).
3. 10 Canonical Categories are present with 0 unexpected or missing categories.
4. Original 30 screenshots are 100% preserved in the 836 pool.
5. Zero External 60 Benchmark Leakage (0 filename overlaps, 0 SHA-256 byte matches, 0 near-duplicates).
6. Production safety flags verify zero code, database, or production modifications.
"""

import json
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JSON_OUT = PROJECT_ROOT / "experiments" / "intelligence" / "dataset_transition_audit.json"
MD_OUT = PROJECT_ROOT / "experiments" / "intelligence" / "dataset_transition_audit.md"

CANONICAL_CATEGORIES = [
    "Coding", "Communication", "Connectivity", "Education",
    "Entertainment", "General", "Payment", "Shopping", "Travel", "Work & Career"
]


def test_audit_artifacts_exist():
    """Verifies JSON and Markdown audit report files exist."""
    assert JSON_OUT.exists(), f"Missing audit JSON file: {JSON_OUT}"
    assert MD_OUT.exists(), f"Missing audit Markdown file: {MD_OUT}"


def test_audit_dataset_size_metrics():
    """Verifies training pool size is exactly 836 and matches metadata rows 1-to-1."""
    with open(JSON_OUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    audit_1 = data["audit_1_dataset_size"]
    assert audit_1["expected_image_count"] == 836
    assert audit_1["actual_image_files_count"] == 836
    assert audit_1["actual_metadata_csv_rows"] == 836
    assert audit_1["one_to_one_correspondence"] is True
    assert audit_1["missing_in_csv_count"] == 0
    assert audit_1["missing_in_folder_count"] == 0
    assert audit_1["duplicate_filenames_in_csv"] == 0


def test_audit_category_distribution():
    """Verifies all 10 canonical categories are present with valid distribution."""
    with open(JSON_OUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    audit_2 = data["audit_2_category_distribution"]
    assert audit_2["total_canonical_categories"] == 10
    assert audit_2["is_distribution_valid"] is True
    assert len(audit_2["unexpected_categories"]) == 0
    assert len(audit_2["missing_categories"]) == 0

    counts = audit_2["category_counts"]
    for cat in CANONICAL_CATEGORIES:
        assert cat in counts
        assert counts[cat] > 0

    assert sum(counts.values()) == 836


def test_audit_original_30_data_inclusion():
    """Verifies all 30 original screenshots are preserved in the expanded training pool."""
    with open(JSON_OUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    audit_3 = data["audit_3_original_30_data"]
    assert audit_3["expected_original_count"] == 30
    assert audit_3["found_original_count"] == 30
    assert audit_3["are_all_30_included"] is True


def test_audit_external_60_zero_leakage():
    """Verifies zero data leakage between 836 training pool and frozen 60 external benchmark."""
    with open(JSON_OUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    audit_4 = data["audit_4_external_60_leakage"]
    assert audit_4["total_external_images_checked"] == 60
    assert audit_4["filename_overlap_count"] == 0
    assert audit_4["exact_sha256_duplicates_count"] == 0
    assert audit_4["near_duplicates_count"] == 0
    assert audit_4["has_data_leakage"] is False


def test_audit_production_safety():
    """Verifies zero production code, ChromaDB, or Git commit modifications were made."""
    with open(JSON_OUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    audit_10 = data["audit_10_production_safety"]
    assert audit_10["production_ocr_code_modified"] is False
    assert audit_10["production_storage_code_modified"] is False
    assert audit_10["chromadb_modified"] is False
    assert audit_10["external_60_benchmark_modified"] is False
    assert audit_10["production_model_replaced"] is False
    assert audit_10["git_commit_or_push_performed"] is False
    assert audit_10["model_training_performed"] is False
