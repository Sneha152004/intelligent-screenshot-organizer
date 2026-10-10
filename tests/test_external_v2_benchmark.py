"""
Unit Tests for External V2 Benchmark Structure, Populated Manifest & Governance Integrity
========================================================================================
Verifies:
1. Exactly 244 image records exist.
2. IDs are ext2_001 through ext2_244 sequentially.
3. Every manifest image exists on disk and is non-empty.
4. No duplicate IDs.
5. No duplicate filenames.
6. All populated categories belong strictly to the 9 canonical categories.
7. Coding is strictly rejected as a top-level category.
8. Coding appears as a tag, distinct from category.
9. Every record has a valid non-empty label_reason.
10. Every record has a valid non-empty label_confidence (High/Medium/Low).
11. Benchmark loader does not import or invoke classifier modules.
12. Historical External V1 benchmark directory is untouched and contains 60 images.
13. No missing labels remain after labeling (allow_empty_categories=False passes).
14. All tags are stored separately from the category field.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

from experiments.intelligence.external_v2.evaluation_targets import (
    CANONICAL_9_CATEGORIES,
    DEFAULT_MANIFEST_PATH,
    EXPECTED_TOTAL_RECORDS,
    load_evaluation_targets,
    validate_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V1_EXTERNAL_DIR = PROJECT_ROOT / "experiments" / "intelligence" / "external"


def test_1_exactly_244_image_records_exist():
    """1. Verify exactly 244 image records exist in External V2."""
    records = load_evaluation_targets(allow_empty_categories=False)
    assert len(records) == EXPECTED_TOTAL_RECORDS == 244


def test_2_ids_are_ext2_001_through_ext2_244():
    """2. Verify IDs are ext2_001 through ext2_244 sequentially."""
    records = load_evaluation_targets(allow_empty_categories=False)
    for idx, rec in enumerate(records, 1):
        expected_id = f"ext2_{idx:03d}"
        assert rec.id == expected_id


def test_3_every_manifest_image_exists_on_disk():
    """3. Verify every manifest image path points to an existing non-empty file."""
    records = load_evaluation_targets(allow_empty_categories=False)
    for rec in records:
        assert rec.image_path.exists()
        assert rec.image_path.is_file()
        assert rec.image_path.stat().st_size > 0


def test_4_no_duplicate_ids():
    """4. Verify there are no duplicate IDs in the benchmark."""
    records = load_evaluation_targets(allow_empty_categories=False)
    ids = [rec.id for rec in records]
    assert len(ids) == len(set(ids)) == 244


def test_5_no_duplicate_filenames():
    """5. Verify there are no duplicate filenames in the benchmark."""
    records = load_evaluation_targets(allow_empty_categories=False)
    filenames = [rec.filename for rec in records]
    assert len(filenames) == len(set(filenames)) == 244


def test_6_all_categories_belong_to_canonical_9_taxonomy():
    """6. Verify all populated categories are one of the 9 canonical categories."""
    records = load_evaluation_targets(allow_empty_categories=False)
    for rec in records:
        assert rec.category in CANONICAL_9_CATEGORIES


def test_7_coding_cannot_appear_as_category(tmp_path):
    """7. Verify Coding is strictly rejected as a top-level category."""
    invalid_csv = tmp_path / "invalid_manifest.csv"
    invalid_csv.write_text(
        "id,filename,image_path,category,tags,label_confidence,label_reason,source\n"
        "ext2_001,ext2_001.jpg,images/ext2_001.jpg,Coding,Python,High,Test,UNKNOWN\n"
        + "".join([f"ext2_{i:03d},ext2_{i:03d}.jpg,images/ext2_{i:03d}.jpg,General,,High,Test,UNKNOWN\n" for i in range(2, 245)]),
        encoding="utf-8",
    )
    (tmp_path / "images").mkdir(exist_ok=True)
    for i in range(1, 245):
        (tmp_path / "images" / f"ext2_{i:03d}.jpg").write_bytes(b"fake_image_bytes")

    with pytest.raises(ValueError, match="Coding is NOT a category"):
        validate_manifest(manifest_path=invalid_csv, allow_empty_categories=False, expected_count=244)


def test_8_coding_may_appear_as_tag_separately():
    """8. Verify Coding appears as a tag distinct from the category field."""
    records = load_evaluation_targets(allow_empty_categories=False)
    coding_tagged_records = [rec for rec in records if "Coding" in rec.tags]
    assert len(coding_tagged_records) > 0
    for rec in coding_tagged_records:
        assert rec.category != "Coding"
        assert rec.category in CANONICAL_9_CATEGORIES


def test_9_every_record_has_non_empty_label_reason():
    """9. Verify every record in the manifest has a descriptive label_reason."""
    records = load_evaluation_targets(allow_empty_categories=False)
    for rec in records:
        assert rec.label_reason != ""
        assert len(rec.label_reason) > 5
        assert "model" not in rec.label_reason.lower()


def test_10_every_record_has_valid_label_confidence():
    """10. Verify every record has a label_confidence value of High, Medium, or Low."""
    records = load_evaluation_targets(allow_empty_categories=False)
    valid_confidences = {"High", "Medium", "Low"}
    for rec in records:
        assert rec.label_confidence in valid_confidences


def test_11_loader_does_not_invoke_classifier():
    """11. Verify loading evaluation targets does not import or invoke classifier modules."""
    initial_modules = set(sys.modules.keys())
    records = load_evaluation_targets(allow_empty_categories=False)
    assert len(records) == 244
    new_modules = set(sys.modules.keys()) - initial_modules

    classifier_module_keywords = ["classifier", "minilm", "easyocr", "model"]
    for mod in new_modules:
        for kw in classifier_module_keywords:
            assert kw not in mod.lower(), f"Loader unexpectedly imported classifier module: {mod}"


def test_12_v1_external_directory_unmodified():
    """12. Verify the historical External V1 benchmark directory exists and is unchanged."""
    assert V1_EXTERNAL_DIR.exists()
    v1_images_dir = V1_EXTERNAL_DIR / "images"
    assert v1_images_dir.exists()
    v1_images = list(v1_images_dir.glob("*.jpg")) + list(v1_images_dir.glob("*.png"))
    assert len(v1_images) == 60


def test_13_no_missing_labels_remain_after_labeling():
    """13. Verify no category or label field is left blank in manifest.csv."""
    df = pd.read_csv(DEFAULT_MANIFEST_PATH)
    assert len(df) == 244
    assert df["category"].isnull().sum() == 0
    assert (df["category"].str.strip() == "").sum() == 0
    assert df["label_confidence"].isnull().sum() == 0
    assert (df["label_confidence"].str.strip() == "").sum() == 0
    assert df["label_reason"].isnull().sum() == 0
    assert (df["label_reason"].str.strip() == "").sum() == 0


def test_14_tags_are_stored_separately_from_category():
    """14. Verify tags are parsed into a distinct list property on records."""
    records = load_evaluation_targets(allow_empty_categories=False)
    for rec in records:
        assert isinstance(rec.tags, list)
        assert rec.category not in rec.tags
