"""
Unit test suite for Dynamic Dataset Loader (experiments/intelligence/dataset_loader.py)
========================================================================================
Verifies:
1. load_dataset(exclude_invalid=True) returns exactly 833 valid samples.
2. load_dataset(exclude_invalid=False) returns all 836 labeled records.
3. Invalid zero-byte files (ss_441.png, ss_443.png, ss_560.png) are accurately flagged and excluded when requested.
4. All sample categories are normalized to the canonical 10-category taxonomy.
5. get_stratified_kfold_splits produces 5 balanced Stratified K-Fold splits on 833 valid samples.
"""

import pytest
from pathlib import Path
from experiments.intelligence.dataset_loader import (
    load_dataset,
    get_stratified_kfold_splits,
    CANONICAL_CATEGORIES,
    INVALID_ZERO_BYTE_FILES,
)


def test_load_dataset_valid_and_total_counts():
    """Verifies valid dataset count is 833 and total record count is 836."""
    valid_samples = load_dataset(exclude_invalid=True)
    all_samples = load_dataset(exclude_invalid=False)

    assert len(valid_samples) == 833
    assert len(all_samples) == 836
    assert len(all_samples) - len(valid_samples) == 3


def test_invalid_zero_byte_files_flagged():
    """Verifies zero-byte files (ss_441.png, ss_443.png, ss_560.png) are flagged as is_valid=False."""
    all_samples = load_dataset(exclude_invalid=False)
    invalid_samples = [s for s in all_samples if not s["is_valid"]]

    assert len(invalid_samples) == 3
    invalid_filenames = set(s["filename"] for s in invalid_samples)
    assert invalid_filenames == INVALID_ZERO_BYTE_FILES

    for s in invalid_samples:
        assert s["is_valid"] is False


def test_canonical_10_categories_normalization():
    """Verifies all categories belong to the 10 canonical category taxonomy."""
    all_samples = load_dataset(exclude_invalid=False)
    sample_categories = set(s["category"] for s in all_samples)

    assert len(sample_categories) == 10
    assert sample_categories == set(CANONICAL_CATEGORIES)


def test_stratified_kfold_splits():
    """Verifies get_stratified_kfold_splits produces 5 balanced folds on 833 samples."""
    valid_samples = load_dataset(exclude_invalid=True)
    splits = get_stratified_kfold_splits(valid_samples, n_splits=5, random_state=42)

    assert len(splits) == 5

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        assert len(train_idx) + len(val_idx) == 833
        assert len(set(train_idx) & set(val_idx)) == 0  # Disjoint
        assert 165 <= len(val_idx) <= 170  # ~166-167 val samples per fold
