"""
Dynamic Dataset Loader for Screenshot Classification & Intelligence Experiments
=============================================================================
Dynamically loads screenshot dataset records from data/metadata.csv and data/screenshots/.

Features:
- Loads full 836 dataset records with normalized category names (canonical 10 categories).
- Identifies and excludes invalid/unreadable zero-byte images (ss_441.png, ss_443.png, ss_560.png),
  returning exactly 833 valid samples for model training and feature extraction.
- Provides dynamic Stratified 5-Fold Cross-Validation splits.
- Does NOT rely on legacy hardcoded 30-sample arrays, DEV_TARGETS, or UNSEEN_GROUND_TRUTH_ANNOTATIONS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
METADATA_CSV = DATA_DIR / "metadata.csv"

CANONICAL_CATEGORIES = [
    "Coding",
    "Communication",
    "Connectivity",
    "Education",
    "Entertainment",
    "General",
    "Payment",
    "Shopping",
    "Travel",
    "Work & Career",
]

INVALID_ZERO_BYTE_FILES = {"ss_441.png", "ss_443.png", "ss_560.png"}


def load_dataset(
    csv_path: Path = METADATA_CSV,
    screenshots_dir: Path = SCREENSHOTS_DIR,
    exclude_invalid: bool = True,
) -> List[Dict[str, Any]]:
    """
    Loads dataset samples dynamically from metadata CSV and screenshots directory.

    Args:
        csv_path: Path to metadata.csv
        screenshots_dir: Path to screenshots/ directory
        exclude_invalid: If True, excludes the 3 unreadable/zero-byte images (ss_441.png, ss_443.png, ss_560.png),
                        returning 833 valid samples. If False, returns all 836 samples.

    Returns:
        List of sample dicts containing 'filename', 'screenshot_id', 'image_path', 'category', 'confidence', 'notes', 'is_valid'.
    """
    df = pd.read_csv(csv_path)
    samples = []

    for _, row in df.iterrows():
        fname = str(row["file"]).strip()
        category = str(row["category"]).strip()
        if category == "Work and career":
            category = "Work & Career"

        img_path = screenshots_dir / fname

        # Check validity: zero-byte file or unreadable
        is_zero_byte = fname in INVALID_ZERO_BYTE_FILES or (img_path.exists() and img_path.stat().st_size == 0)
        is_valid = img_path.exists() and not is_zero_byte

        if exclude_invalid and not is_valid:
            continue

        confidence = str(row.get("confidence", "high")).strip().lower()
        notes = str(row.get("notes", "")).strip() if pd.notna(row.get("notes")) else ""

        samples.append({
            "filename": fname,
            "screenshot_id": Path(fname).stem,
            "image_path": str(img_path),
            "category": category,
            "confidence": confidence,
            "notes": notes,
            "is_valid": is_valid,
        })

    return samples


def get_stratified_kfold_splits(
    samples: List[Dict[str, Any]],
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """
    Generates reproducible Stratified K-Fold Cross-Validation splits for input samples.

    Returns:
        List of (train_indices, val_indices) tuples.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    labels = [s["category"] for s in samples]
    dummy_x = list(range(len(samples)))

    splits = []
    for train_idx, val_idx in skf.split(dummy_x, labels):
        splits.append((train_idx.tolist(), val_idx.tolist()))

    return splits
