"""
Dataset Split Generator for Visual Model Research.
Creates a reproducible, stratified 60 / 20 / 20 (Train / Calibration / Validation) split
on the local dataset (data/metadata_v2.csv and data/screenshots/).

STRICT LEAKAGE PREVENTION:
- No validation screenshot IDs are used in training or calibration.
- External V1 and External V2 benchmarks are completely excluded and untouched.
"""

from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "metadata_v2.csv"
SCREENSHOTS_DIR = PROJECT_ROOT / "data" / "screenshots"
SPLIT_JSON_PATH = Path(__file__).resolve().parent / "dataset_split.json"

CANONICAL_CATEGORIES = [
    "Education",
    "Shopping",
    "Payment",
    "Travel",
    "Communication",
    "Entertainment",
    "Work & Career",
    "Connectivity",
    "General",
]


def create_dataset_splits(seed: int = 42):
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file '{METADATA_PATH}' not found.")

    df = pd.read_csv(METADATA_PATH)

    # Filter for valid existing screenshots and canonical categories
    valid_rows = []
    for idx, row in df.iterrows():
        fname = row["file"]
        cat = row["category"]
        img_path = SCREENSHOTS_DIR / fname

        if img_path.exists() and img_path.is_file() and img_path.stat().st_size > 0:
            if cat in CANONICAL_CATEGORIES:
                valid_rows.append({"file": fname, "category": cat, "path": str(img_path)})

    valid_df = pd.DataFrame(valid_rows)
    print(f"Total valid screenshots in dataset: {len(valid_df)}")

    # 60% Train, 20% Calibration, 20% Validation
    train_cal_df, val_df = train_test_split(
        valid_df, test_size=0.20, random_state=seed, stratify=valid_df["category"]
    )
    train_df, cal_df = train_test_split(
        train_cal_df, test_size=0.25, random_state=seed, stratify=train_cal_df["category"]
    )

    splits = {
        "random_seed": seed,
        "total_images": len(valid_df),
        "canonical_categories": CANONICAL_CATEGORIES,
        "train_ids": train_df["file"].tolist(),
        "calibration_ids": cal_df["file"].tolist(),
        "validation_ids": val_df["file"].tolist(),
    }

    SPLIT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(f"Dataset split generated successfully:")
    print(f"  Train Set:       {len(train_df)} images (60%)")
    print(f"  Calibration Set: {len(cal_df)} images (20%)")
    print(f"  Validation Set:  {len(val_df)} images (20%)")
    print(f"Saved split metadata to: {SPLIT_JSON_PATH}")


if __name__ == "__main__":
    create_dataset_splits()
