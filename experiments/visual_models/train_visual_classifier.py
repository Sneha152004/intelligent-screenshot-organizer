"""
Training Script for Lightweight Candidate Vision Models.
Trains 9-class screenshot classification heads on frozen/fine-tuned CNN feature extractors
using the 499-image training partition and 167-image calibration partition.

STRICT LEAKAGE PREVENTION:
- Training and early-stopping checkpoint selection rely ONLY on Train & Calibration splits.
- The 167-image validation partition is strictly isolated and held out.
- External V1 and External V2 benchmarks are 100% untouched.
"""

from pathlib import Path
import sys

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from experiments.visual_models.model_registry import (
    CNNVisualClassifier,
    CANONICAL_CATEGORIES,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPLIT_JSON_PATH = Path(__file__).resolve().parent / "dataset_split.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

CATEGORY_TO_IDX = {cat: i for i, cat in enumerate(CANONICAL_CATEGORIES)}
IDX_TO_CATEGORY = {i: cat for i, cat in enumerate(CANONICAL_CATEGORIES)}


class ScreenshotDataset(Dataset):
    """PyTorch Dataset for Screenshot Images."""

    def __init__(self, metadata_df: pd.DataFrame, transform=None):
        self.df = metadata_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        category = row["category"]
        label = CATEGORY_TO_IDX[category]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label, row["file"]


def load_partition_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads metadata and splits into train, calibration, and validation DataFrames."""
    if not SPLIT_JSON_PATH.exists():
        from experiments.visual_models.dataset_split import create_dataset_splits
        create_dataset_splits()

    with open(SPLIT_JSON_PATH, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    metadata_path = PROJECT_ROOT / "data" / "metadata_v2.csv"
    df = pd.read_csv(metadata_path)
    df["path"] = df["file"].apply(lambda f: str(PROJECT_ROOT / "data" / "screenshots" / f))

    train_ids = set(split_data["train_ids"])
    cal_ids = set(split_data["calibration_ids"])
    val_ids = set(split_data["validation_ids"])

    train_df = df[df["file"].isin(train_ids)].copy()
    cal_df = df[df["file"].isin(cal_ids)].copy()
    val_df = df[df["file"].isin(val_ids)].copy()

    return train_df, cal_df, val_df


def train_model(
    backbone_name: str,
    epochs: int = 12,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Trains classification head for candidate model on train set,
    evaluating on calibration set for best checkpoint selection.
    """
    print(f"\n--- Training Candidate Model: {backbone_name} ---")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, cal_df, _ = load_partition_datasets()

    model = CNNVisualClassifier(backbone_name=backbone_name, num_classes=len(CANONICAL_CATEGORIES))
    model.to(device)

    train_dataset = ScreenshotDataset(train_df, transform=model.transform)
    cal_dataset = ScreenshotDataset(cal_df, transform=model.transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    cal_loader = DataLoader(cal_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

    best_cal_acc = 0.0
    best_checkpoint_path = RESULTS_DIR / f"{backbone_name}_best.pt"
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            total_train += images.size(0)

        train_acc = train_correct / total_train

        # Calibration evaluation
        model.eval()
        cal_correct = 0
        total_cal = 0
        with torch.no_grad():
            for images, labels, _ in cal_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                cal_correct += (preds == labels).sum().item()
                total_cal += images.size(0)

        cal_acc = cal_correct / total_cal
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Acc: {train_acc*100:.2f}% | Cal Acc: {cal_acc*100:.2f}%")

        if cal_acc > best_cal_acc:
            best_cal_acc = cal_acc
            torch.save(model.state_dict(), best_checkpoint_path)

    training_time_sec = time.perf_counter() - t0
    info = model.get_model_info()

    return {
        "backbone_name": backbone_name,
        "best_cal_acc": round(best_cal_acc, 4),
        "checkpoint_path": str(best_checkpoint_path),
        "training_time_sec": round(training_time_sec, 2),
        "model_info": info,
    }


def train_all_candidates(epochs: int = 10):
    candidates = ["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "shufflenet_v2", "mobilenet_v2"]
    results = {}
    for cand in candidates:
        res = train_model(cand, epochs=epochs)
        results[cand] = res
    
    summary_path = RESULTS_DIR / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll candidate models trained successfully. Summary saved to: {summary_path}")


if __name__ == "__main__":
    train_all_candidates()
