"""
Phase 7I: Dataset Preprocessing and Data Loading Module.

Implements aspect-ratio-preserving letterboxing, RGBA-to-RGB alpha composite handling,
and ImageNet normalization for lightweight visual screen-type model benchmarking.
"""

import os
import io
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCREENSHOTS_DIR = PROJECT_ROOT / "data" / "screenshots"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def letterbox_resize(
    img: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    fill_color: Tuple[int, int, int] = (128, 128, 128)
) -> Image.Image:
    """Resizes an image while preserving its aspect ratio by adding letterbox/pillarbox padding."""
    # Convert RGBA to RGB if necessary
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    target_w, target_h = target_size

    # Calculate scale factor preserving aspect ratio
    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Create target canvas with neutral background fill
    canvas = Image.new("RGB", target_size, fill_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized_img, (paste_x, paste_y))

    return canvas


class VSUEDataset(Dataset):
    """PyTorch Dataset loading visual screen-type screenshots using deterministic letterbox preprocessing."""
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        class_to_idx: Dict[str, int],
        target_size: Tuple[int, int] = (224, 224),
        fill_color: Tuple[int, int, int] = (128, 128, 128),
        is_training: bool = False
    ):
        self.df = manifest_df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.target_size = target_size
        self.fill_color = fill_color
        self.is_training = is_training

        # Standard ImageNet normalization transform
        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        fname = row["filename"]
        stype = row["screen_type"]
        img_path = SCREENSHOTS_DIR / fname

        if not img_path.exists():
            raise FileNotFoundError(f"Screenshot image file missing at {img_path}")

        try:
            with Image.open(img_path) as img:
                img.load()
                processed_img = letterbox_resize(img, target_size=self.target_size, fill_color=self.fill_color)
        except Exception as e:
            raise IOError(f"Error loading screenshot {fname}: {e}")

        tensor_img = self.norm_transform(processed_img)
        label_idx = self.class_to_idx.get(stype, -1)

        return tensor_img, label_idx, fname


def get_data_loaders(
    manifest_df: pd.DataFrame,
    class_to_idx: Dict[str, int],
    batch_size: int = 16,
    num_workers: int = 0,
    target_size: Tuple[int, int] = (224, 224)
) -> Dict[str, DataLoader]:
    """Creates PyTorch DataLoaders for Train, Calibration, and Validation partitions."""
    loaders = {}
    for p in ["train", "calibration", "validation"]:
        p_df = manifest_df[manifest_df["partition"] == p]
        if p_df.empty:
            continue

        dataset = VSUEDataset(
            manifest_df=p_df,
            class_to_idx=class_to_idx,
            target_size=target_size,
            is_training=(p == "train")
        )
        shuffle = (p == "train")
        loaders[p] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    return loaders


if __name__ == "__main__":
    print("Testing VSUEDataset and letterbox preprocessing...")
    manifest_path = PROJECT_ROOT / "experiments" / "visual_understanding" / "phase7hg" / "phase7hg_primary_manifest.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        classes = sorted(df["screen_type"].unique())
        c2i = {c: i for i, c in enumerate(classes)}
        ds = VSUEDataset(df[df["partition"] == "train"], class_to_idx=c2i)
        img_t, lbl, fn = ds[0]
        print(f"Sample Image Tensor Shape: {img_t.shape}, Label: {lbl}, Filename: {fn}")
        print("Preprocessing module test passed.")
