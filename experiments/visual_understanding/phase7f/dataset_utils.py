import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = r"C:\Users\KIIT0001\Documents\intelligent-screenshot-organizer"
PHASE7F_DIR = os.path.join(PROJECT_ROOT, "experiments", "visual_understanding", "phase7f")

# CANONICAL TAXONOMY (13 CLASSES)
FULL_TAXONOMY = [
    "product_listing",
    "product_detail",
    "cart_checkout_payment",
    "login_authentication",
    "messaging_chat",
    "email_view",
    "media_player_library",
    "code_editor_terminal",
    "document_article_view",
    "travel_booking",
    "dashboard_settings",
    "error_dialog_alert",
    "other_screen",
]

# 5 SUPPORTED CLASSES IN THIS EXPERIMENT
SUPPORTED_CLASSES = [
    "document_article_view",
    "product_detail",
    "code_editor_terminal",
    "dashboard_settings",
    "cart_checkout_payment",
]

UNSUPPORTED_CLASSES = [c for c in FULL_TAXONOMY if c not in SUPPORTED_CLASSES]

CLASS_TO_IDX = {cat: i for i, cat in enumerate(SUPPORTED_CLASSES)}
IDX_TO_CLASS = {i: cat for i, cat in enumerate(SUPPORTED_CLASSES)}


def letterbox_pad_and_resize(image: Image.Image, target_size: int = 224, bg_color=(240, 240, 240)) -> Image.Image:
    """
    Safely resizes image while preserving original aspect ratio by applying letterbox padding.
    Handles RGBA transparency by compositing onto bg_color.
    """
    if image.mode in ("RGBA", "LA", "P"):
        canvas = Image.new("RGB", image.size, bg_color)
        canvas.paste(image, mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None)
        image = canvas
    elif image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    max_dim = max(w, h)
    scale = target_size / float(max_dim)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized_img = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    new_canvas = Image.new("RGB", (target_size, target_size), bg_color)
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    new_canvas.paste(resized_img, (pad_left, pad_top))
    return new_canvas


class ScreenshotScreenTypeDataset(Dataset):
    """
    PyTorch Dataset for Phase 7F Visual Screen-Type Classification.
    Loads images deterministically, applies letterbox padding, and maps labels to 5 supported classes.
    """

    def __init__(self, manifest_df: pd.DataFrame, split_ids: list, is_training: bool = False, target_size: int = 224):
        self.manifest_df = manifest_df[manifest_df["screenshot_id"].isin(split_ids)].copy()
        # Filter for supported classes only
        self.manifest_df = self.manifest_df[self.manifest_df["screen_type"].isin(SUPPORTED_CLASSES)].reset_index(drop=True)
        self.is_training = is_training
        self.target_size = target_size

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.to_tensor = transforms.ToTensor()

        # Mild training augmentations
        self.color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)

    def __len__(self):
        return len(self.manifest_df)

    def __getitem__(self, idx):
        row = self.manifest_df.iloc[idx]
        sid = row["screenshot_id"]
        rel_path = row["image_path"]
        stype = row["screen_type"]
        full_path = os.path.join(PROJECT_ROOT, rel_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")

        image = Image.open(full_path)
        orig_w, orig_h = image.size
        aspect_ratio = round(orig_w / float(orig_h), 3)
        orientation = "portrait" if orig_h > orig_w else ("landscape" if orig_w > orig_h else "square")

        # Process image with letterbox padding
        padded_img = letterbox_pad_and_resize(image, target_size=self.target_size)

        if self.is_training:
            padded_img = self.color_jitter(padded_img)

        tensor_img = self.normalize(self.to_tensor(padded_img))
        class_idx = CLASS_TO_IDX[stype]

        metadata_dict = {
            "screenshot_id": sid,
            "filename": sid,
            "screen_type": stype,
            "aspect_ratio": aspect_ratio,
            "orientation": orientation,
            "orig_size": (orig_w, orig_h)
        }

        return tensor_img, class_idx, metadata_dict


def calculate_inverse_class_weights(manifest_df: pd.DataFrame, train_ids: list, device: torch.device) -> torch.Tensor:
    """
    Computes inverse class weights exclusively from the training partition data.
    Formula: W_c = N_train / (K * N_train,c)
    """
    train_df = manifest_df[manifest_df["screenshot_id"].isin(train_ids)].copy()
    train_df = train_df[train_df["screen_type"].isin(SUPPORTED_CLASSES)]
    counts = train_df["screen_type"].value_counts().to_dict()

    n_train = len(train_df)
    k = len(SUPPORTED_CLASSES)
    weights = []

    for cat in SUPPORTED_CLASSES:
        n_c = counts.get(cat, 1)
        w_c = n_train / (k * float(n_c))
        weights.append(w_c)

    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return weights_tensor
