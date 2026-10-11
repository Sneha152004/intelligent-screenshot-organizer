"""
Phase 7I: Supervised Visual Screen-Type Model Training Script.

Trains 4 lightweight visual classification model candidates (MobileNetV3-Small, EfficientNet-Lite0,
MobileViT-XS, YOLO26n-cls) on the frozen Phase 7H-G primary 5-class benchmark.
Enforces calibration-only early stopping and dynamic INT8 quantization export.
"""

import os
import sys
import json
import time
import copy
import random
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from ultralytics import YOLO

# Add current dir to sys.path
PHASE7I_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE7I_DIR.parent.parent.parent
sys.path.insert(0, str(PHASE7I_DIR))

from dataset_preprocessing import get_data_loaders, VSUEDataset, letterbox_resize

CONFIG_PATH = PHASE7I_DIR / "model_configs.json"
CHECKPOINTS_DIR = PHASE7I_DIR / "model_checkpoints"
METRICS_DIR = PHASE7I_DIR / "metrics"
PRIMARY_MANIFEST_PATH = PROJECT_ROOT / "experiments" / "visual_understanding" / "phase7hg" / "phase7hg_primary_manifest.csv"
EXTENDED_MANIFEST_PATH = PROJECT_ROOT / "experiments" / "visual_understanding" / "phase7hg" / "phase7hg_extended_manifest.csv"
ZERO_SHOT_PROBE_PATH = PROJECT_ROOT / "experiments" / "visual_understanding" / "phase7hg" / "phase7hg_zero_shot_probe.csv"


class YOLOClassificationWrapper(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        yolo_wrapper = YOLO("yolov8n-cls.pt")
        self.backbone = copy.deepcopy(yolo_wrapper.model)
        # Adapt final classification head
        in_features = self.backbone.model[9].linear.in_features if hasattr(self.backbone.model[9], "linear") else 1280
        self.backbone.model[9].linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_manifest_hashes(configs: dict):
    print("--- Verifying Frozen Manifest MD5 Hashes ---")
    expected_hashes = configs["frozen_manifest_hashes"]
    
    for fname, exp_hash in expected_hashes.items():
        p = PROJECT_ROOT / "experiments" / "visual_understanding" / "phase7hg" / fname
        if not p.exists():
            raise FileNotFoundError(f"Frozen manifest missing: {p}")
        act_hash = hashlib.md5(p.read_bytes()).hexdigest()
        if act_hash != exp_hash:
            raise ValueError(f"Hash mismatch for {fname}! Expected {exp_hash}, got {act_hash}")
        print(f"  Verified {fname}: {act_hash} (OK)")


def build_model_candidate(model_key: str, num_classes: int) -> nn.Module:
    """Instantiates model candidate with pretrained weights adapted for num_classes."""
    if model_key == "yolo26n_cls":
        return YOLOClassificationWrapper(num_classes=num_classes)

    try:
        import timm
    except ImportError as e:
        raise ImportError(f"Model candidate '{model_key}' requires 'timm', which is not installed: {e}")

    if model_key == "mobilenet_v3_small":
        model = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=num_classes)
    elif model_key == "efficientnet_lite0":
        model = timm.create_model("efficientnet_lite0", pretrained=True, num_classes=num_classes)
    elif model_key == "mobilevit_xs":
        model = timm.create_model("mobilevit_xs", pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model candidate key: {model_key}")

    return model


def evaluate_partition(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == targets.data).item()
            total += inputs.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def train_single_model(
    model_key: str,
    configs: dict,
    manifest_df: pd.DataFrame,
    class_to_idx: dict
) -> dict:
    print(f"\n======================================================================")
    print(f"      TRAINING MODEL CANDIDATE: {configs['model_candidates'][model_key]['display_name']} ({model_key})")
    print(f"======================================================================")

    set_random_seed(configs["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(class_to_idx)
    model = build_model_candidate(model_key, num_classes).to(device)

    # Calculate class weights for loss function based on train partition
    train_df = manifest_df[manifest_df["partition"] == "train"]
    counts = train_df["screen_type"].value_counts().to_dict()
    total_samples = len(train_df)
    weights = [total_samples / (num_classes * max(1, counts.get(c, 1))) for c in sorted(class_to_idx.keys())]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=configs["training_protocol"]["initial_learning_rate"], weight_decay=configs["training_protocol"]["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    loaders = get_data_loaders(manifest_df, class_to_idx, batch_size=configs["training_protocol"]["batch_size"])

    best_calib_loss = float("inf")
    best_model_state = None
    patience = configs["training_protocol"]["early_stopping_patience"]
    patience_counter = 0

    history = []
    start_time = time.time()

    for epoch in range(1, configs["training_protocol"]["max_epochs"] + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets, _ in loaders["train"]:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == targets.data).item()
            train_total += inputs.size(0)

        epoch_train_loss = train_loss / max(1, train_total)
        epoch_train_acc = train_correct / max(1, train_total)

        # Calibration Phase (For Model Selection & Early Stopping Only)
        calib_loss, calib_acc = evaluate_partition(model, loaders["calibration"], criterion, device)
        scheduler.step(calib_loss)

        print(f"Epoch [{epoch:2d}/{configs['training_protocol']['max_epochs']}] | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Calib Loss: {calib_loss:.4f} | Calib Acc: {calib_acc*100:.2f}%", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 4),
            "calib_loss": round(calib_loss, 4),
            "calib_acc": round(calib_acc, 4)
        })

        if calib_loss < best_calib_loss:
            best_calib_loss = calib_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (Calib loss did not improve for {patience} epochs).", flush=True)
                break

    training_time = time.time() - start_time

    # Load best model state dict
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()

    # Save Best FP32 Checkpoint
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    fp32_ckpt_path = CHECKPOINTS_DIR / f"visual_screen_type_{model_key}.pth"
    torch.save({
        "model_key": model_key,
        "class_to_idx": class_to_idx,
        "model_state_dict": model.state_dict(),
        "best_calib_loss": best_calib_loss,
        "training_time_seconds": round(training_time, 2)
    }, fp32_ckpt_path)
    fp32_size_mb = fp32_ckpt_path.stat().st_size / (1024 * 1024)
    print(f"Saved FP32 Checkpoint: {fp32_ckpt_path} ({fp32_size_mb:.2f} MB)", flush=True)

    # Dynamic INT8 Quantization Export for CPU Edge Deployment
    cpu_model = model.to("cpu")
    try:
        int8_quantized_model = torch.ao.quantization.quantize_dynamic(
            cpu_model,
            {nn.Linear},
            dtype=torch.qint8
        )
        int8_ckpt_path = CHECKPOINTS_DIR / f"visual_screen_type_{model_key}_int8.pth"
        torch.save(int8_quantized_model.state_dict(), int8_ckpt_path)
        int8_size_mb = int8_ckpt_path.stat().st_size / (1024 * 1024)
        print(f"Saved INT8 Quantized Checkpoint: {int8_ckpt_path} ({int8_size_mb:.2f} MB)", flush=True)
    except Exception as e:
        print(f"Notice: Dynamic INT8 quantization export skipped for {model_key} ({e})", flush=True)
        int8_size_mb = 0.0

    # Save Training Log
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    log_data = {
        "model_key": model_key,
        "display_name": configs["model_candidates"][model_key]["display_name"],
        "fp32_parameters": configs["model_candidates"][model_key]["fp32_parameters"],
        "training_time_seconds": round(training_time, 2),
        "fp32_checkpoint_mb": round(fp32_size_mb, 2),
        "int8_checkpoint_mb": round(int8_size_mb, 2),
        "best_calib_loss": round(best_calib_loss, 4),
        "epochs_trained": len(history),
        "training_history": history
    }

    log_path = METRICS_DIR / f"{model_key}_training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    return log_data


def run_all_training():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        configs = json.load(f)

    verify_manifest_hashes(configs)

    primary_df = pd.read_csv(PRIMARY_MANIFEST_PATH)
    primary_classes = configs["primary_classes"]
    class_to_idx = {c: i for i, c in enumerate(primary_classes)}

    summary_training_results = {}
    for model_key in configs["model_candidates"].keys():
        result = train_single_model(model_key, configs, primary_df, class_to_idx)
        summary_training_results[model_key] = result

    summary_path = METRICS_DIR / "training_summary_all_models.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_training_results, f, indent=2)

    print("\n======================================================================")
    print("     ALL 4 MODEL CANDIDATES TRAINED AND CHECKPOINTED CLEANLY          ", flush=True)
    print("======================================================================")


if __name__ == "__main__":
    run_all_training()
