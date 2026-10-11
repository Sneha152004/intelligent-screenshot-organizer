import json
import os
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

PROJECT_ROOT = r"C:\Users\KIIT0001\Documents\intelligent-screenshot-organizer"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PHASE7F_DIR = os.path.join(PROJECT_ROOT, "experiments", "visual_understanding", "phase7f")
os.makedirs(PHASE7F_DIR, exist_ok=True)

from experiments.visual_understanding.phase7f.dataset_utils import (
    ScreenshotScreenTypeDataset,
    calculate_inverse_class_weights,
    SUPPORTED_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
)
from experiments.visual_understanding.phase7f.vsue_model import VisualScreenTypeClassifier


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training():
    print("======================================================================")
    print("     PHASE 7F: VSUE MOBILENETV3-SMALL BASELINE TRAINING               ")
    print("======================================================================")

    set_seed(42)
    device = torch.device("cpu")
    print(f"Execution Device: {device}")

    # Paths to Phase 7E artifacts
    manifest_path = os.path.join(PROJECT_ROOT, "experiments", "visual_understanding", "screen_type_manifest_final.csv")
    split_path = os.path.join(PROJECT_ROOT, "experiments", "visual_understanding", "vsue_dataset_split_final.json")

    manifest_df = pd.read_csv(manifest_path)
    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    train_ids = split_data["train_ids"]
    cal_ids = split_data["calibration_ids"]
    val_ids = split_data["validation_ids"]

    print(f"Dataset Split Loaded:")
    print(f"  Training Split:    {len(train_ids)} screenshots")
    print(f"  Calibration Split: {len(cal_ids)} screenshots")
    print(f"  Validation Split:  {len(val_ids)} screenshots (HELD-OUT)")

    # Datasets and Loaders
    train_dataset = ScreenshotScreenTypeDataset(manifest_df, train_ids, is_training=True)
    cal_dataset = ScreenshotScreenTypeDataset(manifest_df, cal_ids, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    cal_loader = DataLoader(cal_dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f"Supported 5-Class Training Samples:    {len(train_dataset)}")
    print(f"Supported 5-Class Calibration Samples: {len(cal_dataset)}")

    # Class weights calculated exclusively from training partition
    class_weights = calculate_inverse_class_weights(manifest_df, train_ids, device)
    print("Computed Inverse Class Weights (Training Partition Only):")
    for i, cat in enumerate(SUPPORTED_CLASSES):
        print(f"  {cat:<23}: {class_weights[i].item():.4f}")

    # Instantiate Model
    model = VisualScreenTypeClassifier(num_classes=len(SUPPORTED_CLASSES), pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    num_epochs = 25
    patience = 7
    best_cal_f1 = -1.0
    best_epoch = -1
    best_model_state = None
    epochs_no_improve = 0

    history = []

    print("\n--- Starting Controlled Training Loop ---")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for imgs, targets, _ in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(targets)
            preds = outputs.argmax(dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(targets.cpu().numpy())

        train_loss /= len(train_dataset)
        train_acc = accuracy_score(train_targets, train_preds)

        # Evaluate Calibration Partition ONLY (for model selection)
        model.eval()
        cal_loss = 0.0
        cal_preds, cal_targets = [], []

        with torch.no_grad():
            for imgs, targets, _ in cal_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)

                cal_loss += loss.item() * len(targets)
                preds = outputs.argmax(dim=1).cpu().numpy()
                cal_preds.extend(preds)
                cal_targets.extend(targets.cpu().numpy())

        cal_loss /= len(cal_dataset)
        cal_acc = accuracy_score(cal_targets, cal_preds)
        cal_f1 = f1_score(cal_targets, cal_preds, average="macro", zero_division=0)

        scheduler.step(cal_f1)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "cal_loss": round(cal_loss, 4),
            "cal_acc": round(cal_acc, 4),
            "cal_macro_f1": round(cal_f1, 4),
            "lr": optimizer.param_groups[0]["lr"]
        })

        print(f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | Cal Loss: {cal_loss:.4f} Acc: {cal_acc*100:5.2f}% Macro F1: {cal_f1*100:5.2f}%")

        if cal_f1 > best_cal_f1:
            best_cal_f1 = cal_f1
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at Epoch {epoch} (Best Calibration Macro F1: {best_cal_f1*100:.2f}% at Epoch {best_epoch}).")
                break

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds.")
    print(f"Best Checkpoint Selected at Epoch {best_epoch} with Calibration Macro F1: {best_cal_f1*100:.2f}%.")

    # Save Best Checkpoint
    checkpoint_path = os.path.join(PHASE7F_DIR, "visual_screen_type_model.pth")
    torch.save(best_model_state, checkpoint_path)
    print(f"Saved best model checkpoint to: {checkpoint_path}")

    # Save Model Configuration
    config = {
        "architecture": "MobileNetV3-Small",
        "pretrained_weights": "MobileNet_V3_Small_Weights.DEFAULT (ImageNet1K_V1)",
        "input_dimensions": [3, 224, 224],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "supported_classes": SUPPORTED_CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "class_weights_inverse_freq": [round(w, 4) for w in class_weights.cpu().numpy().tolist()],
        "hyperparameters": {
            "random_seed": 42,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 16,
            "epochs_trained": len(history),
            "best_epoch": best_epoch,
            "early_stopping_patience": patience
        },
        "best_epoch_calibration_metrics": {
            "best_epoch": best_epoch,
            "cal_macro_f1": round(best_cal_f1, 4)
        },
        "training_history": history
    }

    config_path = os.path.join(PHASE7F_DIR, "model_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved model configuration to: {config_path}")

    print("=== TRAINING COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_training()
