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
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ultralytics import YOLO

# ---------------------------------------------------------
# SETUP DIRECTORIES
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_MANIFEST_PATH = BASE_DIR / "dataset" / "final_13class_manifest.csv"
MODELS_DIR = BASE_DIR / "models"
BENCHMARKS_DIR = BASE_DIR / "benchmarks"
CALIBRATION_DIR = BASE_DIR / "calibration"
REPORTS_DIR = BASE_DIR / "reports"
DATA_SCREENSHOTS_DIR = Path("data/screenshots")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# DETERMINISTIC DATASET PREPROCESSING & DATASET CLASS
# ---------------------------------------------------------
def letterbox_resize(img: Image.Image, target_size: int = 224) -> Image.Image:
    """Letterbox resize with padding to target_size x target_size."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    
    img_resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
    new_img = Image.new('RGB', (target_size, target_size), (114, 114, 114))
    paste_x = (target_size - nw) // 2
    paste_y = (target_size - nh) // 2
    new_img.paste(img_resized, (paste_x, paste_y))
    return new_img

transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Final13ClassDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: dict):
        self.records = []
        self.class_to_idx = class_to_idx
        
        for _, row in df.iterrows():
            fn = str(row['filename'])
            img_path = DATA_SCREENSHOTS_DIR / fn
            st = str(row['screen_type'])
            if img_path.exists() and st in class_to_idx:
                self.records.append((str(img_path), class_to_idx[st], fn, row['screenshot_id']))
                
    def __len__(self):
        return len(self.records)
        
    def __getitem__(self, idx):
        img_path, label_idx, fn, ss_id = self.records[idx]
        try:
            with Image.open(img_path) as img:
                img_lb = letterbox_resize(img, 224)
                tensor_img = transform_pipeline(img_lb)
        except Exception:
            tensor_img = torch.zeros((3, 224, 224), dtype=torch.float32)
            
        return tensor_img, label_idx, fn, ss_id

# ---------------------------------------------------------
# YOLOV8N-CLS 13-CLASS MODEL WRAPPER
# ---------------------------------------------------------
class YOLO13ClassWrapper(nn.Module):
    def __init__(self, num_classes: int = 13):
        super().__init__()
        yolo_base = YOLO("yolov8n-cls.pt")
        self.backbone = copy.deepcopy(yolo_base.model)
        # Final classification linear head
        in_features = self.backbone.model[9].linear.in_features if hasattr(self.backbone.model[9], "linear") else 1280
        self.backbone.model[9].linear = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

# ---------------------------------------------------------
# TEMPERATURE SCALING MODEL FOR CALIBRATION
# ---------------------------------------------------------
class ModelWithTemperature(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, calib_loader: DataLoader, device: torch.device):
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, targets, _, _ in calib_loader:
                inputs = inputs.to(device)
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(targets)
        
        logits_all = torch.cat(logits_list).to(device)
        labels_all = torch.cat(labels_list).to(device)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits_all), labels_all)
            loss.backward()
            return loss

        optimizer.step(eval_step)

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies = preds == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)

# ---------------------------------------------------------
# MAIN TRAINING & EVALUATION PIPELINE
# ---------------------------------------------------------
def run_training_and_benchmarking():
    print("Starting Part C, D, E: 13-Class YOLO Model Training, Calibration & Benchmarking...")
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    # Load Manifest
    df_manifest = pd.read_csv(DATASET_MANIFEST_PATH)
    all_classes = sorted(df_manifest['screen_type'].unique())
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    
    print(f"Taxonomy Classes ({len(all_classes)}): {all_classes}")
    
    # DataLoaders
    train_df = df_manifest[df_manifest['partition'] == 'train']
    calib_df = df_manifest[df_manifest['partition'] == 'calibration']
    val_df = df_manifest[df_manifest['partition'] == 'validation']
    
    train_dataset = Final13ClassDataset(train_df, class_to_idx)
    calib_dataset = Final13ClassDataset(calib_df, class_to_idx)
    val_dataset = Final13ClassDataset(val_df, class_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Model Initialization
    model = YOLO13ClassWrapper(num_classes=len(all_classes)).to(device)
    
    # Class weights for loss
    counts = train_df['screen_type'].value_counts().to_dict()
    total_samples = len(train_df)
    weights = [total_samples / (len(all_classes) * max(1, counts.get(c, 1))) for c in all_classes]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    # Training Loop
    epochs = 15
    best_calib_loss = float('inf')
    best_state = None
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets, _, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == targets).item()
            train_total += inputs.size(0)
            
        train_epoch_loss = train_loss / max(1, train_total)
        train_epoch_acc = train_correct / max(1, train_total)
        
        # Evaluate Calib
        model.eval()
        calib_loss = 0.0
        calib_correct = 0
        calib_total = 0
        with torch.no_grad():
            for inputs, targets, _, _ in calib_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                calib_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                calib_correct += torch.sum(preds == targets).item()
                calib_total += inputs.size(0)
                
        calib_epoch_loss = calib_loss / max(1, calib_total)
        calib_epoch_acc = calib_correct / max(1, calib_total)
        
        scheduler.step(calib_epoch_loss)
        print(f"Epoch [{epoch:2d}/{epochs}] | Train Loss: {train_epoch_loss:.4f}, Acc: {train_epoch_acc*100:.2f}% | Calib Loss: {calib_epoch_loss:.4f}, Acc: {calib_epoch_acc*100:.2f}%")
        
        if calib_epoch_loss < best_calib_loss:
            best_calib_loss = calib_epoch_loss
            best_state = copy.deepcopy(model.state_dict())
            
    training_duration = time.time() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)
        
    # Save FP32 Checkpoint
    fp32_ckpt_path = MODELS_DIR / "yolov8n_cls_13class.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'training_time_seconds': round(training_duration, 2)
    }, fp32_ckpt_path)
    fp32_mb = fp32_ckpt_path.stat().st_size / (1024 * 1024)
    print(f"Saved 13-Class FP32 Model Checkpoint: {fp32_ckpt_path} ({fp32_mb:.2f} MB)")

    # Dynamic INT8 Quantization Export
    try:
        cpu_model = copy.deepcopy(model).to("cpu")
        int8_model = torch.ao.quantization.quantize_dynamic(cpu_model, {nn.Linear}, dtype=torch.qint8)
        int8_ckpt_path = MODELS_DIR / "yolov8n_cls_13class_int8.pth"
        torch.save(int8_model.state_dict(), int8_ckpt_path)
        int8_mb = int8_ckpt_path.stat().st_size / (1024 * 1024)
        print(f"Saved 13-Class INT8 Model Checkpoint: {int8_ckpt_path} ({int8_mb:.2f} MB)")
    except Exception as e:
        print(f"Notice: INT8 dynamic quantization export notice: {e}")
        int8_mb = 0.0

    # ---------------------------------------------------------
    # PART E: CALIBRATION ON CALIBRATION PARTITION
    # ---------------------------------------------------------
    print("\n--- Calibrating Model with Temperature Scaling ---")
    calib_model = ModelWithTemperature(model)
    calib_model.set_temperature(calib_loader, device)
    opt_temp = float(calib_model.temperature.item())
    print(f"Learned Optimal Temperature T = {opt_temp:.4f}")

    # Evaluate pre and post calibration ECE on calibration partition
    model.eval()
    calib_logits, calib_labels = [], []
    with torch.no_grad():
        for inputs, targets, _, _ in calib_loader:
            inputs = inputs.to(device)
            calib_logits.append(model(inputs))
            calib_labels.append(targets)
    c_logits = torch.cat(calib_logits)
    c_labels = torch.cat(calib_labels).numpy()
    
    pre_probs = F.softmax(c_logits, dim=1).cpu().numpy()
    post_probs = F.softmax(c_logits / opt_temp, dim=1).cpu().numpy()
    
    pre_ece = compute_ece(pre_probs, c_labels)
    post_ece = compute_ece(post_probs, c_labels)
    
    print(f"Calibration Set Pre-ECE:  {pre_ece*100:.2f}%")
    print(f"Calibration Set Post-ECE: {post_ece*100:.2f}%")

    calibration_metrics = {
        "optimal_temperature": round(opt_temp, 4),
        "pre_calibration_ece": round(pre_ece, 4),
        "post_calibration_ece": round(post_ece, 4)
    }
    with open(CALIBRATION_DIR / "calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(calibration_metrics, f, indent=2)

    # ---------------------------------------------------------
    # PART D: BENCHMARK ON ISOLATED VALIDATION PARTITION
    # ---------------------------------------------------------
    print("\n--- Benchmarking 13-Class Model on Isolated Validation Partition ---")
    val_logits, val_labels, val_filenames = [], [], []
    
    # Latency measurement
    latencies = []
    model.eval()
    with torch.no_grad():
        for inputs, targets, fns, _ in val_loader:
            inputs = inputs.to(device)
            for i in range(inputs.size(0)):
                single_in = inputs[i:i+1]
                t0 = time.perf_counter()
                out = model(single_in)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)
            
            val_logits.append(model(inputs))
            val_labels.append(targets)
            val_filenames.extend(fns)

    v_logits = torch.cat(val_logits)
    v_labels = torch.cat(val_labels).numpy()
    v_probs = F.softmax(v_logits / opt_temp, dim=1).cpu().numpy()
    
    preds_top1 = np.argmax(v_probs, axis=1)
    
    # Top-1, Top-2, Top-3
    top1_acc = np.mean(preds_top1 == v_labels)
    top2_correct = 0
    top3_correct = 0
    for i in range(len(v_labels)):
        top_k = np.argsort(v_probs[i])[::-1]
        if v_labels[i] in top_k[:2]: top2_correct += 1
        if v_labels[i] in top_k[:3]: top3_correct += 1
        
    top2_acc = top2_correct / max(1, len(v_labels))
    top3_acc = top3_correct / max(1, len(v_labels))
    
    # Precision, Recall, Macro F1, Weighted F1, Balanced Acc
    from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, confusion_matrix
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(v_labels, preds_top1, average='macro', zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(v_labels, preds_top1, average='weighted', zero_division=0)
    bal_acc = balanced_accuracy_score(v_labels, preds_top1)
    val_ece = compute_ece(v_probs, v_labels)
    cm = confusion_matrix(v_labels, preds_top1, labels=list(range(len(all_classes))))
    
    warm_median_lat = float(np.median(latencies[1:])) if len(latencies) > 1 else latencies[0]
    warm_p95_lat = float(np.percentile(latencies[1:], 95)) if len(latencies) > 1 else latencies[0]
    cold_lat = latencies[0]
    
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n13-Class Model Validation Results:")
    print(f"  Top-1 Accuracy:    {top1_acc*100:.2f}%")
    print(f"  Top-2 Accuracy:    {top2_acc*100:.2f}%")
    print(f"  Top-3 Accuracy:    {top3_acc*100:.2f}%")
    print(f"  Macro Precision:   {p_macro*100:.2f}%")
    print(f"  Macro Recall:      {r_macro*100:.2f}%")
    print(f"  Macro F1:          {f1_macro:.4f}")
    print(f"  Weighted F1:       {f1_w:.4f}")
    print(f"  Balanced Accuracy: {bal_acc*100:.2f}%")
    print(f"  Validation ECE:    {val_ece*100:.2f}%")
    print(f"  Warm Median CPU Latency: {warm_median_lat:.2f} ms")
    
    benchmark_data = {
        "model_name": "YOLOv8n-cls (13-class)",
        "num_classes": len(all_classes),
        "parameters": num_params,
        "fp32_checkpoint_mb": round(fp32_mb, 2),
        "int8_checkpoint_mb": round(int8_mb, 2),
        "top1_accuracy": round(top1_acc, 4),
        "top2_accuracy": round(top2_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "macro_precision": round(p_macro, 4),
        "macro_recall": round(r_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_f1": round(f1_w, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "validation_ece": round(val_ece, 4),
        "cold_start_latency_ms": round(cold_lat, 2),
        "warm_median_latency_ms": round(warm_median_lat, 2),
        "warm_p95_latency_ms": round(warm_p95_lat, 2),
        "confusion_matrix": cm.tolist(),
        "taxonomy_classes": all_classes
    }
    
    with open(BENCHMARKS_DIR / "13class_yolo_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

if __name__ == '__main__':
    run_training_and_benchmarking()
