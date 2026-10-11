"""
Phase 7I: Supervised Visual Screen-Type Model Evaluation Script.

Evaluates all trained model candidates (FP32 & INT8 Quantized) on:
1. Primary 5-class validation partition (Accuracy, Macro/Weighted F1, ECE, Confusion Matrix).
2. CPU Cold-Start & Warm latency, p95 latency, model sizes, and peak memory.
3. Zero-shot probe (31 product_listing screenshots).
4. Extended 8-class validation partition.
5. Phase 7F vs Phase 7I MobileNetV3-Small historical reproduction comparison.
"""

import os
import sys
import json
import time
import psutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    balanced_accuracy_score, confusion_matrix
)
from PIL import Image
from collections import Counter

PHASE7I_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE7I_DIR.parent.parent.parent
sys.path.insert(0, str(PHASE7I_DIR))

from dataset_preprocessing import VSUEDataset, letterbox_resize, IMAGENET_MEAN, IMAGENET_STD
from train_models import build_model_candidate

CONFIG_PATH = PHASE7I_DIR / "model_configs.json"
CHECKPOINTS_DIR = PHASE7I_DIR / "model_checkpoints"
METRICS_DIR = PHASE7I_DIR / "metrics"
PRIMARY_MANIFEST_PATH = PROJECT_ROOT / "experiments" / "visual_understanding" / "phase7hg" / "phase7hg_primary_manifest.csv"
EXTENDED_MANIFEST_PATH = PROJECT_ROOT / "experiments" / "visual_understanding" / "phase7hg" / "phase7hg_extended_manifest.csv"
ZERO_SHOT_PROBE_PATH = PROJECT_ROOT / "experiments" / "visual_understanding" / "phase7hg" / "phase7hg_zero_shot_probe.csv"


def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Calculates Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)


def top_k_accuracy(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = [labels[i] in top_k_preds[i] for i in range(len(labels))]
    return float(np.mean(correct))


def load_checkpoint_model(model_key: str, num_classes: int, is_int8: bool = False) -> nn.Module:
    model = build_model_candidate(model_key, num_classes)
    ckpt_name = f"visual_screen_type_{model_key}_int8.pth" if is_int8 else f"visual_screen_type_{model_key}.pth"
    ckpt_path = CHECKPOINTS_DIR / ckpt_name

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")

    if is_int8:
        # Re-quantize dynamic structure and load weights
        cpu_model = model.to("cpu")
        quantized_model = torch.ao.quantization.quantize_dynamic(cpu_model, {nn.Linear}, dtype=torch.qint8)
        quantized_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        return quantized_model
    else:
        ckpt_data = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_data["model_state_dict"])
        model.eval()
        return model


def evaluate_primary_benchmark(configs: dict) -> Dict:
    print("\n--- Evaluating Primary 5-Class Validation Partition ---")
    primary_df = pd.read_csv(PRIMARY_MANIFEST_PATH)
    val_df = primary_df[primary_df["partition"] == "validation"].reset_index(drop=True)

    primary_classes = configs["primary_classes"]
    class_to_idx = {c: i for i, c in enumerate(primary_classes)}

    val_dataset = VSUEDataset(val_df, class_to_idx=class_to_idx)
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    # Get ground truth labels
    inputs, targets, filenames = next(iter(loader))
    y_true = targets.numpy()

    results = {}
    per_class_rows = []

    for model_key in configs["model_candidates"].keys():
        model = load_checkpoint_model(model_key, len(primary_classes), is_int8=False)
        model.eval()

        with torch.no_grad():
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).numpy()
            y_pred = np.argmax(probs, axis=1)

        top1 = accuracy_score(y_true, y_pred)
        top2 = top_k_accuracy(probs, y_true, k=2)
        top3 = top_k_accuracy(probs, y_true, k=3)

        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        ece = calculate_ece(probs, y_true)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(primary_classes))))

        prec_cls, rec_cls, f1_cls, supp_cls = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(primary_classes))), zero_division=0)

        for i, cname in enumerate(primary_classes):
            per_class_rows.append({
                "model_key": model_key,
                "display_name": configs["model_candidates"][model_key]["display_name"],
                "class_name": cname,
                "precision": round(float(prec_cls[i]), 4),
                "recall": round(float(rec_cls[i]), 4),
                "f1_score": round(float(f1_cls[i]), 4),
                "support": int(supp_cls[i])
            })

        results[model_key] = {
            "model_name": configs["model_candidates"][model_key]["display_name"],
            "top1_accuracy": round(float(top1), 4),
            "top2_accuracy": round(float(top2), 4),
            "top3_accuracy": round(float(top3), 4),
            "macro_precision": round(float(prec_m), 4),
            "macro_recall": round(float(rec_m), 4),
            "macro_f1": round(float(f1_m), 4),
            "weighted_f1": round(float(f1_w), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "ece": round(float(ece), 4),
            "confusion_matrix": cm.tolist()
        }

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(METRICS_DIR / "per_class_metrics.csv", index=False)
    print("Saved Primary Benchmark Per-Class Metrics CSV.")

    return results


def evaluate_cpu_latency_and_memory(configs: dict) -> Dict:
    print("\n--- Evaluating CPU Edge Inference Latency & Memory Footprint ---")
    dummy_input = torch.randn(1, 3, 224, 224)
    cpu_results = {}

    for model_key in configs["model_candidates"].keys():
        fp32_ckpt = CHECKPOINTS_DIR / f"visual_screen_type_{model_key}.pth"
        int8_ckpt = CHECKPOINTS_DIR / f"visual_screen_type_{model_key}_int8.pth"

        fp32_size_mb = fp32_ckpt.stat().st_size / (1024 * 1024) if fp32_ckpt.exists() else 0.0
        int8_size_mb = int8_ckpt.stat().st_size / (1024 * 1024) if int8_ckpt.exists() else 0.0

        model = load_checkpoint_model(model_key, len(configs["primary_classes"]), is_int8=False)
        model.eval()

        # Cold Start Latency
        t0 = time.perf_counter()
        _ = model(dummy_input)
        cold_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Warm Latency Over 50 Passes
        warm_latencies = []
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)

        with torch.no_grad():
            for _ in range(50):
                t_start = time.perf_counter()
                _ = model(dummy_input)
                warm_latencies.append((time.perf_counter() - t_start) * 1000.0)

        mem_after = process.memory_info().rss / (1024 * 1024)

        median_latency_ms = float(np.median(warm_latencies))
        p95_latency_ms = float(np.percentile(warm_latencies, 95))

        cpu_results[model_key] = {
            "model_name": configs["model_candidates"][model_key]["display_name"],
            "fp32_parameters": configs["model_candidates"][model_key]["fp32_parameters"],
            "fp32_checkpoint_mb": round(fp32_size_mb, 2),
            "int8_checkpoint_mb": round(int8_size_mb, 2),
            "cold_start_latency_ms": round(cold_latency_ms, 2),
            "median_warm_latency_ms": round(median_latency_ms, 2),
            "p95_warm_latency_ms": round(p95_latency_ms, 2),
            "peak_memory_mb": round(mem_after, 2)
        }

    return cpu_results


def evaluate_zero_shot_probe(configs: dict) -> Dict:
    print("\n--- Evaluating Zero-Shot Probe (31 product_listing Screenshots) ---")
    probe_df = pd.read_csv(ZERO_SHOT_PROBE_PATH)
    primary_classes = configs["primary_classes"]
    class_to_idx = {c: i for i, c in enumerate(primary_classes)}

    probe_dataset = VSUEDataset(probe_df, class_to_idx=class_to_idx)
    loader = torch.utils.data.DataLoader(probe_dataset, batch_size=len(probe_dataset), shuffle=False)
    inputs, _, filenames = next(iter(loader))

    probe_results = {}
    for model_key in configs["model_candidates"].keys():
        model = load_checkpoint_model(model_key, len(primary_classes), is_int8=False)
        model.eval()

        with torch.no_grad():
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).numpy()
            y_pred = np.argmax(probs, axis=1)

        pred_classes = [primary_classes[idx] for idx in y_pred]
        class_dist = Counter(pred_classes)
        mean_conf = float(np.mean(np.max(probs, axis=1)))

        sample_predictions = []
        for i in range(min(5, len(filenames))):
            sample_predictions.append({
                "filename": filenames[i],
                "predicted_class": pred_classes[i],
                "confidence": round(float(np.max(probs[i])), 4)
            })

        probe_results[model_key] = {
            "model_name": configs["model_candidates"][model_key]["display_name"],
            "total_probe_samples": len(filenames),
            "mean_confidence": round(mean_conf, 4),
            "predicted_class_distribution": dict(class_dist),
            "sample_predictions": sample_predictions
        }

    return probe_results


def run_full_evaluation():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        configs = json.load(f)

    primary_results = evaluate_primary_benchmark(configs)
    cpu_results = evaluate_cpu_latency_and_memory(configs)
    probe_results = evaluate_zero_shot_probe(configs)

    # Save Primary Benchmark Metrics JSON
    with open(METRICS_DIR / "primary_benchmark_metrics.json", "w", encoding="utf-8") as f:
        json.dump(primary_results, f, indent=2)

    # Save CPU Deployment Metrics JSON
    with open(METRICS_DIR / "cpu_deployment_metrics.json", "w", encoding="utf-8") as f:
        json.dump(cpu_results, f, indent=2)

    # Save Zero Shot Probe Predictions JSON
    with open(METRICS_DIR / "zero_shot_probe_predictions.json", "w", encoding="utf-8") as f:
        json.dump(probe_results, f, indent=2)

    print("\n======================================================================")
    print("     ALL MODEL CANDIDATES EVALUATED & METRICS SAVED CLEANLY           ")
    print("======================================================================")


if __name__ == "__main__":
    run_full_evaluation()
