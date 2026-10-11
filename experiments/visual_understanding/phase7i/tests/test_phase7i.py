"""
Unit test suite for Phase 7I Supervised Visual Screen-Type Model Benchmarking.
"""

import os
import sys
import json
import hashlib
import unittest
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path

# Paths
TESTS_DIR = Path(__file__).resolve().parent
PHASE7I_DIR = TESTS_DIR.parent
VISUAL_UNDERSTANDING_DIR = PHASE7I_DIR.parent
EXPERIMENTS_DIR = VISUAL_UNDERSTANDING_DIR.parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent

sys.path.insert(0, str(PHASE7I_DIR))

from dataset_preprocessing import letterbox_resize, VSUEDataset
from train_models import build_model_candidate, verify_manifest_hashes
from evaluate_models import calculate_ece, top_k_accuracy

CONFIG_PATH = PHASE7I_DIR / "model_configs.json"
CHECKPOINTS_DIR = PHASE7I_DIR / "model_checkpoints"
METRICS_DIR = PHASE7I_DIR / "metrics"
PHASE7HG_DIR = VISUAL_UNDERSTANDING_DIR / "phase7hg"


class TestPhase7IModelBenchmarking(unittest.TestCase):
    def setUp(self):
        self.assertTrue(CONFIG_PATH.exists(), f"Missing config: {CONFIG_PATH}")
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self.configs = json.load(f)

    def test_01_frozen_manifest_hashes(self):
        """1. Verify frozen manifest hashes match expected MD5 checksums."""
        verify_manifest_hashes(self.configs)

    def test_02_letterbox_preprocessing(self):
        """2. Verify aspect-ratio-preserving letterbox resizing for RGB and RGBA images."""
        # Test wide RGBA image
        wide_rgba = Image.new("RGBA", (800, 400), (255, 0, 0, 128))
        resized_wide = letterbox_resize(wide_rgba, target_size=(224, 224))
        self.assertEqual(resized_wide.size, (224, 224))
        self.assertEqual(resized_wide.mode, "RGB")

        # Test tall RGB image
        tall_rgb = Image.new("RGB", (300, 900), (0, 255, 0))
        resized_tall = letterbox_resize(tall_rgb, target_size=(224, 224))
        self.assertEqual(resized_tall.size, (224, 224))
        self.assertEqual(resized_tall.mode, "RGB")

    def test_03_model_candidate_instantiation(self):
        """3. Verify all 4 model candidates instantiate and produce expected 5-class logits."""
        dummy_input = torch.randn(2, 3, 224, 224)
        for model_key in self.configs["model_candidates"].keys():
            model = build_model_candidate(model_key, num_classes=5)
            model.eval()
            with torch.no_grad():
                out = model(dummy_input)
            self.assertEqual(out.shape, (2, 5), f"Model {model_key} output shape should be (2, 5), got {out.shape}")

    def test_04_ece_and_topk_metrics(self):
        """4. Verify calibration ECE and top-k accuracy metric computations."""
        probs = np.array([
            [0.8, 0.1, 0.05, 0.03, 0.02],
            [0.1, 0.7, 0.1, 0.05, 0.05],
            [0.05, 0.05, 0.6, 0.2, 0.1]
        ])
        labels = np.array([0, 1, 2])

        top1 = top_k_accuracy(probs, labels, k=1)
        top2 = top_k_accuracy(probs, labels, k=2)
        ece = calculate_ece(probs, labels, n_bins=5)

        self.assertEqual(top1, 1.0)
        self.assertEqual(top2, 1.0)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

    def test_05_zero_shot_probe_manifest_isolation(self):
        """5. Verify zero-shot probe manifest is isolated and contains only product_listing."""
        probe_csv = PHASE7HG_DIR / "phase7hg_zero_shot_probe.csv"
        self.assertTrue(probe_csv.exists(), f"Missing probe CSV: {probe_csv}")
        import pandas as pd
        df = pd.read_csv(probe_csv)
        self.assertEqual(len(df), 31)
        self.assertEqual(set(df["screen_type"]), {"product_listing"})

    def test_06_model_checkpoints_exist(self):
        """6. Verify FP32 model checkpoints exist in model_checkpoints directory."""
        for model_key in self.configs["model_candidates"].keys():
            fp32_ckpt = CHECKPOINTS_DIR / f"visual_screen_type_{model_key}.pth"
            self.assertTrue(fp32_ckpt.exists(), f"Missing FP32 checkpoint for {model_key}: {fp32_ckpt}")
            self.assertGreater(fp32_ckpt.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
