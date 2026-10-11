"""
Unit test suite for Phase 7I-A Model Benchmark Integrity, Quantization & OOD Audit.
"""

import os
import sys
import json
import hashlib
import unittest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Setup paths
TESTS_DIR = Path(__file__).resolve().parent
PHASE7IA_DIR = TESTS_DIR.parent
VISUAL_UNDERSTANDING_DIR = PHASE7IA_DIR.parent
EXPERIMENTS_DIR = VISUAL_UNDERSTANDING_DIR.parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
PHASE7I_DIR = VISUAL_UNDERSTANDING_DIR / "phase7i"

sys.path.insert(0, str(PHASE7I_DIR))
sys.path.insert(0, str(PHASE7IA_DIR))

from dataset_preprocessing import VSUEDataset
from train_models import build_model_candidate, YOLOClassificationWrapper


class TestPhase7IABenchmarkAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primary_manifest_path = VISUAL_UNDERSTANDING_DIR / "phase7hg" / "phase7hg_primary_manifest.csv"
        cls.extended_manifest_path = VISUAL_UNDERSTANDING_DIR / "phase7hg" / "phase7hg_extended_manifest.csv"
        cls.zero_shot_probe_path = VISUAL_UNDERSTANDING_DIR / "phase7hg" / "phase7hg_zero_shot_probe.csv"

        with open(PHASE7I_DIR / "model_configs.json", "r", encoding="utf-8") as f:
            cls.configs = json.load(f)

    def test_01_manifest_hashes_and_counts(self):
        """1. Verify frozen manifest hashes match expected MD5 checksums and partition counts."""
        expected_hashes = self.configs["frozen_manifest_hashes"]
        for fname, exp_hash in expected_hashes.items():
            p = VISUAL_UNDERSTANDING_DIR / "phase7hg" / fname
            self.assertTrue(p.exists(), f"Missing manifest: {p}")
            act_hash = hashlib.md5(p.read_bytes()).hexdigest()
            self.assertEqual(act_hash, exp_hash, f"Hash mismatch for {fname}")

        df = pd.read_csv(self.primary_manifest_path)
        counts = df["partition"].value_counts().to_dict()
        self.assertEqual(counts["train"], 83)
        self.assertEqual(counts["calibration"], 27)
        self.assertEqual(counts["validation"], 25)

    def test_02_recalculated_fp32_metrics(self):
        """2. Verify recalculated validation metrics for YOLOv8n-cls match 72.0% Top-1 accuracy."""
        recalc_path = PHASE7IA_DIR / "phase7ia_metric_recalculation.json"
        self.assertTrue(recalc_path.exists(), f"Missing recalculation JSON: {recalc_path}")
        with open(recalc_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("yolo26n_cls", data)
        yolo_metrics = data["yolo26n_cls"]["fp32_metrics"]
        self.assertEqual(yolo_metrics["top1_accuracy"], 0.7200)
        self.assertEqual(yolo_metrics["macro_f1"], 0.5686)
        self.assertEqual(yolo_metrics["weighted_f1"], 0.6963)

    def test_03_yolo_architecture_and_parameters(self):
        """3. Verify YOLOv8n-cls architecture instantiation and exact 5-class parameter count."""
        yolo_model = build_model_candidate("yolo26n_cls", num_classes=5)
        self.assertIsInstance(yolo_model, YOLOClassificationWrapper)

        params = sum(p.numel() for p in yolo_model.parameters())
        self.assertEqual(params, 1444693, f"Expected 1,444,693 parameters for YOLOv8n-cls, got {params}")

    def test_04_int8_quantization_and_agreement(self):
        """4. Verify INT8 quantization comparison artifact and 100% validation agreement."""
        csv_path = PHASE7IA_DIR / "phase7ia_fp32_int8_comparison.csv"
        self.assertTrue(csv_path.exists(), f"Missing quantization CSV: {csv_path}")
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            self.assertEqual(row["prediction_agreement_pct"], 100.0, f"Disagreement for {row['model_key']}")

        # Verify MobileViT size reduction
        mobilevit_row = df[df["model_key"] == "mobilevit_xs"].iloc[0]
        self.assertGreater(mobilevit_row["size_reduction_pct"], 40.0)

    def test_05_ood_confidence_metrics(self):
        """5. Verify zero-shot probe confidence metrics artifact."""
        ood_path = PHASE7IA_DIR / "phase7ia_ood_confidence_analysis.json"
        self.assertTrue(ood_path.exists(), f"Missing OOD analysis JSON: {ood_path}")
        with open(ood_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("yolo26n_cls", data)
        self.assertEqual(data["yolo26n_cls"]["out_of_distribution_probe"]["mean_max_confidence"], 0.3488)

    def test_06_audit_deliverable_artifacts_exist(self):
        """6. Verify all 8 Phase 7I-A deliverable artifacts exist."""
        deliverables = [
            "phase7ia_dataset_reproducibility_audit.md",
            "phase7ia_metric_recalculation.json",
            "phase7ia_architecture_verification.md",
            "phase7ia_quantization_audit.json",
            "phase7ia_fp32_int8_comparison.csv",
            "phase7ia_cpu_benchmark_audit.md",
            "phase7ia_ood_confidence_analysis.json",
            "phase7ia_completion_report.md"
        ]
        for fname in deliverables:
            p = PHASE7IA_DIR / fname
            self.assertTrue(p.exists(), f"Missing deliverable artifact: {p}")
            self.assertGreater(p.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
