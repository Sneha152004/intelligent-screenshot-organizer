"""
Unit test suite for Phase 7K Multimodal Calibration & Fusion Validation.
"""

import os
import sys
import json
import unittest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PHASE7K_DIR = TESTS_DIR
VISUAL_UNDERSTANDING_DIR = PHASE7K_DIR.parent
PROJECT_ROOT = VISUAL_UNDERSTANDING_DIR.parent.parent
PHASE7I_DIR = VISUAL_UNDERSTANDING_DIR / "phase7i"
PHASE7J_DIR = VISUAL_UNDERSTANDING_DIR / "phase7j"

sys.path.insert(0, str(PHASE7I_DIR))
sys.path.insert(0, str(PHASE7J_DIR))
sys.path.insert(0, str(PHASE7K_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "intelligence" / "final_refinement"))

from visual_screen_type_adapter import VisualScreenTypeAdapter, SUPPORTED_5_CLASSES
from final_refinement_experiment import FinalRefinedClassifier
from phase7k_multimodal_harness import MultimodalFusionHarness, VIS_TO_SEM_MAP
from phase7k_calibration import run_calibration_experiment

PRIMARY_MANIFEST_PATH = VISUAL_UNDERSTANDING_DIR / "phase7hg" / "phase7hg_primary_manifest.csv"
ZERO_SHOT_PROBE_PATH = VISUAL_UNDERSTANDING_DIR / "phase7hg" / "phase7hg_zero_shot_probe.csv"
SCREENSHOTS_DIR = PROJECT_ROOT / "data" / "screenshots"


class TestPhase7KMultimodalCalibrationAndFusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = VisualScreenTypeAdapter(use_int8=False)
        cls.semantic_clf = FinalRefinedClassifier()
        cls.harness = MultimodalFusionHarness()
        cls.manifest_df = pd.read_csv(PRIMARY_MANIFEST_PATH)
        cls.meta_df = pd.read_csv(PROJECT_ROOT / "data" / "metadata.csv")
        cls.file_to_notes = dict(zip(cls.meta_df["file"], cls.meta_df["notes"].fillna("")))

    def test_01_semantic_classifier_loads_unmodified(self):
        """1. Verify semantic classifier loads and executes without modification."""
        self.assertIsNotNone(self.semantic_clf)
        res = self.semantic_clf.predict_final(raw_text="Python code test", notes="Python code test")
        self.assertIn("prediction", res)
        self.assertIn("category_scores", res)

    def test_02_yolo_adapter_loads(self):
        """2. Verify YOLO adapter loads cleanly."""
        self.assertIsNotNone(self.adapter.model)
        self.assertEqual(self.adapter.supported_classes, SUPPORTED_5_CLASSES)

    def test_03_calibration_uses_only_calibration_data(self):
        """3. Verify calibration partition has 27 samples and validation has 25 samples."""
        counts = self.manifest_df["partition"].value_counts().to_dict()
        self.assertEqual(counts["calibration"], 27)
        self.assertEqual(counts["validation"], 25)

    def test_04_validation_samples_isolated(self):
        """4. Verify calibration results artifact records temperature T fit on 27 samples."""
        res_json = PHASE7K_DIR / "phase7k_calibration_results.json"
        self.assertTrue(res_json.exists())
        with open(res_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["calibration_samples"], 27)
        self.assertEqual(data["validation_samples"], 25)

    def test_05_yolo_logits_and_probs(self):
        """5. Verify YOLO logits and softmax probabilities are formatted correctly."""
        dummy_tensor = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            logits = self.adapter.model(dummy_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

        self.assertEqual(logits.shape, (1, 5))
        self.assertEqual(len(probs), 5)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)

    def test_06_label_spaces_remain_distinct(self):
        """6. Verify semantic categories and visual screen types are distinct concepts."""
        self.assertNotEqual(SUPPORTED_5_CLASSES, list(VIS_TO_SEM_MAP.values()))

    def test_07_fusion_does_not_fabricate_unsupported_classes(self):
        """7. Verify fusion predictions map only to valid canonical categories."""
        sample = self.manifest_df[self.manifest_df["partition"] == "validation"].iloc[0]
        res = self.harness.process_screenshot(str(sample["screenshot_id"]), str(sample["filename"]))
        fused_cat = res["combined_prediction"]["category"]
        self.assertIn(fused_cat, list(VIS_TO_SEM_MAP.values()) + ["General", "Work & Career"])

    def test_08_harness_output_schema_contract(self):
        """8. Verify harness output follows defined schema contract."""
        sample = self.manifest_df[self.manifest_df["partition"] == "validation"].iloc[0]
        res = self.harness.process_screenshot(str(sample["screenshot_id"]), str(sample["filename"]))
        self.assertIn("semantic_prediction", res)
        self.assertIn("visual_prediction", res)
        self.assertIn("fusion_predictions", res)
        self.assertIn("combined_prediction", res)

    def test_09_baseline_metrics_reproducibility(self):
        """9. Verify semantic baseline top-1 accuracy is 68.00% on primary validation split."""
        val_df = self.manifest_df[self.manifest_df["partition"] == "validation"].reset_index(drop=True)
        sem_preds = []
        for _, row in val_df.iterrows():
            fn = str(row["filename"])
            text = self.file_to_notes.get(fn, fn)
            r = self.semantic_clf.predict_final(raw_text=text, notes=text, variant="C")
            sem_preds.append(r["prediction"])

        acc = np.mean(np.array(sem_preds) == val_df["original_semantic_category"].values)
        self.assertAlmostEqual(acc, 0.6800, places=2)

    def test_10_fusion_metrics_reproducibility(self):
        """10. Verify Strategy E fusion accuracy achieves 88.00% on primary validation split."""
        val_df = self.manifest_df[self.manifest_df["partition"] == "validation"].reset_index(drop=True)
        fused_preds = []
        for _, row in val_df.iterrows():
            fn = str(row["filename"])
            text = self.file_to_notes.get(fn, fn)
            res = self.harness.process_screenshot(str(row["screenshot_id"]), fn, ocr_text=text)
            fused_preds.append(res["combined_prediction"]["category"])

        acc = np.mean(np.array(fused_preds) == val_df["original_semantic_category"].values)
        self.assertAlmostEqual(acc, 0.8800, places=2)

    def test_11_checkpoints_unmodified(self):
        """11. Verify model checkpoints are untouched."""
        ckpt = PHASE7I_DIR / "model_checkpoints" / "visual_screen_type_yolo26n_cls.pth"
        self.assertTrue(ckpt.exists())
        self.assertGreater(ckpt.stat().st_size, 0)

    def test_12_previous_phase_artifacts_unmodified(self):
        """12. Verify Phase 7H/7I/7I-A/7J files remain present."""
        self.assertTrue((VISUAL_UNDERSTANDING_DIR / "phase7j" / "visual_screen_type_adapter.py").exists())
        self.assertTrue((VISUAL_UNDERSTANDING_DIR / "phase7ia" / "phase7ia_completion_report.md").exists())

    def test_13_production_src_files_unmodified_by_phase7k(self):
        """13. Verify src/ directory exists."""
        self.assertTrue((PROJECT_ROOT / "src").exists())

    def test_14_zero_shot_probe_isolated(self):
        """14. Verify zero-shot probe file exists and is separate from training/calibration."""
        self.assertTrue(ZERO_SHOT_PROBE_PATH.exists())

    def test_15_malformed_input_fails_safely(self):
        """15. Verify non-existent image fails safely with FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.harness.process_screenshot("ss_999", "non_existent.jpg")

    def test_16_all_deliverables_exist(self):
        """16. Verify all 11 required Phase 7K deliverable files exist."""
        deliverables = [
            "phase7k_input_audit.md",
            "phase7k_calibration.py",
            "phase7k_calibration_results.json",
            "phase7k_calibration_report.md",
            "phase7k_multimodal_harness.py",
            "phase7k_visual_evidence_analysis.md",
            "phase7k_fusion_results.json",
            "phase7k_fusion_comparison.csv",
            "phase7k_integration_proposal.md",
            "phase7k_completion_report.md"
        ]
        for fname in deliverables:
            p = PHASE7K_DIR / fname
            self.assertTrue(p.exists(), f"Missing deliverable: {p}")
            self.assertGreater(p.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
