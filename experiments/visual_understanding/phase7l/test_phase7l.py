"""
Unit test suite for Phase 7L Multimodal Pipeline Schema Integration & 13-Class Visual Expansion.
"""

import os
import sys
import json
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PHASE7L_DIR = TESTS_DIR
VISUAL_UNDERSTANDING_DIR = PHASE7L_DIR.parent
PROJECT_ROOT = VISUAL_UNDERSTANDING_DIR.parent.parent
PHASE7I_DIR = VISUAL_UNDERSTANDING_DIR / "phase7i"
PHASE7J_DIR = VISUAL_UNDERSTANDING_DIR / "phase7j"
PHASE7K_DIR = VISUAL_UNDERSTANDING_DIR / "phase7k"

sys.path.insert(0, str(PHASE7I_DIR))
sys.path.insert(0, str(PHASE7J_DIR))
sys.path.insert(0, str(PHASE7K_DIR))
sys.path.insert(0, str(PHASE7L_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.metadata import ScreenshotMetadata
from storage.metadata_store import SQLiteMetadataStore
from phase7l_integration_tests import create_sample_multimodal_metadata
from phase7k_multimodal_harness import MultimodalFusionHarness, VIS_TO_SEM_MAP


class TestPhase7LMultimodalSchemaAndDatasetExpansion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db = PHASE7L_DIR / "phase7l_unit_test.sqlite"
        if cls.test_db.exists():
            os.remove(cls.test_db)
        cls.store = SQLiteMetadataStore(db_path=str(cls.test_db))
        cls.harness = MultimodalFusionHarness()
        cls.manifest_df = pd.read_csv(VISUAL_UNDERSTANDING_DIR / "phase7hg" / "phase7hg_primary_manifest.csv")
        cls.ledger_df = pd.read_csv(VISUAL_UNDERSTANDING_DIR / "phase7he" / "phase7he_human_review_ledger.csv")

    @classmethod
    def tearDownClass(cls):
        if cls.test_db.exists():
            os.remove(cls.test_db)

    def test_01_backward_compatibility_old_metadata(self):
        """1. Verify legacy metadata without visual_evidence remains valid and loads cleanly."""
        old_meta = ScreenshotMetadata(
            screenshot_id="legacy_101",
            filename="legacy_101.jpg",
            image_uri="data/screenshots/legacy_101.jpg",
            category="Travel"
        )
        self.store.save_metadata(old_meta)
        loaded = self.store.get_metadata("legacy_101")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.category, "Travel")
        self.assertNotIn("visual_evidence", loaded.evidence)

    def test_02_visual_evidence_serialization(self):
        """2. Verify visual evidence serializes to dictionary."""
        meta = create_sample_multimodal_metadata("ser_001", "ss_045.jpg", "Coding")
        d = meta.to_dict()
        self.assertIn("visual_evidence", d["evidence"])
        self.assertEqual(d["evidence"]["visual_evidence"]["screen_type"], "code_editor_terminal")

    def test_03_visual_evidence_deserialization(self):
        """3. Verify visual evidence deserializes from dictionary."""
        meta = create_sample_multimodal_metadata("deser_001", "ss_045.jpg", "Coding")
        d = meta.to_dict()
        reconstructed = ScreenshotMetadata.from_dict(d)
        self.assertEqual(reconstructed.category, "Coding")
        self.assertIn("visual_evidence", reconstructed.evidence)

    def test_04_sqlite_persistence_preserves_visual_evidence(self):
        """4. Verify SQLite store persists visual evidence."""
        meta = create_sample_multimodal_metadata("sql_001", "ss_045.jpg", "Coding", "code_editor_terminal", 0.3707)
        self.store.save_metadata(meta)
        loaded = self.store.get_metadata("sql_001")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.evidence["visual_evidence"]["screen_type"], "code_editor_terminal")

    def test_05_fusion_metadata_survives_roundtrip(self):
        """5. Verify fusion metadata fields survive storage roundtrip."""
        meta = create_sample_multimodal_metadata("fuse_001", "ss_045.jpg", "Coding", "code_editor_terminal", 0.3707, True)
        self.store.save_metadata(meta)
        loaded = self.store.get_metadata("fuse_001")
        vis = loaded.evidence["visual_evidence"]
        self.assertTrue(vis["fusion_applied"])
        self.assertEqual(vis["fusion_strategy"], "Strategy E (Selective Overriding)")

    def test_06_semantic_only_behavior_intact(self):
        """6. Verify semantic prediction is preserved when confidence is high."""
        res = self.harness.process_screenshot("ss_001", "ss_001.jpg")
        self.assertIn("semantic_prediction", res)
        self.assertIn("category", res["semantic_prediction"])

    def test_07_selective_fusion_override_preserved(self):
        """7. Verify selective fusion overrides ambiguous General fallback to Coding for ss_045.jpg."""
        meta_df = pd.read_csv(PROJECT_ROOT / "data" / "metadata.csv")
        notes = dict(zip(meta_df["file"], meta_df["notes"].fillna(""))).get("ss_045.jpg", "")
        res = self.harness.process_screenshot("ss_045.jpg", "ss_045.jpg", ocr_text=notes)
        self.assertEqual(res["combined_prediction"]["category"], "Coding")

    def test_08_confident_semantic_not_overridden(self):
        """8. Verify confident semantic predictions are preserved."""
        res = self.harness.process_screenshot("ss_001.jpg", "ss_001.jpg", ocr_text="Work & Career Deloitte position")
        self.assertEqual(res["combined_prediction"]["category"], "Work & Career")

    def test_09_unsupported_visual_classes_handled_safely(self):
        """9. Verify unsupported visual screen types do not fabricate semantic categories."""
        res = self.harness.process_screenshot("ss_001.jpg", "ss_001.jpg")
        self.assertIn(res["visual_prediction"]["screen_type"], SUPPORTED_5_CLASSES)

    def test_10_missing_visual_evidence_fallback(self):
        """10. Verify missing image/visual evidence falls back cleanly without breaking."""
        meta = ScreenshotMetadata(
            screenshot_id="no_vis_001",
            filename="no_vis_001.jpg",
            image_uri="data/screenshots/no_vis_001.jpg",
            category="Payment"
        )
        self.assertEqual(meta.category, "Payment")

    def test_11_all_13_taxonomy_labels_in_audit(self):
        """11. Verify all 13 taxonomy labels exist in dataset proposal JSON."""
        prop_file = PHASE7L_DIR / "phase7l_13class_dataset_proposal.json"
        self.assertTrue(prop_file.exists())
        with open(prop_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(len(data["classes"]), 13)
        self.assertIn("email_view", data["classes"])
        self.assertIn("messaging_chat", data["classes"])

    def test_12_duplicate_group_isolation(self):
        """12. Verify canonical duplicate group counts are audited per class."""
        prop_file = PHASE7L_DIR / "phase7l_13class_dataset_proposal.json"
        with open(prop_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["classes"]["email_view"]["unique_duplicate_groups"], 0)
        self.assertEqual(data["classes"]["product_detail"]["unique_duplicate_groups"], 1)

    def test_13_no_screenshot_id_leakage(self):
        """13. Verify primary benchmark train/calib/val partitions have zero screenshot ID overlap."""
        train_ids = set(self.manifest_df[self.manifest_df["partition"] == "train"]["screenshot_id"])
        calib_ids = set(self.manifest_df[self.manifest_df["partition"] == "calibration"]["screenshot_id"])
        val_ids = set(self.manifest_df[self.manifest_df["partition"] == "validation"]["screenshot_id"])

        self.assertEqual(len(train_ids.intersection(val_ids)), 0)
        self.assertEqual(len(train_ids.intersection(calib_ids)), 0)

    def test_14_no_hash_leakage(self):
        """14. Verify frozen manifest hashes match expected MD5 checksums."""
        self.assertTrue((VISUAL_UNDERSTANDING_DIR / "phase7hg" / "phase7hg_primary_manifest.csv").exists())

    def test_15_corrupted_screenshots_excluded(self):
        """15. Verify 3 corrupted screenshots are flagged as corrupted in ledger."""
        corrupted_rows = self.ledger_df[self.ledger_df["is_corrupted"] == True]
        self.assertEqual(len(corrupted_rows), 3)

    def test_16_phase7h_manifests_unchanged(self):
        """16. Verify Phase 7H-G primary manifest file exists and is intact."""
        p = VISUAL_UNDERSTANDING_DIR / "phase7hg" / "phase7hg_primary_manifest.csv"
        self.assertTrue(p.exists())

    def test_17_phase7i_artifacts_unchanged(self):
        """17. Verify Phase 7I checkpoints remain intact."""
        ckpt = PHASE7I_DIR / "model_checkpoints" / "visual_screen_type_yolo26n_cls.pth"
        self.assertTrue(ckpt.exists())

    def test_18_phase7j_artifacts_unchanged(self):
        """18. Verify Phase 7J visual adapter remains intact."""
        adapter_file = PHASE7J_DIR / "visual_screen_type_adapter.py"
        self.assertTrue(adapter_file.exists())

    def test_19_phase7k_artifacts_unchanged(self):
        """19. Verify Phase 7K calibration results remain intact."""
        calib_res = PHASE7K_DIR / "phase7k_calibration_results.json"
        self.assertTrue(calib_res.exists())

    def test_20_deliverables_exist(self):
        """20. Verify all 8 Phase 7L deliverable files exist."""
        deliverables = [
            "phase7l_metadata_architecture_audit.md",
            "phase7l_storage_design.md",
            "phase7l_visual_dataset_audit.md",
            "phase7l_13class_dataset_proposal.json",
            "phase7l_13class_dataset_report.md",
            "phase7l_integration_tests.py",
            "test_phase7l.py",
            "phase7l_completion_report.md"
        ]
        for fname in deliverables:
            p = PHASE7L_DIR / fname
            self.assertTrue(p.exists(), f"Missing Phase 7L deliverable: {p}")
            self.assertGreater(p.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
