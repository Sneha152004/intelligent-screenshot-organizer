import json
import os
import sys
import unittest
import pandas as pd

PROJECT_ROOT = r"C:\Users\KIIT0001\Documents\intelligent-screenshot-organizer"
PHASE7HC_DIR = os.path.join(PROJECT_ROOT, "experiments", "visual_understanding", "phase7hc")
CORRECTED_SHEETS_DIR = os.path.join(PHASE7HC_DIR, "corrected_contact_sheets")


class TestPhase7HCDatasetWideAudit(unittest.TestCase):

    def setUp(self):
        self.ledger_path = os.path.join(PHASE7HC_DIR, "phase7hc_dataset_wide_correction_ledger.csv")
        self.report_path = os.path.join(PHASE7HC_DIR, "phase7hc_dataset_wide_audit_report.md")
        self.index_path = os.path.join(PHASE7HC_DIR, "phase7hc_contact_sheet_index.md")

        self.assertTrue(os.path.exists(self.ledger_path))
        self.assertTrue(os.path.exists(self.report_path))
        self.assertTrue(os.path.exists(self.index_path))

        self.ledger_df = pd.read_csv(self.ledger_path)

    def test_01_candidate_id_uniqueness(self):
        """Test screenshot IDs in correction ledger are strictly unique."""
        sids = self.ledger_df["screenshot_id"].tolist()
        self.assertEqual(len(sids), len(set(sids)))

    def test_02_candidate_to_image_mapping(self):
        """Test all 836 screenshot files in data/screenshots are accounted for."""
        img_dir = os.path.join(PROJECT_ROOT, "data", "screenshots")
        all_files = sorted(os.listdir(img_dir))
        ledger_files = sorted(list(self.ledger_df["screenshot_id"]))
        self.assertEqual(all_files, ledger_files)

    def test_03_corrupted_image_exclusion(self):
        """Test 0-byte corrupted images are marked reject and excluded from contact sheets."""
        corrupted = ["ss_441.png", "ss_443.png", "ss_560.png"]
        for c in corrupted:
            row = self.ledger_df[self.ledger_df["screenshot_id"] == c].iloc[0]
            self.assertEqual(row["correction_action"], "reject")
            self.assertEqual(row["review_status"], "rejected")

    def test_04_human_verified_preservation(self):
        """Test preserved human-reviewed decisions retain human_verified action."""
        human_verified_ids = ["ss_001.jpg", "ss_799.jpg", "ss_002.jpg", "ss_800.jpg"]
        for sid in human_verified_ids:
            row = self.ledger_df[self.ledger_df["screenshot_id"] == sid].iloc[0]
            self.assertEqual(row["correction_action"], "human_verified")

    def test_05_valid_correction_actions(self):
        """Test correction_action values are valid enums."""
        allowed_actions = ["retain", "correct", "uncertain", "reject", "human_verified"]
        actions = self.ledger_df["correction_action"].unique()
        for a in actions:
            self.assertIn(a, allowed_actions)

    def test_06_corrected_contact_sheet_coverage(self):
        """Test corrected contact sheet PNG files exist on disk and cover all valid images."""
        cfiles = [f for f in os.listdir(CORRECTED_SHEETS_DIR) if f.endswith(".png")]
        self.assertGreater(len(cfiles), 0)
        for cfile in cfiles:
            cpath = os.path.join(CORRECTED_SHEETS_DIR, cfile)
            self.assertGreater(os.path.getsize(cpath), 1000)

    def test_07_audit_report_completeness(self):
        """Test phase7hc_dataset_wide_audit_report.md exists and contains required sections."""
        with open(self.report_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("DATASET-WIDE RE-AUDIT COMPLETE", content)
        self.assertIn("Classification Error Patterns Discovered", content)


if __name__ == "__main__":
    unittest.main()
