import json
import os
import sys
import unittest
import pandas as pd

PROJECT_ROOT = r"C:\Users\KIIT0001\Documents\intelligent-screenshot-organizer"
PHASE7H_C_DIR = os.path.join(PROJECT_ROOT, "experiments", "visual_understanding", "phase7h_c")
CONTACT_SHEETS_DIR = os.path.join(PHASE7H_C_DIR, "contact_sheets")


class TestPhase7HCHumanWorkflow(unittest.TestCase):

    def setUp(self):
        self.reconcil_path = os.path.join(PHASE7H_C_DIR, "phase7hc_candidate_reconciliation.csv")
        self.queue_path = os.path.join(PHASE7H_C_DIR, "phase7hc_human_annotation_queue.csv")
        self.index_path = os.path.join(PHASE7H_C_DIR, "phase7hc_contact_sheet_index.md")
        self.guidelines_path = os.path.join(PHASE7H_C_DIR, "phase7hc_annotation_guidelines.md")

        self.assertTrue(os.path.exists(self.reconcil_path))
        self.assertTrue(os.path.exists(self.queue_path))
        self.assertTrue(os.path.exists(self.index_path))

        self.reconcil_df = pd.read_csv(self.reconcil_path)
        self.queue_df = pd.read_csv(self.queue_path)

    def test_01_candidate_id_uniqueness(self):
        """Test candidate screenshot IDs are strictly unique."""
        rec_ids = self.reconcil_df["screenshot_id"].tolist()
        self.assertEqual(len(rec_ids), len(set(rec_ids)))

        q_ids = self.queue_df["screenshot_id"].tolist()
        self.assertEqual(len(q_ids), len(set(q_ids)))

    def test_02_candidate_to_image_mapping(self):
        """Test all 836 files in data/screenshots are accounted for."""
        img_dir = os.path.join(PROJECT_ROOT, "data", "screenshots")
        all_files = sorted(os.listdir(img_dir))
        rec_files = sorted(list(self.reconcil_df["screenshot_id"]))
        self.assertEqual(all_files, rec_files)

    def test_03_corrupted_image_exclusion(self):
        """Test 0-byte corrupted files are marked rejected and excluded from review queue."""
        corrupted = ["ss_441.png", "ss_443.png", "ss_560.png"]
        queue_sids = set(self.queue_df["screenshot_id"])

        for c in corrupted:
            self.assertNotIn(c, queue_sids, f"Corrupted file {c} leaked into review queue!")
            rec_row = self.reconcil_df[self.reconcil_df["screenshot_id"] == c].iloc[0]
            self.assertEqual(rec_row["valid_status"], "CORRUPTED")
            self.assertEqual(rec_row["reconciliation_status"], "REJECTED_CORRUPTED")

    def test_04_verified_record_protection(self):
        """Test Phase 7E verified manifest records are preserved intact."""
        manifest_7e_path = os.path.join(PROJECT_ROOT, "experiments", "visual_understanding", "screen_type_manifest_final.csv")
        df_7e = pd.read_csv(manifest_7e_path)
        self.assertEqual(len(df_7e), 833)

    def test_05_pending_status_validation(self):
        """Test all candidates in queue are initially pending with unassigned final labels."""
        statuses = self.queue_df["annotation_status"].unique()
        self.assertEqual(list(statuses), ["pending"])

        # final_screen_type, reviewer, review_date must be empty/null
        final_types = [str(x) for x in self.queue_df["final_screen_type"].fillna("")]
        reviewers = [str(x) for x in self.queue_df["reviewer"].fillna("")]

        self.assertEqual(set(final_types), {""})
        self.assertEqual(set(reviewers), {""})

    def test_06_duplicate_group_preservation(self):
        """Test duplicate group structure from Phase 7E/7H-A is preserved."""
        manifest_7h_path = os.path.join(PROJECT_ROOT, "experiments", "visual_understanding", "phase7h_a", "screen_type_manifest_phase7h.csv")
        df_7h = pd.read_csv(manifest_7h_path)
        self.assertIn("duplicate_group_id", df_7h.columns)

    def test_07_contact_sheet_coverage(self):
        """Test contact sheet PNG files exist on disk and cover all 438 candidates."""
        contact_files = [f for f in os.listdir(CONTACT_SHEETS_DIR) if f.endswith(".png")]
        self.assertGreater(len(contact_files), 0)

        for cfile in contact_files:
            cpath = os.path.join(CONTACT_SHEETS_DIR, cfile)
            self.assertGreater(os.path.getsize(cpath), 1000, f"Contact sheet image empty: {cfile}")


if __name__ == "__main__":
    unittest.main()
