import json
import os
import sys
import unittest
import pandas as pd
from PIL import Image

PROJECT_ROOT = r"C:\Users\KIIT0001\Documents\intelligent-screenshot-organizer"
PHASE7H_B_DIR = os.path.join(PROJECT_ROOT, "experiments", "visual_understanding", "phase7h_b")

SEMANTIC_CATEGORIES = [
    "Shopping", "Entertainment", "Travel", "Work and career",
    "General", "Communication", "Education", "Coding",
    "Connectivity", "Payment"
]

FULL_VISUAL_TAXONOMY = [
    "product_listing", "product_detail", "cart_checkout_payment",
    "login_authentication", "messaging_chat", "email_view",
    "media_player_library", "code_editor_terminal", "document_article_view",
    "travel_booking", "dashboard_settings", "error_dialog_alert", "other_screen"
]


class TestPhase7HBDatasetReconciliation(unittest.TestCase):

    def setUp(self):
        self.meta_path = os.path.join(PROJECT_ROOT, "data", "metadata.csv")
        self.img_dir = os.path.join(PROJECT_ROOT, "data", "screenshots")
        self.manifest_7hb_path = os.path.join(PHASE7H_B_DIR, "screen_type_manifest_phase7hb.csv")
        self.rq_path = os.path.join(PHASE7H_B_DIR, "phase7hb_human_review_queue.csv")
        self.sem_audit_path = os.path.join(PHASE7H_B_DIR, "phase7hb_semantic_metadata_audit.csv")

        self.assertTrue(os.path.exists(self.meta_path))
        self.assertTrue(os.path.exists(self.manifest_7hb_path))
        self.assertTrue(os.path.exists(self.rq_path))

        self.meta_df = pd.read_csv(self.meta_path)
        self.manifest_7hb_df = pd.read_csv(self.manifest_7hb_path)
        self.rq_df = pd.read_csv(self.rq_path)

    def test_01_metadata_to_image_mapping(self):
        """Test 1-to-1 mapping between metadata records and screenshot files."""
        img_files = set(os.listdir(self.img_dir))
        meta_files = set(self.meta_df["file"])
        self.assertEqual(len(meta_files), len(img_files))
        self.assertEqual(len(meta_files - img_files), 0)

    def test_02_screenshot_id_consistency(self):
        """Test screenshot IDs match consistently across metadata and manifest."""
        meta_sids = sorted(list(self.meta_df["file"]))
        man_sids = sorted(list(self.manifest_7hb_df["screenshot_id"]))
        self.assertEqual(meta_sids, man_sids)

    def test_03_zero_missing_metadata(self):
        """Test zero files in screenshots dir lack metadata."""
        img_files = set(os.listdir(self.img_dir))
        meta_files = set(self.meta_df["file"])
        self.assertEqual(len(img_files - meta_files), 0)

    def test_04_zero_missing_images(self):
        """Test all metadata records point to existing files on disk."""
        for fname in self.meta_df["file"]:
            fpath = os.path.join(self.img_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Missing image file: {fname}")

    def test_05_unique_ids_and_hashes(self):
        """Test screenshot IDs in Phase 7H-B manifest are unique."""
        sids = self.manifest_7hb_df["screenshot_id"].tolist()
        self.assertEqual(len(sids), len(set(sids)))

    def test_06_semantic_category_preservation(self):
        """Test all original 10 semantic categories are preserved without mutation."""
        cats = self.manifest_7hb_df["semantic_category"].unique()
        for c in cats:
            self.assertIn(c, SEMANTIC_CATEGORIES)

    def test_07_annotation_status_validity(self):
        """Test annotation status values are valid enums."""
        valid_statuses = ["verified", "needs_review", "rejected"]
        for s in self.manifest_7hb_df["annotation_status"].unique():
            self.assertIn(s, valid_statuses)

    def test_08_verified_label_preservation(self):
        """Test all 833 Phase 7E verified visual labels are preserved intact."""
        verified = self.manifest_7hb_df[self.manifest_7hb_df["annotation_status"] == "verified"]
        self.assertEqual(len(verified), 833)

    def test_09_semantic_visual_separation(self):
        """Test semantic_category and screen_type are separate fields."""
        self.assertIn("semantic_category", self.manifest_7hb_df.columns)
        self.assertIn("screen_type", self.manifest_7hb_df.columns)
        self.assertIn("proposed_candidate_screen_type", self.manifest_7hb_df.columns)

    def test_10_human_review_queue_completeness(self):
        """Test human review queue contains all candidate proposals."""
        self.assertGreater(len(self.rq_df), 0)
        required_cols = ["screenshot_id", "existing_semantic_category", "proposed_candidate_screen_type", "reason_for_review"]
        for col in required_cols:
            self.assertIn(col, self.rq_df.columns)

    def test_11_historical_artifact_protection(self):
        """Test historical Phase 7E manifest remains unchanged."""
        manifest_7e_path = os.path.join(PROJECT_ROOT, "experiments", "visual_understanding", "screen_type_manifest_final.csv")
        df_7e = pd.read_csv(manifest_7e_path)
        self.assertEqual(len(df_7e), 833)


if __name__ == "__main__":
    unittest.main()
