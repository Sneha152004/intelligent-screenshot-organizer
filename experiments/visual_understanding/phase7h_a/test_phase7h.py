import json
import os
import sys
import hashlib
import unittest
import pandas as pd
from PIL import Image

PROJECT_ROOT = r"C:\Users\KIIT0001\Documents\intelligent-screenshot-organizer"
PHASE7H_A_DIR = os.path.join(PROJECT_ROOT, "experiments", "visual_understanding", "phase7h_a")

FULL_TAXONOMY = [
    "product_listing",
    "product_detail",
    "cart_checkout_payment",
    "login_authentication",
    "messaging_chat",
    "email_view",
    "media_player_library",
    "code_editor_terminal",
    "document_article_view",
    "travel_booking",
    "dashboard_settings",
    "error_dialog_alert",
    "other_screen",
]

ALLOWED_STATUSES = ["verified", "ambiguous", "needs_review", "rejected"]


class TestPhase7HDatasetIntegrity(unittest.TestCase):

    def setUp(self):
        self.manifest_path = os.path.join(PHASE7H_A_DIR, "screen_type_manifest_phase7h.csv")
        self.split_path = os.path.join(PHASE7H_A_DIR, "vsue_dataset_split_phase7h.json")
        self.dup_audit_path = os.path.join(PHASE7H_A_DIR, "duplicate_audit.csv")

        self.assertTrue(os.path.exists(self.manifest_path), "Manifest file missing!")
        self.assertTrue(os.path.exists(self.split_path), "Split JSON file missing!")
        self.assertTrue(os.path.exists(self.dup_audit_path), "Duplicate audit file missing!")

        self.manifest_df = pd.read_csv(self.manifest_path)
        with open(self.split_path, "r", encoding="utf-8") as f:
            self.split_data = json.load(f)

    def test_01_manifest_schema(self):
        """Test schema validity of Phase 7H manifest."""
        required_cols = [
            "screenshot_id", "image_path", "screen_type", "annotation_status",
            "annotation_confidence", "visual_evidence", "annotator_notes",
            "duplicate_group_id", "source", "review_status", "md5_hash"
        ]
        for col in required_cols:
            self.assertIn(col, self.manifest_df.columns, f"Missing required column: {col}")

    def test_02_unique_screenshot_ids(self):
        """Test all screenshot IDs in manifest are strictly unique."""
        sids = self.manifest_df["screenshot_id"].tolist()
        self.assertEqual(len(sids), len(set(sids)), "Duplicate screenshot IDs detected in manifest!")

    def test_03_valid_taxonomy_values(self):
        """Test all screen_type labels belong to the canonical 13-class taxonomy."""
        types = self.manifest_df["screen_type"].unique()
        for t in types:
            self.assertIn(t, FULL_TAXONOMY, f"Invalid taxonomy label: {t}")

    def test_04_valid_annotation_statuses(self):
        """Test annotation_status values are valid enum strings."""
        statuses = self.manifest_df["annotation_status"].unique()
        for s in statuses:
            self.assertIn(s, ALLOWED_STATUSES, f"Invalid annotation status: {s}")

    def test_05_verified_evidence(self):
        """Test all verified records have non-empty visual evidence descriptions."""
        verified = self.manifest_df[self.manifest_df["annotation_status"] == "verified"]
        for _, row in verified.iterrows():
            ev = str(row["visual_evidence"]).strip()
            self.assertTrue(len(ev) > 10, f"Missing or trivial visual evidence for {row['screenshot_id']}")

    def test_06_image_readability(self):
        """Test all verified images exist and are valid readable PIL images."""
        verified = self.manifest_df[self.manifest_df["annotation_status"] == "verified"]
        for _, row in verified.iterrows():
            full_path = os.path.join(PROJECT_ROOT, row["image_path"])
            self.assertTrue(os.path.exists(full_path), f"Image file missing: {full_path}")
            img = Image.open(full_path)
            self.assertGreater(img.width, 0)
            self.assertGreater(img.height, 0)

    def test_07_duplicate_label_consistency(self):
        """Test 100% label consistency within duplicate groups."""
        verified = self.manifest_df[self.manifest_df["annotation_status"] == "verified"]
        grouped = verified.groupby("duplicate_group_id")["screen_type"].nunique()
        inconsistent = grouped[grouped > 1]
        self.assertEqual(len(inconsistent), 0, f"Inconsistent duplicate group labels found: {inconsistent.to_dict()}")

    def test_08_group_aware_partition_isolation(self):
        """Test duplicate group members are assigned to the exact same partition."""
        verified = self.manifest_df[self.manifest_df["annotation_status"] == "verified"]
        grouped = verified.groupby("duplicate_group_id")["partition_split"].nunique()
        split_inconsistent = grouped[grouped > 1]
        self.assertEqual(len(split_inconsistent), 0, f"Duplicate group split across partitions: {split_inconsistent.to_dict()}")

    def test_09_zero_leakage(self):
        """Test zero screenshot ID and zero MD5 hash leakage across partitions."""
        train_ids = set(self.split_data["train_ids"])
        cal_ids = set(self.split_data["calibration_ids"])
        val_ids = set(self.split_data["validation_ids"])

        self.assertEqual(len(train_ids & cal_ids), 0, "Train and Cal ID leakage detected!")
        self.assertEqual(len(train_ids & val_ids), 0, "Train and Val ID leakage detected!")
        self.assertEqual(len(cal_ids & val_ids), 0, "Cal and Val ID leakage detected!")

        # MD5 leakage
        id_to_md5 = dict(zip(self.manifest_df["screenshot_id"], self.manifest_df["md5_hash"]))
        train_md5 = {id_to_md5[i] for i in train_ids}
        cal_md5 = {id_to_md5[i] for i in cal_ids}
        val_md5 = {id_to_md5[i] for i in val_ids}

        self.assertEqual(len(train_md5 & cal_md5), 0, "Train and Cal MD5 hash leakage detected!")
        self.assertEqual(len(train_md5 & val_md5), 0, "Train and Val MD5 hash leakage detected!")
        self.assertEqual(len(cal_md5 & val_md5), 0, "Cal and Val MD5 hash leakage detected!")

    def test_10_split_exclusion(self):
        """Test only verified screenshots are included in split JSON."""
        unverified_ids = set(self.manifest_df[self.manifest_df["annotation_status"] != "verified"]["screenshot_id"])
        all_split_ids = set(self.split_data["train_ids"]) | set(self.split_data["calibration_ids"]) | set(self.split_data["validation_ids"])

        leaked_unverified = unverified_ids & all_split_ids
        self.assertEqual(len(leaked_unverified), 0, f"Unverified/rejected records leaked into split: {leaked_unverified}")


if __name__ == "__main__":
    unittest.main()
