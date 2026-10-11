"""
Unit test suite for Phase 7H-G Group-Aware Dataset Split & Benchmark Construction.
"""

import unittest
import os
import sys
import json
import pandas as pd
from pathlib import Path

# Paths
TESTS_DIR = Path(__file__).resolve().parent
PHASE7HG_DIR = TESTS_DIR.parent
VISUAL_UNDERSTANDING_DIR = PHASE7HG_DIR.parent
PHASE7HE_DIR = VISUAL_UNDERSTANDING_DIR / "phase7he"
EXPERIMENTS_DIR = VISUAL_UNDERSTANDING_DIR.parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent

sys_path_added = str(PHASE7HG_DIR)
if sys_path_added not in os.sys.path:
    os.sys.path.insert(0, sys_path_added)

from build_phase7hg_splits import audit_leakage, perform_group_aware_split, PRIMARY_CLASSES, EXTENDED_CLASSES


class TestPhase7HGIntegrity(unittest.TestCase):
    def setUp(self):
        self.primary_csv = PHASE7HG_DIR / "phase7hg_primary_manifest.csv"
        self.extended_csv = PHASE7HG_DIR / "phase7hg_extended_manifest.csv"
        self.zero_shot_csv = PHASE7HG_DIR / "phase7hg_zero_shot_probe.csv"
        self.split_summary_json = PHASE7HG_DIR / "phase7hg_split_summary.json"
        self.class_counts_csv = PHASE7HG_DIR / "phase7hg_class_partition_counts.csv"
        self.leakage_audit_json = PHASE7HG_DIR / "phase7hg_leakage_audit.json"
        self.readiness_md = PHASE7HG_DIR / "phase7hg_benchmark_readiness.md"
        self.comp_report = PHASE7HG_DIR / "phase7hg_completion_report.md"

    def test_01_artifacts_exist(self):
        """1. Verify all Phase 7H-G deliverables exist on disk."""
        self.assertTrue(self.primary_csv.exists(), f"Missing {self.primary_csv}")
        self.assertTrue(self.extended_csv.exists(), f"Missing {self.extended_csv}")
        self.assertTrue(self.zero_shot_csv.exists(), f"Missing {self.zero_shot_csv}")
        self.assertTrue(self.split_summary_json.exists(), f"Missing {self.split_summary_json}")
        self.assertTrue(self.class_counts_csv.exists(), f"Missing {self.class_counts_csv}")
        self.assertTrue(self.leakage_audit_json.exists(), f"Missing {self.leakage_audit_json}")
        self.assertTrue(self.readiness_md.exists(), f"Missing {self.readiness_md}")
        self.assertTrue(self.comp_report.exists(), f"Missing {self.comp_report}")

    def test_02_primary_manifest_integrity_and_counts(self):
        """2. Verify Primary 5-class manifest counts and class filtering."""
        df = pd.read_csv(self.primary_csv)
        self.assertEqual(len(df), 135, "Primary benchmark must contain 135 items.")
        self.assertTrue(set(df["screen_type"]).issubset(set(PRIMARY_CLASSES)), "Primary manifest classes must be a subset of PRIMARY_CLASSES.")

        partition_counts = df["partition"].value_counts().to_dict()
        self.assertEqual(partition_counts["train"], 83, "Train split must contain 83 items.")
        self.assertEqual(partition_counts["calibration"], 27, "Calibration split must contain 27 items.")
        self.assertEqual(partition_counts["validation"], 25, "Validation split must contain 25 items.")

    def test_03_extended_manifest_integrity_and_counts(self):
        """3. Verify Extended 8-class manifest counts and class filtering."""
        df = pd.read_csv(self.extended_csv)
        self.assertEqual(len(df), 147, "Extended benchmark must contain 147 items.")
        self.assertTrue(set(df["screen_type"]).issubset(set(EXTENDED_CLASSES)), "Extended manifest classes must be a subset of EXTENDED_CLASSES.")

        partition_counts = df["partition"].value_counts().to_dict()
        self.assertEqual(partition_counts["train"], 89, "Train split must contain 89 items.")
        self.assertEqual(partition_counts["calibration"], 30, "Calibration split must contain 30 items.")
        self.assertEqual(partition_counts["validation"], 28, "Validation split must contain 28 items.")

    def test_04_zero_shot_probe_isolation(self):
        """4. Verify zero-shot probe contains only product_listing screenshots."""
        df = pd.read_csv(self.zero_shot_csv)
        self.assertEqual(len(df), 31, "Zero-shot probe must contain 31 product_listing items.")
        self.assertEqual(set(df["screen_type"]), {"product_listing"}, "Zero-shot probe must strictly contain product_listing.")

    def test_05_group_aware_splitting_and_zero_leakage(self):
        """5. Verify zero group, MD5, screenshot ID, or filename leakage across partitions."""
        for csv_file, name in [(self.primary_csv, "Primary"), (self.extended_csv, "Extended")]:
            df = pd.read_csv(csv_file)
            audit = audit_leakage(df, name)
            self.assertTrue(audit["overall_leakage_audit_pass"], f"Leakage audit failed for {name} benchmark.")
            for chk_name, chk_res in audit["checks"].items():
                self.assertTrue(chk_res["pass"], f"Leakage check {chk_name} failed for {name} benchmark.")

    def test_06_deterministic_reproducibility(self):
        """6. Verify re-running split with seed 42 produces identical partitions."""
        df_p1 = pd.read_csv(self.primary_csv)
        df_e1 = pd.read_csv(self.extended_csv)

        ledger_df = pd.read_csv(PHASE7HE_DIR / "phase7he_human_review_ledger.csv")
        eligible_df = ledger_df[(ledger_df["human_decision"].isin(["approve", "correct"])) | (ledger_df["review_status"] == "verified")].copy()
        
        def get_effective_label(row):
            if row["human_decision"] == "correct" and pd.notna(row.get("final_human_screen_type")) and str(row["final_human_screen_type"]).strip():
                return str(row["final_human_screen_type"]).strip()
            return str(row["phase7hc_proposed_screen_type"]).strip()

        import hashlib
        eligible_df["effective_screen_type"] = eligible_df.apply(get_effective_label, axis=1)
        eligible_df["md5_hash"] = eligible_df["filename"].apply(lambda f: hashlib.md5(((PROJECT_ROOT / "data" / "screenshots") / f).read_bytes()).hexdigest())
        eligible_df["canonical_group_id"] = "GRP_" + eligible_df["md5_hash"].str[:12]

        split_p2 = perform_group_aware_split(eligible_df, PRIMARY_CLASSES, seed=42)
        
        map_p1 = dict(zip(df_p1["screenshot_id"], df_p1["partition"]))
        map_p2 = dict(zip(split_p2["screenshot_id"], split_p2["partition"]))

        self.assertEqual(map_p1, map_p2, "Primary benchmark split must be 100% deterministically reproducible.")

    def test_07_failure_behavior_on_simulated_leakage(self):
        """7. Verify leakage audit function fails when artificial leakage is introduced."""
        df = pd.read_csv(self.primary_csv).copy()
        
        # Inject artificial leakage: duplicate a train screenshot into validation
        train_row = df[df["partition"] == "train"].iloc[0].copy()
        train_row["partition"] = "validation"
        leaky_df = pd.concat([df, pd.DataFrame([train_row])], ignore_index=True)

        audit_result = audit_leakage(leaky_df, "Leaky Benchmark Test")
        self.assertFalse(audit_result["overall_leakage_audit_pass"], "Audit must fail when leakage is present.")

    def test_08_image_file_existence_and_non_zero_size(self):
        """8. Verify all benchmark screenshots exist on disk with non-zero byte size."""
        for csv_file in [self.primary_csv, self.extended_csv, self.zero_shot_csv]:
            df = pd.read_csv(csv_file)
            for fname in df["filename"]:
                img_path = PROJECT_ROOT / "data" / "screenshots" / fname
                self.assertTrue(img_path.exists(), f"Image {fname} missing from disk.")
                self.assertGreater(img_path.stat().st_size, 0, f"Image {fname} has 0-byte size.")


if __name__ == "__main__":
    unittest.main()
