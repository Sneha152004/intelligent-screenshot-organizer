"""
Unit test suite for Phase 7H-G-A Benchmark Manifest Reconciliation & Integrity Audit.
"""

import unittest
import os
import sys
import json
import pandas as pd
from pathlib import Path

# Paths
TESTS_DIR = Path(__file__).resolve().parent
PHASE7HGA_DIR = TESTS_DIR
VISUAL_UNDERSTANDING_DIR = PHASE7HGA_DIR.parent
EXPERIMENTS_DIR = VISUAL_UNDERSTANDING_DIR.parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent


class TestPhase7HGAReconciliation(unittest.TestCase):
    def setUp(self):
        self.report_md = PHASE7HGA_DIR / "phase7hga_reconciliation_report.md"
        self.missing_primary_csv = PHASE7HGA_DIR / "phase7hga_missing_primary_records.csv"
        self.missing_extended_csv = PHASE7HGA_DIR / "phase7hga_missing_extended_records.csv"
        self.unexpected_csv = PHASE7HGA_DIR / "phase7hga_unexpected_manifest_records.csv"
        self.zero_shot_audit_csv = PHASE7HGA_DIR / "phase7hga_zero_shot_probe_audit.csv"
        self.class_comp_csv = PHASE7HGA_DIR / "phase7hga_class_count_comparison.csv"
        self.partition_arithmetic_json = PHASE7HGA_DIR / "phase7hga_partition_arithmetic.json"

    def test_01_deliverables_exist(self):
        """1. Verify all Phase 7H-G-A deliverables exist on disk."""
        self.assertTrue(self.report_md.exists(), f"Missing {self.report_md}")
        self.assertTrue(self.missing_primary_csv.exists(), f"Missing {self.missing_primary_csv}")
        self.assertTrue(self.missing_extended_csv.exists(), f"Missing {self.missing_extended_csv}")
        self.assertTrue(self.unexpected_csv.exists(), f"Missing {self.unexpected_csv}")
        self.assertTrue(self.zero_shot_audit_csv.exists(), f"Missing {self.zero_shot_audit_csv}")
        self.assertTrue(self.class_comp_csv.exists(), f"Missing {self.class_comp_csv}")
        self.assertTrue(self.partition_arithmetic_json.exists(), f"Missing {self.partition_arithmetic_json}")

    def test_02_zero_missing_primary_records(self):
        """2. Verify 0 eligible ground truth primary records were omitted."""
        df = pd.read_csv(self.missing_primary_csv)
        self.assertEqual(len(df), 0, "Missing primary records CSV must contain 0 rows.")

    def test_03_zero_missing_extended_records(self):
        """3. Verify 0 eligible ground truth extended records were omitted."""
        df = pd.read_csv(self.missing_extended_csv)
        self.assertEqual(len(df), 0, "Missing extended records CSV must contain 0 rows.")

    def test_04_zero_unexpected_manifest_records(self):
        """4. Verify 0 ineligible/unverified records entered the benchmark manifests."""
        df = pd.read_csv(self.unexpected_csv)
        self.assertEqual(len(df), 0, "Unexpected manifest records CSV must contain 0 rows.")

    def test_05_zero_shot_probe_audit(self):
        """5. Verify zero-shot probe audit contains 31 valid product_listing records."""
        df = pd.read_csv(self.zero_shot_audit_csv)
        self.assertEqual(len(df), 31, "Zero-shot probe audit must contain 31 records.")
        self.assertTrue((df["probe_eligibility_verdict"] == "VALID_ZERO_SHOT_PROBE").all(),
                        "All probe records must have verdict VALID_ZERO_SHOT_PROBE.")

    def test_06_partition_arithmetic_accuracy(self):
        """6. Verify partition arithmetic JSON matches expected counts."""
        with open(self.partition_arithmetic_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        prim = data["primary_benchmark_arithmetic"]
        self.assertEqual(prim["total_screenshots"], 135)
        self.assertEqual(prim["total_groups"], 128)
        self.assertEqual(prim["train"]["screenshot_count"], 83)
        self.assertEqual(prim["calibration"]["screenshot_count"], 27)
        self.assertEqual(prim["validation"]["screenshot_count"], 25)

        ext = data["extended_benchmark_arithmetic"]
        self.assertEqual(ext["total_screenshots"], 147)
        self.assertEqual(ext["total_groups"], 139)
        self.assertEqual(ext["train"]["screenshot_count"], 89)
        self.assertEqual(ext["calibration"]["screenshot_count"], 30)
        self.assertEqual(ext["validation"]["screenshot_count"], 28)

    def test_07_reconciliation_verdict(self):
        """7. Verify final reconciliation report verdict is RECONCILIATION PASSED."""
        with open(self.report_md, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("RECONCILIATION PASSED", content)


if __name__ == "__main__":
    unittest.main()
