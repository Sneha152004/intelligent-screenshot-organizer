"""
Unit test suite for Phase 7H-F Partial Ground-Truth Audit & Model Benchmark Readiness.
"""

import unittest
import os
import csv
import pandas as pd
from pathlib import Path

# Paths
TESTS_DIR = Path(__file__).resolve().parent
PHASE7HF_DIR = TESTS_DIR.parent
VISUAL_UNDERSTANDING_DIR = PHASE7HF_DIR.parent
EXPERIMENTS_DIR = VISUAL_UNDERSTANDING_DIR.parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent

class TestPhase7HFIntegrity(unittest.TestCase):
    def setUp(self):
        self.comp_report = PHASE7HF_DIR / "phase7hf_completion_report.md"
        self.class_dist = PHASE7HF_DIR / "phase7hf_class_distribution.csv"
        self.eligibility = PHASE7HF_DIR / "phase7hf_training_eligibility.csv"
        self.rejected_adj = PHASE7HF_DIR / "phase7hf_rejected_adjudication_candidates.csv"
        self.readiness_md = PHASE7HF_DIR / "phase7hf_benchmark_readiness.md"

    def test_01_artifacts_exist(self):
        """1. Verify all Phase 7H-F deliverables exist on disk."""
        self.assertTrue(self.comp_report.exists(), f"Missing {self.comp_report}")
        self.assertTrue(self.class_dist.exists(), f"Missing {self.class_dist}")
        self.assertTrue(self.eligibility.exists(), f"Missing {self.eligibility}")
        self.assertTrue(self.rejected_adj.exists(), f"Missing {self.rejected_adj}")
        self.assertTrue(self.readiness_md.exists(), f"Missing {self.readiness_md}")

    def test_02_class_distribution_schema_and_counts(self):
        """2. Verify class distribution CSV has 13 canonical taxonomy classes."""
        df = pd.read_csv(self.class_dist)
        self.assertEqual(len(df), 13, "Class distribution must contain 13 taxonomy classes.")
        self.assertEqual(df["eligible_human_verified_count"].sum(), 443, "Sum of eligible human verified count must equal 443.")

    def test_03_training_eligibility_counts(self):
        """3. Verify training eligibility CSV has 836 records and exact status counts."""
        df = pd.read_csv(self.eligibility)
        self.assertEqual(len(df), 836, "Eligibility file must contain 836 records.")

        counts = df["training_eligibility_status"].value_counts().to_dict()
        self.assertEqual(counts.get("ELIGIBLE_GROUND_TRUTH", 0), 443, "Must have 443 ELIGIBLE_GROUND_TRUTH records.")
        self.assertEqual(counts.get("EXCLUDED_CORRUPTED", 0), 3, "Must have 3 EXCLUDED_CORRUPTED records.")
        self.assertEqual(counts.get("EXCLUDED_REJECTED", 0), 51, "Must have 51 EXCLUDED_REJECTED records.")
        self.assertEqual(counts.get("EXCLUDED_AMBIGUOUS", 0), 1, "Must have 1 EXCLUDED_AMBIGUOUS record.")
        self.assertEqual(counts.get("EXCLUDED_PENDING", 0), 338, "Must have 338 EXCLUDED_PENDING records.")

    def test_04_rejected_adjudication_candidates(self):
        """4. Verify rejected adjudication candidates CSV has 51 records."""
        df = pd.read_csv(self.rejected_adj)
        self.assertEqual(len(df), 51, "Must contain 51 rejected valid candidates.")

    def test_05_benchmark_readiness_verdict(self):
        """5. Verify audit verdict is READY WITH CLASS RESTRICTIONS."""
        with open(self.comp_report, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("READY WITH CLASS RESTRICTIONS", content)

        with open(self.readiness_md, "r", encoding="utf-8") as f:
            content_bench = f.read()
        self.assertIn("READY WITH CLASS RESTRICTIONS", content_bench)


if __name__ == "__main__":
    unittest.main()
