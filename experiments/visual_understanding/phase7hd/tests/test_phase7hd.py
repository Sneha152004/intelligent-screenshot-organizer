"""
Unit test suite for Phase 7H-D Human Verification Preparation and Ground-Truth Integrity Audit.
"""

import unittest
import os
import csv
from pathlib import Path

# Paths
TESTS_DIR = Path(__file__).resolve().parent
PHASE7HD_DIR = TESTS_DIR.parent
VISUAL_UNDERSTANDING_DIR = PHASE7HD_DIR.parent
EXPERIMENTS_DIR = VISUAL_UNDERSTANDING_DIR.parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent

PHASE7HC_DIR = VISUAL_UNDERSTANDING_DIR / "phase7hc"
CORRECTED_CS_DIR = PHASE7HC_DIR / "corrected_contact_sheets"

class TestPhase7HDIntegrity(unittest.TestCase):
    def setUp(self):
        self.mapping_file = PHASE7HD_DIR / "phase7hd_unique_screenshot_mapping.csv"
        self.ledger_file = PHASE7HD_DIR / "phase7hd_human_review_ledger.csv"
        self.previous_decisions_file = PHASE7HD_DIR / "phase7hd_previous_human_decisions_audit.csv"
        self.coverage_report = PHASE7HD_DIR / "phase7hd_contact_sheet_coverage_report.md"
        self.priority_plan = PHASE7HD_DIR / "phase7hd_review_priority_plan.md"
        self.reconciliation_report = PHASE7HD_DIR / "phase7hd_reconciliation_report.md"

    def test_artifacts_exist(self):
        """Verify all Phase 7H-D deliverables exist."""
        self.assertTrue(self.mapping_file.exists(), f"Missing {self.mapping_file}")
        self.assertTrue(self.ledger_file.exists(), f"Missing {self.ledger_file}")
        self.assertTrue(self.previous_decisions_file.exists(), f"Missing {self.previous_decisions_file}")
        self.assertTrue(self.coverage_report.exists(), f"Missing {self.coverage_report}")
        self.assertTrue(self.priority_plan.exists(), f"Missing {self.priority_plan}")
        self.assertTrue(self.reconciliation_report.exists(), f"Missing {self.reconciliation_report}")

    def test_unique_screenshot_mapping(self):
        """Verify screenshot mapping has 836 unique records and exact totals."""
        with open(self.mapping_file, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        
        self.assertEqual(len(reader), 836, "Mapping must contain exactly 836 entries.")
        
        screenshot_ids = [r["screenshot_id"] for r in reader]
        self.assertEqual(len(screenshot_ids), len(set(screenshot_ids)), "Screenshot IDs must be unique.")
        
        filenames = [r["filename"] for r in reader]
        self.assertEqual(len(filenames), len(set(filenames)), "Filenames must be 1-to-1 unique.")

    def test_corrupted_image_handling(self):
        """Verify corrupted 0-byte images are properly flagged and rejected."""
        corrupted_files = {"ss_441.png", "ss_443.png", "ss_560.png"}
        with open(self.ledger_file, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        
        corrupted_in_ledger = [r for r in reader if r["screenshot_id"] in corrupted_files]
        self.assertEqual(len(corrupted_in_ledger), 3, "Exactly 3 corrupted files must be present in ledger.")
        
        for record in corrupted_in_ledger:
            self.assertEqual(record["human_decision"], "reject", f"{record['screenshot_id']} must have human_decision=reject")
            self.assertEqual(record["human_verified_screen_type"], "rejected", f"{record['screenshot_id']} must have human_verified_screen_type=rejected")
            self.assertEqual(record["reviewer"], "system_audit", f"{record['screenshot_id']} must have reviewer=system_audit")

    def test_human_decision_preservation(self):
        """Verify all 12 previously human-reviewed decisions are 100% preserved."""
        with open(self.previous_decisions_file, "r", encoding="utf-8") as f:
            audit_records = list(csv.DictReader(f))
        
        self.assertEqual(len(audit_records), 12, "Audit file must contain 12 human decisions.")
        
        with open(self.ledger_file, "r", encoding="utf-8") as f:
            ledger_dict = {r["screenshot_id"]: r for r in csv.DictReader(f)}
        
        for audit in audit_records:
            sid = audit["screenshot_id"]
            self.assertIn(sid, ledger_dict, f"Human verified file {sid} missing from review ledger.")
            ledger_rec = ledger_dict[sid]
            self.assertEqual(ledger_rec["human_decision"], "approve", f"File {sid} human_decision should be approve.")
            self.assertEqual(ledger_rec["human_verified_screen_type"], audit["human_approved_screen_type"],
                             f"File {sid} visual screen type mismatch.")
            self.assertEqual(audit["phase7hc_preservation_status"], "PRESERVED_EXACT",
                             f"File {sid} preservation status must be PRESERVED_EXACT.")

    def test_ledger_schema_and_counts(self):
        """Verify human review ledger schema and record counts."""
        expected_columns = {
            "screenshot_id", "original_semantic_category", "phase7hc_proposed_screen_type",
            "human_verified_screen_type", "human_decision", "reviewer",
            "review_notes", "verification_date", "duplicate_group_id"
        }
        with open(self.ledger_file, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        
        self.assertEqual(len(reader), 836, "Ledger must contain 836 rows.")
        self.assertTrue(expected_columns.issubset(set(reader[0].keys())), f"Missing expected columns in ledger: {expected_columns - set(reader[0].keys())}")
        
        decision_counts = {"approve": 0, "reject": 0, "pending": 0}
        for r in reader:
            decision_counts[r["human_decision"]] += 1
            
        self.assertEqual(decision_counts["approve"], 12, "Should be exactly 12 approved records.")
        self.assertEqual(decision_counts["reject"], 3, "Should be exactly 3 rejected records.")
        self.assertEqual(decision_counts["pending"], 821, "Should be exactly 821 pending records.")

    def test_contact_sheet_files_exist(self):
        """Verify all 57 contact sheet files exist on disk."""
        self.assertTrue(CORRECTED_CS_DIR.exists(), f"Directory missing: {CORRECTED_CS_DIR}")
        cs_files = list(CORRECTED_CS_DIR.glob("*.png"))
        self.assertEqual(len(cs_files), 57, f"Expected 57 contact sheet PNG files, found {len(cs_files)}.")

if __name__ == "__main__":
    unittest.main()
