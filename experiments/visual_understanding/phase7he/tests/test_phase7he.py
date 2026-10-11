"""
Unit test suite for Phase 7H-E Human Review Workflow Engine & Ground-Truth Finalization.
"""

import unittest
import os
import csv
import shutil
import tempfile
from pathlib import Path

# Path setup
TESTS_DIR = Path(__file__).resolve().parent
PHASE7HE_DIR = TESTS_DIR.parent
VISUAL_UNDERSTANDING_DIR = PHASE7HE_DIR.parent
EXPERIMENTS_DIR = VISUAL_UNDERSTANDING_DIR.parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent

sys_path_added = str(PHASE7HE_DIR)
if sys_path_added not in os.sys.path:
    os.sys.path.insert(0, sys_path_added)

from human_review_engine import (
    Phase7HEReviewEngine,
    FULL_VISUAL_TAXONOMY,
    LOCKED_PHASE7E_SCREENSHOT_IDS,
    CORRUPTED_SCREENSHOT_IDS,
    PRIORITY_TIER_SEQUENCE
)


class TestPhase7HEEngine(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_ledger_path = Path(self.temp_dir) / "test_phase7he_ledger.csv"
        self.engine = Phase7HEReviewEngine(ledger_path=self.temp_ledger_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ledger_initialization_and_counts(self):
        """Verify ledger initializes with exact corpus counts and pre-verified items."""
        rep = self.engine.generate_review_report()
        self.assertEqual(rep["total_records"], 836, "Ledger must contain 836 total records.")
        self.assertEqual(rep["total_valid"], 833, "Must have 833 valid screenshots.")
        self.assertEqual(rep["corrupted_count"], 3, "Must exclude 3 corrupted screenshots.")
        self.assertEqual(rep["status_counts"]["verified"], 12, "Must pre-verify 12 Phase 7E decisions.")
        self.assertEqual(rep["status_counts"]["pending"], 821, "Must have 821 pending annotations.")

    def test_preservation_of_previous_human_decisions(self):
        """Verify pre-verified 12 Phase 7E human decisions are locked and raise error if altered."""
        for sid in LOCKED_PHASE7E_SCREENSHOT_IDS:
            with self.assertRaises(ValueError, msg=f"Should raise ValueError for locked ID {sid}"):
                self.engine.record_decision(
                    screenshot_id=sid,
                    decision="correct",
                    final_screen_type="dashboard_settings"
                )

    def test_corrupted_screenshot_exclusion(self):
        """Verify corrupted 0-byte screenshots cannot be reviewed."""
        for sid in CORRUPTED_SCREENSHOT_IDS:
            with self.assertRaises(ValueError, msg=f"Should raise ValueError for corrupted ID {sid}"):
                self.engine.record_decision(
                    screenshot_id=sid,
                    decision="approve"
                )

    def test_valid_taxonomy_enforcement(self):
        """Verify invalid screen-type strings raise ValueError."""
        pending_items = self.engine.get_pending_queue()
        target_sid = pending_items[0]["screenshot_id"]

        with self.assertRaises(ValueError, msg="Should raise ValueError for invalid taxonomy class"):
            self.engine.record_decision(
                screenshot_id=target_sid,
                decision="correct",
                final_screen_type="invalid_custom_screen_class"
            )

    def test_incremental_persistence_and_resume(self):
        """Verify decisions persist incrementally to disk and reload upon engine re-instantiation."""
        pending_items = self.engine.get_pending_queue()
        target_sid = pending_items[0]["screenshot_id"]

        # Record decision
        self.engine.record_decision(
            screenshot_id=target_sid,
            decision="approve",
            reviewer="test_annotator",
            notes="Unit test approval"
        )

        # Confirm file updated on disk
        self.assertTrue(self.temp_ledger_path.exists())
        with open(self.temp_ledger_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            row = next(r for r in reader if r["screenshot_id"] == target_sid)
            self.assertEqual(row["human_decision"], "approve")
            self.assertEqual(row["reviewer"], "test_annotator")
            self.assertEqual(row["review_status"], "completed")

        # Reload engine from disk (Resume test)
        resumed_engine = Phase7HEReviewEngine(ledger_path=self.temp_ledger_path)
        resumed_rec = resumed_engine.records[target_sid]
        self.assertEqual(resumed_rec["human_decision"], "approve")
        self.assertEqual(resumed_rec["review_status"], "completed")

    def test_priority_queue_order(self):
        """Verify pending queue follows strict 12-tier priority sequence."""
        queue = self.engine.get_pending_queue()
        self.assertEqual(len(queue), 821, "Queue should start with 821 pending items.")
        
        # First items must belong to Tier 1 (dashboard_settings)
        self.assertEqual(queue[0]["priority_tier"], "Tier 1")
        self.assertEqual(queue[0]["phase7hc_proposed_screen_type"], "dashboard_settings")

    def test_final_manifest_generation_conditions(self):
        """Verify manifest generation fails if pending items remain, unless forced."""
        manifest_dest = Path(self.temp_dir) / "vsue_ground_truth_manifest_v1.csv"
        
        # Should fail with pending items
        with self.assertRaises(ValueError, msg="Manifest generation should fail when pending items remain"):
            self.engine.generate_final_manifest(force=False)

        # Force mode should generate manifest for valid items
        manifest_path = self.engine.generate_final_manifest(force=True)
        self.assertTrue(manifest_path.exists())
        
        with open(manifest_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 833, "Manifest must contain all 833 valid screenshots.")

if __name__ == "__main__":
    unittest.main()
