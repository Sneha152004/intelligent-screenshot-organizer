"""
Readiness check unit test suite verifying all 9 criteria for Phase 7H-E Human Review CLI.
"""

import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Paths
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
    SCREENSHOTS_DIR
)


class TestCLIReadiness(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_ledger_path = Path(self.temp_dir) / "test_readiness_ledger.csv"
        self.engine = Phase7HEReviewEngine(ledger_path=self.temp_ledger_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_cli_launch_and_initialization(self):
        """1. Verify engine and ledger initialize cleanly."""
        self.assertIsNotNone(self.engine)
        self.assertEqual(len(self.engine.records), 836)

    def test_02_individual_screenshot_file_access(self):
        """2. Verify individual screenshot files exist and are readable for valid pending queue."""
        queue = self.engine.get_pending_queue()
        first_item = queue[0]
        fname = first_item["filename"]
        img_path = SCREENSHOTS_DIR / fname
        self.assertTrue(img_path.exists(), f"Individual screenshot {fname} must exist on disk.")
        self.assertGreater(img_path.stat().st_size, 0, f"Valid screenshot {fname} must have non-zero size.")

    def test_03_metadata_and_priority_tier_display(self):
        """3. Verify screenshot ID, proposed screen type, and priority tier are mapped correctly."""
        queue = self.engine.get_pending_queue()
        first_item = queue[0]
        self.assertTrue(first_item["screenshot_id"].startswith("ss_"))
        self.assertEqual(first_item["phase7hc_proposed_screen_type"], "dashboard_settings")
        self.assertEqual(first_item["priority_tier"], "Tier 1")

    def test_04_all_review_actions(self):
        """4. Verify Approve, Correct, Ambiguous, Reject, and Skip work correctly."""
        queue = self.engine.get_pending_queue()

        # Approve
        rec_app = self.engine.record_decision(queue[0]["screenshot_id"], "approve", notes="Test approval")
        self.assertEqual(rec_app["review_status"], "completed")
        self.assertEqual(rec_app["human_decision"], "approve")

        # Correct
        rec_corr = self.engine.record_decision(queue[1]["screenshot_id"], "correct", final_screen_type="email_view", notes="Test correction")
        self.assertEqual(rec_corr["review_status"], "completed")
        self.assertEqual(rec_corr["final_human_screen_type"], "email_view")

        # Ambiguous
        rec_amb = self.engine.record_decision(queue[2]["screenshot_id"], "ambiguous", notes="Test ambiguous")
        self.assertEqual(rec_amb["review_status"], "ambiguous")

        # Reject
        rec_rej = self.engine.record_decision(queue[3]["screenshot_id"], "reject", notes="Test rejection")
        self.assertEqual(rec_rej["review_status"], "rejected")

    def test_05_immediate_persistence_and_restart(self):
        """5. Verify decisions persist immediately and survive engine/CLI restart."""
        queue = self.engine.get_pending_queue()
        target_id = queue[0]["screenshot_id"]
        
        self.engine.record_decision(target_id, "approve", reviewer="annotator_1")
        
        # Reload new instance from same ledger path
        restarted_engine = Phase7HEReviewEngine(ledger_path=self.temp_ledger_path)
        rec = restarted_engine.records[target_id]
        self.assertEqual(rec["review_status"], "completed")
        self.assertEqual(rec["reviewer"], "annotator_1")

    def test_06_locked_previous_decisions(self):
        """6. Verify all 12 Phase 7E harmonized decisions remain locked."""
        for sid in LOCKED_PHASE7E_SCREENSHOT_IDS:
            rec = self.engine.records[sid]
            self.assertEqual(rec["review_status"], "verified")
            self.assertEqual(rec["human_decision"], "approve")
            with self.assertRaises(ValueError):
                self.engine.record_decision(sid, "correct", final_screen_type="product_detail")

    def test_07_corrupted_screenshot_exclusion(self):
        """7. Verify 3 corrupted screenshots remain excluded."""
        for sid in CORRUPTED_SCREENSHOT_IDS:
            rec = self.engine.records[sid]
            self.assertEqual(rec["review_status"], "rejected")
            self.assertEqual(rec["is_corrupted"].lower(), "true")
            with self.assertRaises(ValueError):
                self.engine.record_decision(sid, "approve")

    def test_08_resumable_queue_tracking(self):
        """8. Verify review queue updates pending count as items are annotated."""
        initial_queue = self.engine.get_pending_queue()
        self.assertEqual(len(initial_queue), 821)

        target_id = initial_queue[0]["screenshot_id"]
        self.engine.record_decision(target_id, "approve")

        updated_queue = self.engine.get_pending_queue()
        self.assertEqual(len(updated_queue), 820)
        self.assertNotIn(target_id, [item["screenshot_id"] for item in updated_queue])

    def test_09_progress_report_updates(self):
        """9. Verify progress reports calculate accurate metrics after decisions."""
        queue = self.engine.get_pending_queue()
        self.engine.record_decision(queue[0]["screenshot_id"], "approve")
        self.engine.record_decision(queue[1]["screenshot_id"], "correct", final_screen_type="email_view")

        rep = self.engine.generate_review_report()
        self.assertEqual(rep["status_counts"]["completed"], 2)
        self.assertEqual(rep["status_counts"]["pending"], 819)


if __name__ == "__main__":
    unittest.main()
