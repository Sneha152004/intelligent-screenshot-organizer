"""
Unit Tests for Phase 3G.2 External Dataset Evaluation Harness
==============================================================
Tests external dataset discovery, target coverage, manifest consistency, evaluator output schema,
result count (60 items), source image immutability, and missing file exception handling.
"""

from pathlib import Path
import unittest

import pandas as pd

from experiments.intelligence.external.evaluation_targets import (
    get_evaluation_targets,
    get_target_by_filename,
    EVALUATION_TARGET_FILENAMES,
)
from experiments.intelligence.evaluate_external import (
    evaluate_external_dataset,
    check_dataset_integrity_and_overlap,
)

EXTERNAL_DIR = Path(__file__).resolve().parent.parent / "experiments" / "intelligence" / "external"
EXTERNAL_IMAGES_DIR = EXTERNAL_DIR / "images"


class TestExternalDatasetDiscoveryAndIntegrity(unittest.TestCase):
    """Verifies external dataset structure, manifest consistency, and source image immutability."""

    def test_external_dataset_discovery_and_count(self):
        """Verify exactly 60 external image files exist on disk."""
        images = sorted(list(EXTERNAL_IMAGES_DIR.glob("*.jpg")))
        self.assertEqual(len(images), 60, f"Expected 60 images, found {len(images)}")
        self.assertEqual(len(EVALUATION_TARGET_FILENAMES), 60)

    def test_annotation_coverage(self):
        """Verify all 60 images have corresponding evaluation targets."""
        targets = get_evaluation_targets()
        self.assertEqual(len(targets), 60)
        target_filenames = {t["filename"] for t in targets}
        for fname in EVALUATION_TARGET_FILENAMES:
            self.assertIn(fname, target_filenames)

    def test_manifest_consistency(self):
        """Verify manifest.csv has 60 valid rows matching filenames and unique IDs."""
        manifest_path = EXTERNAL_DIR / "manifest.csv"
        self.assertTrue(manifest_path.exists())
        df = pd.read_csv(manifest_path)

        self.assertEqual(len(df), 60)
        self.assertEqual(df["id"].nunique(), 60)
        self.assertEqual(df["filename"].nunique(), 60)

        # Check each manifest image exists
        for fname in df["filename"]:
            self.assertTrue((EXTERNAL_IMAGES_DIR / fname).exists())

    def test_zero_exact_sha256_overlap_with_original(self):
        """Verify 0 exact SHA-256 file overlaps with the original 30 screenshots."""
        integ = check_dataset_integrity_and_overlap()
        self.assertEqual(len(integ["exact_sha256_overlaps_with_original"]), 0)

    def test_source_image_immutability(self):
        """Verify source images are untouched and unmutated during evaluation."""
        sample_img = EXTERNAL_IMAGES_DIR / "ext_001.jpg"
        self.assertTrue(sample_img.exists())
        size_before = sample_img.stat().st_size

        # Run mock evaluation
        _ = evaluate_external_dataset(use_mock_ocr=True)

        size_after = sample_img.stat().st_size
        self.assertEqual(size_before, size_after, "Source image file was mutated during evaluation!")

    def test_get_target_by_filename_lookup_and_invalid_handling(self):
        """Verify target lookup and missing file handling."""
        target = get_target_by_filename("ext_001.jpg")
        self.assertEqual(target["filename"], "ext_001.jpg")
        self.assertEqual(target["category"], "Education")

        with self.assertRaises(KeyError):
            get_target_by_filename("non_existent_file.jpg")


class TestExternalEvaluatorExecution(unittest.TestCase):
    """Tests evaluate_external.py execution and output schema."""

    def test_evaluate_external_dataset_run(self):
        """Executes full external evaluation run and validates result count and schema."""
        data = evaluate_external_dataset(use_mock_ocr=True)

        self.assertEqual(data["dataset_size"], 60)
        self.assertEqual(len(data["per_image_predictions"]), 60)

        # Validate metrics structure
        m = data["aggregate_metrics"]
        self.assertIn("category_accuracy", m)
        self.assertIn("tag_f1", m)
        self.assertEqual(m["schema_validity_rate"], 1.0)

        # Validate latencies structure
        l = data["latency_percentiles"]
        for stage in ["preprocessing_ms", "ocr_ms", "rules_ms", "embedding_ms", "classification_ms", "total_ms"]:
            self.assertIn(stage, l)
            self.assertIn("mean", l[stage])
            self.assertIn("p50", l[stage])
            self.assertIn("p95", l[stage])

        self.assertIn("cold_start_initialization_ms", l)

        # Validate result files generated
        results_dir = Path(__file__).resolve().parent.parent / "experiments" / "intelligence" / "results"
        self.assertTrue((results_dir / "external_evaluation_results.json").exists())
        self.assertTrue((results_dir / "external_evaluation_report.md").exists())


if __name__ == "__main__":
    unittest.main()
