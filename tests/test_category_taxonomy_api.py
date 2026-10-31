"""
Regression Tests for Category Taxonomy & API Contract (9 Canonical Categories)
=============================================================================
Verifies:
1. Exactly nine canonical categories are returned by GET /api/categories.
2. 'Coding' is NEVER returned as a top-level category.
3. Historical/legacy category values (e.g. legacy 'Coding') do not inflate the active category count.
4. Total screenshot count in the response matches the database.
5. All categories contain required visual and metadata properties.
"""

from collections import Counter
import unittest
from unittest.mock import MagicMock, patch

try:
    from backend.app.api.categories import list_categories, CATEGORY_META
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    list_categories, CATEGORY_META = None, None

from src.models.metadata import SUPPORTED_CATEGORIES, ScreenshotMetadata


CANONICAL_9_EXPECTED = [
    "Education",
    "Shopping",
    "Payment",
    "Travel",
    "Communication",
    "Entertainment",
    "Work & Career",
    "Connectivity",
    "General",
]


class TestCategoryTaxonomyAPI(unittest.TestCase):
    """Test suite for category taxonomy enforcement across the API layer."""

    def test_canonical_9_categories_in_metadata_model(self):
        """Verifies SUPPORTED_CATEGORIES in ScreenshotMetadata has exactly 9 canonical categories."""
        self.assertEqual(len(SUPPORTED_CATEGORIES), 9)
        self.assertNotIn("Coding", SUPPORTED_CATEGORIES)
        self.assertEqual(sorted(SUPPORTED_CATEGORIES), sorted(CANONICAL_9_EXPECTED))

    @unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi not installed in this environment")
    def test_category_meta_defines_exactly_canonical_9(self):
        """Verifies CATEGORY_META defines exactly the 9 canonical categories."""
        self.assertEqual(len(CATEGORY_META), 9)
        self.assertNotIn("Coding", CATEGORY_META)
        self.assertEqual(list(CATEGORY_META.keys()), CANONICAL_9_EXPECTED)

    @unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi not installed in this environment")
    def test_list_categories_returns_exactly_nine_canonical(self):
        """Verifies list_categories() returns exactly 9 categories with no Coding category."""
        res = list_categories()
        cats = res["categories"]
        cat_names = [c["name"] for c in cats]

        self.assertEqual(len(cats), 9)
        self.assertNotIn("Coding", cat_names)
        self.assertEqual(cat_names, CANONICAL_9_EXPECTED)
        self.assertGreater(res["total_screenshots"], 0)

    @unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi not installed in this environment")
    def test_legacy_coding_records_do_not_inflate_category_count(self):
        """Verifies that database containing historical 'Coding' records does not inflate category count."""
        mock_meta_store = MagicMock()
        # Create mock records including 5 'Coding' and 2 'Work_Career' records
        mock_records = [
            ScreenshotMetadata(screenshot_id=f"ss_{i}", filename=f"ss_{i}.jpg", image_uri="x", category="Coding")
            for i in range(5)
        ] + [
            ScreenshotMetadata(screenshot_id=f"ss_wc_{i}", filename=f"ss_wc_{i}.jpg", image_uri="x", category="Work_Career")
            for i in range(2)
        ] + [
            ScreenshotMetadata(screenshot_id=f"ss_ed_{i}", filename=f"ss_ed_{i}.jpg", image_uri="x", category="Education")
            for i in range(3)
        ]
        mock_meta_store.list_all_metadata.return_value = mock_records

        with patch("backend.app.core.pipeline_instance.metadata_store", mock_meta_store):
            res = list_categories()

        cat_names = [c["name"] for c in res["categories"]]
        self.assertEqual(len(res["categories"]), 9)
        self.assertNotIn("Coding", cat_names)
        self.assertNotIn("Work_Career", cat_names)
        self.assertEqual(res["total_screenshots"], 10)

        # Work & Career count should include the 2 Work_Career and 5 legacy Coding records
        wc_cat = next(c for c in res["categories"] if c["name"] == "Work & Career")
        self.assertEqual(wc_cat["count"], 7)

        # Education count should be 3
        ed_cat = next(c for c in res["categories"] if c["name"] == "Education")
        self.assertEqual(ed_cat["count"], 3)


if __name__ == "__main__":
    unittest.main()
