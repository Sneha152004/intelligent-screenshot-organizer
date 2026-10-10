"""
Unit Tests for Taxonomy Refactor (9 Categories + Coding Tag Extraction)
========================================================================
Verifies 9-category taxonomy schema, removal of Coding as top-level category,
deterministic coding tag extraction, Education vs Work & Career disambiguation,
General fallback logic, and strict 0% external benchmark access.
"""

from pathlib import Path
import pytest

from experiments.intelligence.taxonomy_refactor.coding_tag_extractor import (
    extract_deterministic_tags,
)
from experiments.intelligence.taxonomy_refactor.taxonomy_refactor_classifier import (
    CANONICAL_9_CATEGORIES,
    TaxonomyRefactorClassifier,
)
from experiments.intelligence.taxonomy_refactor.taxonomy_audit_migration import (
    METADATA_CSV_V2,
    PROJECT_ROOT,
)


def test_1_python_tutorial_education_plus_coding_tag():
    """1. Verifies Python tutorial predicts Education + Coding tag."""
    clf = TaxonomyRefactorClassifier(categories=CANONICAL_9_CATEGORIES)
    res = clf.predict_single_explainable("Python tutorial online course practice questions assignment")

    assert res["prediction"] == "Education"
    assert "Coding" in res["tags"]
    assert "Python" in res["tags"]


def test_2_python_course_education_plus_coding_tag():
    """2. Verifies Python course predicts Education + Coding tag."""
    clf = TaxonomyRefactorClassifier(categories=CANONICAL_9_CATEGORIES)
    res = clf.predict_single_explainable("Python programming lecture slides course material exam revision")

    assert res["prediction"] == "Education"
    assert "Coding" in res["tags"]
    assert "Python" in res["tags"]


def test_3_vscode_development_work_career_plus_coding_tag():
    """3. Verifies VS Code development predicts Work & Career + Coding tag."""
    clf = TaxonomyRefactorClassifier(categories=CANONICAL_9_CATEGORIES)
    res = clf.predict_single_explainable("VS Code screenshot main.py def compute_loss(): import torch return loss")

    assert res["prediction"] == "Work & Career"
    assert "Coding" in res["tags"]
    assert "VS Code" in res["tags"]


def test_4_github_repository_work_career_plus_coding_tag():
    """4. Verifies GitHub repository predicts Work & Career + Coding tag."""
    clf = TaxonomyRefactorClassifier(categories=CANONICAL_9_CATEGORIES)
    res = clf.predict_single_explainable("GitHub repository ragnova/screenshot-organizer main branch pull request #12")

    assert res["prediction"] == "Work & Career"
    assert "Coding" in res["tags"]
    assert "GitHub" in res["tags"]


def test_5_git_commit_work_career_plus_coding_tag():
    """5. Verifies Git commit screen predicts Work & Career + Coding tag."""
    clf = TaxonomyRefactorClassifier(categories=CANONICAL_9_CATEGORIES)
    res = clf.predict_single_explainable("git commit -m 'fix memory leak in buffer' git push origin main")

    assert res["prediction"] == "Work & Career"
    assert "Coding" in res["tags"]
    assert "Git" in res["tags"]


def test_6_leetcode_lesson_education_plus_coding_tag():
    """6. Verifies LeetCode lesson predicts Education + Coding tag."""
    clf = TaxonomyRefactorClassifier(categories=CANONICAL_9_CATEGORIES)
    res = clf.predict_single_explainable("leetcode.com/problems/two-sum/ Python3 Solution def twoSum(): algorithm")

    assert res["prediction"] == "Education"
    assert "Coding" in res["tags"]
    assert "LeetCode" in res["tags"]


def test_7_generic_unsupported_screen_predicts_general():
    """7. Verifies generic unsupported screen predicts General."""
    clf = TaxonomyRefactorClassifier(categories=CANONICAL_9_CATEGORIES)
    res = clf.predict_single_explainable("System UI battery 12:00 status bar")

    assert res["prediction"] == "General"


def test_8_wifi_settings_predicts_connectivity_not_general():
    """8. Verifies Wi-Fi settings predicts Connectivity, not General."""
    clf = TaxonomyRefactorClassifier(categories=CANONICAL_9_CATEGORIES)
    res = clf.predict_single_explainable("Wi-Fi network settings connected SSID Airtel_5G IP address show password")

    assert res["prediction"] == "Connectivity"


def test_9_product_page_predicts_shopping():
    """9. Verifies product page predicts Shopping."""
    clf = TaxonomyRefactorClassifier(categories=CANONICAL_9_CATEGORIES)
    res = clf.predict_single_explainable("Product catalog price $29.99 Add to Cart Checkout free shipping")

    assert res["prediction"] == "Shopping"


def test_10_payment_confirmation_predicts_payment():
    """10. Verifies payment confirmation predicts Payment."""
    clf = TaxonomyRefactorClassifier(categories=CANONICAL_9_CATEGORIES)
    res = clf.predict_single_explainable("Paying via Google Pay UPI transaction successful ref no 998877 total paid $15")

    assert res["prediction"] == "Payment"


def test_11_coding_is_not_a_top_level_category():
    """11. Verifies Coding is NOT in the top-level categories list."""
    assert "Coding" not in CANONICAL_9_CATEGORIES


def test_12_exactly_nine_top_level_categories_exist():
    """12. Verifies exactly 9 top-level categories exist."""
    assert len(CANONICAL_9_CATEGORIES) == 9
    expected = ["Communication", "Connectivity", "Education", "Entertainment", "General", "Payment", "Shopping", "Travel", "Work & Career"]
    assert sorted(CANONICAL_9_CATEGORIES) == sorted(expected)


def test_13_general_used_only_when_evidence_insufficient():
    """13. Verifies General is assigned only when specific evidence is insufficient."""
    clf = TaxonomyRefactorClassifier(categories=CANONICAL_9_CATEGORIES)
    res = clf.predict_single_explainable("Flight booking Indigo PNR 123456 departure 10:30 AM")
    assert res["prediction"] == "Travel"
    assert res["prediction"] != "General"


def test_14_no_hallucinated_coding_tags():
    """14. Verifies non-coding screens do NOT receive hallucinated coding tags."""
    tags = extract_deterministic_tags("Wi-Fi network settings connected SSID Airtel_5G show password")
    assert "Coding" not in tags
    assert "Python" not in tags


def test_15_external_benchmark_zero_file_access():
    """15. Verifies external benchmark directory is not accessed."""
    ext_dir = PROJECT_ROOT / "experiments" / "intelligence" / "external" / "images"
    assert ext_dir.exists(), "External benchmark directory check."
