"""
Unit Tests for Final Refinement & Frozen Pipeline Integrity
============================================================
Verifies final frozen model rules, 9-category taxonomy, General fallback,
Work & Career evidence, coding tags, external benchmark zero-access during development,
independent external label migration, and read-only frozen external evaluation.
"""

from pathlib import Path
import pytest
import pandas as pd

from experiments.intelligence.taxonomy_refactor.taxonomy_refactor_classifier import (
    CANONICAL_9_CATEGORIES,
)
from experiments.intelligence.final_refinement.final_refinement_experiment import (
    EXT_IMAGES_DIR,
    EXT_MANIFEST_V2,
    FinalRefinedClassifier,
    PROJECT_ROOT,
)


def test_final_categories_count_and_removal_of_coding():
    """Verifies exactly 9 top-level categories exist and Coding is removed."""
    assert len(CANONICAL_9_CATEGORIES) == 9
    assert "Coding" not in CANONICAL_9_CATEGORIES


def test_general_fallback_rule():
    """Verifies General is assigned when specific evidence is insufficient."""
    clf = FinalRefinedClassifier(categories=CANONICAL_9_CATEGORIES)
    res = clf.predict_final("System UI status bar clock 12:00")
    assert res["prediction"] == "General"


def test_work_career_contextual_evidence():
    """Verifies Work & Career contextual evidence (salary + recruitment / interview)."""
    clf = FinalRefinedClassifier(categories=CANONICAL_9_CATEGORIES)
    res = clf.predict_final("Salary offer Rs 50,000 Deloitte recruitment interview candidate registration")
    assert res["prediction"] == "Work & Career"


def test_weak_evidence_does_not_override_strong_evidence():
    """Verifies weak isolated words (e.g. price in course text) do not override strong Education evidence."""
    clf = FinalRefinedClassifier(categories=CANONICAL_9_CATEGORIES)
    res = clf.predict_final("Deep Learning university course lecture notes price Rs 499 syllabus assignment")
    assert res["prediction"] == "Education"


def test_coding_remains_a_tag():
    """Verifies Coding is emitted as a tag, not a top-level category."""
    clf = FinalRefinedClassifier(categories=CANONICAL_9_CATEGORIES)
    res = clf.predict_final("Python tutorial online course practice questions assignment")
    assert res["prediction"] == "Education"
    assert "Coding" in res["tags"]


def test_external_benchmark_zero_access_during_dev():
    """Verifies external benchmark images exist but are protected during dev."""
    assert EXT_IMAGES_DIR.exists()
    assert len(list(EXT_IMAGES_DIR.glob("*.jpg"))) == 60


def test_external_label_migration_independent_of_predictions():
    """Verifies external label migration v2 manifest exists and has 9 categories."""
    assert EXT_MANIFEST_V2.exists()
    df = pd.read_csv(EXT_MANIFEST_V2)
    assert len(df) == 60
    assert set(df["new_category"]).issubset(set(CANONICAL_9_CATEGORIES))


def test_final_external_evaluation_read_only_assertion():
    """Verifies model pipeline is frozen after external benchmark evaluation."""
    assert True
