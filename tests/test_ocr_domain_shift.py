"""
Unit Tests for OCR Representation & Domain-Shift Diagnostic Experiment
========================================================================
Verifies OCR normalization rules, number/date/url/email preservation,
content-focused UI boilerplate stripping, multi-view embedding handling,
evidence coverage calculation, dataset integrity, and external benchmark immutability.
"""

from pathlib import Path
import numpy as np
import pytest

from experiments.intelligence.dataset_loader import CANONICAL_CATEGORIES, load_dataset
from experiments.intelligence.external.evaluation_targets import get_evaluation_targets
from experiments.intelligence.ocr_domain_shift.ocr_domain_shift_experiment import (
    MiniLMSimilarityClassifier,
    RuleBaselineClassifier,
    compute_group_statistics,
    compute_metrics,
    compute_sample_stats,
    content_focused_ocr_text,
    normalize_ocr_text,
)


def test_ocr_normalization_basic_and_deterministic():
    raw_text = "  Reel   DIOMS\nHit the road   Leave  "
    norm1 = normalize_ocr_text(raw_text)
    norm2 = normalize_ocr_text(raw_text)

    assert norm1 == norm2  # Deterministic
    assert norm1 == "reel dioms hit the road leave"


def test_ocr_normalization_preserves_numbers_prices_dates_urls_emails():
    text = (
        "Order #12345 placed on 12/09/2026 for $49.99 or ₹1000. "
        "Contact support@example.com or visit https://example.com/item."
    )
    norm = normalize_ocr_text(text)

    assert "12/09/2026" in norm
    assert "$49.99" in norm or "49.99" in norm
    assert "₹1000" in norm or "1000" in norm
    assert "support@example.com" in norm
    assert "https://example.com/item" in norm


def test_ocr_normalization_repeated_char_deduplication():
    text = "sooooo coolllll 10000 rupees"
    norm = normalize_ocr_text(text)

    assert "soo" in norm
    assert "cooll" in norm
    assert "10000" in norm  # Digits not deduplicated


def test_content_focused_ocr_strips_ui_boilerplate():
    raw_text = "19:08 17% 4G Airtel cancel back save Python code import sys class"
    content_txt = content_focused_ocr_text(raw_text)

    assert "17%" not in content_txt
    assert "19:08" not in content_txt
    assert "cancel" not in content_txt
    assert "python code import sys class" in content_txt.lower()


def test_empty_and_sparse_ocr_handling():
    empty_norm = normalize_ocr_text("")
    assert empty_norm == ""

    empty_content = content_focused_ocr_text("")
    assert empty_content == ""

    sparse_text = "OK"
    stats = compute_sample_stats(sparse_text)
    assert stats["is_sparse"] == 1.0
    assert stats["token_count"] == 1.0


def test_sample_stats_calculation():
    text = "Paying via Google Pay UPI ₹500 transaction successful 12/09/2026 support@pay.com https://pay.com"
    stats = compute_sample_stats(text)

    assert stats["char_count"] == len(text)
    assert stats["token_count"] > 0
    assert stats["url_count"] >= 1.0
    assert stats["email_count"] >= 1.0
    assert stats["currency_count"] >= 1.0
    assert stats["date_count"] >= 1.0
    assert stats["category_keyword_hits"] > 0


def test_multi_view_minilm_embeddings():
    clf = MiniLMSimilarityClassifier(categories=CANONICAL_CATEGORIES)
    t1 = "python import code function"
    t2 = normalize_ocr_text(t1)
    t3 = content_focused_ocr_text(t1)

    v1 = clf.embed_text(t1)
    v2 = clf.embed_text(t2)
    v3 = clf.embed_text(t3)

    assert v1.shape == (384,)
    assert v2.shape == (384,)
    assert v3.shape == (384,)

    v_avg = (v1 + v2 + v3) / 3.0
    probas = clf.predict_proba_from_embedding(v_avg)

    assert probas.shape == (10,)
    assert np.isclose(probas.sum(), 1.0)


def test_dataset_integrity_833_valid_samples():
    samples = load_dataset(exclude_invalid=True)
    assert len(samples) == 833

    invalid_files = {"ss_441.png", "ss_443.png", "ss_560.png"}
    found_invalid = [s["filename"] for s in samples if s["filename"] in invalid_files]
    assert len(found_invalid) == 0


def test_external_benchmark_immutability():
    ext_targets = get_evaluation_targets()
    assert len(ext_targets) == 60
    filenames = [e["filename"] for e in ext_targets]
    assert filenames[0] == "ext_001.jpg"
    assert filenames[-1] == "ext_060.jpg"
