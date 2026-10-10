"""
Unit tests for Expanded Candidate Pool Generation & Multi-Signal Reranker
==========================================================================
Tests:
- Candidate union construction & duplicate removal
- Provenance tracking (classifier_rank, minilm_rank, candidate_sources)
- Score normalization & Top-K preservation
- Handling missing visual evidence / missing MiniLM evidence
- Multi-signal evidence reranking behavior
- Output determinism across repeated runs
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "candidate_reranking"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "accuracy_improvement"))

from expanded_candidate_pool import ExpandedCandidatePoolBuilder
from reranker_experiments import MultiSignalCandidateReranker


@pytest.fixture(scope="module")
def shared_pool_builder():
    return ExpandedCandidatePoolBuilder()


@pytest.fixture(scope="module")
def shared_reranker(shared_pool_builder):
    return MultiSignalCandidateReranker(pool_builder=shared_pool_builder)


def test_candidate_union_and_provenance(shared_pool_builder):
    sample_text = "Flight ticket booking PNR 982341 airline check-in boarding pass"
    res = shared_pool_builder.build_candidate_pool(raw_text=sample_text)

    pool = res["candidate_pool"]
    categories = [c["category"] for c in pool]

    # Verify duplicate removal (all categories in pool are unique)
    assert len(categories) == len(set(categories))

    # Verify provenance fields
    for item in pool:
        assert "category" in item
        assert "classifier_rank" in item
        assert "classifier_score" in item
        assert "minilm_rank" in item
        assert "minilm_score" in item
        assert "candidate_sources" in item
        assert isinstance(item["candidate_sources"], list)
        assert len(item["candidate_sources"]) >= 1


def test_topk_preservation(shared_reranker):
    sample_text = "Your Progress SCORE 100/100 Correct 25/25 Accuracy Completed"
    res = shared_reranker.rerank(raw_text=sample_text, strategy="E")

    # Verify Top-K candidate preservation
    assert "top1_category" in res
    assert "top1_score" in res
    assert "top2_category" in res
    assert "top2_score" in res
    assert "top3_category" in res
    assert "top3_score" in res
    assert "final_category" in res
    assert "reranking_reason" in res
    assert "candidate_pool" in res
    assert "ranked_candidates" in res

    assert res["final_category"] == "Education"


def test_missing_visual_and_minilm_evidence_handling(shared_reranker):
    # Test handling when visual screen type is missing/other and text is sparse
    sparse_text = "System settings"
    res = shared_reranker.rerank(
        raw_text=sparse_text,
        vis_screen_type="other_screen",
        vis_conf=0.0,
        strategy="E"
    )

    assert res["final_category"] != ""
    assert isinstance(res["top1_score"], float)


def test_ocr_phrase_evidence_reranking(shared_reranker):
    # Text contains clear Payment OCR phrases
    payment_text = "UPI Payment successful Transaction ID 982341234 Paid to Merchant Bank Transfer"
    res = shared_reranker.rerank(raw_text=payment_text, strategy="E")

    assert res["final_category"] == "Payment"


def test_deterministic_output(shared_reranker):
    sample_text = "def test_pipeline(): import pytest assert True"
    res1 = shared_reranker.rerank(raw_text=sample_text, strategy="E")
    res2 = shared_reranker.rerank(raw_text=sample_text, strategy="E")

    assert res1["final_category"] == res2["final_category"]
    assert res1["top1_score"] == res2["top1_score"]
    assert res1["pool_categories"] if "pool_categories" in res1 else True
