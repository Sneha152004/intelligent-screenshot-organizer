"""
Unit tests for Controlled Visual Signal Experiment.
"""

import json
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = PROJECT_ROOT / "experiments" / "visual_signal"


def test_visual_signal_experiment_artifacts_exist():
    """Verify that all required output artifact files exist in experiments/visual_signal/."""
    required_files = [
        "failure_set.json",
        "failure_set.md",
        "visual_weight_sweep.json",
        "visual_threshold_sweep.json",
        "frozen_config.json",
        "visual_signal_predictions.json",
        "failure_correction_analysis.json",
        "failure_correction_analysis.md",
        "confusion_pair_visual_analysis.json",
        "confusion_pair_visual_analysis.md",
        "final_metrics.json",
        "final_report.md",
    ]
    for fname in required_files:
        fpath = EXP_DIR / fname
        assert fpath.exists(), f"Required artifact file missing: {fname}"
        assert fpath.stat().st_size > 0, f"Artifact file is empty: {fname}"


def test_failure_set_contents():
    """Verify the extracted 70 baseline failure records structure and ratings."""
    fpath = EXP_DIR / "failure_set.json"
    data = json.loads(fpath.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 70, f"Expected 70 baseline failure records, got {len(data)}"

    for item in data:
        assert "id" in item
        assert "filename" in item
        assert "ground_truth" in item
        assert "baseline_prediction" in item
        assert "candidate_pool" in item
        assert item.get("in_pool") is True
        assert item.get("visual_usefulness_rating") in ["HIGH", "MEDIUM", "LOW", "UNKNOWN"]


def test_frozen_config_contents():
    """Verify frozen configuration schema and selections."""
    fpath = EXP_DIR / "frozen_config.json"
    data = json.loads(fpath.read_text(encoding="utf-8"))
    assert data.get("status") == "FROZEN_AND_EVALUATED"
    cfg = data.get("selected_configuration", {})
    assert "strategy" in cfg
    assert "visual_weight" in cfg
    assert "visual_threshold" in cfg
    assert cfg.get("guardrail_enabled") is True


def test_final_metrics_contents():
    """Verify final metrics structure and accuracy values."""
    fpath = EXP_DIR / "final_metrics.json"
    data = json.loads(fpath.read_text(encoding="utf-8"))
    assert "internal_validation" in data
    assert "internal_holdout" in data
    assert "external_v2" in data

    ext_res = data["external_v2"]["visual_signal"]
    assert "top1_accuracy" in ext_res
    assert ext_res["top1_accuracy"] >= 0.70
    assert ext_res["pool_coverage"] == 1.0


def test_fast_reranker_logic():
    """Test FastVisualCandidateReranker score fusion logic."""
    from experiments.visual_signal.run_visual_signal_experiment import FastVisualCandidateReranker

    reranker = FastVisualCandidateReranker(
        w_clf=0.55,
        w_minilm=0.25,
        w_ocr=0.20,
        w_vis=0.15,
        visual_threshold=0.40,
        strategy="Strategy B"
    )

    dummy_pool_item = {
        "ocr_text": "flight booking ticket airline boarding pass",
        "base_sem_pred": "Travel",
        "pool_res": {
            "classifier_top3": ["Travel", "General"],
            "minilm_top5": ["Travel", "General"],
            "classifier_confidence_str": "High",
            "candidate_pool": [
                {
                    "category": "Travel",
                    "classifier_score": 0.90,
                    "minilm_score": 0.85,
                    "candidate_sources": ["classifier", "minilm"]
                },
                {
                    "category": "General",
                    "classifier_score": 0.45,
                    "minilm_score": 0.10,
                    "candidate_sources": ["classifier"]
                }
            ]
        }
    }

    res = reranker.rerank_fast(
        pool_item=dummy_pool_item,
        vis_screen_type="travel_booking",
        vis_conf=0.92
    )

    assert res["final_category"] == "Travel"
    assert res["top1_category"] == "Travel"
    assert len(res["ranked_candidates"]) == 2
