"""
Unit tests for Lightweight Fine-Tuning Investigation Module
============================================================
"""

import os
import json
import torch
import pytest
from pathlib import Path
from experiments.fine_tuning.train_and_evaluate_fine_tuning import LightweightMLPHead, FineTunedPoolBuilder
from expanded_candidate_pool import CANONICAL_CATEGORIES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FT_DIR = PROJECT_ROOT / "experiments" / "fine_tuning"


def test_lightweight_mlp_head_shape():
    model = LightweightMLPHead(in_dim=384, num_classes=10)
    dummy_input = torch.randn(4, 384)
    output = model(dummy_input)
    assert output.shape == (4, 10)


def test_fine_tuning_artifacts_exist():
    assert (FT_DIR / "error_taxonomy.json").exists()
    assert (FT_DIR / "error_taxonomy.md").exists()
    assert (FT_DIR / "frozen_config.json").exists()
    assert (FT_DIR / "final_metrics.json").exists()
    assert (FT_DIR / "predictions_comparison.json").exists()
    assert (FT_DIR / "final_report.md").exists()


def test_error_taxonomy_json_structure():
    tax_path = FT_DIR / "error_taxonomy.json"
    with open(tax_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "total_external_v2_screenshots" in data
    assert "total_failures" in data
    assert "category_counts" in data
    assert "failure_details" in data
    assert isinstance(data["failure_details"], list)


def test_frozen_config_json_structure():
    cfg_path = FT_DIR / "frozen_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["experiment_name"] == "fine_tuning_investigation"
    assert "models_evaluated" in data
    assert "hyperparameters" in data
