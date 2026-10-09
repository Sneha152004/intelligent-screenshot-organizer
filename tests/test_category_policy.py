"""
Unit tests for Category Decision Policy V1, Policy V2, and Policy V3
=====================================================================
Tests indicator scoring, primary-object rules, confusion-pair tie-breakers,
Policy V2/V3 rules, empty OCR handling, General protection, determinism, and backward compatibility.
"""

import os
import json
import pytest
from pathlib import Path

from experiments.category_policy.policy_engine import CategoryPolicyEngine
from experiments.category_policy_v2.run_category_policy_v2_experiment import CategoryPolicyEngineV2
from experiments.category_policy_v3.run_category_policy_v3_experiment import CategoryPolicyEngineV3


@pytest.fixture
def policy_engine():
    return CategoryPolicyEngine()


@pytest.fixture
def policy_engine_v2():
    return CategoryPolicyEngineV2()


@pytest.fixture
def policy_engine_v3():
    return CategoryPolicyEngineV3()


def test_policy_engine_initialization(policy_engine):
    """Verifies policy engine loads categories from default policy file."""
    assert policy_engine.policy_data is not None
    assert len(policy_engine.categories_policy) == 10
    assert "Communication" in policy_engine.categories_policy
    assert "Payment" in policy_engine.categories_policy
    assert "General" in policy_engine.categories_policy


def test_positive_and_negative_indicators(policy_engine):
    """Verifies positive indicator scoring and negative indicator penalization."""
    text = "Payment successful via Google Pay. Transaction ID: 12345. UPI transfer complete."
    result = policy_engine.evaluate_evidence(text)

    payment_ev = result["category_evidence"]["Payment"]
    assert payment_ev["positive_score"] > 0
    assert len(payment_ev["matched_strong_positive"]) > 0 or len(payment_ev["matched_supporting_positive"]) > 0
    assert result["total_policy_boost"]["Payment"] > result["total_policy_boost"]["Education"]


def test_primary_object_communication(policy_engine):
    """Verifies Email/Chat correspondence primary object trigger."""
    text = "From: Alice <alice@example.com>\nTo: Bob\nSubject: Project Sync\nInbox message received on WhatsApp"
    result = policy_engine.evaluate_evidence(text)

    assert result["primary_object_scores"]["Communication"] == 2.5
    assert any("Email/Chat correspondence layout" in r for r in result["primary_object_reasons"])


def test_primary_object_work_and_career(policy_engine):
    """Verifies Job vacancy / Career primary object trigger."""
    text = "We are hiring! Job Application submitted. Interview invitation for Senior Engineer candidate."
    result = policy_engine.evaluate_evidence(text)

    assert result["primary_object_scores"]["Work & Career"] == 2.5
    assert any("Job portal / Career vacancy layout" in r for r in result["primary_object_reasons"])


def test_primary_object_shopping(policy_engine):
    """Verifies E-commerce product catalog primary object trigger."""
    text = "Product Details: Wireless Headphones. Add to Cart. Buy Now. Free shipping available."
    result = policy_engine.evaluate_evidence(text)

    assert result["primary_object_scores"]["Shopping"] == 2.5
    assert any("E-commerce product catalog" in r for r in result["primary_object_reasons"])


def test_primary_object_coding(policy_engine):
    """Verifies IDE editor / Terminal traceback primary object trigger."""
    text = "VS Code workspace. Commit changes to git. Error traceback: File main.py, line 42. def main():"
    result = policy_engine.evaluate_evidence(text)

    assert result["primary_object_scores"]["Coding"] == 2.5
    assert any("IDE editor / Terminal traceback" in r for r in result["primary_object_reasons"])


def test_confusion_pair_shopping_vs_payment(policy_engine):
    """Verifies Shopping vs Payment transaction completion tie-breaker."""
    # Completed payment
    text_paid = "Payment successful for order #8812. Ref No: 9912837. Money transferred."
    res_paid = policy_engine.evaluate_evidence(text_paid)
    assert res_paid["tiebreaker_scores"]["Payment"] > res_paid["tiebreaker_scores"]["Shopping"]

    # In-cart shopping intent
    text_cart = "Item added to cart. Add to cart. Buy Now for Rs 499."
    res_cart = policy_engine.evaluate_evidence(text_cart)
    assert res_cart["tiebreaker_scores"]["Shopping"] > res_cart["tiebreaker_scores"]["Payment"]


def test_policy_v2_education_rule(policy_engine_v2):
    """Verifies Policy V2 Education coursework platform indicator rule."""
    text = "Weka Explorer data mining tool dataset preprocess. Coursera quiz submission complete."
    res = policy_engine_v2.evaluate_evidence_v2(
        raw_text=text,
        rule_config={"enable_rule1_education_calib": True}
    )
    assert "V2_Rule1_Education_Calib" in res["v2_rules_fired"]
    assert res["total_policy_boost"]["Education"] > 2.0


def test_policy_v2_work_dashboard_calibration(policy_engine_v2):
    """Verifies Policy V2 Work & Career dashboard calibration rule."""
    text = "Projects timeline view. Business partner client name CRM update."
    res = policy_engine_v2.evaluate_evidence_v2(
        raw_text=text,
        rule_config={"enable_rule2_work_dash_calib": True}
    )
    assert "V2_Rule2_Work_Dashboard_Calib" in res["v2_rules_fired"]
    assert res["primary_object_scores"]["Work & Career"] == 0.0


def test_policy_v3_banking_dashboard_safeguard(policy_engine_v3):
    """Verifies Policy V3 Rule 1: Banking Account Summary Safeguard for General."""
    text = "Net worth overview: Total assets $127,087.15, Operating balance $25,185,107.28. Total available balance."
    res = policy_engine_v3.evaluate_evidence_v3(
        raw_text=text,
        rule_config_v3={"enable_v3_rule1_banking_dashboard": True}
    )
    assert "V3_Rule1_Banking_Dashboard_Safeguard" in res["v3_rules_fired"]
    assert res["primary_object_scores"]["General"] == 2.0


def test_policy_v3_dev_admin_tools(policy_engine_v3):
    """Verifies Policy V3 Rule 2: Developer control panel & DB admin tools for Work & Career."""
    text = "XAMPP Control Panel v3.3.0 phpMyAdmin localhost/ MySQL error."
    res = policy_engine_v3.evaluate_evidence_v3(
        raw_text=text,
        rule_config_v3={"enable_v3_rule2_dev_admin_tools": True}
    )
    assert "V3_Rule2_Dev_Admin_Tools" in res["v3_rules_fired"]
    assert res["primary_object_scores"]["Work & Career"] >= 2.5


def test_policy_v3_empty_ocr_text_handling(policy_engine_v3):
    """Verifies Policy V3 gracefully handles empty or whitespace OCR text."""
    res = policy_engine_v3.evaluate_evidence_v3(raw_text="   ")
    assert "category_evidence" in res
    assert res["total_policy_boost"]["General"] == 0.0


def test_determinism_and_backward_compatibility(policy_engine_v3):
    """Verifies evaluation is deterministic and backward compatible."""
    text = "Sample text for testing determinism in policy evaluation."
    res1 = policy_engine_v3.evaluate_evidence_v3(text)
    res2 = policy_engine_v3.evaluate_evidence_v3(text)

    assert res1 == res2
    assert "category_evidence" in res1
    assert "total_policy_boost" in res1
