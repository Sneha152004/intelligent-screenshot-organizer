"""
Category Decision Policy Engine V1
===================================
Implements Category Decision Policy V1 featuring:
1. Positive Evidence Scorer (+2.0 for strong indicators, +1.0 for supporting indicators)
2. Negative Evidence Penalizer (-1.5 for negative indicators)
3. Primary Object / Purpose Classifier (layout & intent multipliers)
4. Confusion-Pair Tie-Breakers (resolves General vs Domain, Shopping vs Payment, Work vs Education)
5. Structured Evidence Provenance Logging
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "best_of_all_categorization"))

POLICY_JSON_PATH = Path(__file__).resolve().parent / "category_decision_policy_v1.json"


class CategoryPolicyEngine:
    """Rule-based evidence engine executing Category Decision Policy V1."""

    def __init__(self, policy_data: Optional[Dict[str, Any]] = None):
        if policy_data is None:
            if POLICY_JSON_PATH.exists():
                with open(POLICY_JSON_PATH, "r", encoding="utf-8") as f:
                    policy_data = json.load(f)
            else:
                policy_data = {}

        self.policy_data = policy_data
        self.categories_policy = policy_data.get("categories", {})

    def evaluate_evidence(
        self,
        raw_text: str,
        notes: str = "",
        candidates: Optional[List[Dict[str, Any]]] = None,
        vis_screen_type: str = "other_screen",
        vis_conf: float = 0.0
    ) -> Dict[str, Any]:
        """
        Evaluates positive/negative policy evidence, primary object indicators,
        and confusion-pair tie-breakers across all candidate categories.
        """
        text_lower = (raw_text + " " + notes).lower().strip()
        tokens = set(re.findall(r"\b[a-z0-9_-]+\b", text_lower))

        category_evidence: Dict[str, Dict[str, Any]] = {}

        for cat, policy in self.categories_policy.items():
            strong_pos = policy.get("strong_positive_indicators", [])
            supp_pos = policy.get("supporting_indicators", [])
            neg_inds = policy.get("negative_indicators", [])

            matched_strong_pos = [ind for ind in strong_pos if ind in text_lower]
            matched_supp_pos = [ind for ind in supp_pos if ind in text_lower or ind in tokens]
            matched_neg = [ind for ind in neg_inds if ind in text_lower]

            pos_score = (len(matched_strong_pos) * 2.0) + (len(matched_supp_pos) * 1.0)
            neg_score = len(matched_neg) * 1.5
            net_score = max(0.0, pos_score - neg_score)

            category_evidence[cat] = {
                "positive_score": round(pos_score, 2),
                "negative_score": round(neg_score, 2),
                "net_score": round(net_score, 2),
                "matched_strong_positive": matched_strong_pos,
                "matched_supporting_positive": matched_supp_pos,
                "matched_negative": matched_neg
            }

        # Global Rule: Primary Object / Purpose Multiplier Evaluation
        primary_object_scores: Dict[str, float] = {c: 0.0 for c in self.categories_policy.keys()}
        primary_object_reasons: List[str] = []

        # Case A: Communication (Email / Chat frame)
        if any(hdr in text_lower for hdr in ["subject:", "reply to", "inbox", "whatsapp", "telegram", "from:", "to:"]):
            if not any(receipt_kw in text_lower for receipt_kw in ["transaction successful", "boarding pass"]):
                primary_object_scores["Communication"] += 2.5
                primary_object_reasons.append("Email/Chat correspondence layout primary object")

        # Case B: Work & Career (Job posting / Portal layout)
        if any(kw in text_lower for kw in ["job application", "job vacancy", "campus placement", "interview invitation", "applicant", "recruiter"]):
            primary_object_scores["Work & Career"] += 2.5
            primary_object_reasons.append("Job portal / Career vacancy layout primary object")

        # Case C: Shopping (Cart / E-commerce checkout)
        if any(kw in text_lower for kw in ["add to cart", "buy now", "product details", "free shipping", "wishlist"]):
            primary_object_scores["Shopping"] += 2.5
            primary_object_reasons.append("E-commerce product catalog / cart primary object")

        # Case D: Payment (Completed transaction receipt)
        if any(kw in text_lower for kw in ["payment successful", "transaction successful", "money transferred", "upi id", "ref no", "paid successfully"]):
            primary_object_scores["Payment"] += 2.5
            primary_object_reasons.append("Completed transaction receipt primary object")

        # Case E: Education (Course material / Syllabus)
        if any(kw in text_lower for kw in ["online course", "practice questions", "lecture notes", "syllabus", "classroom", "coursera"]):
            primary_object_scores["Education"] += 2.5
            primary_object_reasons.append("Academic lecture / Course syllabus primary object")

        # Case F: Coding (IDE / Code Syntax / Traceback)
        if any(kw in text_lower for kw in ["vs code", "vscode", "error traceback", "commit changes", "git commit", "stack trace", "def ", "class ", "compiler error"]):
            primary_object_scores["Coding"] += 2.5
            primary_object_reasons.append("IDE editor / Terminal traceback primary object")

        # Case G: Connectivity (Wi-Fi / Bluetooth settings)
        if any(kw in text_lower for kw in ["connected to wi-fi", "network settings", "bluetooth devices", "hotspot settings"]):
            primary_object_scores["Connectivity"] += 2.5
            primary_object_reasons.append("Network / Wi-Fi configuration primary object")

        # Confusion-Pair Tie-Breakers
        tiebreaker_scores: Dict[str, float] = {c: 0.0 for c in self.categories_policy.keys()}
        tiebreaker_reasons: List[str] = []

        # Confusion Pair 1: General vs Work & Career
        if "Work & Career" in category_evidence and any(kw in text_lower for kw in ["position", "role", "company", "hiring", "team", "manager", "project", "work", "office"]):
            tiebreaker_scores["Work & Career"] += 1.5
            tiebreaker_reasons.append("General vs Work & Career: Workplace context tiebreaker applied to Work & Career")

        # Confusion Pair 2: General vs Communication
        if "Communication" in category_evidence and any(kw in text_lower for kw in ["reply", "sender", "message", "sent", "inbox", "mail", "contact"]):
            tiebreaker_scores["Communication"] += 1.5
            tiebreaker_reasons.append("General vs Communication: Messaging context tiebreaker applied to Communication")

        # Confusion Pair 3: General vs Shopping
        if "Shopping" in category_evidence and any(kw in text_lower for kw in ["rupees", "₹", "price", "item", "order", "store", "buy", "deal"]):
            tiebreaker_scores["Shopping"] += 1.5
            tiebreaker_reasons.append("General vs Shopping: Retail item context tiebreaker applied to Shopping")

        # Confusion Pair 4: Shopping vs Payment
        if any(kw in text_lower for kw in ["payment successful", "transaction successful", "ref no"]):
            tiebreaker_scores["Payment"] += 2.0
            tiebreaker_scores["Shopping"] -= 1.0
            tiebreaker_reasons.append("Shopping vs Payment: Transaction completion status tiebreaker applied to Payment")
        elif any(kw in text_lower for kw in ["add to cart", "buy now"]):
            tiebreaker_scores["Shopping"] += 2.0
            tiebreaker_scores["Payment"] -= 1.0
            tiebreaker_reasons.append("Shopping vs Payment: Purchase intent tiebreaker applied to Shopping")

        # Compute Total Policy Boost per Category
        total_policy_boost: Dict[str, float] = {}
        for cat in self.categories_policy.keys():
            ev = category_evidence[cat]
            total_policy_boost[cat] = round(ev["net_score"] + primary_object_scores[cat] + tiebreaker_scores[cat], 2)

        return {
            "category_evidence": category_evidence,
            "primary_object_scores": primary_object_scores,
            "primary_object_reasons": primary_object_reasons,
            "tiebreaker_scores": tiebreaker_scores,
            "tiebreaker_reasons": tiebreaker_reasons,
            "total_policy_boost": total_policy_boost
        }
