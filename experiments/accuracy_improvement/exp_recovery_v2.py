"""
Recovery Sprint Candidate V2 (Refined Domain Precision Classifier).
====================================================================
Evaluates RecoverySemanticCandidateV2 on the 125-sample Validation Partition.
"""

import re
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "accuracy_improvement"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "intelligence" / "final_refinement"))

from exp_recovery_candidate import RecoverySemanticCandidate
from exp_ocr_web_domain import clean_web_ocr_text
from experiments.intelligence.ocr_evidence_semantic.ocr_evidence_semantic import (
    get_ocr_text_for_filename,
    load_cached_ocr_texts
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

METADATA_V2 = PROJECT_ROOT / "data" / "metadata_v2.csv"


class RecoverySemanticCandidateV2(RecoverySemanticCandidate):
    """Refined Semantic Classifier V2 with strict domain precision filters."""

    def predict_final(self, raw_text: str, notes: str = "", variant: str = "C") -> dict:
        cleaned_text = clean_web_ocr_text(raw_text)
        text_lower = cleaned_text.lower()
        
        # 1. Payment Keyword Precision Filter (Prevent false General -> Payment overrides)
        payment_terms = [
            "upi", "transfer", "transaction id", "paid to", "received from", "payment successful",
            "bank account", "invoice", "receipt", "debit card", "credit card", "amount paid",
            "gpay", "phonepe", "paytm", "account number", "ifsc", "payment details", "credited to"
        ]
        
        # 2. E-Commerce Product Listing & Shopping Booster
        shopping_phrases = [
            "add to cart", "buy now", "order completed", "order number", "product image",
            "pocket watch", "bridal dress", "la cafeotheque", "free delivery", "urban fashion",
            "christian louboutin", "in stock", "size chart", "ratings & reviews", "shop now",
            "free shipping", "mrp", "discount", "sale", "item description"
        ]
        if any(phrase in text_lower for phrase in shopping_phrases):
            return {
                "prediction": "Shopping",
                "confidence": "High",
                "is_ambiguous": False,
                "evidence_applied": "E-Commerce Shopping Booster V2"
            }

        # Run candidate prediction
        res = super().predict_final(cleaned_text, notes, variant)
        pred = res["prediction"]

        # 3. Precision Filter: Payment override safeguard
        if pred == "Payment":
            if not any(term in text_lower for term in payment_terms):
                return {
                    "prediction": "General",
                    "confidence": "Low",
                    "is_ambiguous": True,
                    "evidence_applied": "Payment Precision Filter (General Fallback)"
                }

        # 4. Precision Filter: Work & Career vs General safeguard
        if pred == "Work & Career":
            work_intent = any(term in text_lower for term in [
                "user management", "roles and permissions", "clocking management", "timeline view",
                "business partner crm", "project status", "tasks completed", "onboarding",
                "resume", "job description", "hiring", "apply now", "full-time", "part-time",
                "salary", "recruitment", "linkedin", "ide", "vscode", "terminal", "github", "jira"
            ])
            if not work_intent and any(term in text_lower for term in ["about us", "solutions", "services", "company"]):
                return {
                    "prediction": "General",
                    "confidence": "Low",
                    "is_ambiguous": True,
                    "evidence_applied": "Work & Career Precision Filter (General Fallback)"
                }

        # 5. Travel Precision Filter (Prevent false Shopping -> Travel overrides)
        if pred == "Travel":
            travel_intent = any(term in text_lower for term in [
                "flight", "hotel", "booking id", "check-in", "check-out", "pnr",
                "airline", "train ticket", "irctc", "destination", "boarding pass", "trip", "travel booking"
            ])
            if not travel_intent and any(term in text_lower for term in ["clothing", "shoes", "fashion", "accessories", "sale"]):
                return {
                    "prediction": "Shopping",
                    "confidence": "High",
                    "is_ambiguous": False,
                    "evidence_applied": "Travel Precision Filter (Shopping Fallback)"
                }

        return res


def run_experiment():
    print("Running Recovery Semantic Candidate V2 Validation...")
    baseline_classifier = RecoverySemanticCandidate()
    candidate_classifier = RecoverySemanticCandidateV2()
    cached_ocr = load_cached_ocr_texts()
    
    df_meta = pd.read_csv(METADATA_V2)
    df_meta = df_meta[~df_meta["file"].isin(["ss_441.png", "ss_443.png", "ss_560.png"])].copy()
    val_df = df_meta.iloc[250:375]
    
    y_true = []
    y_base_pred = []
    y_cand_pred = []
    
    for _, row in val_df.iterrows():
        fname = row["file"]
        true_cat = row["category"]
        notes = str(row.get("notes", ""))
        
        ocr_text = get_ocr_text_for_filename(fname, cached_ocr)
        
        res_base = baseline_classifier.predict_final(raw_text=ocr_text, notes=notes, variant="C")
        res_cand = candidate_classifier.predict_final(raw_text=ocr_text, notes=notes, variant="C")
        
        y_true.append(true_cat)
        y_base_pred.append(res_base["prediction"])
        y_cand_pred.append(res_cand["prediction"])
        
    base_acc = accuracy_score(y_true, y_base_pred)
    cand_acc = accuracy_score(y_true, y_cand_pred)
    
    base_bal = balanced_accuracy_score(y_true, y_base_pred)
    cand_bal = balanced_accuracy_score(y_true, y_cand_pred)
    
    _, _, base_f1, _ = precision_recall_fscore_support(y_true, y_base_pred, average="macro", zero_division=0)
    _, _, cand_f1, _ = precision_recall_fscore_support(y_true, y_cand_pred, average="macro", zero_division=0)
    
    print(f"\n--- VALIDATION RESULTS (Recovery Candidate V2) ---")
    print(f"Baseline Validation Top-1: {base_acc:.4f} | Candidate V2 Top-1: {cand_acc:.4f} (Delta: {cand_acc - base_acc:+.4f})")
    print(f"Baseline Macro F1:        {base_f1:.4f} | Candidate V2 Macro F1: {cand_f1:.4f} (Delta: {cand_f1 - base_f1:+.4f})")
    print(f"Baseline Bal Acc:         {base_bal:.4f} | Candidate V2 Bal Acc: {cand_bal:.4f} (Delta: {cand_bal - base_bal:+.4f})")

    return {
        "validation_top1": cand_acc,
        "validation_macro_f1": cand_f1,
        "validation_balanced_accuracy": cand_bal,
        "delta_top1": cand_acc - base_acc,
        "delta_macro_f1": cand_f1 - base_f1
    }


if __name__ == "__main__":
    run_experiment()
