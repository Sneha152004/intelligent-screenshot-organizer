"""
EXP-007: Web-Domain OCR Normalization & Boilerplate Filter Candidate.
======================================================================
Evaluates web boilerplate OCR filtering (removing social feed metadata,
sponsored tags, view counts, navigation lines) on the 125-sample Validation Partition.
"""

import re
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "accuracy_improvement"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "intelligence" / "final_refinement"))

from exp_semantic_refinement import RefinedSemanticCandidate
from final_refinement_experiment import FinalRefinedClassifier
from experiments.intelligence.ocr_evidence_semantic.ocr_evidence_semantic import (
    get_ocr_text_for_filename,
    load_cached_ocr_texts
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

METADATA_V2 = PROJECT_ROOT / "data" / "metadata_v2.csv"

# Web UI / Social Feed boilerplate patterns to filter
BOILERPLATE_PATTERNS = [
    r"^view (all )?\d+ comments.*$",
    r"^\d+ (lakh|k|m) views.*$",
    r"^\d+ (hours?|days?|months?|years?) ago.*$",
    r"^sponsored\b.*$",
    r"^just now\b.*$",
    r"^tap to unmute.*$",
    r"^follow\b.*$",
    r"^share\b.*$",
    r"^like\b.*$",
    r"^subscribe\b.*$"
]

def clean_web_ocr_text(raw_text: str) -> str:
    """Removes social feed metadata and web navbar noise while preserving core domain text."""
    lines = raw_text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue
        # Check boilerplate match
        is_boilerplate = any(re.match(pattern, line_strip, re.IGNORECASE) for pattern in BOILERPLATE_PATTERNS)
        if is_boilerplate:
            continue
        cleaned_lines.append(line_strip)
        
    return "\n".join(cleaned_lines)


def run_experiment():
    print("Running EXP-007: Web-Domain OCR Normalization Experiment...")
    candidate_classifier = RefinedSemanticCandidate()
    cached_ocr = load_cached_ocr_texts()
    
    df_meta = pd.read_csv(METADATA_V2)
    df_meta = df_meta[~df_meta["file"].isin(["ss_441.png", "ss_443.png", "ss_560.png"])].copy()
    val_df = df_meta.iloc[250:375]
    
    y_true = []
    y_base_pred = []
    y_clean_pred = []
    
    for _, row in val_df.iterrows():
        fname = row["file"]
        true_cat = row["category"]
        notes = str(row.get("notes", ""))
        
        raw_ocr = get_ocr_text_for_filename(fname, cached_ocr)
        cleaned_ocr = clean_web_ocr_text(raw_ocr)
        
        res_base = candidate_classifier.predict_final(raw_text=raw_ocr, notes=notes, variant="C")
        res_clean = candidate_classifier.predict_final(raw_text=cleaned_ocr, notes=notes, variant="C")
        
        y_true.append(true_cat)
        y_base_pred.append(res_base["prediction"])
        y_clean_pred.append(res_clean["prediction"])
        
    base_acc = accuracy_score(y_true, y_base_pred)
    clean_acc = accuracy_score(y_true, y_clean_pred)
    
    base_bal = balanced_accuracy_score(y_true, y_base_pred)
    clean_bal = balanced_accuracy_score(y_true, y_clean_pred)
    
    _, _, base_f1, _ = precision_recall_fscore_support(y_true, y_base_pred, average="macro", zero_division=0)
    _, _, clean_f1, _ = precision_recall_fscore_support(y_true, y_clean_pred, average="macro", zero_division=0)
    
    print(f"\n--- VALIDATION RESULTS (EXP-007) ---")
    print(f"Baseline Validation Top-1: {base_acc:.4f} | Cleaned OCR Top-1: {clean_acc:.4f} (Delta: {clean_acc - base_acc:+.4f})")
    print(f"Baseline Macro F1:        {base_f1:.4f} | Cleaned OCR Macro F1: {clean_f1:.4f} (Delta: {clean_f1 - base_f1:+.4f})")
    print(f"Baseline Bal Acc:         {base_bal:.4f} | Cleaned OCR Bal Acc: {clean_bal:.4f} (Delta: {clean_bal - base_bal:+.4f})")
    
    return {
        "validation_top1": clean_acc,
        "validation_macro_f1": clean_f1,
        "validation_balanced_accuracy": clean_bal,
        "delta_top1": clean_acc - base_acc,
        "delta_macro_f1": clean_f1 - base_f1,
        "delta_balanced_accuracy": clean_bal - base_bal
    }


if __name__ == "__main__":
    run_experiment()
