"""
EXP-003: OCR Preprocessing & Status-Bar Cleanup Experiment.
============================================================
Evaluates raw OCR vs Cleaned OCR text on the 125-sample Validation Partition.
"""

import sys
import json
import re
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "intelligence" / "final_refinement"))

from final_refinement_experiment import FinalRefinedClassifier
from experiments.intelligence.ocr_evidence_semantic.ocr_evidence_semantic import (
    get_ocr_text_for_filename,
    load_cached_ocr_texts,
    normalize_ocr_text
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

METADATA_V2 = PROJECT_ROOT / "data" / "metadata_v2.csv"


def clean_ocr_text(raw_text: str) -> str:
    """
    Cleans top status bar noise (battery %, time, signal icons) and deduplicates repeated UI lines.
    """
    lines = raw_text.split("\n")
    cleaned_lines = []
    
    for i, line in enumerate(lines):
        line_strip = line.strip()
        # Filter top 2 lines status bar artifacts: e.g. "12:34 PM 85%" or "10:56 Wed, 6 May"
        if i < 2 and re.match(r"^(\d{1,2}:\d{2}\s*(AM|PM)?|\d{1,3}%\s*|\w{3},\s*\d{1,2}\s*\w+)*$", line_strip, re.IGNORECASE):
            continue
        cleaned_lines.append(line_strip)
        
    return "\n".join(cleaned_lines)


def run_experiment():
    print("Running EXP-003: OCR Cleaning Experiment...")
    classifier = FinalRefinedClassifier()
    cached_ocr = load_cached_ocr_texts()
    
    df_meta = pd.read_csv(METADATA_V2)
    df_meta = df_meta[~df_meta["file"].isin(["ss_441.png", "ss_443.png", "ss_560.png"])].copy()
    val_df = df_meta.iloc[250:375]
    
    y_true = []
    y_baseline_pred = []
    y_cleaned_pred = []
    
    for _, row in val_df.iterrows():
        fname = row["file"]
        true_cat = row["category"]
        notes = str(row.get("notes", ""))
        
        raw_ocr = get_ocr_text_for_filename(fname, cached_ocr)
        cleaned_ocr = clean_ocr_text(raw_ocr)
        
        res_base = classifier.predict_final(raw_text=raw_ocr, notes=notes, variant="C")
        res_clean = classifier.predict_final(raw_text=cleaned_ocr, notes=notes, variant="C")
        
        y_true.append(true_cat)
        y_baseline_pred.append(res_base["prediction"])
        y_cleaned_pred.append(res_clean["prediction"])
        
    base_acc = accuracy_score(y_true, y_baseline_pred)
    clean_acc = accuracy_score(y_true, y_cleaned_pred)
    
    base_bal = balanced_accuracy_score(y_true, y_baseline_pred)
    clean_bal = balanced_accuracy_score(y_true, y_cleaned_pred)
    
    _, _, base_f1, _ = precision_recall_fscore_support(y_true, y_baseline_pred, average="macro", zero_division=0)
    _, _, clean_f1, _ = precision_recall_fscore_support(y_true, y_cleaned_pred, average="macro", zero_division=0)
    
    print(f"Baseline Validation Top-1: {base_acc:.4f} | Cleaned Top-1: {clean_acc:.4f} (Delta: {clean_acc - base_acc:+.4f})")
    print(f"Baseline Macro F1: {base_f1:.4f} | Cleaned Macro F1: {clean_f1:.4f} (Delta: {clean_f1 - base_f1:+.4f})")
    print(f"Baseline Bal Acc: {base_bal:.4f} | Cleaned Bal Acc: {clean_bal:.4f} (Delta: {clean_bal - base_bal:+.4f})")


if __name__ == "__main__":
    run_experiment()
