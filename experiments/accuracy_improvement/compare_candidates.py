"""
Compare Candidates Script
=========================
Parses experiment_log.csv, sorts candidates by validation performance metrics,
and prints structured decision tables for Semantic, Visual, and Multimodal components.
"""

import pandas as pd
from pathlib import Path

EXPERIMENT_LOG_PATH = Path(__file__).resolve().parent / "experiment_log.csv"

def compare_candidates():
    if not EXPERIMENT_LOG_PATH.exists():
        print(f"Error: {EXPERIMENT_LOG_PATH} not found.")
        return

    df = pd.read_csv(EXPERIMENT_LOG_PATH)
    
    print("=" * 80)
    print("ACCURACY IMPROVEMENT SPRINT: CANDIDATE MODEL COMPARISON & SELECTION SUMMARY")
    print("=" * 80)

    for component in ["Semantic", "Visual", "Multimodal", "Preprocessing"]:
        comp_df = df[df["component"] == component].copy()
        if comp_df.empty:
            continue
        
        comp_df.sort_values(by=["validation_top1", "validation_macro_f1"], ascending=False, inplace=True)
        
        print(f"\n--- COMPONENT: {component.upper()} ---")
        header = f"{'EXP ID':<10} {'Change Description':<45} {'Status':<10} {'Top-1':<8} {'Macro F1':<10} {'Bal Acc':<10} {'Delta Top-1':<12}"
        print(header)
        print("-" * len(header))
        
        for _, row in comp_df.iterrows():
            exp_id = row["experiment_id"]
            change = str(row["change"])[:44]
            status = row["status"]
            top1 = f"{row['validation_top1']*100:.2f}%"
            f1 = f"{row['validation_macro_f1']:.4f}"
            bal = f"{row['validation_balanced_accuracy']:.4f}"
            delta = f"{row['delta_top1']*100:+.2f}%"
            
            print(f"{exp_id:<10} {change:<45} {status:<10} {top1:<8} {f1:<10} {bal:<10} {delta:<12}")

    print("\n" * 1)
    print("=" * 80)
    print("SELECTED WINNING CANDIDATE MODELS (FROZEN FOR HELD-OUT TEST EVALUATION):")
    print("=" * 80)
    print("1. SEMANTIC CLASSIFIER : EXP-004 (RefinedSemanticCandidate) [Top-1: 93.60%, Macro F1: 0.9516]")
    print("2. VISUAL ADAPTER     : EXP-001 (Baseline 5-Class YOLOv8n-cls) [Top-1: 72.00%, Macro F1: 0.5686]")
    print("3. MULTIMODAL SYSTEM   : EXP-006 (Candidate Multimodal System - Strategy E+) [Top-1: 84.00%, Macro F1: 0.6889]")
    print("=" * 80)

if __name__ == "__main__":
    compare_candidates()
