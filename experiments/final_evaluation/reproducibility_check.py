"""
Reproducibility Verification Script.
=====================================
Runs semantic, visual, and multimodal evaluations twice and verifies 100% identity
in metric output JSON files to guarantee test result determinism.
"""

import sys
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "experiments" / "final_evaluation"


def run_eval_script(script_name: str):
    """Executes evaluation script via subprocess."""
    script_path = EVAL_DIR / script_name
    print(f"Executing {script_name}...")
    res = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running {script_name}:\n{res.stderr}")
        sys.exit(1)


def strip_latencies(d):
    """Recursively removes variable latency fields from metric dict for deterministic comparison."""
    if isinstance(d, dict):
        return {
            k: strip_latencies(v)
            for k, v in d.items()
            if "latency" not in k and "time" not in k
        }
    elif isinstance(d, list):
        return [strip_latencies(item) for item in d]
    return d


def compare_json_files(json_name: str) -> bool:
    """Verifies that two JSON files produced across runs are identical for all predictions and metrics."""
    p1 = EVAL_DIR / json_name
    p_run1 = EVAL_DIR / f"run1_{json_name}"
    
    with open(p1, "r", encoding="utf-8") as f:
        data2 = json.load(f)
    with open(p_run1, "r", encoding="utf-8") as f:
        data1 = json.load(f)
        
    return strip_latencies(data1) == strip_latencies(data2)


def main():
    print("=" * 60)
    print("RUNNING REPRODUCIBILITY AUDIT (2 EVALUATION RUNS)")
    print("=" * 60)
    
    scripts_and_jsons = [
        ("evaluate_semantic.py", "semantic_metrics.json"),
        ("evaluate_visual.py", "visual_metrics.json"),
        ("evaluate_multimodal.py", "multimodal_metrics.json")
    ]
    
    # Run 1
    print("\n--- RUN 1 ---")
    for script_name, json_name in scripts_and_jsons:
        run_eval_script(script_name)
        # Copy to run1_
        orig_p = EVAL_DIR / json_name
        run1_p = EVAL_DIR / f"run1_{json_name}"
        run1_p.write_bytes(orig_p.read_bytes())
        
    # Run 2
    print("\n--- RUN 2 ---")
    for script_name, json_name in scripts_and_jsons:
        run_eval_script(script_name)
        
    # Verify exact identity
    print("\n--- VERIFYING DETERMINISTIC IDENTITY ---")
    all_matched = True
    for _, json_name in scripts_and_jsons:
        matched = compare_json_files(json_name)
        status = "PASSED (100% Identical)" if matched else "FAILED (Discrepancy Detected)"
        print(f" - {json_name}: {status}")
        if not matched:
            all_matched = False
            
    # Cleanup run1_ files
    for _, json_name in scripts_and_jsons:
        run1_p = EVAL_DIR / f"run1_{json_name}"
        if run1_p.exists():
            run1_p.unlink()
            
    if all_matched:
        print("\nREPRODUCIBILITY CHECK PASSED! All evaluation runs produce identical results.")
    else:
        print("\nREPRODUCIBILITY CHECK FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
