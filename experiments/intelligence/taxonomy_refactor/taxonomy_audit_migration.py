"""
Taxonomy Audit & Migration Script for Screenshot Intelligence System
====================================================================
Audits the existing 833 valid screenshot records (and 836 metadata rows)
to refactor the taxonomy from 10 categories to 9 canonical categories by
removing Coding as a top-level category and representing coding attributes via tags.

Migration Rules:
1. Coding top-level category is REMOVED.
2. Coding screenshots are re-mapped based on broad primary context:
   - Educational coding (LeetCode, Kaggle, AI Lab, tutorials, courses) -> Education
   - Software development (VS Code projects, Git repositories, codebases) -> Work & Career
   - Generic/sparse coding content -> General
3. Coding tag extraction adds 'Coding' and specific language/tool tags.
4. Output is saved to data/metadata_v2.csv and isolated experiment directory.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
METADATA_CSV_V1 = DATA_DIR / "metadata.csv"
METADATA_CSV_V2 = DATA_DIR / "metadata_v2.csv"
CACHE_FILE = PROJECT_ROOT / "experiments" / "intelligence" / "ocr_semantic_classification" / "ocr_text_cache.json"

EXP_DIR = Path(__file__).resolve().parent
AUDIT_JSON = EXP_DIR / "taxonomy_audit_summary.json"

CANONICAL_9_CATEGORIES = [
    "Communication",
    "Connectivity",
    "Education",
    "Entertainment",
    "General",
    "Payment",
    "Shopping",
    "Travel",
    "Work & Career",
]

INVALID_ZERO_BYTE_FILES = {"ss_441.png", "ss_443.png", "ss_560.png"}


def extract_coding_tags(text: str, notes: str) -> List[str]:
    """Extracts deterministic coding and technology tags based on text evidence."""
    combined = (text + " " + notes).lower()
    tags = []

    code_indicators = [
        "def ", "class ", "import ", "vs code", "vscode", "github", "git ", "git status",
        "leetcode", "kaggle", "terminal", "python", "javascript", "typescript", "html",
        "css", "sql", "programming", "code editor", "syntaxerror", "traceback", "public static",
        "function", "const ", "let ", "var ", "return ", "npm ", "pip "
    ]

    if any(ind in combined for ind in code_indicators):
        tags.append("Coding")
        tags.append("Software Development")

        if any(k in combined for k in ["python", "def ", "import ", "pip "]):
            tags.append("Python")
        if any(k in combined for k in ["javascript", "js", "const ", "function", "npm "]):
            tags.append("JavaScript")
        if any(k in combined for k in ["vs code", "vscode"]):
            tags.append("VS Code")
        if any(k in combined for k in ["github", "git ", "git status", "git pull", "git fetch"]):
            tags.append("GitHub")
            tags.append("Git")
        if "leetcode" in combined:
            tags.append("LeetCode")
            tags.append("Algorithm")
        if "kaggle" in combined:
            tags.append("Kaggle")
            tags.append("Machine Learning")
        if any(k in combined for k in ["terminal", "bash", "cmd", "powershell"]):
            tags.append("Terminal")
        if any(k in combined for k in ["html", "css", "web"]):
            tags.append("Web Development")

    # Deduplicate tags preserving order
    return list(dict.fromkeys(tags))


def run_taxonomy_audit_and_migration() -> Dict[str, Any]:
    """Audits metadata.csv and produces metadata_v2.csv with 9 categories and tags."""
    if not METADATA_CSV_V1.exists():
        raise FileNotFoundError(f"Original metadata CSV not found: {METADATA_CSV_V1}")

    ocr_cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            ocr_cache = json.load(f)

    df_v1 = pd.read_csv(METADATA_CSV_V1)

    v2_records = []
    remap_summary: Dict[str, int] = {}
    audited_coding_details = []

    for _, row in df_v1.iterrows():
        fname = str(row["file"]).strip()
        old_cat = str(row["category"]).strip()
        if old_cat == "Work and career":
            old_cat = "Work & Career"

        notes = str(row["notes"]) if pd.notna(row.get("notes")) else ""
        confidence = str(row.get("confidence", "high")).strip().lower()

        ocr_text = ocr_cache.get(fname, "")
        if not ocr_text:
            stem = Path(fname).stem
            ocr_text = ocr_cache.get(f"{stem}.jpg", ocr_cache.get(f"{stem}.png", ""))

        tags = extract_coding_tags(ocr_text, notes)

        # General non-coding tag additions
        text_lower = ocr_text.lower()
        if old_cat == "Payment" or "upi" in text_lower or "paid" in text_lower:
            tags.append("Payment")
        if old_cat == "Shopping" or "cart" in text_lower or "price" in text_lower:
            tags.append("Shopping")
        if old_cat == "Connectivity" or "wifi" in text_lower or "ip address" in text_lower:
            tags.append("Connectivity")

        tags = list(dict.fromkeys(tags))
        tags_str = ", ".join(tags)

        new_cat = old_cat
        reason = "Retained original 9-taxonomy category."

        if old_cat == "Coding":
            text_combo = (ocr_text + " " + notes).lower()
            if any(k in text_combo for k in ["leetcode", "kaggle", "ailab", "labz", "tutorial", "course", "lecture", "assignment", "homework", "practice"]):
                new_cat = "Education"
                reason = "Educational programming assignment, LeetCode practice, or course material."
            elif any(k in text_combo for k in ["vs code", "github", "git ", "git status", "git pull", "mcp-rag", "stock-predictor", "questlog", "project", "script.js", "style.css"]):
                new_cat = "Work & Career"
                reason = "Professional software development project, repository, or workspace."
            else:
                if "ragnova" in text_combo or "python/matplotlib" in text_combo or "data-science" in text_combo or "cse22" in text_combo:
                    new_cat = "Work & Career"
                    reason = "Software project codebase or data science development."
                else:
                    new_cat = "General"
                    reason = "Coding content without distinct educational or workplace context."

            remap_summary[new_cat] = remap_summary.get(new_cat, 0) + 1
            audited_coding_details.append({
                "file": fname,
                "old_category": old_cat,
                "new_category": new_cat,
                "tags": tags,
                "reason": reason,
            })

        v2_records.append({
            "file": fname,
            "category": new_cat,
            "old_category": old_cat,
            "tags": tags_str,
            "confidence": confidence,
            "notes": notes,
            "migration_reason": reason,
        })

    v2_df = pd.DataFrame(v2_records)
    v2_df.to_csv(METADATA_CSV_V2, index=False)
    print(f"[OK] Saved versioned dataset metadata to {METADATA_CSV_V2}")

    # Also save inside experiment directory
    exp_csv = EXP_DIR / "metadata_v2.csv"
    v2_df.to_csv(exp_csv, index=False)

    summary_data = {
        "total_records": len(v2_df),
        "valid_records_count": len(v2_df[~v2_df["file"].isin(INVALID_ZERO_BYTE_FILES)]),
        "old_coding_count": len(audited_coding_details),
        "coding_remap_breakdown": remap_summary,
        "new_category_distribution": v2_df["category"].value_counts().to_dict(),
        "audited_coding_details": audited_coding_details,
    }

    with open(AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"[OK] Saved taxonomy audit JSON summary to {AUDIT_JSON}")
    return summary_data


if __name__ == "__main__":
    res = run_taxonomy_audit_and_migration()
    print("Migration Summary:", res["coding_remap_breakdown"])
