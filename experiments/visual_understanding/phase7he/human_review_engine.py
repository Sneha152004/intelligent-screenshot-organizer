"""
Phase 7H-E: Visual Screen-Type Human Review Workflow Engine & Ground-Truth Finalization.

Provides programmatic and CLI tools for inspecting, annotating, filtering, and reporting on the 836 screenshot dataset while strictly maintaining data integrity.
"""

import os
import sys
import csv
import json
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Path Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PHASE7HD_DIR = PROJECT_ROOT / "experiments" / "visual_understanding" / "phase7hd"
PHASE7HC_DIR = PROJECT_ROOT / "experiments" / "visual_understanding" / "phase7hc"
PHASE7HE_DIR = PROJECT_ROOT / "experiments" / "visual_understanding" / "phase7he"
CONTACT_SHEETS_DIR = PHASE7HC_DIR / "corrected_contact_sheets"
SCREENSHOTS_DIR = PROJECT_ROOT / "data" / "screenshots"

# Taxonomy Definition (13 Canonical Screen Types)
FULL_VISUAL_TAXONOMY = [
    "product_listing",
    "product_detail",
    "cart_checkout_payment",
    "login_authentication",
    "messaging_chat",
    "email_view",
    "media_player_library",
    "code_editor_terminal",
    "document_article_view",
    "travel_booking",
    "dashboard_settings",
    "error_dialog_alert",
    "other_screen"
]

# Locked 12 Phase 7E Human Decisions
LOCKED_PHASE7E_SCREENSHOT_IDS = {
    "ss_001.jpg", "ss_799.jpg", "ss_002.jpg", "ss_800.jpg",
    "ss_003.jpg", "ss_801.jpg", "ss_004.jpg", "ss_802.jpg",
    "ss_005.jpg", "ss_803.jpg", "ss_007.jpg", "ss_805.jpg"
}

# Locked 3 Corrupted 0-Byte Screenshots
CORRUPTED_SCREENSHOT_IDS = {"ss_441.png", "ss_443.png", "ss_560.png"}

# 12-Tier Review Priority Sequence
PRIORITY_TIER_SEQUENCE = [
    ("Tier 1", "dashboard_settings"),
    ("Tier 2", "email_view"),
    ("Tier 3", "product_listing"),
    ("Tier 4", "login_authentication"),
    ("Tier 5", "cart_checkout_payment"),
    ("Tier 6", "error_dialog_alert"),
    ("Tier 7", "product_detail"),
    ("Tier 8", "messaging_chat"),
    ("Tier 9", "code_editor_terminal"),
    ("Tier 10", "media_player_library"),
    ("Tier 11", "travel_booking"),
    ("Tier 12", "document_article_view")
]

PRIORITY_TIER_MAP = {stype: tier_name for tier_name, stype in PRIORITY_TIER_SEQUENCE}

VALID_DECISIONS = {"approve", "correct", "ambiguous", "reject", "pending"}

LEDGER_HEADER = [
    "screenshot_id",
    "filename",
    "original_semantic_category",
    "phase7hc_proposed_screen_type",
    "final_human_screen_type",
    "human_decision",
    "reviewer",
    "review_notes",
    "verification_date",
    "duplicate_group_id",
    "priority_tier",
    "contact_sheet_name",
    "is_corrupted",
    "review_status"
]


class Phase7HEReviewEngine:
    def __init__(self, ledger_path: Optional[Path] = None):
        PHASE7HE_DIR.mkdir(parents=True, exist_ok=True)
        self.ledger_path = ledger_path or (PHASE7HE_DIR / "phase7he_human_review_ledger.csv")
        self.records: Dict[str, Dict] = {}
        self._initialize_or_load_ledger()

    def _initialize_or_load_ledger(self):
        """Initializes ledger from Phase 7H-D if not present, otherwise loads existing ledger for resume."""
        if self.ledger_path.exists():
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.records[row["screenshot_id"]] = row
        else:
            # Seed from Phase 7H-D human review ledger
            source_hd_ledger = PHASE7HD_DIR / "phase7hd_human_review_ledger.csv"
            if not source_hd_ledger.exists():
                raise FileNotFoundError(f"Source Phase 7H-D ledger not found at {source_hd_ledger}")

            with open(source_hd_ledger, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sid = row["screenshot_id"]
                    prop_stype = row["phase7hc_proposed_screen_type"]
                    is_corr = (sid in CORRUPTED_SCREENSHOT_IDS)
                    is_p7e = (sid in LOCKED_PHASE7E_SCREENSHOT_IDS)

                    if is_corr:
                        final_stype = "other_screen"
                        decision = "reject"
                        reviewer = "system_audit"
                        notes = "Corrupted 0-byte file."
                        vdate = "2026-10-02"
                        status = "rejected"
                    elif is_p7e:
                        final_stype = prop_stype
                        decision = "approve"
                        reviewer = "human_annotator_phase7e"
                        notes = "Preserved human-verified decision from Phase 7E harmonization."
                        vdate = "2026-10-02"
                        status = "verified"
                    else:
                        final_stype = row.get("human_verified_screen_type", "")
                        decision = row.get("human_decision", "pending")
                        reviewer = row.get("reviewer", "")
                        notes = row.get("review_notes", "Pending human review.")
                        vdate = row.get("verification_date", "")
                        status = "pending"

                    tier_name = PRIORITY_TIER_MAP.get(prop_stype, "Tier 13 (Other)")
                    cs_name = f"contact_sheet_{prop_stype}.png"

                    self.records[sid] = {
                        "screenshot_id": sid,
                        "filename": sid,
                        "original_semantic_category": row.get("original_semantic_category", "unknown"),
                        "phase7hc_proposed_screen_type": prop_stype,
                        "final_human_screen_type": final_stype,
                        "human_decision": decision,
                        "reviewer": reviewer,
                        "review_notes": notes,
                        "verification_date": vdate,
                        "duplicate_group_id": row.get("duplicate_group_id", f"singleton_{sid.split('.')[0]}"),
                        "priority_tier": tier_name,
                        "contact_sheet_name": cs_name,
                        "is_corrupted": str(is_corr),
                        "review_status": status
                    }
            self.save()

    def save(self):
        """Persists records atomically to CSV file."""
        temp_path = self.ledger_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LEDGER_HEADER)
            writer.writeheader()
            for sid in sorted(self.records.keys()):
                writer.writerow(self.records[sid])
        temp_path.replace(self.ledger_path)

    def record_decision(
        self,
        screenshot_id: str,
        decision: str,
        final_screen_type: Optional[str] = None,
        reviewer: str = "human_annotator",
        notes: str = "",
        force_override_locked: bool = False
    ) -> Dict:
        """Records a human review decision with strict integrity validation."""
        if screenshot_id not in self.records:
            raise KeyError(f"Screenshot ID '{screenshot_id}' not found in review ledger.")

        if screenshot_id in CORRUPTED_SCREENSHOT_IDS:
            raise ValueError(f"Cannot review corrupted 0-byte screenshot '{screenshot_id}'.")

        if screenshot_id in LOCKED_PHASE7E_SCREENSHOT_IDS and not force_override_locked:
            raise ValueError(f"Cannot modify pre-verified Phase 7E human decision for '{screenshot_id}'.")

        decision = decision.lower().strip()
        if decision not in VALID_DECISIONS:
            raise ValueError(f"Invalid decision '{decision}'. Must be one of {VALID_DECISIONS}.")

        rec = self.records[screenshot_id]
        prop_stype = rec["phase7hc_proposed_screen_type"]

        if decision == "approve":
            target_stype = final_screen_type or prop_stype
            if target_stype not in FULL_VISUAL_TAXONOMY:
                raise ValueError(f"Invalid screen type '{target_stype}'. Must be in taxonomy.")
            status = "completed"
        elif decision == "correct":
            if not final_screen_type:
                raise ValueError("Must provide final_screen_type when correcting a label.")
            if final_screen_type not in FULL_VISUAL_TAXONOMY:
                raise ValueError(f"Invalid screen type '{final_screen_type}'. Must be in taxonomy.")
            target_stype = final_screen_type
            status = "completed"
        elif decision == "ambiguous":
            target_stype = final_screen_type or prop_stype
            status = "ambiguous"
        elif decision == "reject":
            target_stype = "unclassifiable"
            status = "rejected"
        else: # pending
            target_stype = ""
            status = "pending"

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        rec["final_human_screen_type"] = target_stype
        rec["human_decision"] = decision
        rec["reviewer"] = reviewer
        rec["review_notes"] = notes or f"Decision set to {decision}"
        rec["verification_date"] = now_str
        rec["review_status"] = status

        self.save()
        return rec

    def get_pending_queue(
        self,
        priority_tier: Optional[str] = None,
        proposed_screen_type: Optional[str] = None
    ) -> List[Dict]:
        """Returns pending screenshots in strict 12-tier priority sequence."""
        pending_items = []
        for tier_name, stype in PRIORITY_TIER_SEQUENCE:
            if priority_tier and tier_name != priority_tier:
                continue
            if proposed_screen_type and stype != proposed_screen_type:
                continue

            tier_items = [
                r for r in self.records.values()
                if r["review_status"] == "pending"
                and r["phase7hc_proposed_screen_type"] == stype
                and r["is_corrupted"].lower() == "false"
            ]
            tier_items.sort(key=lambda x: x["screenshot_id"])
            pending_items.extend(tier_items)

        return pending_items

    def filter_records(
        self,
        status: Optional[str] = None,
        proposed_screen_type: Optional[str] = None,
        screenshot_id: Optional[str] = None
    ) -> List[Dict]:
        """Filters ledger records based on criteria."""
        results = []
        for r in self.records.values():
            if screenshot_id and r["screenshot_id"] != screenshot_id:
                continue
            if status and r["review_status"] != status:
                continue
            if proposed_screen_type and r["phase7hc_proposed_screen_type"] != proposed_screen_type:
                continue
            results.append(r)
        return results

    def generate_review_report(self) -> Dict:
        """Generates comprehensive review progress report."""
        total_all = len(self.records)
        corrupted_count = sum(1 for r in self.records.values() if r["is_corrupted"].lower() == "true")
        total_valid = total_all - corrupted_count

        status_counts = {"verified": 0, "completed": 0, "ambiguous": 0, "rejected": 0, "pending": 0}
        decision_counts = {"approve": 0, "correct": 0, "ambiguous": 0, "reject": 0, "pending": 0}
        
        transition_matrix: Dict[str, Dict[str, int]] = {
            stype: {target: 0 for target in FULL_VISUAL_TAXONOMY + ["unclassifiable", "pending", "rejected"]}
            for stype in FULL_VISUAL_TAXONOMY
        }

        tier_summary = []
        for tier_name, stype in PRIORITY_TIER_SEQUENCE:
            tier_recs = [r for r in self.records.values() if r["phase7hc_proposed_screen_type"] == stype and r["is_corrupted"].lower() == "false"]
            tier_total = len(tier_recs)
            tier_completed = sum(1 for r in tier_recs if r["review_status"] in ("verified", "completed"))
            tier_pending = sum(1 for r in tier_recs if r["review_status"] == "pending")
            tier_ambig = sum(1 for r in tier_recs if r["review_status"] == "ambiguous")
            tier_rej = sum(1 for r in tier_recs if r["review_status"] == "rejected")
            
            pct = (tier_completed / tier_total * 100.0) if tier_total > 0 else 0.0

            tier_summary.append({
                "tier": tier_name,
                "screen_type": stype,
                "total": tier_total,
                "completed": tier_completed,
                "pending": tier_pending,
                "ambiguous": tier_ambig,
                "rejected": tier_rej,
                "completion_pct": round(pct, 2)
            })

        for r in self.records.values():
            if r["is_corrupted"].lower() == "true":
                continue
            st = r["review_status"]
            dec = r["human_decision"]
            status_counts[st] = status_counts.get(st, 0) + 1
            decision_counts[dec] = decision_counts.get(dec, 0) + 1

            prop = r["phase7hc_proposed_screen_type"]
            final_stype = r["final_human_screen_type"] or "pending"
            if prop in transition_matrix and final_stype in transition_matrix[prop]:
                transition_matrix[prop][final_stype] += 1

        report_data = {
            "total_records": total_all,
            "total_valid": total_valid,
            "corrupted_count": corrupted_count,
            "status_counts": status_counts,
            "decision_counts": decision_counts,
            "tier_summary": tier_summary,
            "transition_matrix": transition_matrix
        }

        # Write markdown report
        report_md_path = PHASE7HE_DIR / "phase7he_review_progress_report.md"
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(f"""# Phase 7H-E: Human Review Progress & Integrity Report

**Date**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Total Dataset Records**: {total_all}  
**Total Valid Screenshots**: {total_valid}  
**Corrupted Excluded**: {corrupted_count}  

---

## 1. Overall Progress Metrics

* **Phase 7E Pre-Verified (Preserved)**: {status_counts.get("verified", 0)}
* **Completed New Annotations**: {status_counts.get("completed", 0)}
* **Ambiguous / Second-Opinion**: {status_counts.get("ambiguous", 0)}
* **Rejected Valid Screenshots**: {status_counts.get("rejected", 0)}
* **Pending Human Verification**: **{status_counts.get("pending", 0)}**

---

## 2. Priority Tier Execution Progress

| Priority Tier | Visual Screen Type | Total Valid | Verified/Completed | Pending | Ambiguous | Rejected | Completion % |
| :--- | :--- | ----: | ----: | ----: | ----: | ----: | ----: |
""")
            for t in tier_summary:
                f.write(f"| **{t['tier']}** | **`{t['screen_type']}`** | {t['total']} | {t['completed']} | {t['pending']} | {t['ambiguous']} | {t['rejected']} | {t['completion_pct']}% |\n")

        return report_data

    def generate_final_manifest(self, force: bool = False) -> Path:
        """Generates the finalized ground-truth manifest vsue_ground_truth_manifest_v1.csv when criteria are met."""
        pending_items = [
            r for r in self.records.values()
            if r["review_status"] == "pending" and r["is_corrupted"].lower() == "false"
        ]

        if pending_items and not force:
            raise ValueError(f"Cannot generate final manifest: {len(pending_items)} valid screenshots are still pending human review.")

        manifest_path = PHASE7HE_DIR / "vsue_ground_truth_manifest_v1.csv"
        manifest_header = [
            "screenshot_id",
            "filename",
            "screen_type",
            "original_semantic_category",
            "phase7hc_proposed_screen_type",
            "human_decision",
            "reviewer",
            "verification_date",
            "duplicate_group_id",
            "is_corrupted",
            "review_status"
        ]

        valid_manifest_rows = []
        for sid in sorted(self.records.keys()):
            r = self.records[sid]
            if r["is_corrupted"].lower() == "true":
                continue
            
            final_stype = r["final_human_screen_type"] or r["phase7hc_proposed_screen_type"]
            valid_manifest_rows.append({
                "screenshot_id": r["screenshot_id"],
                "filename": r["filename"],
                "screen_type": final_stype,
                "original_semantic_category": r["original_semantic_category"],
                "phase7hc_proposed_screen_type": r["phase7hc_proposed_screen_type"],
                "human_decision": r["human_decision"],
                "reviewer": r["reviewer"],
                "verification_date": r["verification_date"],
                "duplicate_group_id": r["duplicate_group_id"],
                "is_corrupted": r["is_corrupted"],
                "review_status": r["review_status"]
            })

        with open(manifest_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=manifest_header)
            writer.writeheader()
            for row in valid_manifest_rows:
                writer.writerow(row)

        return manifest_path


if __name__ == "__main__":
    print("======================================================================")
    print("     PHASE 7H-E: HUMAN REVIEW ENGINE INITIALIZATION & REPORT          ")
    print("======================================================================")
    engine = Phase7HEReviewEngine()
    rep = engine.generate_review_report()
    print(f"Ledger Path: {engine.ledger_path}")
    print(f"Total Records: {rep['total_records']}")
    print(f"Total Valid Screenshots: {rep['total_valid']}")
    print(f"Corrupted Excluded: {rep['corrupted_count']}")
    print(f"Pre-Verified Preserved: {rep['status_counts']['verified']}")
    print(f"Pending Annotations: {rep['status_counts']['pending']}")
    print("======================================================================")
