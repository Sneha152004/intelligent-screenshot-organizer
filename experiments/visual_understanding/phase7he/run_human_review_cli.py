"""
Phase 7H-E: Interactive Human Review CLI Tool with Visual Screenshot Display.

Enables human annotators to inspect individual screenshots directly using the system image viewer
and record decisions according to the 12-tier priority plan.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from human_review_engine import (
    Phase7HEReviewEngine,
    FULL_VISUAL_TAXONOMY,
    PRIORITY_TIER_SEQUENCE,
    PROJECT_ROOT,
    SCREENSHOTS_DIR
)


def open_individual_screenshot(img_filename: str):
    """Launches the system default image viewer to display the individual screenshot clearly."""
    img_path = SCREENSHOTS_DIR / img_filename
    if not img_path.exists():
        print(f"Warning: Screenshot file not found at {img_path}")
        return

    try:
        if sys.platform == "win32":
            os.startfile(str(img_path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(img_path)])
        else:
            subprocess.Popen(["xdg-open", str(img_path)])
    except Exception as e:
        print(f"Notice: Could not automatically open image viewer ({e}). File location: {img_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 7H-E Interactive Human Review CLI Tool")
    parser.add_argument("--tier", type=str, help="Filter queue by priority tier (e.g., 'Tier 1')")
    parser.add_argument("--type", type=str, help="Filter queue by proposed screen type (e.g., 'dashboard_settings')")
    parser.add_argument("--id", type=str, help="Jump to specific screenshot ID (e.g., 'ss_010.jpg')")
    parser.add_argument("--reviewer", type=str, default="human_annotator", help="Reviewer name")
    parser.add_argument("--non-interactive", dest="non_interactive", action="store_true", help="Run status report without prompting")
    parser.add_argument("--no-view", dest="no_view", action="store_true", help="Disable automatic opening of image viewer")

    args = parser.parse_args()

    engine = Phase7HEReviewEngine()
    report = engine.generate_review_report()

    print("======================================================================")
    print("      PHASE 7H-E: VISUAL SCREEN-TYPE HUMAN REVIEW INTERFACE           ")
    print("======================================================================")
    print(f"Total Valid Screenshots: {report['total_valid']}")
    print(f"Pre-Verified Preserved:  {report['status_counts']['verified']}")
    print(f"Completed Annotations:  {report['status_counts']['completed']}")
    print(f"Pending Review:         {report['status_counts']['pending']}")
    print("======================================================================")

    if args.non_interactive:
        print("\nReview Status Report generated in Phase 7H-E directory.")
        return

    if args.id:
        items = engine.filter_records(screenshot_id=args.id)
    else:
        items = engine.get_pending_queue(priority_tier=args.tier, proposed_screen_type=args.type)

    if not items:
        print("\nNo pending items match the specified criteria.")
        return

    print(f"\nLoaded {len(items)} items for review session.")
    print("Commands: (A)pprove | (C)orrect | (M)ark Ambiguous | (R)eject | (S)kip | (Q)uit\n")

    idx = 0
    while idx < len(items):
        item = items[idx]
        sid = item["screenshot_id"]
        fname = item["filename"]
        prop_stype = item["phase7hc_proposed_screen_type"]
        tier = item["priority_tier"]
        cs_name = item["contact_sheet_name"]
        sem_cat = item["original_semantic_category"]
        cur_status = item["review_status"]
        img_full_path = SCREENSHOTS_DIR / fname

        print("----------------------------------------------------------------------")
        print(f"Item [{idx+1}/{len(items)}]: Screenshot ID: {sid} ({fname})")
        print(f"Priority Tier:      {tier}")
        print(f"Proposed Type:      {prop_stype}")
        print(f"Semantic Category:  {sem_cat}")
        print(f"Contact Sheet:      {cs_name}")
        print(f"Image File Location: {img_full_path}")
        print(f"Current Status:     {cur_status}")
        print("----------------------------------------------------------------------")

        # Open individual screenshot image viewer if enabled
        if not args.no_view:
            open_individual_screenshot(fname)

        try:
            choice = input("Select Action [A/C/M/R/S/Q]: ").strip().upper()
        except EOFError:
            print("\nNon-interactive session ended.")
            break

        if choice == "Q":
            print("Exiting review session.")
            break
        elif choice == "S":
            idx += 1
            continue
        elif choice == "A":
            notes = input("Optional Notes: ").strip()
            engine.record_decision(
                screenshot_id=sid,
                decision="approve",
                reviewer=args.reviewer,
                notes=notes
            )
            print(f"APPROVED: {sid} -> {prop_stype}")
            idx += 1
        elif choice == "C":
            print("\nValid Visual Taxonomy Options:")
            for i, stype in enumerate(FULL_VISUAL_TAXONOMY, 1):
                print(f"  [{i:2d}] {stype}")
            
            val = input("Select Correct Type Number or Name: ").strip()
            if val.isdigit() and 1 <= int(val) <= len(FULL_VISUAL_TAXONOMY):
                corrected_stype = FULL_VISUAL_TAXONOMY[int(val)-1]
            elif val in FULL_VISUAL_TAXONOMY:
                corrected_stype = val
            else:
                print("Invalid taxonomy selection. Action cancelled.")
                continue

            notes = input("Optional Notes for Correction: ").strip()
            engine.record_decision(
                screenshot_id=sid,
                decision="correct",
                final_screen_type=corrected_stype,
                reviewer=args.reviewer,
                notes=notes
            )
            print(f"CORRECTED: {sid} -> {corrected_stype}")
            idx += 1
        elif choice == "M":
            notes = input("Notes explaining ambiguity / second opinion: ").strip()
            engine.record_decision(
                screenshot_id=sid,
                decision="ambiguous",
                reviewer=args.reviewer,
                notes=notes
            )
            print(f"MARKED AMBIGUOUS: {sid}")
            idx += 1
        elif choice == "R":
            notes = input("Reason for rejection: ").strip()
            engine.record_decision(
                screenshot_id=sid,
                decision="reject",
                reviewer=args.reviewer,
                notes=notes
            )
            print(f"REJECTED: {sid}")
            idx += 1
        else:
            print("Invalid choice. Please enter A, C, M, R, S, or Q.")

    engine.generate_review_report()
    print("\nReview session complete. Progress report updated.")


if __name__ == "__main__":
    main()
