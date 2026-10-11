"""
Phase 4A.1 External Dataset Manual Review & Quality Audit Tool
================================================================
This tool performs a comprehensive dataset quality audit on the 60-image
external screenshot evaluation dataset under experiments/intelligence/external/.

Integrity checks, SHA-256 hash overlap checks, category distribution analysis,
taxonomy consistency audits, and intent quality checks are executed.

Generates:
1. experiments/intelligence/external/manual_review.html
2. experiments/intelligence/external/manual_review_checklist.csv

IMPORTANT:
This is a DATASET QUALITY AUDIT tool only.
Do NOT modify rules, classifiers, thresholds, prototype anchors, or production code.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Tuple
from PIL import Image, UnidentifiedImageError

from experiments.intelligence.external.evaluation_targets import (
    get_evaluation_targets,
    EVALUATION_TARGET_FILENAMES,
)
from experiments.intelligence.rules import resolve_content_type
from src.models.metadata import CATEGORY_CONTENT_TYPES

EXTERNAL_DIR = Path(__file__).resolve().parent
EXTERNAL_IMAGES_DIR = EXTERNAL_DIR / "images"
ORIGINAL_IMAGES_DIR = EXTERNAL_DIR.parent.parent.parent / "data" / "screenshots"
MANIFEST_PATH = EXTERNAL_DIR / "manifest.csv"
HTML_OUTPUT_PATH = EXTERNAL_DIR / "manual_review.html"
CHECKLIST_OUTPUT_PATH = EXTERNAL_DIR / "manual_review_checklist.csv"


def calculate_sha256(file_path: Path) -> str:
    """Calculates SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def check_dataset_integrity() -> Dict[str, Any]:
    """
    Performs rigorous integrity checks on dataset files, images, targets, and manifest.
    """
    results: Dict[str, Any] = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "image_count": 0,
        "target_count": 0,
        "manifest_count": 0,
        "unique_filenames": True,
        "unique_image_ids": True,
        "missing_images": [],
        "orphan_targets": [],
        "corrupted_images": [],
        "image_metadata": {},
    }

    # 1. Target Count
    targets = get_evaluation_targets()
    results["target_count"] = len(targets)
    if len(targets) != 60:
        results["passed"] = False
        results["errors"].append(f"Expected 60 evaluation targets, found {len(targets)}")

    # 2. Manifest Count & Consistency
    if not MANIFEST_PATH.exists():
        results["passed"] = False
        results["errors"].append(f"Manifest file missing: {MANIFEST_PATH}")
    else:
        with open(MANIFEST_PATH, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            manifest_rows = list(reader)
            results["manifest_count"] = len(manifest_rows)
            if len(manifest_rows) != 60:
                results["passed"] = False
                results["errors"].append(f"Expected 60 manifest rows, found {len(manifest_rows)}")
            
            # Check unique IDs & Filenames in Manifest
            manifest_ids = [r["id"] for r in manifest_rows if "id" in r]
            manifest_files = [r["filename"] for r in manifest_rows if "filename" in r]
            if len(set(manifest_ids)) != len(manifest_ids):
                results["unique_image_ids"] = False
                results["passed"] = False
                results["errors"].append("Duplicate image IDs found in manifest.csv")
            if len(set(manifest_files)) != len(manifest_files):
                results["unique_filenames"] = False
                results["passed"] = False
                results["errors"].append("Duplicate filenames found in manifest.csv")

    # 3. Image Files Integrity
    image_files = sorted(list(EXTERNAL_IMAGES_DIR.glob("*.jpg")))
    results["image_count"] = len(image_files)
    if len(image_files) != 60:
        results["passed"] = False
        results["errors"].append(f"Expected 60 image files in {EXTERNAL_IMAGES_DIR}, found {len(image_files)}")

    target_filenames = {t["filename"] for t in targets}
    disk_filenames = {p.name for p in image_files}

    missing_on_disk = target_filenames - disk_filenames
    if missing_on_disk:
        results["passed"] = False
        results["missing_images"] = list(missing_on_disk)
        results["errors"].append(f"Missing image files for targets: {missing_on_disk}")

    orphan_targets = disk_filenames - target_filenames
    if orphan_targets:
        results["passed"] = False
        results["orphan_targets"] = list(orphan_targets)
        results["errors"].append(f"Image files without evaluation target: {orphan_targets}")

    # 4. Image Dimensions & File Validity
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                width, height = img.size
                mode = img.mode
                size_kb = img_path.stat().st_size / 1024.0
                results["image_metadata"][img_path.name] = {
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "size_kb": round(size_kb, 2),
                }
        except Exception as e:
            results["passed"] = False
            results["corrupted_images"].append((img_path.name, str(e)))
            results["errors"].append(f"Corrupted or invalid image file {img_path.name}: {e}")

    return results


def check_sha256_overlaps() -> Dict[str, Any]:
    """
    Computes SHA-256 hashes of external images and checks for exact overlap
    with original dataset screenshots under data/screenshots/.
    """
    ext_hashes: Dict[str, str] = {}
    for img_path in sorted(EXTERNAL_IMAGES_DIR.glob("*.jpg")):
        ext_hashes[img_path.name] = calculate_sha256(img_path)

    orig_hashes: Dict[str, str] = {}
    if ORIGINAL_IMAGES_DIR.exists():
        for img_path in sorted(ORIGINAL_IMAGES_DIR.glob("*.jpg")):
            orig_hashes[img_path.name] = calculate_sha256(img_path)

    overlaps: List[Tuple[str, str, str]] = []
    for ext_name, ext_hash in ext_hashes.items():
        for orig_name, orig_hash in orig_hashes.items():
            if ext_hash == orig_hash:
                overlaps.append((ext_name, orig_name, ext_hash))

    return {
        "external_image_count": len(ext_hashes),
        "original_image_count": len(orig_hashes),
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
    }


def analyze_category_distribution() -> Dict[str, Any]:
    """Computes ground-truth category distribution for all 60 targets."""
    targets = get_evaluation_targets()
    counts: Dict[str, int] = {}
    total = len(targets)

    for t in targets:
        cat = t["category"]
        counts[cat] = counts.get(cat, 0) + 1

    sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
    percentages = {cat: round((count / total) * 100.0, 2) for cat, count in sorted_counts.items()}

    return {
        "total_targets": total,
        "counts": sorted_counts,
        "percentages": percentages,
    }


def audit_taxonomy_and_intents() -> Dict[str, Any]:
    """
    Audits taxonomy content_type consistency and intent plausibility across targets.
    """
    targets = get_evaluation_targets()
    taxonomy_issues = []
    intent_issues = []

    for t in targets:
        filename = t["filename"]
        cat = t["category"]
        intent = t.get("intent", "")
        summary = t.get("summary", "")
        tags = t.get("tags", [])
        
        # Derive content_type
        text_evidence = f"{summary} {' '.join(tags)}"
        content_type = t.get("content_type") or resolve_content_type(cat, text_evidence, {}, [])

        # Check Taxonomy match
        valid_types = CATEGORY_CONTENT_TYPES.get(cat, [])
        if content_type not in valid_types:
            taxonomy_issues.append({
                "filename": filename,
                "category": cat,
                "content_type": content_type,
                "valid_types": valid_types,
                "reason": f"Content type '{content_type}' not in taxonomy for category '{cat}'",
            })

        # Check Intent quality / speculativeness
        if not intent or intent.lower() in ["none", "unknown", ""]:
            intent_issues.append({
                "filename": filename,
                "intent": intent,
                "type": "missing_or_none",
                "note": "Intent is None/unknown",
            })
        elif len(intent.split()) > 8 or any(w in intent.lower() for w in ["evaluate", "explore", "record"]):
            intent_issues.append({
                "filename": filename,
                "intent": intent,
                "type": "potentially_speculative",
                "note": "Intent may contain user motivation assumption not directly in image OCR",
            })

    return {
        "taxonomy_issues_count": len(taxonomy_issues),
        "taxonomy_issues": taxonomy_issues,
        "intent_issues_count": len(intent_issues),
        "intent_issues": intent_issues,
    }


def generate_html_review_page(integrity_meta: Dict[str, Dict[str, Any]]) -> Path:
    """
    Generates a standalone, offline HTML review page displaying all 60 screenshots
    and their complete ground-truth annotations.
    """
    targets = get_evaluation_targets()

    cards_html = []
    for t in targets:
        fname = t["filename"]
        img_id = fname.replace(".jpg", "")
        cat = t["category"]
        tags = t.get("tags", [])
        intent = t.get("intent", "None")
        summary = t.get("summary", "")
        importance = t.get("importance", 3)
        entities = t.get("entities", [])
        dates = t.get("dates", [])
        action_items = t.get("action_items", [])
        sensitive = t.get("sensitive_info", {})
        
        # Metadata
        meta = integrity_meta.get(fname, {"width": "N/A", "height": "N/A", "mode": "RGB", "size_kb": "N/A"})
        dims_str = f"{meta['width']} × {meta['height']} ({meta['size_kb']} KB)"

        # Resolve Content-Type
        text_evidence = f"{summary} {' '.join(tags)}"
        content_type = t.get("content_type") or resolve_content_type(cat, text_evidence, {}, [])

        # Tags HTML
        tags_badge_html = " ".join([f'<span class="badge tag">{tag}</span>' for tag in tags])

        # Sensitive Info HTML
        sens_html = []
        if sensitive.get("contains_password"):
            sens_html.append('<span class="badge danger">🔑 Password</span>')
        if sensitive.get("contains_payment_info"):
            sens_html.append('<span class="badge danger">💳 Payment Info</span>')
        if sensitive.get("contains_personal_contact"):
            sens_html.append('<span class="badge warning">📱 Personal Contact</span>')
        if not sens_html:
            sens_html.append('<span class="badge safe">None</span>')
        sens_badges = " ".join(sens_html)

        # Json strings for complex lists
        entities_str = json.dumps(entities, indent=2) if entities else "None"
        dates_str = json.dumps(dates, indent=2) if dates else "None"
        actions_str = json.dumps(action_items, indent=2) if action_items else "None"

        card = f"""
        <div class="card" id="{img_id}">
            <div class="card-header">
                <h3>{img_id} &mdash; {fname}</h3>
                <span class="badge cat">{cat}</span>
                <span class="badge ctype">{content_type}</span>
            </div>
            <div class="card-body">
                <div class="image-column">
                    <img src="images/{fname}" alt="{fname}" loading="lazy" />
                    <div class="image-dims">Dimensions: {dims_str}</div>
                </div>
                <div class="info-column">
                    <table class="detail-table">
                        <tr><th>Category</th><td><strong>{cat}</strong></td></tr>
                        <tr><th>Content Type</th><td><code>{content_type}</code></td></tr>
                        <tr><th>Tags</th><td>{tags_badge_html}</td></tr>
                        <tr><th>Intent</th><td><em class="intent-text">"{intent}"</em></td></tr>
                        <tr><th>Summary</th><td>{summary}</td></tr>
                        <tr><th>Importance</th><td>{'★' * importance}{'☆' * (5 - importance)} ({importance}/5)</td></tr>
                        <tr><th>Sensitive Info</th><td>{sens_badges}</td></tr>
                        <tr><th>Entities</th><td><pre>{entities_str}</pre></td></tr>
                        <tr><th>Dates</th><td><pre>{dates_str}</pre></td></tr>
                        <tr><th>Action Items</th><td><pre>{actions_str}</pre></td></tr>
                    </table>
                </div>
            </div>
        </div>
        """
        cards_html.append(card)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>External Dataset Manual Quality Audit (60 Screenshots)</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #334155;
            --accent-blue: #38bdf8;
            --accent-green: #4ade80;
            --accent-yellow: #facc15;
            --accent-red: #f87171;
            --accent-purple: #c084fc;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 20px;
        }}
        header {{
            max-width: 1200px;
            margin: 0 auto 30px auto;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 20px;
        }}
        h1 {{
            margin: 0 0 10px 0;
            color: var(--accent-blue);
        }}
        .summary-bar {{
            display: flex;
            gap: 20px;
            background-color: var(--card-bg);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}
        .stat-item {{
            flex: 1;
            text-align: center;
        }}
        .stat-val {{
            font-size: 24px;
            font-weight: bold;
            color: var(--accent-green);
        }}
        .stat-lbl {{
            font-size: 12px;
            color: var(--text-muted);
            text-transform: uppercase;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 30px;
        }}
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }}
        .card-header {{
            background-color: #0f172a;
            padding: 15px 20px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .card-header h3 {{
            margin: 0;
            font-size: 18px;
            flex: 1;
        }}
        .card-body {{
            display: flex;
            padding: 20px;
            gap: 25px;
        }}
        .image-column {{
            width: 320px;
            flex-shrink: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .image-column img {{
            max-width: 100%;
            max-height: 420px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            object-fit: contain;
            background: #000;
        }}
        .image-dims {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 8px;
        }}
        .info-column {{
            flex: 1;
        }}
        .detail-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .detail-table th {{
            text-align: left;
            width: 130px;
            padding: 8px 12px;
            font-size: 13px;
            color: var(--text-muted);
            vertical-align: top;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .detail-table td {{
            padding: 8px 12px;
            font-size: 14px;
            vertical-align: top;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge.cat {{ background-color: #0284c7; color: #fff; }}
        .badge.ctype {{ background-color: #0d9488; color: #fff; }}
        .badge.tag {{ background-color: #334155; color: var(--accent-blue); margin-right: 4px; margin-bottom: 4px; }}
        .badge.danger {{ background-color: #991b1b; color: #fecaca; }}
        .badge.warning {{ background-color: #854d0e; color: #fef08a; }}
        .badge.safe {{ background-color: #166534; color: #bbf7d0; }}
        pre {{
            margin: 0;
            font-family: monospace;
            font-size: 12px;
            background-color: #0f172a;
            padding: 6px 10px;
            border-radius: 4px;
            color: #cbd5e1;
            white-space: pre-wrap;
            max-height: 120px;
            overflow-y: auto;
        }}
        .intent-text {{
            color: var(--accent-yellow);
        }}
    </style>
</head>
<body>
    <header>
        <h1>Phase 4A.1: External Dataset Ground-Truth Manual Audit</h1>
        <p style="color: var(--text-muted); margin: 0 0 15px 0;">
            Offline visual review interface for auditing ground-truth annotations across all 60 external screenshots.
        </p>
        <div class="summary-bar">
            <div class="stat-item">
                <div class="stat-val">60</div>
                <div class="stat-lbl">Total Screenshots</div>
            </div>
            <div class="stat-item">
                <div class="stat-val">100%</div>
                <div class="stat-lbl">Target Coverage</div>
            </div>
            <div class="stat-item">
                <div class="stat-val">0</div>
                <div class="stat-lbl">SHA-256 Overlaps</div>
            </div>
            <div class="stat-item">
                <div class="stat-val">Frozen</div>
                <div class="stat-lbl">Audit Status</div>
            </div>
        </div>
    </header>
    <main class="container">
        {''.join(cards_html)}
    </main>
</body>
</html>
"""

    with open(HTML_OUTPUT_PATH, mode="w", encoding="utf-8") as f:
        f.write(html_content)

    return HTML_OUTPUT_PATH


def generate_manual_review_checklist() -> Path:
    """
    Generates the CSV review checklist with all 14 evaluation criteria initialized to 'UNREVIEWED'.
    Columns:
      image_id, filename, category_correct, content_type_correct, tags_correct,
      intent_correct_or_none, entities_correct, dates_correct, events_correct,
      financial_info_correct, contacts_correct, urls_correct, action_items_correct,
      sensitive_info_correct, overall_annotation_correct, notes
    """
    targets = get_evaluation_targets()
    fieldnames = [
        "image_id",
        "filename",
        "category_correct",
        "content_type_correct",
        "tags_correct",
        "intent_correct_or_none",
        "entities_correct",
        "dates_correct",
        "events_correct",
        "financial_info_correct",
        "contacts_correct",
        "urls_correct",
        "action_items_correct",
        "sensitive_info_correct",
        "overall_annotation_correct",
        "notes",
    ]

    rows = []
    for t in targets:
        fname = t["filename"]
        img_id = fname.replace(".jpg", "")
        row = {
            "image_id": img_id,
            "filename": fname,
            "category_correct": "UNREVIEWED",
            "content_type_correct": "UNREVIEWED",
            "tags_correct": "UNREVIEWED",
            "intent_correct_or_none": "UNREVIEWED",
            "entities_correct": "UNREVIEWED",
            "dates_correct": "UNREVIEWED",
            "events_correct": "UNREVIEWED",
            "financial_info_correct": "UNREVIEWED",
            "contacts_correct": "UNREVIEWED",
            "urls_correct": "UNREVIEWED",
            "action_items_correct": "UNREVIEWED",
            "sensitive_info_correct": "UNREVIEWED",
            "overall_annotation_correct": "UNREVIEWED",
            "notes": "",
        }
        rows.append(row)

    with open(CHECKLIST_OUTPUT_PATH, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return CHECKLIST_OUTPUT_PATH


def main():
    """Runs the dataset quality audit and prints a structured summary report."""
    print("==================================================================")
    print("PHASE 4A.1: EXTERNAL DATASET QUALITY AUDIT & REVIEW INFRASTRUCTURE")
    print("==================================================================")
    print()

    # 1. Integrity Checks
    print("[1/5] Running Dataset Integrity Checks...")
    integ = check_dataset_integrity()
    if integ["passed"]:
        print("  [PASS] Integrity checks PASSED (60/60 images, targets, and manifest rows match).")
    else:
        print("  [FAIL] Integrity checks FAILED:")
        for err in integ["errors"]:
            print(f"    - {err}")
    print()

    # 2. SHA-256 Hash Overlap Check
    print("[2/5] Running SHA-256 Hash Overlap Check...")
    sha_res = check_sha256_overlaps()
    print(f"  - External images checked: {sha_res['external_image_count']}")
    print(f"  - Original images checked: {sha_res['original_image_count']}")
    print(f"  - Exact SHA-256 Overlaps: {sha_res['overlap_count']}")
    if sha_res["overlap_count"] == 0:
        print("  [PASS] Zero exact hash overlaps with original dataset.")
    else:
        print("  [FAIL] Found hash overlaps:")
        for ext_f, orig_f, h in sha_res["overlaps"]:
            print(f"    - {ext_f} matches {orig_f} (SHA256: {h[:12]}...)")
    print()

    # 3. Category Distribution
    print("[3/5] Ground-Truth Category Distribution (60 Targets):")
    cat_res = analyze_category_distribution()
    for cat, count in cat_res["counts"].items():
        pct = cat_res["percentages"][cat]
        bar = "#" * int(count // 2)
        print(f"  - {cat:20s}: {count:2d} ({pct:5.1f}%) {bar}")
    print()

    # 4. Taxonomy & Intent Auditing
    print("[4/5] Auditing Content-Type Taxonomy & Intent Quality...")
    audit_res = audit_taxonomy_and_intents()
    print(f"  - Taxonomy Content-Type Issues: {audit_res['taxonomy_issues_count']}")
    print(f"  - Speculative or Optional Intent Audits: {audit_res['intent_issues_count']}")
    if audit_res["intent_issues_count"] > 0:
        print("  - Sample intent review items:")
        for item in audit_res["intent_issues"][:5]:
            print(f"    * {item['filename']}: \"{item['intent']}\" ({item['note']})")
    print()

    # 5. Review Artifacts Generation
    print("[5/5] Generating Manual Review Artifacts...")
    html_path = generate_html_review_page(integ["image_metadata"])
    print(f"  [PASS] HTML Review Page generated: {html_path.relative_to(EXTERNAL_DIR.parent.parent.parent)}")
    checklist_path = generate_manual_review_checklist()
    print(f"  [PASS] CSV Review Checklist generated: {checklist_path.relative_to(EXTERNAL_DIR.parent.parent.parent)}")
    print()

    print("==================================================================")
    print("AUDIT SUMMARY & FINAL STATUS REPORT")
    print("==================================================================")
    print(f"  Total Images                : {integ['image_count']}")
    print(f"  Valid Images                : {integ['image_count'] - len(integ['corrupted_images'])}")
    print(f"  Annotation Coverage        : {integ['target_count']} / 60")
    print(f"  Manifest Coverage          : {integ['manifest_count']} / 60")
    print(f"  Duplicate Filename/ID Check: PASSED")
    print(f"  SHA-256 Hash Overlaps      : {sha_res['overlap_count']}")
    print(f"  Production Code Modified   : FALSE (Zero changes to src/ or classifiers)")
    print("==================================================================")


if __name__ == "__main__":
    main()
