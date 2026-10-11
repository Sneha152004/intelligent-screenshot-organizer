"""
Lightweight Screenshot Intelligence Baseline Coordinator
==========================================================
Coordinates local OCR, evidence extraction, context resolution, and metadata formatting
without requiring a generative VLM or LLM. Follows EVIDENCE FIRST, CLASSIFICATION SECOND.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from src.easy_ocr import EasyOCREngine, BaseOCREngine
from src.mock_ocr import MockOCREngine
from src.embedder import BaseEmbedder
from experiments.intelligence.rules import (
    extract_financial_info,
    extract_dates,
    extract_contacts,
    extract_urls,
    detect_sensitive_info,
    extract_entities,
    extract_events,
    extract_action_items,
    extract_text_topics,
    resolve_content_type,
    generate_deterministic_summary,
)
from experiments.intelligence.classifier import HybridScreenshotClassifier


def generate_tags(category: str, ocr_text: str, entities: List[Dict[str, str]], sensitive_info: Dict[str, bool]) -> List[str]:
    """Generates relevant tags from category, OCR content, entities, and sensitive info flags."""
    tags: List[str] = [category.lower().replace(" & ", "_")]

    ocr_lower = ocr_text.lower() if ocr_text else ""

    if "airtel" in ocr_lower or "wifi" in ocr_lower:
        tags.extend(["wifi", "credentials"])
    if "upi" in ocr_lower or "paid" in ocr_lower or sensitive_info.get("contains_payment_info"):
        tags.extend(["payment", "receipt"])
    if "syllabus" in ocr_lower or "kiit" in ocr_lower:
        tags.extend(["syllabus", "academics"])
    if "indigo" in ocr_lower or "boarding" in ocr_lower:
        tags.extend(["boarding_pass", "travel"])
    if "express" in ocr_lower or "code" in ocr_lower:
        tags.extend(["code", "developer"])
    if "hackathon" in ocr_lower or "openai" in ocr_lower:
        tags.extend(["hackathon", "event"])
    if "whatsapp" in ocr_lower:
        tags.extend(["chat", "messaging"])
    if "combo" in ocr_lower or "menu" in ocr_lower:
        tags.extend(["food", "menu"])

    for ent in entities:
        ent_text = ent.get("text", "").lower().replace(" ", "_")
        if ent_text and ent_text not in tags and len(ent_text) > 2:
            tags.append(ent_text)

    # Deduplicate preserving order
    return list(dict.fromkeys(tags))


def analyze_screenshot(
    image_path: str,
    ocr_engine: Optional[BaseOCREngine] = None,
    provided_ocr_text: Optional[str] = None,
    embedder: Optional[BaseEmbedder] = None,
    classifier: Optional[HybridScreenshotClassifier] = None,
) -> Dict[str, Any]:
    """
    Runs the evidence-first lightweight screenshot intelligence pipeline on a screenshot.

    Args:
        image_path: Path to the screenshot file.
        ocr_engine: OCR Engine instance (defaults to EasyOCREngine or MockOCREngine).
        provided_ocr_text: Optional pre-extracted OCR text to bypass OCR stage.
        embedder: Optional embedding model instance.
        classifier: Optional classifier instance.

    Returns:
        Dict containing structured metadata and timing breakdown.
    """
    total_start = time.perf_counter()

    p = Path(image_path)
    filename = p.name
    screenshot_id = p.stem

    # 1. OCR Extraction Sub-stage
    ocr_start = time.perf_counter()
    if provided_ocr_text is not None:
        ocr_text = provided_ocr_text
    else:
        if ocr_engine is None:
            ocr_engine = EasyOCREngine()
        ocr_text = ocr_engine.extract_text(str(p))
    ocr_latency_ms = (time.perf_counter() - ocr_start) * 1000.0

    # 2. Evidence Extraction Sub-stage (EVIDENCE FIRST)
    rules_start = time.perf_counter()
    financial_info = extract_financial_info(ocr_text)
    dates = extract_dates(ocr_text)
    contacts = extract_contacts(ocr_text)
    urls = extract_urls(ocr_text)
    sensitive_info = detect_sensitive_info(ocr_text)
    entities = extract_entities(ocr_text)
    events = extract_events(ocr_text, financial_info=financial_info, dates=dates)
    action_items = extract_action_items(ocr_text)
    text_topics = extract_text_topics(ocr_text)

    evidence = {
        "financial_info": financial_info,
        "contacts": contacts,
        "urls": urls,
        "dates": dates,
        "sensitive_info": sensitive_info,
        "entities": entities,
        "events": events,
        "action_items": action_items,
        "text_topics": text_topics,
    }
    rules_latency_ms = (time.perf_counter() - rules_start) * 1000.0

    # 3. Context Resolution & Hierarchical Multi-Signal Classification Sub-stage
    class_start = time.perf_counter()
    if classifier is None:
        classifier = HybridScreenshotClassifier(embedder=embedder)

    emb_start = time.perf_counter()
    emb_scores = classifier.compute_embedding_scores(ocr_text)
    embedding_latency_ms = (time.perf_counter() - emb_start) * 1000.0

    # Initial Content-Type Resolution
    initial_ct = resolve_content_type("General", ocr_text, financial_info, events)

    # Multi-Signal Hierarchical Classification
    class_res = classifier.classify_multi_signal(
        ocr_text, evidence=evidence, content_type=initial_ct, top_k=3
    )

    category = class_res["selected_category"]
    confidence = class_res["confidence"]
    ambiguous = class_res["ambiguous"]
    margin = class_res["score_margin"]
    candidates = class_res["candidates"]
    category_scores = class_res["category_scores"]

    # Refine Content Type using top category
    content_type = resolve_content_type(category, ocr_text, financial_info, events)

    # Optional intent inference: None if evidence is weak or insufficient
    raw_intent = classifier.infer_intent(category, ocr_text) if ocr_text and len(ocr_text.strip()) > 10 else None
    intent = raw_intent if (confidence >= 0.35 and category != "General" and not ambiguous) else None

    classification_latency_ms = (time.perf_counter() - class_start) * 1000.0

    summary = generate_deterministic_summary(category, ocr_text, entities)
    tags = generate_tags(category, ocr_text, entities, sensitive_info)

    confidence_scores = {
        "category": round(confidence, 4),
        "content_type": round(min(1.0, confidence * 1.1) if content_type != "unknown" else 0.5, 4),
        "evidence": 0.85 if bool(ocr_text and ocr_text.strip()) else 0.0,
        "margin": round(margin, 4),
    }

    total_latency_ms = (time.perf_counter() - total_start) * 1000.0
    now_iso = datetime.now(timezone.utc).isoformat()

    metadata = {
        "screenshot_id": screenshot_id,
        "filename": filename,
        "image_uri": str(p.resolve()) if p.exists() else str(p),
        "category": category,
        "content_type": content_type,
        "ocr_available": bool(ocr_text and ocr_text.strip()),
        "created_at": now_iso,
        "indexed_at": now_iso,
        "tags": tags,
        "intent": intent,  # Optional: None if evidence is weak or ambiguous
        "summary": summary,
        "importance": None,  # Explicitly None when unassigned
        "evidence": evidence,
        "events": events,
        "financial_info": financial_info,
        "contacts": contacts,
        "urls": urls,
        "text_topics": text_topics,
        "entities": entities,
        "dates": dates,
        "action_items": action_items,
        "sensitive_info": sensitive_info,
        "confidence_scores": confidence_scores,
        "category_candidates": candidates,
        "category_scores": category_scores,
        "classification_confidence": round(confidence, 4),
        "classification_ambiguous": ambiguous,
        "classification_margin": round(margin, 4),
        "classification_evidence": class_res.get("signal_scores", {}),
    }

    timing = {
        "ocr_latency_ms": round(ocr_latency_ms, 2),
        "rules_latency_ms": round(rules_latency_ms, 2),
        "embedding_latency_ms": round(embedding_latency_ms, 2),
        "classification_latency_ms": round(classification_latency_ms, 2),
        "total_latency_ms": round(total_latency_ms, 2),
        "classification_confidence": round(confidence, 4),
        "classification_ambiguous": ambiguous,
        "classification_margin": round(margin, 4),
    }

    return {
        "metadata": metadata,
        "timing": timing,
        "ocr_text": ocr_text,
        "classification_result": class_res,
    }


def main():
    parser = argparse.ArgumentParser(description="Evidence-First Screenshot Intelligence CLI")
    parser.add_argument("image_path", nargs="?", help="Path to screenshot image")
    parser.add_argument("--mock-ocr", action="store_true", help="Use MockOCREngine instead of EasyOCR")

    args = parser.parse_args()

    if not args.image_path:
        print("Usage: python experiments/intelligence/baseline_intelligence.py <image_path> [--mock-ocr]")
        return

    ocr_engine = MockOCREngine() if args.mock_ocr else EasyOCREngine()
    result = analyze_screenshot(args.image_path, ocr_engine=ocr_engine)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
