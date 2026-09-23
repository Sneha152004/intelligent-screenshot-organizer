"""
Model-Agnostic VLM Evaluation Harness
======================================
Provides a standardized framework for evaluating Visual Language Model (VLM) candidates
against representative screenshot targets, calculating category accuracy, tag precision/recall,
intent match, entity/date/action-item recall, sensitive info flag accuracy, schema validity,
and per-image latency without requiring real model downloads during testing.
"""

from abc import ABC, abstractmethod
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mock_ocr import BaseOCREngine
from src.easy_ocr import get_ocr_engine
from src.vlm import BaseVLM, get_vlm_engine
from experiments.vlm.evaluation_targets import get_evaluation_targets

# Centralized Model-Agnostic VLM Prompt
STANDARDIZED_VLM_PROMPT = """Analyze the supplied screenshot image along with its supporting OCR text to understand its visual layout and semantic meaning.

Definitions:
- category: The broad functional category of the screenshot (e.g. Connectivity, Payment, Education, Shopping, Travel, Communication, Coding, Work & Career, Entertainment, General).
- tags: Specific concepts, topics, tools, or keywords visible or relevant to the image content.
- intent: The primary underlying reason why a user likely captured or saved this screenshot.

Perform the following structured analysis and return ONLY a single valid JSON object adhering strictly to the schema below without markdown formatting or conversational text around it:

Schema:
{
    "category": "<broad functional category string>",
    "tags": ["<tag1>", "<tag2>"],
    "intent": "<inferred user intent string or null>",
    "summary": "<concise summary string of visual content or null>",
    "importance": <integer between 1 and 5 indicating priority or null>,
    "entities": [{"text": "<entity name>", "type": "<entity type>"}],
    "dates": [{"text": "<date string>", "type": "<date context>"}],
    "action_items": [{"text": "<task description>", "status": "pending"}],
    "sensitive_info": {
        "contains_password": <true/false>,
        "contains_payment_info": <true/false>,
        "contains_personal_contact": <true/false>
    }
}

Supporting OCR Text:
{ocr_text}
"""


class BaseVLMAdapter(ABC):
    """Abstract adapter interface for VLM model evaluation."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns human-readable model name."""
        pass

    @property
    def model_version(self) -> str:
        """Returns model version identifier string."""
        return "1.0"

    @property
    def model_footprint_mb(self) -> Optional[float]:
        """Returns model size / memory footprint in MB if available."""
        return None

    @abstractmethod
    def analyze(self, image_path: str, ocr_text: str = "") -> Dict[str, Any]:
        """
        Executes visual screenshot analysis using the candidate VLM.

        Args:
            image_path: Path to screenshot image file.
            ocr_text: Supporting OCR text string.

        Returns:
            Dictionary matching the VLM analysis schema.
        """
        pass


class MockVLMAdapter(BaseVLMAdapter):
    """Evaluation adapter wrapping MockVLM engine for testing and demo mode."""

    def __init__(self, vlm_engine: Optional[BaseVLM] = None):
        self.vlm_engine = vlm_engine if vlm_engine is not None else get_vlm_engine("mock")

    @property
    def model_name(self) -> str:
        return "MockVLM Engine (Deterministic Demo)"

    @property
    def model_version(self) -> str:
        return "mock-3d"

    @property
    def model_footprint_mb(self) -> Optional[float]:
        return 0.0

    def analyze(self, image_path: str, ocr_text: str = "") -> Dict[str, Any]:
        return self.vlm_engine.analyze_image(image_path, ocr_text=ocr_text)


def check_schema_validity(output: Any) -> Tuple[bool, List[str]]:
    """
    Validates structural compliance of raw VLM model output against expected schema.

    Returns:
        Tuple of (is_valid: bool, list_of_error_strings: List[str]).
    """
    errors: List[str] = []

    if isinstance(output, str):
        try:
            output = json.loads(output)
        except Exception as e:
            return False, [f"Failed to parse string output as JSON: {e}"]

    if not isinstance(output, dict):
        return False, [f"Output must be a JSON object/dict, got {type(output).__name__}."]

    required_keys = {
        "category",
        "tags",
        "intent",
        "summary",
        "importance",
        "entities",
        "dates",
        "action_items",
        "sensitive_info",
    }
    missing_keys = required_keys - set(output.keys())
    if missing_keys:
        errors.append(f"Missing required schema keys: {sorted(list(missing_keys))}")

    category = output.get("category")
    if not isinstance(category, str):
        errors.append(f"Field 'category' must be a string, got {type(category).__name__}.")

    tags = output.get("tags")
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        errors.append("Field 'tags' must be a list of strings.")

    intent = output.get("intent")
    if intent is not None and not isinstance(intent, str):
        errors.append("Field 'intent' must be None or a string.")

    summary = output.get("summary")
    if summary is not None and not isinstance(summary, str):
        errors.append("Field 'summary' must be None or a string.")

    importance = output.get("importance")
    if importance is not None:
        if isinstance(importance, bool) or not isinstance(importance, int):
            errors.append(f"Field 'importance' must be an integer (1-5) or None, got {type(importance).__name__}.")
        elif not (1 <= importance <= 5):
            errors.append(f"Field 'importance' must be between 1 and 5, got {importance}.")

    for list_field in ("entities", "dates", "action_items"):
        val = output.get(list_field)
        if not isinstance(val, list):
            errors.append(f"Field '{list_field}' must be a list, got {type(val).__name__}.")

    sensitive_info = output.get("sensitive_info")
    if not isinstance(sensitive_info, dict):
        errors.append(f"Field 'sensitive_info' must be a dictionary, got {type(sensitive_info).__name__}.")
    else:
        for flag in ("contains_password", "contains_payment_info", "contains_personal_contact"):
            if flag in sensitive_info and not isinstance(sensitive_info[flag], bool):
                errors.append(f"Field 'sensitive_info.{flag}' must be a boolean.")

    return (len(errors) == 0), errors


def compute_item_metrics(expected: Dict[str, Any], predicted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates detailed field-level comparison metrics between expected and predicted VLM outputs.
    """
    # Category Accuracy
    exp_cat = str(expected.get("category", "")).strip().lower()
    pred_cat = str(predicted.get("category", "")).strip().lower()
    category_match = 1.0 if (exp_cat and exp_cat == pred_cat) else 0.0

    # Tag Precision, Recall, and Overlap
    exp_tags = set(str(t).strip().lower() for t in expected.get("tags", []))
    pred_tags = set(str(t).strip().lower() for t in predicted.get("tags", []) if isinstance(t, str))

    overlap_count = len(exp_tags.intersection(pred_tags))
    tag_precision = (overlap_count / len(pred_tags)) if pred_tags else (1.0 if not exp_tags else 0.0)
    tag_recall = (overlap_count / len(exp_tags)) if exp_tags else (1.0 if not pred_tags else 0.0)

    # Intent Exact Match
    exp_intent = (expected.get("intent") or "").strip().lower()
    pred_intent = (predicted.get("intent") or "").strip().lower()
    intent_match = 1.0 if exp_intent == pred_intent else 0.0

    # Entities Recall
    exp_ents = set(str(e.get("text", "")).strip().lower() for e in expected.get("entities", []) if isinstance(e, dict))
    pred_ents = set(str(e.get("text", "")).strip().lower() for e in predicted.get("entities", []) if isinstance(e, dict))
    entities_recall = (len(exp_ents.intersection(pred_ents)) / len(exp_ents)) if exp_ents else 1.0

    # Dates Recall
    exp_dates = set(str(d.get("text", "")).strip().lower() for d in expected.get("dates", []) if isinstance(d, dict))
    pred_dates = set(str(d.get("text", "")).strip().lower() for d in predicted.get("dates", []) if isinstance(d, dict))
    dates_recall = (len(exp_dates.intersection(pred_dates)) / len(exp_dates)) if exp_dates else 1.0

    # Action Items Recall
    exp_actions = set(str(a.get("text", "")).strip().lower() for a in expected.get("action_items", []) if isinstance(a, dict))
    pred_actions = set(str(a.get("text", "")).strip().lower() for a in predicted.get("action_items", []) if isinstance(a, dict))
    action_items_recall = (len(exp_actions.intersection(pred_actions)) / len(exp_actions)) if exp_actions else 1.0

    # Sensitive Info Flag Accuracy
    exp_sens = expected.get("sensitive_info", {})
    pred_sens = predicted.get("sensitive_info", {})
    flags = ["contains_password", "contains_payment_info", "contains_personal_contact"]
    correct_flags = sum(
        1 for f in flags if bool(exp_sens.get(f)) == bool(pred_sens.get(f))
    )
    sensitive_info_accuracy = correct_flags / len(flags)

    return {
        "category_match": category_match,
        "tag_precision": round(tag_precision, 4),
        "tag_recall": round(tag_recall, 4),
        "tag_overlap_count": overlap_count,
        "intent_match": intent_match,
        "entities_recall": round(entities_recall, 4),
        "dates_recall": round(dates_recall, 4),
        "action_items_recall": round(action_items_recall, 4),
        "sensitive_info_accuracy": round(sensitive_info_accuracy, 4),
    }


def compute_aggregate_metrics(per_target_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes dataset-level aggregate metrics across all processed target results."""
    total = len(per_target_results)
    if total == 0:
        return {}

    valid_count = sum(1 for r in per_target_results if r["schema_validity"]["is_valid"])

    def avg(key: str) -> float:
        vals = [r["metrics"][key] for r in per_target_results if "metrics" in r and key in r["metrics"]]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "schema_validity_rate": round(valid_count / total, 4),
        "category_accuracy": avg("category_match"),
        "tag_precision": avg("tag_precision"),
        "tag_recall": avg("tag_recall"),
        "intent_match_rate": avg("intent_match"),
        "entities_recall": avg("entities_recall"),
        "dates_recall": avg("dates_recall"),
        "action_items_recall": avg("action_items_recall"),
        "sensitive_info_accuracy": avg("sensitive_info_accuracy"),
    }


class VLMEvaluator:
    """Orchestrates model evaluation against benchmark targets and generates reports."""

    def __init__(
        self,
        adapter: BaseVLMAdapter,
        ocr_engine: Optional[BaseOCREngine] = None,
        images_dir: str = "sample",
    ):
        self.adapter = adapter
        self.ocr_engine = ocr_engine if ocr_engine is not None else get_ocr_engine("mock")
        self.images_dir = Path(images_dir)

    def evaluate(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Runs evaluation loop over benchmark targets.
        """
        start_eval_time = time.time()
        targets = get_evaluation_targets()
        if limit is not None and limit > 0:
            targets = targets[:limit]

        per_target_results: List[Dict[str, Any]] = []
        successful_evals = 0
        failed_evals = 0
        failures: List[Dict[str, Any]] = []

        for target in targets:
            filename = target["filename"]
            sid = target["screenshot_id"]
            expected = target["expected"]
            img_path = self.images_dir / filename

            if not img_path.exists():
                failed_evals += 1
                failure_msg = f"Image file not found: {img_path}"
                failures.append({"screenshot_id": sid, "error": failure_msg})
                per_target_results.append({
                    "screenshot_id": sid,
                    "filename": filename,
                    "expected": expected,
                    "predicted": {},
                    "metrics": {},
                    "latency_seconds": 0.0,
                    "schema_validity": {"is_valid": False, "errors": [failure_msg]},
                    "error": failure_msg,
                })
                continue

            try:
                # Extract supporting OCR text
                ocr_text = self.ocr_engine.extract_text(str(img_path))
            except Exception as e:
                ocr_text = ""

            t_start = time.time()
            try:
                predicted = self.adapter.analyze(str(img_path), ocr_text=ocr_text)
                t_latency = time.time() - t_start

                is_valid, schema_errors = check_schema_validity(predicted)
                metrics = compute_item_metrics(expected, predicted)

                successful_evals += 1
                per_target_results.append({
                    "screenshot_id": sid,
                    "filename": filename,
                    "expected": expected,
                    "predicted": predicted,
                    "metrics": metrics,
                    "latency_seconds": round(t_latency, 4),
                    "schema_validity": {"is_valid": is_valid, "errors": schema_errors},
                    "error": None,
                })

            except Exception as e:
                t_latency = time.time() - t_start
                failed_evals += 1
                err_msg = str(e)
                failures.append({"screenshot_id": sid, "error": err_msg})
                per_target_results.append({
                    "screenshot_id": sid,
                    "filename": filename,
                    "expected": expected,
                    "predicted": {},
                    "metrics": {},
                    "latency_seconds": round(t_latency, 4),
                    "schema_validity": {"is_valid": False, "errors": [err_msg]},
                    "error": err_msg,
                })

        total_eval_time = round(time.time() - start_eval_time, 4)
        agg_metrics = compute_aggregate_metrics(per_target_results)

        avg_latency = (
            round(sum(r["latency_seconds"] for r in per_target_results) / len(per_target_results), 4)
            if per_target_results else 0.0
        )

        return {
            "metadata": {
                "model_name": self.adapter.model_name,
                "model_version": self.adapter.model_version,
                "model_footprint_mb": self.adapter.model_footprint_mb,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "device": "CPU",
                "total_screenshots": len(targets),
            },
            "performance": {
                "total_evaluation_time_seconds": total_eval_time,
                "average_latency_seconds": avg_latency,
                "successful_evaluations": successful_evals,
                "failed_evaluations": failed_evals,
            },
            "aggregate_metrics": agg_metrics,
            "failures": failures,
            "per_target_results": per_target_results,
        }

    def save_results(
        self,
        eval_result: Dict[str, Any],
        output_dir: str = "experiments/vlm/results",
    ) -> Tuple[str, str]:
        """
        Saves machine-readable JSON and human-readable Markdown evaluation report files.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        json_path = out_path / "vlm_evaluation_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, indent=2, ensure_ascii=False)

        md_path = out_path / "vlm_evaluation_report.md"
        meta = eval_result["metadata"]
        perf = eval_result["performance"]
        agg = eval_result["aggregate_metrics"]

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Phase 3E: VLM Candidate Evaluation Report\n\n")
            f.write("## Overview & Model Metadata\n")
            f.write(f"- **Model Name:** {meta['model_name']}\n")
            f.write(f"- **Model Version:** {meta['model_version']}\n")
            f.write(f"- **Model Footprint (MB):** {meta['model_footprint_mb'] if meta['model_footprint_mb'] is not None else 'N/A'}\n")
            f.write(f"- **Evaluation Timestamp:** `{meta['timestamp']}`\n")
            f.write(f"- **Runtime Hardware:** `{meta['device']}`\n\n")

            f.write("## Performance & Latency Summary\n")
            f.write(f"- **Total Screenshots Evaluated:** {meta['total_screenshots']}\n")
            f.write(f"- **Total Time:** {perf['total_evaluation_time_seconds']}s\n")
            f.write(f"- **Average Per-Image Latency:** {perf['average_latency_seconds']}s\n")
            f.write(f"- **Successful Evaluations:** {perf['successful_evaluations']}\n")
            f.write(f"- **Failed Evaluations:** {perf['failed_evaluations']}\n\n")

            f.write("## Aggregate Metrics Summary\n\n")
            f.write("| Metric | Score |\n")
            f.write("| :--- | :---: |\n")
            f.write(f"| **Schema Validity Rate** | `{agg.get('schema_validity_rate', 0.0):.2%}` |\n")
            f.write(f"| **Category Accuracy** | `{agg.get('category_accuracy', 0.0):.2%}` |\n")
            f.write(f"| **Tag Precision** | `{agg.get('tag_precision', 0.0):.2%}` |\n")
            f.write(f"| **Tag Recall** | `{agg.get('tag_recall', 0.0):.2%}` |\n")
            f.write(f"| **Intent Exact Match Rate** | `{agg.get('intent_match_rate', 0.0):.2%}` |\n")
            f.write(f"| **Entities Recall** | `{agg.get('entities_recall', 0.0):.2%}` |\n")
            f.write(f"| **Dates Recall** | `{agg.get('dates_recall', 0.0):.2%}` |\n")
            f.write(f"| **Action Items Recall** | `{agg.get('action_items_recall', 0.0):.2%}` |\n")
            f.write(f"| **Sensitive Info Accuracy** | `{agg.get('sensitive_info_accuracy', 0.0):.2%}` |\n\n")

            f.write("## Per-Screenshot Results Benchmark\n\n")
            f.write("| Screenshot ID | Category Match | Tag Precision / Recall | Intent Match | Latency (s) | Schema Valid |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")

            for res in eval_result["per_target_results"]:
                sid = res["screenshot_id"]
                m = res.get("metrics", {})
                lat = res.get("latency_seconds", 0.0)
                valid = "Yes" if res.get("schema_validity", {}).get("is_valid") else "No"
                cat_m = "1.00" if m.get("category_match", 0) == 1.0 else "0.00"
                tag_str = f"{m.get('tag_precision', 0.0):.2f} / {m.get('tag_recall', 0.0):.2f}"
                int_m = "1.00" if m.get("intent_match", 0) == 1.0 else "0.00"
                f.write(f"| `{sid}` | {cat_m} | {tag_str} | {int_m} | {lat:.4f}s | {valid} |\n")

            f.write("\n## Observations & Key Takeaways\n")
            f.write("- **Infrastructure Verification:** Demo evaluation ran in MockVLM mode with zero external model dependencies.\n")
            f.write("- **Reference Annotations:** Metrics are evaluated against engineering reference annotations established during Phase 3D.\n")

        return str(json_path), str(md_path)


def main():
    parser = argparse.ArgumentParser(description="Model-Agnostic VLM Evaluation Harness")
    parser.add_argument(
        "--engine",
        type=str,
        default="mock",
        help="VLM engine type / adapter choice (default: 'mock').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/vlm/results",
        help="Output directory path for results (default: 'experiments/vlm/results').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to top N screenshots (optional).",
    )
    args = parser.parse_args()

    engine_clean = args.engine.strip().lower()
    if engine_clean == "mock":
        adapter = MockVLMAdapter()
    else:
        raise ValueError(f"Unsupported evaluation engine: '{args.engine}'. Supported demo engine: 'mock'.")

    evaluator = VLMEvaluator(adapter=adapter)
    eval_results = evaluator.evaluate(limit=args.limit)
    json_path, md_path = evaluator.save_results(eval_results, output_dir=args.output)

    print("=" * 80)
    print("VLM EVALUATION COMPLETED")
    print("=" * 80)
    print(f"Model Name:        {eval_results['metadata']['model_name']}")
    print(f"Screenshots Run:   {eval_results['metadata']['total_screenshots']}")
    print(f"Total Time:        {eval_results['performance']['total_evaluation_time_seconds']}s")
    print(f"Avg Latency:       {eval_results['performance']['average_latency_seconds']}s")
    print(f"Schema Validity:   {eval_results['aggregate_metrics'].get('schema_validity_rate', 0.0):.2%}")
    print(f"Category Accuracy: {eval_results['aggregate_metrics'].get('category_accuracy', 0.0):.2%}")
    print(f"Results Saved To:  {json_path}")
    print(f"Report Saved To:   {md_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
