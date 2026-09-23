"""
VLM Evaluation Targets & Reference Annotations
===============================================
Defines the 8 representative screenshot evaluation targets across functional categories
and loads engineering reference annotations used for benchmarking VLM understanding models.
"""

from typing import Any, Dict, List
from src.vlm import MockVLM

# 8 representative screenshots across 8 distinct functional domains
EVALUATION_TARGET_FILENAMES: List[str] = [
    "ss_001.jpg",   # Connectivity
    "ss_002.jpg",   # Payment
    "ss_003.jpg",   # Education
    "ss_008.jpg",   # Shopping
    "ss_0011.jpg",  # Travel
    "ss_0021.jpg",  # Communication
    "ss_0028.jpg",  # Coding
    "ss_0030.jpg",  # Work & Career
]


def get_evaluation_targets() -> List[Dict[str, Any]]:
    """
    Retrieves the 8 representative screenshot evaluation targets paired with their
    engineering reference annotations.

    Returns:
        List of target dictionaries containing:
        - filename: str
        - screenshot_id: str
        - expected: dict (matching ScreenshotMetadata contract fields)
    """
    mock_mappings = MockVLM.DEFAULT_MOCK_VLM_MAPPINGS
    targets = []

    for filename in EVALUATION_TARGET_FILENAMES:
        sid = filename.split(".")[0]
        expected = mock_mappings.get(filename, MockVLM.FALLBACK_VLM_RESPONSE)
        targets.append({
            "filename": filename,
            "screenshot_id": sid,
            "expected": expected,
        })

    return targets
