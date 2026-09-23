# Phase 3E: VLM Candidate Evaluation Report

## Overview & Model Metadata
- **Model Name:** MockVLM Engine (Deterministic Demo)
- **Model Version:** mock-3d
- **Model Footprint (MB):** 0.0
- **Evaluation Timestamp:** `2026-09-23T17:05:15.252226+00:00`
- **Runtime Hardware:** `CPU`

## Performance & Latency Summary
- **Total Screenshots Evaluated:** 8
- **Total Time:** 0.002s
- **Average Per-Image Latency:** 0.0s
- **Successful Evaluations:** 8
- **Failed Evaluations:** 0

## Aggregate Metrics Summary

| Metric | Score |
| :--- | :---: |
| **Schema Validity Rate** | `100.00%` |
| **Category Accuracy** | `100.00%` |
| **Tag Precision** | `100.00%` |
| **Tag Recall** | `100.00%` |
| **Intent Exact Match Rate** | `100.00%` |
| **Entities Recall** | `100.00%` |
| **Dates Recall** | `100.00%` |
| **Action Items Recall** | `100.00%` |
| **Sensitive Info Accuracy** | `100.00%` |

## Per-Screenshot Results Benchmark

| Screenshot ID | Category Match | Tag Precision / Recall | Intent Match | Latency (s) | Schema Valid |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `ss_001` | 1.00 | 1.00 / 1.00 | 1.00 | 0.0000s | Yes |
| `ss_002` | 1.00 | 1.00 / 1.00 | 1.00 | 0.0000s | Yes |
| `ss_003` | 1.00 | 1.00 / 1.00 | 1.00 | 0.0000s | Yes |
| `ss_008` | 1.00 | 1.00 / 1.00 | 1.00 | 0.0000s | Yes |
| `ss_0011` | 1.00 | 1.00 / 1.00 | 1.00 | 0.0000s | Yes |
| `ss_0021` | 1.00 | 1.00 / 1.00 | 1.00 | 0.0000s | Yes |
| `ss_0028` | 1.00 | 1.00 / 1.00 | 1.00 | 0.0000s | Yes |
| `ss_0030` | 1.00 | 1.00 / 1.00 | 1.00 | 0.0000s | Yes |

## Observations & Key Takeaways
- **Infrastructure Verification:** Demo evaluation ran in MockVLM mode with zero external model dependencies.
- **Reference Annotations:** Metrics are evaluated against engineering reference annotations established during Phase 3D.
