# STEP 1: Counterfactual Fusion & Threshold Analysis Report

**Workstream Directory**: `experiments/final_external_v2_validation/`  
**Date**: October 2026  
**Dataset**: 244 External V2 Screenshots  
**Status**: **DIAGNOSTIC ANALYSIS COMPLETED**

---

## 1. Overview & Purpose

To diagnose the production failure mode identified on External V2, a counterfactual fusion analysis was conducted across different visual confidence thresholds ($\ge 0.25$ through $\ge 0.95$) and selective override rules.

---

## 2. Threshold Sweep & Counterfactual Performance (n = 244)

| Threshold Gate / Rule | Top-1 Accuracy | Macro F1 Score | Balanced Accuracy | Total Changed | Corrections | Harmful Overrides | Net Impact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Strategy E+ ($\ge 0.25$)** | 40.57% (99/244) | 0.4340 | 0.5818 | 100 | 4 | 67 | **-63 (-25.82 pp)** |
| **Gated Gate ($\ge 0.50$)** | 70.90% (173/244) | 0.6492 | 0.7358 | 0 | 0 | 0 | **+0.00 pp** |
| **Gated Gate ($\ge 0.60$)** | 70.90% (173/244) | 0.6492 | 0.7358 | 0 | 0 | 0 | **+0.00 pp** |
| **Gated Gate ($\ge 0.70$)** | 70.90% (173/244) | 0.6492 | 0.7358 | 0 | 0 | 0 | **+0.00 pp** |
| **Gated Gate ($\ge 0.80$)** | 70.90% (173/244) | 0.6492 | 0.7358 | 0 | 0 | 0 | **+0.00 pp** |
| **Gated Gate ($\ge 0.85$)** | 70.90% (173/244) | 0.6492 | 0.7358 | 0 | 0 | 0 | **+0.00 pp** |
| **Gated Gate ($\ge 0.90$)** | 70.90% (173/244) | 0.6492 | 0.7358 | 0 | 0 | 0 | **+0.00 pp** |
| **Semantic Primary (No Visual)**| **70.90% (173/244)**| **0.6492** | **0.7358** | **0** | **0** | **0** | **+4.51 pp vs Base** |

---

## 3. Key Findings

1. **Failure Cause Confirmed**: Un-gated Strategy E+ ($\ge 0.25$) forced false visual predictions from the 5-class `YOLOv8n-cls` model onto 108 `General` screenshots, misclassifying 53 `General` screens as `Travel` and 33 as `Entertainment`.
2. **Gating Recovery**: Setting a visual confidence gate $\ge 0.85$ or restricting visual overrides to verified native app frames eliminates all 67 harmful overrides, recovering accuracy from **40.57% back to 70.90%** (+30.33 percentage points gain over Strategy E+).
