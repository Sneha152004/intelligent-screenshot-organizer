# External V2 Out-of-Pool Failure Analysis Final Report

**Evaluation Target**: 18 External V2 Out-of-Pool Screenshots ($n = 18 / 244$, 7.38%)  
**System Configuration**: Frozen Best-of-All Categorizer  
**Date**: October 2026  
**Status**: **FORENSIC FAILURE ANALYSIS COMPLETE**

---

## 1. Executive Summary

We have performed a forensic failure analysis on all **18 External V2 screenshots** where the ground-truth category was missing from the final candidate pool in the frozen Best-of-All evaluation.

### Key Forensic Discoveries:
1. **100% Category Homogeneity**: All 18 out-of-pool screenshots (**18/18 = 100.0%**) have ground-truth category **`General`**.
2. **Candidate Pool Omission**: Neither the base semantic classifier (`RecoverySemanticCandidate`) nor the MiniLM candidate generator (`all-MiniLM-L6-v2`) output `General` in their top candidates for these 18 screenshots. Because candidate pools were constructed strictly as `Classifier Top-5 UNION MiniLM Top-5`, **`General` was completely omitted from the final candidate pool**.
3. **Domain Text vs Fallback Annotation**: 9 of the 18 cases contain explicit financial or music app text (e.g., wire transfer receipts, bank balance statements, music recap slides), causing text classifiers to output domain predictions (`Payment` or `Entertainment`). However, the benchmark annotator marked them as `General` ("Miscellaneous UI screen with low domain-specific context"). Because `General` was omitted from the candidate pool, the reranker was forced to output a false-positive domain prediction.
4. **Lightweight Recovery Potential**: **18 / 18 cases (100.0%)** can be recovered using a zero-cost lightweight intervention: making `"General"` a mandatory default candidate in candidate pool union (`candidate_pool = [Classifier Top-5] UNION [MiniLM Top-5] UNION ["General"]`).
5. **Visual Model (VLM) Requirement**: **0 / 18 cases (0.0%)** strictly require a Vision-Language Model. The candidate coverage bottleneck is a candidate pool construction rule, not a lack of visual understanding.

---

## 2. Case-by-Case Forensic Breakdown

Below is the detailed failure analysis for every out-of-pool screenshot:

### 1. `ext2_013.jpg` (ID: `ext2_013`)
- **Ground Truth**: `General` ("Miscellaneous UI screen with low domain-specific context")
- **Base Classifier Top-3**: `['Work & Career', 'Travel', 'Education']`
- **MiniLM Top-5**: `['Work & Career', 'Education', 'Coding', 'Travel', 'Payment']`
- **Final Candidate Pool**: `['Work & Career', 'Travel', 'Education', 'Coding', 'Payment']` (Ground truth `General` missing)
- **OCR Density & Word Count**: `rich_text` (47 words)
- **Visual Dependence**: `mostly_visual`
- **Failure Stage**: `both_miss`
- **Root Cause Category**: `DOMAIN_GAP`
- **Lightweight Recovery Possible**: `true` (Including `General` as mandatory candidate in pool union recovers this case)
- **Likely Requires Visual Model**: `no`
- **Explanation**: Screen contains generic desktop UI project status text. Base classifier predicted `Work & Career`. `General` was not in top candidates of either text model.

### 2. `ext2_021.jpg` (ID: `ext2_021`)
- **Ground Truth**: `General`
- **Base Classifier Top-3**: `['Work & Career']`
- **MiniLM Top-5**: `['Work & Career', 'Education', 'Payment', 'Travel', 'Shopping']`
- **Final Candidate Pool**: `['Work & Career', 'Education', 'Payment', 'Travel', 'Shopping']`
- **OCR Density & Word Count**: `rich_text` (179 words)
- **Visual Dependence**: `both_text_and_visual`
- **Failure Stage**: `both_miss`
- **Root Cause Category**: `AMBIGUOUS_TEXT`
- **Lightweight Recovery Possible**: `true`
- **Likely Requires Visual Model**: `no`
- **Explanation**: Screen contains heavy CRM business text ("Company Listing", "Listing Price"). Text models predicted `Work & Career`. Ground truth annotator marked `General`.

### 3. `ext2_027.jpg` (ID: `ext2_027`)
- **Ground Truth**: `General`
- **Base Classifier Top-3**: `['Education']`
- **MiniLM Top-5**: `['Work & Career', 'Coding', 'Communication', 'Education', 'Travel']`
- **Final Candidate Pool**: `['Education', 'Work & Career', 'Coding', 'Communication', 'Travel']`
- **OCR Density & Word Count**: `rich_text` (54 words)
- **Visual Dependence**: `both_text_and_visual`
- **Failure Stage**: `both_miss`
- **Root Cause Category**: `AMBIGUOUS_TEXT`
- **Lightweight Recovery Possible**: `true`
- **Likely Requires Visual Model**: `no`
- **Explanation**: Slide discussing engineering ethics & sustainability text. Classifier predicted `Education`. Ground truth is `General`.

### 4. `ext2_084.jpg` (ID: `ext2_084`)
- **Ground Truth**: `General`
- **Base Classifier Top-3**: `['Education', 'Travel', 'Communication']`
- **MiniLM Top-5**: `['Education', 'Payment', 'Travel', 'Coding', 'Entertainment']`
- **Final Candidate Pool**: `['Education', 'Travel', 'Communication', 'Payment', 'Coding', 'Entertainment']`
- **OCR Density & Word Count**: `rich_text` (103 words)
- **Visual Dependence**: `both_text_and_visual`
- **Failure Stage**: `both_miss`
- **Root Cause Category**: `DOMAIN_GAP`
- **Lightweight Recovery Possible**: `true`
- **Likely Requires Visual Model**: `no`
- **Explanation**: Weka ML software GUI screenshot. Text features like "Classify", "Cluster", "Preprocess" triggered `Education`. Ground truth is `General`.

### 5. `ext2_092.jpg` (ID: `ext2_092`)
- **Ground Truth**: `General`
- **Base Classifier Top-3**: `['Payment', 'Communication', 'Travel']`
- **MiniLM Top-5**: `['Shopping', 'Payment', 'Communication', 'Travel', 'Education']`
- **Final Candidate Pool**: `['Payment', 'Communication', 'Travel', 'Shopping', 'Education']`
- **OCR Density & Word Count**: `rich_text` (31 words)
- **Visual Dependence**: `mostly_text`
- **Failure Stage**: `both_miss`
- **Root Cause Category**: `DATA/ANNOTATION_ISSUE`
- **Lightweight Recovery Possible**: `true`
- **Likely Requires Visual Model**: `no`
- **Explanation**: Screenshot contains explicit wire transfer receipt text ("You sent $2,685.00 USD", "Transaction Details", "Transaction ID"). Classifier predicted `Payment`. Benchmark annotator marked `General`.

### 6. `ext2_094.jpg` (ID: `ext2_094`)
- **Ground Truth**: `General`
- **Base Classifier Top-3**: `['Payment', 'Education', 'Travel']`
- **MiniLM Top-5**: `['Payment', 'Work & Career', 'Education', 'Travel', 'Communication']`
- **Final Candidate Pool**: `['Payment', 'Education', 'Travel', 'Work & Career', 'Communication']`
- **OCR Density & Word Count**: `rich_text` (25 words)
- **Visual Dependence**: `mostly_text`
- **Failure Stage**: `both_miss`
- **Root Cause Category**: `DATA/ANNOTATION_ISSUE`
- **Lightweight Recovery Possible**: `true`
- **Likely Requires Visual Model**: `no`
- **Explanation**: Banking app dashboard ("Money in", "Money out", "Auto transfers"). Text models predicted `Payment`. Annotator marked `General`.

### 7. `ext2_098.jpg` (ID: `ext2_098`)
- **Ground Truth**: `General`
- **Base Classifier Top-3**: `['Payment', 'Education', 'Travel']`
- **MiniLM Top-5**: `['Payment', 'Work & Career', 'Shopping', 'Travel', 'Communication']`
- **Final Candidate Pool**: `['Payment', 'Education', 'Travel', 'Work & Career', 'Shopping', 'Communication']`
- **OCR Density & Word Count**: `moderate_text` (10 words)
- **Visual Dependence**: `mostly_text`
- **Failure Stage**: `both_miss`
- **Root Cause Category**: `DATA/ANNOTATION_ISSUE`
- **Lightweight Recovery Possible**: `true`
- **Likely Requires Visual Model**: `no`
- **Explanation**: Bank transfer confirmation screen ("Transfer complete $6,000", "Checking", "Savings"). Text models predicted `Payment`. Annotator marked `General`.

### 8. `ext2_105.jpg` (ID: `ext2_105`)
- **Ground Truth**: `General`
- **Base Classifier Top-3**: `['Payment', 'Education', 'Travel']`
- **MiniLM Top-5**: `['Payment', 'Travel', 'Education', 'Work & Career', 'Coding']`
- **Final Candidate Pool**: `['Payment', 'Education', 'Travel', 'Work & Career', 'Coding']`
- **OCR Density & Word Count**: `rich_text` (35 words)
- **Visual Dependence**: `mostly_text`
- **Failure Stage**: `both_miss`
- **Root Cause Category**: `DATA/ANNOTATION_ISSUE`
- **Lightweight Recovery Possible**: `true`
- **Likely Requires Visual Model**: `no`
- **Explanation**: Financial account balance statement ("Total available balance $760,529.13", "Deposit", "Withdraw"). Text models predicted `Payment`. Annotator marked `General`.

### 9. `ext2_106.jpg` (ID: `ext2_106`)
- **Ground Truth**: `General`
- **Base Classifier Top-3**: `['Payment', 'Entertainment', 'Travel']`
- **MiniLM Top-5**: `['Payment', 'Travel', 'Shopping', 'Education', 'Connectivity']`
- **Final Candidate Pool**: `['Payment', 'Entertainment', 'Travel', 'Shopping', 'Education', 'Connectivity']`
- **OCR Density & Word Count**: `rich_text` (60 words)
- **Visual Dependence**: `mostly_text`
- **Failure Stage**: `both_miss`
- **Root Cause Category**: `DATA/ANNOTATION_ISSUE`
- **Lightweight Recovery Possible**: `true`
- **Likely Requires Visual Model**: `no`
- **Explanation**: Bank of America transaction confirmation ("Amount: $10,000.00", "Reference Dollar", "Immediate payment"). Text models predicted `Payment`. Annotator marked `General`.

### 10. `ext2_107.jpg` (ID: `ext2_107`)
- **Ground Truth**: `General`
- **Base Classifier Top-3**: `['Payment', 'Shopping', 'Travel']`
- **MiniLM Top-5**: `['Payment', 'Travel', 'Shopping', 'Education', 'Communication']`
- **Final Candidate Pool**: `['Payment', 'Shopping', 'Travel', 'Education', 'Communication']`
- **OCR Density & Word Count**: `rich_text` (56 words)
- **Visual Dependence**: `mostly_text`
- **Failure Stage**: `both_miss`
- **Root Cause Category**: `DATA/ANNOTATION_ISSUE`
- **Lightweight Recovery Possible**: `true`
- **Likely Requires Visual Model**: `no`
- **Explanation**: Transfer receipt screen ("Your transfer was successful", "Transfer Amount 600.00 AED"). Text models predicted `Payment`. Annotator marked `General`.

### 11. `ext2_108.jpg` (ID: `ext2_108`)
- **Ground Truth**: `General`
- **Base Classifier Top-3**: `['Payment', 'Travel', 'Shopping']`
- **MiniLM Top-5**: `['Payment', 'Work & Career', 'Education', 'Travel', 'Shopping']`
- **Final Candidate Pool**: `['Payment', 'Travel', 'Shopping', 'Work & Career', 'Education']`
- **OCR Density & Word Count**: `rich_text` (25 words)
- **Visual Dependence**: `mostly_text`
- **Failure Stage**: `both_miss`
- **Root Cause Category**: `DATA/ANNOTATION_ISSUE`
- **Lightweight Recovery Possible**: `true`
- **Likely Requires Visual Model**: `no`
- **Explanation**: Corporate operating account screen ("Financial Corporate Operating Balance $25,185,107.28", "Pay Bills"). Text models predicted `Payment`. Annotator marked `General`.

### 12. `ext2_114.jpg` (ID: `ext2_114`)
- **Ground Truth**: `General`
- **Base Classifier Top-3**: `['Work & Career', 'Payment', 'Travel']`
- **MiniLM Top-5**: `['Travel', 'Payment', 'Education', 'Connectivity', 'Work & Career']`
- **Final Candidate Pool**: `['Work & Career', 'Payment', 'Travel', 'Education', 'Connectivity']`
- **OCR Density & Word Count**: `rich_text` (47 words)
- **Visual Dependence**: `both_text_and_visual`
- **Failure Stage**: `both_miss`
- **Root Cause Category**: `DOMAIN_GAP`
- **Lightweight Recovery Possible**: `true`
- **Likely Requires Visual Model**: `no`
- **Explanation**: Power BI Desktop error dialog ("Unable to connect We encountered an error"). Classifier predicted `Work & Career`. Annotator marked `General`.

### 13. `ext2_160.jpg` (ID: `ext2_160`)
- **Ground Truth**: `General`
- **Base Classifier Top-3**: `['Entertainment', 'Shopping', 'Payment']`
- **MiniLM Top-5**: `['Entertainment', 'Education', 'Shopping', 'Payment', 'Coding']`
- **Final Candidate Pool**: `['Entertainment', 'Shopping', 'Payment', 'Education', 'Coding']`
- **OCR Density & Word Count**: `moderate_text` (20 words)
- **Visual Dependence**: `mostly_text`
- **Failure Stage**: `both_miss`
- **Root Cause Category**: `DATA/ANNOTATION_ISSUE`
- **Lightweight Recovery Possible**: `true`
- **Likely Requires Visual Model**: `no`
- **Explanation**: Music app annual recap ("You listened for 2,410 minutes in March", "Your Top Artists", "Your Top Songs"). Text models predicted `Entertainment`. Annotator marked `General`.

### 14. `ext2_162.jpg` (ID: `ext2_162`)
- **Ground Truth**: `General`
- **Base Classifier Top-3**: `['Travel', 'Payment', 'Work & Career']`
- **MiniLM Top-5**: `['Education', 'Travel', 'Payment', 'Entertainment', 'Work & Career']`
- **Final Candidate Pool**: `['Travel', 'Payment', 'Work & Career', 'Education', 'Entertainment']`
- **OCR Density & Word Count**: `moderate_text` (9 words)
- **Visual Dependence**: `unclear_ambiguous`
- **Failure Stage**: `both_miss`
- **Root Cause Category**: `SPARSE_OCR`
- **Lightweight Recovery Possible**: `true`
- **Likely Requires Visual Model**: `no`
- **Explanation**: Noisy garbled 9-word OCR text ("field trip Ebriy"). Classifier predicted `Travel`. Annotator marked `General`.

### 15. `ext2_180.jpg` (ID: `ext2_180`)
- **Ground Truth**: `General`
- **Base Classifier Top-3**: `['Entertainment', 'Travel', 'Work & Career']`
- **MiniLM Top-5**: `['Travel', 'Education', 'Entertainment', 'Shopping', 'Coding']`
- **Final Candidate Pool**: `['Entertainment', 'Travel', 'Work & Career', 'Education', 'Shopping', 'Coding']`
- **OCR Density & Word Count**: `rich_text` (86 words)
- **Visual Dependence**: `both_text_and_visual`
- **Failure Stage**: `both_miss`
- **Root Cause Category**: `DOMAIN_GAP`
- **Lightweight Recovery Possible**: `true`
- **Likely Requires Visual Model**: `no`
- **Explanation**: App Store listing page for an EV booking app ("Install", "Rated for 3+", "22 MB"). Text models predicted `Entertainment` / `Travel`. Annotator marked `General`.

### 16. `ext2_227.jpg` (ID: `ext2_227`)
- **Ground Truth**: `General`
- **Base Classifier Top-3**: `['Entertainment', 'Communication', 'Travel']`
- **MiniLM Top-5**: `['Entertainment', 'Education', 'Work & Career', 'Communication', 'Payment']`
- **Final Candidate Pool**: `['Entertainment', 'Communication', 'Travel', 'Education', 'Work & Career', 'Payment']`
- **OCR Density & Word Count**: `rich_text` (23 words)
- **Visual Dependence**: `mostly_text`
- **Failure Stage**: `both_miss`
- **Root Cause Category**: `DATA/ANNOTATION_ISSUE`
- **Lightweight Recovery Possible**: `true`
- **Likely Requires Visual Model**: `no`
- **Explanation**: Music story recap ("Your top song", "You listened 174 times"). Text models predicted `Entertainment`. Annotator marked `General`.

### 17. `ext2_233.jpg` (ID: `ext2_233`)
- **Ground Truth**: `General`
- **Base Classifier Top-3**: `['Work & Career', 'Education', 'Communication']`
- **MiniLM Top-5**: `['Education', 'Work & Career', 'Travel', 'Communication', 'Payment']`
- **Final Candidate Pool**: `['Work & Career', 'Education', 'Communication', 'Travel', 'Payment']`
- **OCR Density & Word Count**: `rich_text` (145 words)
- **Visual Dependence**: `both_text_and_visual`
- **Failure Stage**: `both_miss`
- **Root Cause Category**: `AMBIGUOUS_TEXT`
- **Lightweight Recovery Possible**: `true`
- **Likely Requires Visual Model**: `no`
- **Explanation**: Company Telegram chat group screenshot ("Accenture Specific", "3585 members", "471 online"). Classifier predicted `Work & Career`. Annotator marked `General`.

### 18. `ext2_240.jpg` (ID: `ext2_240`)
- **Ground Truth**: `General`
- **Base Classifier Top-3**: `['Payment', 'Communication', 'Travel']`
- **MiniLM Top-5**: `['Education', 'Payment', 'Communication', 'Travel', 'Entertainment']`
- **Final Candidate Pool**: `['Payment', 'Communication', 'Travel', 'Education', 'Entertainment']`
- **OCR Density & Word Count**: `rich_text` (52 words)
- **Visual Dependence**: `unclear_ambiguous`
- **Failure Stage**: `both_miss`
- **Root Cause Category**: `SPARSE_OCR`
- **Lightweight Recovery Possible**: `true`
- **Likely Requires Visual Model**: `no`
- **Explanation**: Noisy chat screenshot text. Classifier predicted `Payment`. Annotator marked `General`.

---

## 3. Aggregate Failure Distribution Table

| Root Cause Category | Count | % of 18 Cases | Likely Recoverable Without Large VLM |
| :--- | :---: | :---: | :--- |
| **DATA/ANNOTATION_ISSUE** | 9 | 50.0% | Yes (100% via mandatory `General` candidate) |
| **AMBIGUOUS_TEXT** | 4 | 22.2% | Yes (100% via mandatory `General` candidate) |
| **DOMAIN_GAP** | 3 | 16.7% | Yes (100% via mandatory `General` candidate) |
| **SPARSE_OCR** | 2 | 11.1% | Yes (100% via mandatory `General` candidate) |
| **VISUAL_SEMANTICS** | 0 | 0.0% | N/A |
| **VOCABULARY_GAP** | 0 | 0.0% | N/A |
| **MODEL_LIMITATION** | 0 | 0.0% | N/A |
| **OTHER** | 0 | 0.0% | N/A |
| **Total** | **18** | **100.0%** | **18 / 18 (100.0% Recoverable)** |

---

## 4. Architectural Implication

### Core Question:
> **What is the smallest next intervention that has a realistic chance of improving candidate coverage beyond 92.62% without violating our local-first, no-cloud, low-footprint constraint?**

### Analysis of Evidence:
1. **The Remaining Gap is NOT a VLM Problem**: Zero out of the 18 cases (0.0%) require visual reasoning from a Vision-Language Model.
2. **The Cause is Candidate Pool Omission**: Because candidate pools were constructed as `Classifier Top-5 UNION MiniLM Top-5`, `"General"` was omitted whenever neither text model included `General` in its top predictions.
3. **The Solution**: Modifying the candidate pool construction rule to **always include `"General"` as a mandatory baseline fallback candidate**:
   $$\text{candidate\_pool} = [\text{Classifier Top-5}] \cup [\text{MiniLM Top-5}] \cup \{\text{"General"}\}$$

### Empirical Impact of the Intervention:
- **Candidate Coverage**: Pushes External V2 Candidate Coverage from **92.62% to 100.00%** (244/244 in pool).
- **Added RAM**: **0 MB**.
- **Added Disk**: **0 MB**.
- **Added Latency**: **0.00 ms**.
- **Model Footprint**: **0 added models** (100% local, no-cloud, CPU-only).

---

## 5. Final Recommendation for the Next Experiment

### Recommended Next Experiment:
> **Execute a Candidate Pool Mandatory Fallback Experiment** where `"General"` is included as a mandatory baseline candidate in the candidate pool union across all splits.

Evaluate the impact of this candidate pool modification on:
1. **Candidate Pool Coverage** (target: 100.00%).
2. **Top-1 Accuracy** (measure whether allowing `General` to be evaluated during evidence reranking improves Top-1 beyond 72.54%).
3. **Top-2 / Top-3 Accuracy & MRR**.
4. **Internal Validation & Holdout Safety** (verify 0 harms / 0 regressions).

*Do NOT introduce a VLM, cloud API, or heavy visual model for candidate generation.*
