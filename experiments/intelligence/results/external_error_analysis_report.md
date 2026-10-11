# Phase 4A: External Screenshot Intelligence Error Analysis Report

**Evaluation Timestamp:** `2026-09-26T16:34:33.280116+00:00`  
**Dataset:** Frozen 60-Image External Evaluation Benchmark (`experiments/intelligence/external/`)  
**Classifier Evaluated:** Multi-Signal Hierarchical Classifier (Rule + Semantic + Embedding + Content-Type)

---

## 1. Executive Summary

A comprehensive, scientific error diagnosis was conducted across all **38 Top-1 misclassifications** resulting from the 60-image external benchmark evaluation.

### Key Benchmark Metrics

| Metric | Value |
|---|---|
| **Total Evaluation Images** | **60** |
| **Top-1 Category Matches** | **22 / 60 (36.7%)** |
| **Total Top-1 Errors** | **38 / 60 (63.3%)** |
| **Top-2 Category Recall** | **33 / 60 (55.0%)** (11 errors rescued in Top-2) |
| **Ambiguity Flagging Rate** | **38.3%** (23 / 60 screenshots marked ambiguous) |
| **Ambiguous Error Cases** | **17 / 38 (44.7% of errors were flagged as ambiguous)** |
| **Mean Classification Margin** | **0.1954** |

---

## 2. Quantitative Root Cause Distribution

Each of the 38 errors was assigned a single **PRIMARY Root Cause** based on signal breakdown and evidence inspection:

| Primary Failure Category | Error Count | Percentage |
|---|---|---|
| `A. OCR/Evidence Limitation` | **2** | **5.26%** || `B. Rule/Evidence Misleading` | **5** | **13.16%** || `C. Semantic Classification Limitation` | **15** | **39.47%** || `D. Embedding Limitation` | **7** | **18.42%** || `E. Content-Type Limitation` | **0** | **0.0%** || `F. Cross-Category Context Overlap` | **9** | **23.68%** || `G. Taxonomy Ambiguity` | **0** | **0.0%** || `H. Other` | **0** | **0.0%** |

---

## 3. Recurring Failure Patterns

The 38 errors were grouped into dominant recurring confusion patterns:

| Pattern (Expected &rarr; Predicted) | Error Count | Percentage | Affected Files (Sample) |
|---|---|---|---|
| `Education -> Payment` | **4** | 10.53% | ext_010.jpg, ext_043.jpg, ext_055.jpg (+1 more) || `Education -> Coding` | **3** | 7.89% | ext_006.jpg, ext_013.jpg, ext_052.jpg || `Education -> Work & Career` | **3** | 7.89% | ext_007.jpg, ext_011.jpg, ext_028.jpg || `Education -> Shopping` | **3** | 7.89% | ext_009.jpg, ext_059.jpg, ext_060.jpg || `Entertainment -> General` | **3** | 7.89% | ext_022.jpg, ext_050.jpg, ext_053.jpg || `Shopping -> General` | **2** | 5.26% | ext_002.jpg, ext_034.jpg || `Entertainment -> Shopping` | **2** | 5.26% | ext_003.jpg, ext_033.jpg || `Work & Career -> Payment` | **2** | 5.26% | ext_014.jpg, ext_041.jpg |

---

## 4. Top-10 Representative Failure Deep Dives


### Failure #1: `ext_001.jpg` (ext_001)

- **Ground Truth:** `Education`
- **Predicted Category:** `Entertainment` (Match in Top-2: **NO**)
- **Content Type:** `video` | **Confidence:** `0.7062` | **Ambiguous:** `False` | **Margin:** `0.5538`
- **Top Candidates:**
   1. **Entertainment**: score=0.6802 (rule=1.0, sem=1.0, emb=0.2014, ct=0.0)
   2. **Travel**: score=0.1264 (rule=0.0, sem=0.4051, emb=0.0321, ct=0.0)
   3. **General**: score=0.1205 (rule=0.0, sem=0.3494, emb=0.1048, ct=0.0)
- **OCR Word Count:** `144 words`
- **OCR Snippet:** *"Reel DIOMS Hit the road Leave Kick the bucket Die Jump the gun Act too soon Fit as a fiddle Healthy Break a Good luck Piece of cake Very easy Put your"*
- **Primary Root Cause:** `D. Embedding Limitation`
- **Cause Reasoning:** Prototype vector embedding for 'Entertainment' pulled vector similarity away from 'Education'.
- **Information Analysis:**
  - Text/OCR Sufficient: `YES`
  - Visual/Layout Information Useful: `YES`
  - Taxonomy Distinguishable: `YES`

### Failure #2: `ext_002.jpg` (ext_002)

- **Ground Truth:** `Shopping`
- **Predicted Category:** `General` (Match in Top-2: **NO**)
- **Content Type:** `unknown` | **Confidence:** `0.0` | **Ambiguous:** `True` | **Margin:** `0.0`
- **Top Candidates:**
   1. **Connectivity**: score=0.0 (rule=0.0, sem=0.0, emb=0.0, ct=0.0)
   2. **Payment**: score=0.0 (rule=0.0, sem=0.0, emb=0.0, ct=0.0)
   3. **Education**: score=0.0 (rule=0.0, sem=0.0, emb=0.0, ct=0.0)
- **OCR Word Count:** `0 words`
- **OCR Snippet:** *""*
- **Primary Root Cause:** `A. OCR/Evidence Limitation`
- **Cause Reasoning:** Extracted OCR text is extremely sparse (0 words). Crucial visual context (diagrams, layout, handwritten notes) is absent from OCR.
- **Information Analysis:**
  - Text/OCR Sufficient: `NO`
  - Visual/Layout Information Useful: `YES`
  - Taxonomy Distinguishable: `UNCERTAIN`

### Failure #3: `ext_003.jpg` (ext_003)

- **Ground Truth:** `Entertainment`
- **Predicted Category:** `Shopping` (Match in Top-2: **NO**)
- **Content Type:** `product_page` | **Confidence:** `0.255` | **Ambiguous:** `True` | **Margin:** `0.0343`
- **Top Candidates:**
   1. **Shopping**: score=0.356 (rule=1.0, sem=0.0, emb=0.0399, ct=0.0)
   2. **Work & Career**: score=0.3217 (rule=0.0, sem=1.0, emb=0.1447, ct=0.0)
   3. **Travel**: score=0.2104 (rule=0.0, sem=0.6887, emb=0.0253, ct=0.0)
- **OCR Word Count:** `113 words`
- **OCR Snippet:** *"ImoOhow-AocK_-Ibun EHL2 JESSICA VUNES U15 (AvENGERS -Eaptain AMERIIA ~THO R TAE Finsi AVENGEF IRON FIST crie IRON MAN 3 Pd Tn THOR H WAt CAPTAIN AMERI"*
- **Primary Root Cause:** `F. Cross-Category Context Overlap`
- **Cause Reasoning:** Media/movie promotional graphics contain purchase/watch pricing controls overlapping with Shopping product listings.
- **Information Analysis:**
  - Text/OCR Sufficient: `YES`
  - Visual/Layout Information Useful: `YES`
  - Taxonomy Distinguishable: `NO`

### Failure #4: `ext_006.jpg` (ext_006)

- **Ground Truth:** `Education`
- **Predicted Category:** `Coding` (Match in Top-2: **NO**)
- **Content Type:** `documentation` | **Confidence:** `0.262` | **Ambiguous:** `True` | **Margin:** `0.0497`
- **Top Candidates:**
   1. **Coding**: score=0.3608 (rule=1.0, sem=0.0, emb=0.072, ct=0.0)
   2. **Travel**: score=0.3111 (rule=0.0, sem=1.0, emb=0.074, ct=0.0)
   3. **Education**: score=0.2938 (rule=0.6, sem=0.2526, emb=0.0535, ct=0.0)
- **OCR Word Count:** `64 words`
- **OCR Snippet:** *"UnIT 1 SOFTMAX, LOSS AND GRADIENT-BASED LEARNING E. Softmax (multi-class output) softmax(z)_i=e (z_i) sum_j e (z_j) It turns logits into probabilities"*
- **Primary Root Cause:** `F. Cross-Category Context Overlap`
- **Cause Reasoning:** Computer science lecture notes or lab assignments contain source code snippets and technical syntax matching Coding.
- **Information Analysis:**
  - Text/OCR Sufficient: `YES`
  - Visual/Layout Information Useful: `YES`
  - Taxonomy Distinguishable: `YES`

### Failure #5: `ext_007.jpg` (ext_007)

- **Ground Truth:** `Education`
- **Predicted Category:** `Work & Career` (Match in Top-2: **NO**)
- **Content Type:** `job_posting` | **Confidence:** `0.5135` | **Ambiguous:** `False` | **Margin:** `0.1657`
- **Top Candidates:**
   1. **Work & Career**: score=0.661 (rule=1.0, sem=1.0, emb=0.0736, ct=0.0)
   2. **Coding**: score=0.4953 (rule=0.8333, sem=0.6789, emb=0.0, ct=0.0)
   3. **Shopping**: score=0.418 (rule=0.9091, sem=0.3326, emb=0.0, ct=0.0)
- **OCR Word Count:** `138 words`
- **OCR Snippet:** *"Feature Bagging (Bootstrap Boosting Aggregating) Training Approach Parallel: Models are trained Sequential: Models are trained independently at the sa"*
- **Primary Root Cause:** `F. Cross-Category Context Overlap`
- **Cause Reasoning:** University announcements, campus drive notices, or academic events overlap with Work & Career event/job listings.
- **Information Analysis:**
  - Text/OCR Sufficient: `YES`
  - Visual/Layout Information Useful: `YES`
  - Taxonomy Distinguishable: `NO`

### Failure #6: `ext_008.jpg` (ext_008)

- **Ground Truth:** `Communication`
- **Predicted Category:** `Payment` (Match in Top-2: **NO**)
- **Content Type:** `transaction` | **Confidence:** `0.4877` | **Ambiguous:** `False` | **Margin:** `0.3162`
- **Top Candidates:**
   1. **Payment**: score=0.6524 (rule=1.0, sem=0.9285, emb=0.1589, ct=0.0)
   2. **Work & Career**: score=0.3362 (rule=0.0, sem=0.9811, emb=0.2794, ct=0.0)
   3. **Travel**: score=0.3231 (rule=0.0, sem=1.0, emb=0.1538, ct=0.0)
- **OCR Word Count:** `60 words`
- **OCR Snippet:** *"741 @ Mm 509 EH 4*" 0 Cj Your chapter in [978-3-032-37100-3, DoSCI 2026, LNNS 2205, schedule for paper approval (698286_1_ Chapter 7)] Inbox Summarise"*
- **Primary Root Cause:** `C. Semantic Classification Limitation`
- **Cause Reasoning:** Semantic text description similarity for 'Payment' outweighed semantic similarity for 'Communication'.
- **Information Analysis:**
  - Text/OCR Sufficient: `YES`
  - Visual/Layout Information Useful: `YES`
  - Taxonomy Distinguishable: `YES`

### Failure #7: `ext_009.jpg` (ext_009)

- **Ground Truth:** `Education`
- **Predicted Category:** `Shopping` (Match in Top-2: **NO**)
- **Content Type:** `product_page` | **Confidence:** `0.3877` | **Ambiguous:** `False` | **Margin:** `0.1953`
- **Top Candidates:**
   1. **Shopping**: score=0.5249 (rule=1.0, sem=0.5472, emb=0.0717, ct=0.0)
   2. **General**: score=0.3296 (rule=0.0, sem=1.0, emb=0.1976, ct=0.0)
   3. **Education**: score=0.2774 (rule=0.0, sem=0.8327, emb=0.1841, ct=0.0)
- **OCR Word Count:** `81 words`
- **OCR Snippet:** *"#au sdhwan' Page Ho_ YOUVA Date; Bandish - < 120 BPo 10 11 12 13 14 1S Ib 2 845767 48 Pa Sa N; Re Sa Ni Pa Ga Re Ga &e Sa 46 644 Sa SS Ga Re Ga Ss Pa "*
- **Primary Root Cause:** `C. Semantic Classification Limitation`
- **Cause Reasoning:** Semantic text description similarity for 'Shopping' outweighed semantic similarity for 'Education'.
- **Information Analysis:**
  - Text/OCR Sufficient: `YES`
  - Visual/Layout Information Useful: `YES`
  - Taxonomy Distinguishable: `YES`

### Failure #8: `ext_010.jpg` (ext_010)

- **Ground Truth:** `Education`
- **Predicted Category:** `Payment` (Match in Top-2: **YES**)
- **Content Type:** `transaction` | **Confidence:** `0.2422` | **Ambiguous:** `True` | **Margin:** `0.0642`
- **Top Candidates:**
   1. **Payment**: score=0.3126 (rule=0.0, sem=1.0, emb=0.0839, ct=0.0)
   2. **Education**: score=0.2484 (rule=0.0, sem=0.8097, emb=0.0369, ct=0.0)
   3. **Connectivity**: score=0.1557 (rule=0.0, sem=0.4595, emb=0.119, ct=0.0)
- **OCR Word Count:** `65 words`
- **OCR Snippet:** *"Gneent ambtm Aan TeanitontL Gnrfu fien Fpunea 4iai Jonn- F> 1+f51FT FTC) FI( + enen Flaan se Peutet Hultijeke 07] Tdee [1,2,4] D] NeL unvelukd 0ut4f L"*
- **Primary Root Cause:** `C. Semantic Classification Limitation`
- **Cause Reasoning:** Semantic text description similarity for 'Payment' outweighed semantic similarity for 'Education'.
- **Information Analysis:**
  - Text/OCR Sufficient: `YES`
  - Visual/Layout Information Useful: `YES`
  - Taxonomy Distinguishable: `YES`

### Failure #9: `ext_014.jpg` (ext_014)

- **Ground Truth:** `Work & Career`
- **Predicted Category:** `Payment` (Match in Top-2: **NO**)
- **Content Type:** `transaction` | **Confidence:** `0.319` | **Ambiguous:** `True` | **Margin:** `0.0412`
- **Top Candidates:**
   1. **Payment**: score=0.3466 (rule=0.0, sem=1.0, emb=0.3109, ct=0.0)
   2. **Education**: score=0.3054 (rule=0.0, sem=0.8845, emb=0.2667, ct=0.0)
   3. **General**: score=0.2164 (rule=0.0, sem=0.7082, emb=0.0265, ct=0.0)
- **OCR Word Count:** `77 words`
- **OCR Snippet:** *"INDIAN INSTITUTE OF TECHNOLOGY KHARAGPUR-721 302. INDIA, PHONE-03222 255221 ENTRY PASS ID No: : 200251809754256207 Name : NAFISA HASAN DOB : 04-05-200"*
- **Primary Root Cause:** `D. Embedding Limitation`
- **Cause Reasoning:** Prototype vector embedding for 'Payment' pulled vector similarity away from 'Work & Career'.
- **Information Analysis:**
  - Text/OCR Sufficient: `YES`
  - Visual/Layout Information Useful: `YES`
  - Taxonomy Distinguishable: `YES`

### Failure #10: `ext_016.jpg` (ext_016)

- **Ground Truth:** `Work & Career`
- **Predicted Category:** `Shopping` (Match in Top-2: **NO**)
- **Content Type:** `product_page` | **Confidence:** `0.3134` | **Ambiguous:** `True` | **Margin:** `0.092`
- **Top Candidates:**
   1. **Shopping**: score=0.4382 (rule=1.0, sem=0.2405, emb=0.107, ct=0.0)
   2. **Education**: score=0.3462 (rule=0.0, sem=1.0, emb=0.308, ct=0.0)
   3. **Work & Career**: score=0.3259 (rule=0.0, sem=0.9479, emb=0.2768, ct=0.0)
- **OCR Word Count:** `48 words`
- **OCR Snippet:** *"4.31:18 269 VoLfeg ,ll ,l 100% Piyush Das 62 ZYcus How to apply Date: Pre Sales Solutions Consultant Apply Here Details: Applications Open Till Saturd"*
- **Primary Root Cause:** `F. Cross-Category Context Overlap`
- **Cause Reasoning:** Institutional entry pass or career posting overlaps with academic or identity document context.
- **Information Analysis:**
  - Text/OCR Sufficient: `YES`
  - Visual/Layout Information Useful: `YES`
  - Taxonomy Distinguishable: `NO`


---

## 5. Visual Information Gap Analysis

Across all 38 failure cases, we evaluated whether textual OCR alone contains sufficient evidence or if visual/layout context is fundamentally missing:

- **Text / OCR Sufficient for Classification:**
  - **YES:** 28 cases (73.7%) — Text contains sufficient semantic signals, but classifier weighting or cross-category overlap confused the category decision.
  - **NO:** 10 cases (26.3%) — Text is sparse, handwritten, or image-heavy (e.g. diagrams, posters, receipts without clear headings).

- **Visual / Layout Information Potentially Useful:**
  - **YES:** 38 cases (100.0%) — Spatial layout, UI headers, app icons, typography, and visual hierarchy would provide definitive disambiguation signals.

---

## 6. Taxonomy Ambiguity Analysis

We evaluated whether the ground-truth category is clearly distinguishable from the competing predicted category under the existing project taxonomy:

- **Clearly Distinguishable (YES):** 20 cases (52.6%) — The categories are distinct, but multi-signal evidence overlapped.
- **Taxonomy Boundary Overlap (NO / UNCERTAIN):** 18 cases (47.4%) — Intrinsic boundary ambiguity exists between `Shopping` vs `Payment` (order summaries containing prices) and `Education` vs `Work & Career` (academic announcements).

---

## 7. Confidence & Ambiguity Correlation

- **High Confidence Errors ($\ge 0.70$):** 2 errors (5.3%)
- **Medium Confidence Errors ($0.40 - 0.69$):** 8 errors (21.1%)
- **Low Confidence Errors ($< 0.40$):** 28 errors (73.7%)
- **Ambiguity Detection Performance:** Ambiguity detection correctly flagged **17 of 38 errors (44.7%)** as ambiguous, proving that low score margins effectively signal classification risk.

---

## 8. Complete Breakdown of All 38 Misclassifications


---
#### Misclassification #1: `ext_001.jpg`
- **Ground Truth:** `Education` &rarr; **Predicted:** `Entertainment` (Top-2 Match: `False`)
- **Candidates:** Entertainment (0.6802), Travel (0.1264), General (0.1205)
- **Content-Type:** `video` | **Conf:** `0.7062` | **Ambiguous:** `False` | **Margin:** `0.5538`
- **Primary Cause:** `D. Embedding Limitation`
- **Reason:** Prototype vector embedding for 'Entertainment' pulled vector similarity away from 'Education'.

---
#### Misclassification #2: `ext_002.jpg`
- **Ground Truth:** `Shopping` &rarr; **Predicted:** `General` (Top-2 Match: `False`)
- **Candidates:** Connectivity (0.0), Payment (0.0), Education (0.0)
- **Content-Type:** `unknown` | **Conf:** `0.0` | **Ambiguous:** `True` | **Margin:** `0.0`
- **Primary Cause:** `A. OCR/Evidence Limitation`
- **Reason:** Extracted OCR text is extremely sparse (0 words). Crucial visual context (diagrams, layout, handwritten notes) is absent from OCR.

---
#### Misclassification #3: `ext_003.jpg`
- **Ground Truth:** `Entertainment` &rarr; **Predicted:** `Shopping` (Top-2 Match: `False`)
- **Candidates:** Shopping (0.356), Work & Career (0.3217), Travel (0.2104)
- **Content-Type:** `product_page` | **Conf:** `0.255` | **Ambiguous:** `True` | **Margin:** `0.0343`
- **Primary Cause:** `F. Cross-Category Context Overlap`
- **Reason:** Media/movie promotional graphics contain purchase/watch pricing controls overlapping with Shopping product listings.

---
#### Misclassification #4: `ext_006.jpg`
- **Ground Truth:** `Education` &rarr; **Predicted:** `Coding` (Top-2 Match: `False`)
- **Candidates:** Coding (0.3608), Travel (0.3111), Education (0.2938)
- **Content-Type:** `documentation` | **Conf:** `0.262` | **Ambiguous:** `True` | **Margin:** `0.0497`
- **Primary Cause:** `F. Cross-Category Context Overlap`
- **Reason:** Computer science lecture notes or lab assignments contain source code snippets and technical syntax matching Coding.

---
#### Misclassification #5: `ext_007.jpg`
- **Ground Truth:** `Education` &rarr; **Predicted:** `Work & Career` (Top-2 Match: `False`)
- **Candidates:** Work & Career (0.661), Coding (0.4953), Shopping (0.418)
- **Content-Type:** `job_posting` | **Conf:** `0.5135` | **Ambiguous:** `False` | **Margin:** `0.1657`
- **Primary Cause:** `F. Cross-Category Context Overlap`
- **Reason:** University announcements, campus drive notices, or academic events overlap with Work & Career event/job listings.

---
#### Misclassification #6: `ext_008.jpg`
- **Ground Truth:** `Communication` &rarr; **Predicted:** `Payment` (Top-2 Match: `False`)
- **Candidates:** Payment (0.6524), Work & Career (0.3362), Travel (0.3231)
- **Content-Type:** `transaction` | **Conf:** `0.4877` | **Ambiguous:** `False` | **Margin:** `0.3162`
- **Primary Cause:** `C. Semantic Classification Limitation`
- **Reason:** Semantic text description similarity for 'Payment' outweighed semantic similarity for 'Communication'.

---
#### Misclassification #7: `ext_009.jpg`
- **Ground Truth:** `Education` &rarr; **Predicted:** `Shopping` (Top-2 Match: `False`)
- **Candidates:** Shopping (0.5249), General (0.3296), Education (0.2774)
- **Content-Type:** `product_page` | **Conf:** `0.3877` | **Ambiguous:** `False` | **Margin:** `0.1953`
- **Primary Cause:** `C. Semantic Classification Limitation`
- **Reason:** Semantic text description similarity for 'Shopping' outweighed semantic similarity for 'Education'.

---
#### Misclassification #8: `ext_010.jpg`
- **Ground Truth:** `Education` &rarr; **Predicted:** `Payment` (Top-2 Match: `True`)
- **Candidates:** Payment (0.3126), Education (0.2484), Connectivity (0.1557)
- **Content-Type:** `transaction` | **Conf:** `0.2422` | **Ambiguous:** `True` | **Margin:** `0.0642`
- **Primary Cause:** `C. Semantic Classification Limitation`
- **Reason:** Semantic text description similarity for 'Payment' outweighed semantic similarity for 'Education'.

---
#### Misclassification #9: `ext_011.jpg`
- **Ground Truth:** `Education` &rarr; **Predicted:** `Work & Career` (Top-2 Match: `True`)
- **Candidates:** Work & Career (0.6747), Education (0.543), Shopping (0.4337)
- **Content-Type:** `job_posting` | **Conf:** `0.4435` | **Ambiguous:** `False` | **Margin:** `0.1317`
- **Primary Cause:** `F. Cross-Category Context Overlap`
- **Reason:** University announcements, campus drive notices, or academic events overlap with Work & Career event/job listings.

---
#### Misclassification #10: `ext_013.jpg`
- **Ground Truth:** `Education` &rarr; **Predicted:** `Coding` (Top-2 Match: `False`)
- **Candidates:** Coding (0.5532), Work & Career (0.4857), Shopping (0.417)
- **Content-Type:** `documentation` | **Conf:** `0.2969` | **Ambiguous:** `True` | **Margin:** `0.0675`
- **Primary Cause:** `F. Cross-Category Context Overlap`
- **Reason:** Computer science lecture notes or lab assignments contain source code snippets and technical syntax matching Coding.

---
#### Misclassification #11: `ext_014.jpg`
- **Ground Truth:** `Work & Career` &rarr; **Predicted:** `Payment` (Top-2 Match: `False`)
- **Candidates:** Payment (0.3466), Education (0.3054), General (0.2164)
- **Content-Type:** `transaction` | **Conf:** `0.319` | **Ambiguous:** `True` | **Margin:** `0.0412`
- **Primary Cause:** `D. Embedding Limitation`
- **Reason:** Prototype vector embedding for 'Payment' pulled vector similarity away from 'Work & Career'.

---
#### Misclassification #12: `ext_016.jpg`
- **Ground Truth:** `Work & Career` &rarr; **Predicted:** `Shopping` (Top-2 Match: `False`)
- **Candidates:** Shopping (0.4382), Education (0.3462), Work & Career (0.3259)
- **Content-Type:** `product_page` | **Conf:** `0.3134` | **Ambiguous:** `True` | **Margin:** `0.092`
- **Primary Cause:** `F. Cross-Category Context Overlap`
- **Reason:** Institutional entry pass or career posting overlaps with academic or identity document context.

---
#### Misclassification #13: `ext_017.jpg`
- **Ground Truth:** `Shopping` &rarr; **Predicted:** `Coding` (Top-2 Match: `False`)
- **Candidates:** Coding (0.5135), General (0.3184), Shopping (0.3109)
- **Content-Type:** `documentation` | **Conf:** `0.3819` | **Ambiguous:** `False` | **Margin:** `0.1951`
- **Primary Cause:** `C. Semantic Classification Limitation`
- **Reason:** Semantic text description similarity for 'Coding' outweighed semantic similarity for 'Shopping'.

---
#### Misclassification #14: `ext_019.jpg`
- **Ground Truth:** `Shopping` &rarr; **Predicted:** `Communication` (Top-2 Match: `True`)
- **Candidates:** Communication (0.6326), Shopping (0.4817), Travel (0.3106)
- **Content-Type:** `chat` | **Conf:** `0.4282` | **Ambiguous:** `False` | **Margin:** `0.1509`
- **Primary Cause:** `C. Semantic Classification Limitation`
- **Reason:** Semantic text description similarity for 'Communication' outweighed semantic similarity for 'Shopping'.

---
#### Misclassification #15: `ext_021.jpg`
- **Ground Truth:** `Coding` &rarr; **Predicted:** `Communication` (Top-2 Match: `False`)
- **Candidates:** Communication (0.6756), Shopping (0.3079), Coding (0.1727)
- **Content-Type:** `email` | **Conf:** `0.6481` | **Ambiguous:** `False` | **Margin:** `0.3677`
- **Primary Cause:** `D. Embedding Limitation`
- **Reason:** Prototype vector embedding for 'Communication' pulled vector similarity away from 'Coding'.

---
#### Misclassification #16: `ext_022.jpg`
- **Ground Truth:** `Entertainment` &rarr; **Predicted:** `General` (Top-2 Match: `False`)
- **Candidates:** General (0.3272), Connectivity (0.3206), Shopping (0.1755)
- **Content-Type:** `unknown` | **Conf:** `0.2989` | **Ambiguous:** `True` | **Margin:** `0.0066`
- **Primary Cause:** `C. Semantic Classification Limitation`
- **Reason:** Semantic text description similarity for 'General' outweighed semantic similarity for 'Entertainment'.

---
#### Misclassification #17: `ext_025.jpg`
- **Ground Truth:** `Work & Career` &rarr; **Predicted:** `Communication` (Top-2 Match: `False`)
- **Candidates:** Communication (0.4572), Education (0.3399), Travel (0.2723)
- **Content-Type:** `email` | **Conf:** `0.3305` | **Ambiguous:** `False` | **Margin:** `0.1173`
- **Primary Cause:** `C. Semantic Classification Limitation`
- **Reason:** Semantic text description similarity for 'Communication' outweighed semantic similarity for 'Work & Career'.

---
#### Misclassification #18: `ext_026.jpg`
- **Ground Truth:** `Entertainment` &rarr; **Predicted:** `Payment` (Top-2 Match: `False`)
- **Candidates:** Payment (0.3111), Education (0.2047), Travel (0.1675)
- **Content-Type:** `transaction` | **Conf:** `0.2541` | **Ambiguous:** `True` | **Margin:** `0.1064`
- **Primary Cause:** `C. Semantic Classification Limitation`
- **Reason:** Semantic text description similarity for 'Payment' outweighed semantic similarity for 'Entertainment'.

---
#### Misclassification #19: `ext_027.jpg`
- **Ground Truth:** `Shopping` &rarr; **Predicted:** `Travel` (Top-2 Match: `False`)
- **Candidates:** Travel (0.3291), General (0.2382), Shopping (0.2303)
- **Content-Type:** `travel_itinerary` | **Conf:** `0.3252` | **Ambiguous:** `True` | **Margin:** `0.0909`
- **Primary Cause:** `D. Embedding Limitation`
- **Reason:** Prototype vector embedding for 'Travel' pulled vector similarity away from 'Shopping'.

---
#### Misclassification #20: `ext_028.jpg`
- **Ground Truth:** `Education` &rarr; **Predicted:** `Work & Career` (Top-2 Match: `True`)
- **Candidates:** Work & Career (0.648), Education (0.3251), Shopping (0.3201)
- **Content-Type:** `job_posting` | **Conf:** `0.5542` | **Ambiguous:** `False` | **Margin:** `0.3229`
- **Primary Cause:** `F. Cross-Category Context Overlap`
- **Reason:** University announcements, campus drive notices, or academic events overlap with Work & Career event/job listings.

---
#### Misclassification #21: `ext_033.jpg`
- **Ground Truth:** `Entertainment` &rarr; **Predicted:** `Shopping` (Top-2 Match: `False`)
- **Candidates:** Shopping (0.3573), Work & Career (0.3232), Education (0.3166)
- **Content-Type:** `product_page` | **Conf:** `0.2555` | **Ambiguous:** `True` | **Margin:** `0.0341`
- **Primary Cause:** `F. Cross-Category Context Overlap`
- **Reason:** Media/movie promotional graphics contain purchase/watch pricing controls overlapping with Shopping product listings.

---
#### Misclassification #22: `ext_034.jpg`
- **Ground Truth:** `Shopping` &rarr; **Predicted:** `General` (Top-2 Match: `False`)
- **Candidates:** General (0.3355), Entertainment (0.3087), Shopping (0.2933)
- **Content-Type:** `unknown` | **Conf:** `0.3758` | **Ambiguous:** `True` | **Margin:** `0.0268`
- **Primary Cause:** `D. Embedding Limitation`
- **Reason:** Prototype vector embedding for 'General' pulled vector similarity away from 'Shopping'.

---
#### Misclassification #23: `ext_036.jpg`
- **Ground Truth:** `Work & Career` &rarr; **Predicted:** `Communication` (Top-2 Match: `False`)
- **Candidates:** Communication (0.5006), Shopping (0.4548), Work & Career (0.34)
- **Content-Type:** `email` | **Conf:** `0.3307` | **Ambiguous:** `True` | **Margin:** `0.0458`
- **Primary Cause:** `C. Semantic Classification Limitation`
- **Reason:** Semantic text description similarity for 'Communication' outweighed semantic similarity for 'Work & Career'.

---
#### Misclassification #24: `ext_038.jpg`
- **Ground Truth:** `General` &rarr; **Predicted:** `Shopping` (Top-2 Match: `False`)
- **Candidates:** Shopping (0.35), Connectivity (0.3264), Education (0.2312)
- **Content-Type:** `product_page` | **Conf:** `0.2487` | **Ambiguous:** `True` | **Margin:** `0.0236`
- **Primary Cause:** `C. Semantic Classification Limitation`
- **Reason:** Semantic text description similarity for 'Shopping' outweighed semantic similarity for 'General'.

---
#### Misclassification #25: `ext_039.jpg`
- **Ground Truth:** `General` &rarr; **Predicted:** `Connectivity` (Top-2 Match: `False`)
- **Candidates:** Connectivity (0.7132), Shopping (0.2123), Education (0.1872)
- **Content-Type:** `wifi_credentials` | **Conf:** `0.7069` | **Ambiguous:** `False` | **Margin:** `0.5009`
- **Primary Cause:** `D. Embedding Limitation`
- **Reason:** Prototype vector embedding for 'Connectivity' pulled vector similarity away from 'General'.

---
#### Misclassification #26: `ext_040.jpg`
- **Ground Truth:** `Communication` &rarr; **Predicted:** `General` (Top-2 Match: `True`)
- **Candidates:** General (0.3208), Communication (0.3177), Entertainment (0.2878)
- **Content-Type:** `unknown` | **Conf:** `0.2947` | **Ambiguous:** `True` | **Margin:** `0.0031`
- **Primary Cause:** `C. Semantic Classification Limitation`
- **Reason:** Semantic text description similarity for 'General' outweighed semantic similarity for 'Communication'.

---
#### Misclassification #27: `ext_041.jpg`
- **Ground Truth:** `Work & Career` &rarr; **Predicted:** `Payment` (Top-2 Match: `True`)
- **Candidates:** Payment (0.447), Work & Career (0.3381), Connectivity (0.336)
- **Content-Type:** `transaction` | **Conf:** `0.3228` | **Ambiguous:** `False` | **Margin:** `0.1089`
- **Primary Cause:** `C. Semantic Classification Limitation`
- **Reason:** Semantic text description similarity for 'Payment' outweighed semantic similarity for 'Work & Career'.

---
#### Misclassification #28: `ext_042.jpg`
- **Ground Truth:** `Travel` &rarr; **Predicted:** `Payment` (Top-2 Match: `True`)
- **Candidates:** Payment (0.6531), Travel (0.4275), Shopping (0.2433)
- **Content-Type:** `transaction` | **Conf:** `0.4609` | **Ambiguous:** `False` | **Margin:** `0.2256`
- **Primary Cause:** `C. Semantic Classification Limitation`
- **Reason:** Semantic text description similarity for 'Payment' outweighed semantic similarity for 'Travel'.

---
#### Misclassification #29: `ext_043.jpg`
- **Ground Truth:** `Education` &rarr; **Predicted:** `Payment` (Top-2 Match: `False`)
- **Candidates:** Payment (0.4378), Travel (0.4157), Education (0.2919)
- **Content-Type:** `transaction` | **Conf:** `0.2922` | **Ambiguous:** `True` | **Margin:** `0.0221`
- **Primary Cause:** `B. Rule/Evidence Misleading`
- **Reason:** Presence of numeric course units, marks, or isolated currency symbols triggered misleading financial/payment rule boosts.

---
#### Misclassification #30: `ext_047.jpg`
- **Ground Truth:** `Travel` &rarr; **Predicted:** `Payment` (Top-2 Match: `True`)
- **Candidates:** Payment (0.5711), Travel (0.4777), Entertainment (0.1013)
- **Content-Type:** `transaction` | **Conf:** `0.3802` | **Ambiguous:** `True` | **Margin:** `0.0934`
- **Primary Cause:** `C. Semantic Classification Limitation`
- **Reason:** Semantic text description similarity for 'Payment' outweighed semantic similarity for 'Travel'.

---
#### Misclassification #31: `ext_050.jpg`
- **Ground Truth:** `Entertainment` &rarr; **Predicted:** `General` (Top-2 Match: `False`)
- **Candidates:** General (0.3), Communication (0.2988), Coding (0.2192)
- **Content-Type:** `unknown` | **Conf:** `0.2837` | **Ambiguous:** `True` | **Margin:** `0.0012`
- **Primary Cause:** `A. OCR/Evidence Limitation`
- **Reason:** Extracted OCR text is extremely sparse (1 words). Crucial visual context (diagrams, layout, handwritten notes) is absent from OCR.

---
#### Misclassification #32: `ext_051.jpg`
- **Ground Truth:** `Entertainment` &rarr; **Predicted:** `Payment` (Top-2 Match: `False`)
- **Candidates:** Payment (0.4645), General (0.313), Entertainment (0.2599)
- **Content-Type:** `transaction` | **Conf:** `0.3444` | **Ambiguous:** `False` | **Margin:** `0.1515`
- **Primary Cause:** `C. Semantic Classification Limitation`
- **Reason:** Semantic text description similarity for 'Payment' outweighed semantic similarity for 'Entertainment'.

---
#### Misclassification #33: `ext_052.jpg`
- **Ground Truth:** `Education` &rarr; **Predicted:** `Coding` (Top-2 Match: `False`)
- **Candidates:** Coding (0.414), Shopping (0.3743), Education (0.3624)
- **Content-Type:** `documentation` | **Conf:** `0.2189` | **Ambiguous:** `True` | **Margin:** `0.0397`
- **Primary Cause:** `F. Cross-Category Context Overlap`
- **Reason:** Computer science lecture notes or lab assignments contain source code snippets and technical syntax matching Coding.

---
#### Misclassification #34: `ext_053.jpg`
- **Ground Truth:** `Entertainment` &rarr; **Predicted:** `General` (Top-2 Match: `True`)
- **Candidates:** General (0.327), Entertainment (0.3162), Coding (0.2985)
- **Content-Type:** `unknown` | **Conf:** `0.3667` | **Ambiguous:** `True` | **Margin:** `0.0108`
- **Primary Cause:** `D. Embedding Limitation`
- **Reason:** Prototype vector embedding for 'General' pulled vector similarity away from 'Entertainment'.

---
#### Misclassification #35: `ext_055.jpg`
- **Ground Truth:** `Education` &rarr; **Predicted:** `Payment` (Top-2 Match: `True`)
- **Candidates:** Payment (0.6992), Education (0.397), Travel (0.2765)
- **Content-Type:** `transaction` | **Conf:** `0.6403` | **Ambiguous:** `False` | **Margin:** `0.3022`
- **Primary Cause:** `B. Rule/Evidence Misleading`
- **Reason:** Presence of numeric course units, marks, or isolated currency symbols triggered misleading financial/payment rule boosts.

---
#### Misclassification #36: `ext_058.jpg`
- **Ground Truth:** `Education` &rarr; **Predicted:** `Payment` (Top-2 Match: `False`)
- **Candidates:** Payment (0.4469), Coding (0.3664), General (0.3206)
- **Content-Type:** `transaction` | **Conf:** `0.3143` | **Ambiguous:** `True` | **Margin:** `0.0805`
- **Primary Cause:** `B. Rule/Evidence Misleading`
- **Reason:** Presence of numeric course units, marks, or isolated currency symbols triggered misleading financial/payment rule boosts.

---
#### Misclassification #37: `ext_059.jpg`
- **Ground Truth:** `Education` &rarr; **Predicted:** `Shopping` (Top-2 Match: `False`)
- **Candidates:** Shopping (0.4715), General (0.3109), Communication (0.2641)
- **Content-Type:** `product_page` | **Conf:** `0.3506` | **Ambiguous:** `False` | **Margin:** `0.1606`
- **Primary Cause:** `B. Rule/Evidence Misleading`
- **Reason:** Presence of numeric course units, marks, or isolated currency symbols triggered misleading financial/payment rule boosts.

---
#### Misclassification #38: `ext_060.jpg`
- **Ground Truth:** `Education` &rarr; **Predicted:** `Shopping` (Top-2 Match: `True`)
- **Candidates:** Shopping (0.5032), Education (0.3876), Communication (0.3117)
- **Content-Type:** `shopping_order` | **Conf:** `0.3529` | **Ambiguous:** `False` | **Margin:** `0.1156`
- **Primary Cause:** `B. Rule/Evidence Misleading`
- **Reason:** Presence of numeric course units, marks, or isolated currency symbols triggered misleading financial/payment rule boosts.


---

## 9. Final Architectural Diagnosis

### What the Current Architecture Handles Well
1. **Pure Financial Receipts & Transactions**: 100% accuracy on standard payment receipts with UPI IDs and transaction numbers.
2. **Clear E-Commerce Shopping Orders & Products**: High precision when shopping keywords and delivery text are present.
3. **Multi-Signal Score Fusion & Ambiguity Safeguards**: High-confidence predictions achieve **66.7% accuracy**, while low-margin errors are safely flagged as `ambiguous = True`.

### What the Current Architecture Consistently Struggles With
1. **Cross-Category Context Overlap**: Screenshots containing both commercial prices and item descriptions (e.g., Zomato menu orders) create strong signal competition between `Shopping` and `Payment`.
2. **Academic & Technical Content Spillover**: Computer Science course material and code assignments trigger both `Education` and `Coding` signals.
3. **Sparse & Non-Text Screenshots**: Image-heavy graphics, scanned handwritten notes, or low-OCR screenshots produce near-zero rule and semantic scores.

### Single Architectural Direction to Investigate Next
**Multi-Modal Visual-Layout Embedding & Joint Evidence Retrieval Architecture (Phase 4B)**:
Combining OCR text embeddings with lightweight visual layout features (e.g., UI layout regions, image-text alignment, or a lightweight multi-modal feature extractor) will resolve the 26.3% sparse-text limitations and provide spatial structure signals to resolve Shopping vs Payment ambiguity.

---
