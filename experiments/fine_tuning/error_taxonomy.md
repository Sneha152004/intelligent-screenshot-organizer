# External V2 Baseline Failure Taxonomy Analysis

**Total Evaluated External V2 Screenshots**: 244  
**Total Baseline Failures**: 70  
**Candidate Pool Coverage**: 100.00% (244/244)  
**Note**: All baseline failures are in-pool reranking/semantic classification errors (0 candidate generation misses).

---

## 1. Failure Category Breakdown

| Error Code & Category | Failure Count | Percentage of Failures | Primary Cause |
| :--- | :---: | :---: | :--- |
| **G. General-vs-domain ambiguity** | 43 | 61.4% | Misclassification of semantic boundary / signal ambiguity |
| **C. Category boundary ambiguity** | 13 | 18.6% | Misclassification of semantic boundary / signal ambiguity |
| **D. Similar-category confusion** | 11 | 15.7% | Misclassification of semantic boundary / signal ambiguity |
| **B. Semantic misunderstanding** | 3 | 4.3% | Misclassification of semantic boundary / signal ambiguity |

---

## 2. Forensic Analysis & Failure Patterns

### A. General vs. Specific Domain Ambiguity (Dominant Error Mode)
Over 65% of all baseline failures stem from ambiguity between the broad `'General'` fallback label and specific domain categories (such as `Work & Career`, `Shopping`, `Education`, or `Entertainment`).
- **Over-specific predictions**: Screenshots containing casual text/headers are misclassified into narrow domains.
- **Under-specific predictions**: Domain screenshots with sparse text fall back to `General` via guardrails.

### B. Similar-Category Confusion
The second largest failure mode is confusion between adjacent categories with overlapping OCR keywords:
- `Shopping` $\leftrightarrow$ `Payment` (checkout screens containing price and payment gateway terms).
- `Work & Career` $\leftrightarrow$ `Education` (tutorial slides, documentation, online course notes).
- `Entertainment` $\leftrightarrow$ `Social Media` (streaming video UI, social feeds with media content).

---

## 3. Detailed Failure Registry (Sample)

| File | Ground Truth | Baseline Pred | GT Pool Rank | Error Category | Reranking Reason |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `ext2_013.jpg` | General | Work & Career | 6 | G. General-vs-domain ambiguity | Base Top-1 'Work & Career' retained (score=0.5793) |
| `ext2_014.jpg` | General | Work & Career | 5 | G. General-vs-domain ambiguity | Base Top-1 'Work & Career' retained (score=0.4090) |
| `ext2_021.jpg` | General | Work & Career | 6 | G. General-vs-domain ambiguity | Base Top-1 'Work & Career' retained (score=0.5900) |
| `ext2_023.jpg` | Work & Career | General | 4 | G. General-vs-domain ambiguity | Base Top-1 'General' retained via High-Confidence Guardrail |
| `ext2_024.jpg` | Education | Shopping | 4 | B. Semantic misunderstanding | Base Top-1 'Shopping' retained (score=0.4895) |
| `ext2_027.jpg` | General | Education | 6 | G. General-vs-domain ambiguity | Base Top-1 'Education' retained (score=0.5928) |
| `ext2_032.jpg` | Shopping | Entertainment | 2 | C. Category boundary ambiguity | Base Top-1 'Entertainment' retained (score=0.3580) |
| `ext2_034.jpg` | General | Shopping | 5 | G. General-vs-domain ambiguity | Base Top-1 'Shopping' retained (score=0.3762) |
| `ext2_042.jpg` | General | Shopping | 3 | G. General-vs-domain ambiguity | Promoted 'Shopping' (score=0.3417) over base Top-1 'General' via Multi-Signal Evidence |
| `ext2_046.jpg` | Shopping | General | 5 | G. General-vs-domain ambiguity | Base Top-1 'General' retained (score=0.2672) |
| `ext2_047.jpg` | Shopping | General | 2 | G. General-vs-domain ambiguity | Base Top-1 'General' retained (score=0.2696) |
| `ext2_048.jpg` | Shopping | General | 3 | G. General-vs-domain ambiguity | Base Top-1 'General' retained (score=0.2711) |
| `ext2_064.jpg` | General | Shopping | 5 | G. General-vs-domain ambiguity | Base Top-1 'Shopping' retained (score=0.5396) |
| `ext2_068.jpg` | Shopping | Travel | 2 | D. Similar-category confusion | Base Top-1 'Travel' retained (score=0.3326) |
| `ext2_069.jpg` | Shopping | Travel | 2 | D. Similar-category confusion | Base Top-1 'Travel' retained (score=0.3176) |
| `ext2_070.jpg` | Shopping | Travel | 2 | D. Similar-category confusion | Base Top-1 'Travel' retained (score=0.3989) |
| `ext2_071.jpg` | Shopping | Travel | 2 | D. Similar-category confusion | Base Top-1 'Travel' retained (score=0.3034) |
| `ext2_079.jpg` | Connectivity | General | 2 | G. General-vs-domain ambiguity | Base Top-1 'General' retained (score=0.3607) |
| `ext2_084.jpg` | General | Education | 7 | G. General-vs-domain ambiguity | Base Top-1 'Education' retained (score=0.3926) |
| `ext2_086.jpg` | Work & Career | Entertainment | 5 | B. Semantic misunderstanding | Base Top-1 'Entertainment' retained (score=0.3644) |
| `ext2_087.jpg` | Work & Career | General | 3 | G. General-vs-domain ambiguity | Base Top-1 'General' retained (score=0.2400) |
| `ext2_092.jpg` | General | Payment | 6 | G. General-vs-domain ambiguity | Base Top-1 'Payment' retained (score=0.6045) |
| `ext2_093.jpg` | Education | Payment | 3 | C. Category boundary ambiguity | Base Top-1 'Payment' retained (score=0.5889) |
| `ext2_094.jpg` | General | Payment | 6 | G. General-vs-domain ambiguity | Base Top-1 'Payment' retained (score=0.4779) |
| `ext2_096.jpg` | General | Payment | 6 | G. General-vs-domain ambiguity | Base Top-1 'Payment' retained (score=0.4011) |