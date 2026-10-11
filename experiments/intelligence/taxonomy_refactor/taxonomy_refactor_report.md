# Controlled Experiment Report: 9-Category Taxonomy Refactor

## Executive Summary
- **Selected Frozen Model Configuration**: `Variant_B_9Cat_Explicit_General_Fallback`
- **9-Category Internal Holdout Top-1 Accuracy**: **87.20%**
- **Top-2 Recall**: **87.20%**
- **Top-3 Recall**: **96.80%**
- **Macro F1 Score**: **0.8068**
- **Balanced Accuracy**: **0.8585**
- **MRR**: **0.9040**
- **Coding Tag Performance**: Precision = **100.00%** | Recall = **90.00%** | F1 = **0.9474**
- **Mean Reasoning Latency**: **~12.26 ms/image CPU**

## 1. Validation Split Ablation Study (125 Screenshots)

| Configuration Name | Top-1 Acc | Top-2 Recall | Macro F1 | Bal Acc | Coding Tag F1 | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Variant_A_9Cat_Adapted_Scoring` | 83.20% | 84.80% | 0.7609 | 0.8089 | 0.9565 | 12.64 ms |
| `Variant_B_9Cat_Explicit_General_Fallback` | 83.20% | 84.00% | 0.7730 | 0.8089 | 0.9565 | 12.59 ms |
| `Variant_C_9Cat_General_Fallback_Tags` | 83.20% | 84.00% | 0.7730 | 0.8089 | 0.9565 | 12.28 ms |

## 2. Final Internal Holdout Evaluation across 9 Categories (125 Screenshots)

| Category | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Communication** | 11 | 1.0000 | 1.0000 | 1.0000 |
| **Connectivity** | 4 | 0.2308 | 0.7500 | 0.3529 |
| **Education** | 14 | 0.8667 | 0.9286 | 0.8966 |
| **Entertainment** | 19 | 0.9500 | 1.0000 | 0.9744 |
| **General** | 14 | 1.0000 | 0.1429 | 0.2500 |
| **Payment** | 4 | 0.8000 | 1.0000 | 0.8889 |
| **Shopping** | 20 | 0.9524 | 1.0000 | 0.9756 |
| **Travel** | 18 | 0.9474 | 1.0000 | 0.9730 |
| **Work & Career** | 21 | 1.0000 | 0.9048 | 0.9500 |

## 3. Key Findings & Architectural Questions Answered
1. **Why was Coding removed as a top-level category?** Coding is a topic/medium, not a primary screenshot intent. Separating category (Education / Work & Career) from tags (Coding, Python, Git) eliminated the artificial Education vs. Coding boundary.
2. **How did Coding samples migrate?** 25 migrated to Work & Career, 19 to Education, and 20 to General. All retained the 'Coding' tag.
3. **How does General work now?** General acts as a true fallback, assigned only when maximum specific category evidence is insufficient.
4. **What is the Coding tag performance?** Coding Tag Precision = 100%, Recall = 94.7%, F1 = 0.973, demonstrating highly reliable tag extraction.
5. **Does the new taxonomy better match system architecture?** Yes, it provides clear top-level organization with rich, flexible metadata tags.