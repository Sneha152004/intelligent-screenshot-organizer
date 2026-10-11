# Final Experiment Report: Intelligent Screenshot Organizer Classification Pipeline

## Executive Summary
- **Final Frozen Model Configuration**: `Variant_A_Baseline_9Cat`
- **Internal Holdout Top-1 Accuracy (125 imgs)**: **87.20%**
- **External Benchmark Top-1 Accuracy (60 imgs)**: **61.67%**
- **Internal-to-External Generalization Gap**: **25.53%**
- **Internal Macro F1**: **0.8068** | **External Macro F1**: **0.5522**
- **Internal Coding Tag F1**: **0.9474** | **External Coding Tag F1**: **0.0000**
- **Reasoning Latency**: **~12.60 ms/image CPU**
- **Strict Stopping Rule**: Pipeline is 100% frozen. No post-external tuning performed.

## 1. Multi-Stage Performance Comparison

| Evaluation Stage | Top-1 Acc | Top-2 Recall | Top-3 Recall | Macro F1 | Bal Acc | MRR | Coding Tag F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation Split (125 imgs)** | 83.20% | 88.00% | 96.00% | 0.7730 | 0.8250 | 0.8920 | 1.0000 |
| **Internal Holdout (125 imgs)** | **87.20%** | **87.20%** | **96.80%** | **0.8068** | **0.8585** | **0.9040** | **0.9474** |
| **External Benchmark (60 imgs)** | **61.67%** | **73.33%** | **78.33%** | **0.5522** | **0.6606** | **0.6917** | **0.0000** |

## 2. External Benchmark Category Breakdown (60 Screenshots)

| Category | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Communication** | 2 | 0.2500 | 1.0000 | 0.4000 |
| **Connectivity** | 0 | 0.0000 | 0.0000 | 0.0000 |
| **Education** | 23 | 0.9286 | 0.5652 | 0.7027 |
| **Entertainment** | 8 | 0.5000 | 0.2500 | 0.3333 |
| **General** | 2 | 0.0000 | 0.0000 | 0.0000 |
| **Payment** | 3 | 1.0000 | 1.0000 | 1.0000 |
| **Shopping** | 11 | 0.7778 | 0.6364 | 0.7000 |
| **Travel** | 5 | 1.0000 | 1.0000 | 1.0000 |
| **Work & Career** | 6 | 0.8333 | 0.8333 | 0.8333 |

## 3. Final Answers to Key Project Questions
1. **Did the final targeted refinement improve the internal holdout?** Yes, internal holdout accuracy reached **87.20% Top-1** with 0.8068 Macro F1.
2. **Did General improve?** Yes, General recall reached 64.29% on internal holdout by acting as a true evidence-based fallback.
3. **Did Work & Career improve?** Yes, Work & Career recall reached 90.48% on internal holdout.
4. **Did strong categories remain stable?** Yes, Education, Entertainment, Communication, Payment, Shopping, and Travel maintained high recall (90% - 100%).
5. **What is the final independent external accuracy?** **61.67% Top-1** across the 60 frozen external benchmark images.
6. **How large is the internal-to-external generalization gap?** **25.53%** (Internal 87.20% vs External 61.67%).
7. **How well do Coding tags generalize?** Coding tag F1 is **0.0000** on external benchmark (Precision = 0.00%, Recall = 0.00%).
8. **What are the remaining failure modes?** OCR quality variations on dense external academic slides and non-standard domain layouts.
9. **What is the final frozen classification architecture?** Explainable 9-Category OCR + MiniLM Semantic Fusion + Deterministic Tag Extractor (~12.3 ms/image CPU).