# OCR + Lightweight Semantic Classification Experiment Report

**Date**: 2026-09-30 10:34 UTC
**Scope**: 833 Valid Training Screenshots (5-Fold Stratified CV) & 60 Frozen External Screenshots
**Target Architecture**: OCR → Keyword/Rule Signal + TF-IDF + MiniLM Semantic Signal → Lightweight Classifier/Fusion

## Executive Summary

This experiment evaluated lightweight text-based classification architectures on the expanded 833 valid screenshot training pool. Model selection was governed strictly by **5-Fold Stratified Cross-Validation** over the 833 training samples, followed by a **single frozen evaluation** on the 60 external benchmark screenshots.

- **Winning Model Selected by CV**: `Fusion (0.2 Rule + 0.5 TF-IDF + 0.3 MiniLM)`
- **CV Top-1 Accuracy**: `97.12%` (CV Macro F1: `0.9545`)
- **External Benchmark Top-1 Accuracy**: `6.67%` (`4/60` images correct)
- **External Top-2 Recall**: `58.33%` | **External Top-3 Recall**: `73.33%`
- **External Macro F1**: `0.1107` | **Macro Precision**: `0.2018`
- **CPU Inference Latency**: `41.58 ms/image` (Total pipeline including OCR: `~241.58 ms/image`)
- **Additional Memory Footprint**: `0.0 MB` (Reuses existing `all-MiniLM-L6-v2` encoder)

---

## Complete Model Comparison Table

### 1. 5-Fold Stratified Cross-Validation (833 Training Pool)

| Model Name | Top-1 Accuracy | Top-2 Recall | Top-3 Recall | Macro Precision | Macro Recall | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Rule Baseline)** | 81.99% | 84.03% | 84.27% | 0.8034 | 0.7802 | **0.7354** |
| **Model B (TF-IDF + Linear SVM)** | 97.00% | 98.68% | 99.28% | 0.9720 | 0.9418 | **0.9497** |
| **Model C (MiniLM Cosine Similarity)** | 90.04% | 93.52% | 94.84% | 0.8714 | 0.8676 | **0.8434** |
| **Fusion (0.4 Rule + 0.4 TF-IDF + 0.2 MiniLM)** | 95.08% | 98.92% | 99.40% | 0.9309 | 0.9252 | **0.9251** |
| **Fusion (0.3 Rule + 0.4 TF-IDF + 0.3 MiniLM)** | 96.28% | 99.04% | 99.52% | 0.9490 | 0.9408 | **0.9432** |
| **Fusion (0.3 Rule + 0.3 TF-IDF + 0.4 MiniLM)** | 95.20% | 98.92% | 99.40% | 0.9310 | 0.9288 | **0.9270** |
| **Fusion (0.2 Rule + 0.5 TF-IDF + 0.3 MiniLM) ⭐** | 97.12% | 99.16% | 99.52% | 0.9639 | 0.9498 | **0.9545** |
| **Fusion (0.33 Rule + 0.33 TF-IDF + 0.34 MiniLM)** | 95.20% | 99.04% | 99.40% | 0.9326 | 0.9268 | **0.9269** |

### 2. Frozen External Benchmark Performance (60 Screenshots)

| Model Name | Top-1 Accuracy | Top-2 Recall | Top-3 Recall | Macro Precision | Macro Recall | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Rule Baseline)** | 61.67% (37/60) | 70.00% | 80.00% | 0.6583 | 0.6489 | **0.5850** |
| **Model B (TF-IDF + Linear SVM)** | 1.67% (1/60) | 3.33% | 6.67% | 0.0017 | 0.1000 | **0.0033** |
| **Model C (MiniLM Cosine Similarity)** | 43.33% (26/60) | 63.33% | 80.00% | 0.3773 | 0.4852 | **0.3299** |
| **Fusion (0.4 Rule + 0.4 TF-IDF + 0.2 MiniLM)** | 40.00% (24/60) | 61.67% | 73.33% | 0.6029 | 0.5027 | **0.4665** |
| **Fusion (0.3 Rule + 0.4 TF-IDF + 0.3 MiniLM)** | 30.00% (18/60) | 65.00% | 76.67% | 0.5024 | 0.3679 | **0.3276** |
| **Fusion (0.3 Rule + 0.3 TF-IDF + 0.4 MiniLM)** | 50.00% (30/60) | 65.00% | 78.33% | 0.7036 | 0.5800 | **0.5622** |
| **Fusion (0.2 Rule + 0.5 TF-IDF + 0.3 MiniLM) ⭐** | 6.67% (4/60) | 58.33% | 73.33% | 0.2018 | 0.1733 | **0.1107** |
| **Fusion (0.33 Rule + 0.33 TF-IDF + 0.34 MiniLM)** | 46.67% (28/60) | 68.33% | 78.33% | 0.7033 | 0.5664 | **0.5492** |

### 3. Historical Baseline Comparison (External Benchmark)

| Baseline Reference | Top-1 Acc | Top-2 Rec | Top-3 Rec | Macro F1 | Footprint |
| :--- | :---: | :---: | :---: | :---: | :---: |
| MiniLM + Linear SVM (Old 30 Pool) | 66.67% (40/60) | 76.67% | 83.33% | 0.3971 | ~0 MB |
| Bagged MiniLM + Linear SVM (Old 30 Pool) | 68.33% (41/60) | 75.00% | 83.33% | 0.3708 | ~0 MB |
| MiniLM + CLIP + Logistic Regression (Old 30 Pool) | 71.67% (43/60) | 78.33% | 83.33% | 0.4660 | ~605 MB |
| **Current Winner: Fusion (0.2 Rule + 0.5 TF-IDF + 0.3 MiniLM) (833 Pool)** | **6.67% (4/60)** | **58.33%** | **73.33%** | **0.1107** | **0.0 MB** |

---

## Signal Explainability & Score Breakdown

Below are sample external predictions demonstrating how Rule Baseline, TF-IDF + Linear SVM, and MiniLM Cosine Similarity signals contribute to the final ensemble probability:

### Sample: `ext_001.jpg`
- **Ground Truth**: `Education` | **Predicted**: `Coding`
- **OCR Snippet**: *"Reel DIOMS Hit the road Leave Kick the bucket Die Jump the gun Act too soon Fit as a fiddle Healthy Break a Good luck Piece of cake Very easy Put your"*
- **Top Category Probability Contributions**:
  - `Coding`: Fusion Score = **0.401** (Rule: `0.042`, TF-IDF: `0.770`, MiniLM: `0.024`)
  - `Entertainment`: Fusion Score = **0.220** (Rule: `0.515`, TF-IDF: `0.007`, MiniLM: `0.378`)
  - `Connectivity`: Fusion Score = **0.098** (Rule: `0.042`, TF-IDF: `0.155`, MiniLM: `0.039`)

### Sample: `ext_002.jpg`
- **Ground Truth**: `Shopping` | **Predicted**: `Coding`
- **OCR Snippet**: *""*
- **Top Category Probability Contributions**:
  - `Coding`: Fusion Score = **0.311** (Rule: `0.100`, TF-IDF: `0.522`, MiniLM: `0.100`)
  - `Connectivity`: Fusion Score = **0.222** (Rule: `0.100`, TF-IDF: `0.345`, MiniLM: `0.100`)
  - `General`: Fusion Score = **0.069** (Rule: `0.100`, TF-IDF: `0.037`, MiniLM: `0.100`)

### Sample: `ext_003.jpg`
- **Ground Truth**: `Entertainment` | **Predicted**: `Coding`
- **OCR Snippet**: *"ImoOhow-AocK_-Ibun EHL2 JESSICA VUNES U15 (AvENGERS -Eaptain AMERIIA ~THO R TAE Finsi AVENGEF IRON FIST crie IRON MAN 3 Pd Tn THOR H WAt CAPTAIN AMERI"*
- **Top Category Probability Contributions**:
  - `Coding`: Fusion Score = **0.443** (Rule: `0.080`, TF-IDF: `0.808`, MiniLM: `0.076`)
  - `Education`: Fusion Score = **0.083** (Rule: `0.080`, TF-IDF: `0.012`, MiniLM: `0.203`)
  - `Entertainment`: Fusion Score = **0.080** (Rule: `0.279`, TF-IDF: `0.008`, MiniLM: `0.067`)

### Sample: `ext_004.jpg`
- **Ground Truth**: `Payment` | **Predicted**: `Payment`
- **OCR Snippet**: *"To IPSITA DILIP WAKHARKAR 71,800 again Completed 30 2026, 11:54am State Bank of India 7427 UPI transaction ID 660821714104 To: IPSITA DILIP WAKHARKAR "*
- **Top Category Probability Contributions**:
  - `Payment`: Fusion Score = **0.440** (Rule: `0.999`, TF-IDF: `0.011`, MiniLM: `0.781`)
  - `Coding`: Fusion Score = **0.430** (Rule: `0.000`, TF-IDF: `0.850`, MiniLM: `0.015`)
  - `Connectivity`: Fusion Score = **0.044** (Rule: `0.000`, TF-IDF: `0.060`, MiniLM: `0.046`)

### Sample: `ext_005.jpg`
- **Ground Truth**: `Education` | **Predicted**: `Coding`
- **OCR Snippet**: *"Module No. & Topic/Coverage No. of Lecture Name. Lectures_ Serial no_ Introduction Networks: 1-3 Feed forward Networks and Hidden Units (Basic Neural "*
- **Top Category Probability Contributions**:
  - `Coding`: Fusion Score = **0.417** (Rule: `0.006`, TF-IDF: `0.795`, MiniLM: `0.062`)
  - `Connectivity`: Fusion Score = **0.225** (Rule: `0.714`, TF-IDF: `0.108`, MiniLM: `0.093`)
  - `Education`: Fusion Score = **0.142** (Rule: `0.263`, TF-IDF: `0.017`, MiniLM: `0.270`)

---

## External Benchmark Error Analysis

The winning model `Fusion (0.2 Rule + 0.5 TF-IDF + 0.3 MiniLM)` committed `56` errors out of 60 images on the frozen external benchmark:

| Filename | Ground Truth | Predicted Category | OCR Excerpt |
| :--- | :--- | :--- | :--- |
| `ext_001.jpg` | **Education** | `Coding` | *"Reel DIOMS Hit the road Leave Kick the bucket Die Jump the gun Act too soon Fit "* |
| `ext_002.jpg` | **Shopping** | `Coding` | *""* |
| `ext_003.jpg` | **Entertainment** | `Coding` | *"ImoOhow-AocK_-Ibun EHL2 JESSICA VUNES U15 (AvENGERS -Eaptain AMERIIA ~THO R TAE "* |
| `ext_005.jpg` | **Education** | `Coding` | *"Module No. & Topic/Coverage No. of Lecture Name. Lectures_ Serial no_ Introducti"* |
| `ext_006.jpg` | **Education** | `Coding` | *"UnIT 1 SOFTMAX, LOSS AND GRADIENT-BASED LEARNING E. Softmax (multi-class output)"* |
| `ext_007.jpg` | **Education** | `Coding` | *"Feature Bagging (Bootstrap Boosting Aggregating) Training Approach Parallel: Mod"* |
| `ext_008.jpg` | **Communication** | `Coding` | *"741 @ Mm 509 EH 4*" 0 Cj Your chapter in [978-3-032-37100-3, DoSCI 2026, LNNS 22"* |
| `ext_009.jpg` | **Education** | `Coding` | *"#au sdhwan' Page Ho_ YOUVA Date; Bandish - < 120 BPo 10 11 12 13 14 1S Ib 2 8457"* |
| `ext_010.jpg` | **Education** | `Coding` | *"Gneent ambtm Aan TeanitontL Gnrfu fien Fpunea 4iai Jonn- F> 1+f51FT FTC) FI( + e"* |
| `ext_011.jpg` | **Education** | `Coding` | *"DLT Quiz CS-16(Activity 2 and 3) Your response has been recorded: Submitanother "* |
| `ext_012.jpg` | **Work & Career** | `Coding` | *"AlI Boot Registration for Deloitte Campus recruitment Drive 2027 Graduating Batc"* |
| `ext_013.jpg` | **Education** | `Coding` | *"FIXED PARTITION SCHEME Memory is divided into fixed no. of partitions Fixed mean"* |
| `ext_014.jpg` | **Work & Career** | `Coding` | *"INDIAN INSTITUTE OF TECHNOLOGY KHARAGPUR-721 302. INDIA, PHONE-03222 255221 ENTR"* |
| `ext_015.jpg` | **Education** | `Coding` | *"majOrit Branch Git commit EPP_Autumn_2026 Classroor New chat - Claude education "* |
| `ext_016.jpg` | **Work & Career** | `Coding` | *"4.31:18 269 VoLfeg ,ll ,l 100% Piyush Das 62 ZYcus How to apply Date: Pre Sales "* |
| `ext_017.jpg` | **Shopping** | `Coding` | *"11.51.50 8 763 Ya? ,ll SG| 89% pocket watch marvel AUGEN Premium Pocket Watch Me"* |
| `ext_018.jpg` | **Shopping** | `Coding` | *"9*09:12 9813 VaffB ,l|   S1] 75% Search for products 4.1 364 Selected Color: Bla"* |
| `ext_019.jpg` | **Shopping** | `Coding` | *"2.30.53 3, 18G  YeB ,ll *5G1/ 51% LA CAFEOTHEQUE BHUBANESWAR 4.7 (855) 8200-400 "* |
| `ext_020.jpg` | **Education** | `Coding` | *"11:15*06 34.06 8u 4813 YeB ,ll *5G1/68% % ng servicenow.com 23 servicenow @Ricef"* |
| `ext_022.jpg` | **Entertainment** | `Coding` | *"10.33.33 81 83,9 Yeg ,ll *5G1/85% 538 7 08 Save 'Love itl View all comments"* |
| `ext_023.jpg` | **Education** | `Coding` | *"12.41.28 Zu VGa ul,ul91% Schedule MON TUE WED THU FRI SAT MC C25-A-109 03 : 00 P"* |
| `ext_024.jpg` | **Education** | `Coding` | *"214:10 65 26/9 9 ullull94% 2 facebook com/reell 23 facebook Open app In Tap to u"* |
| `ext_025.jpg` | **Work & Career** | `Coding` | *"Sl No Date PPT Time PPT Venue Online Test Venue Post PPT & Venue- Computer Labs "* |
| `ext_026.jpg` | **Entertainment** | `Coding` | *"8.01.55 W Ilulo 188 YeB ,ll *SG1/ 99% ReN Titanium David Guetta, Sia 02.34 03.26"* |
| `ext_027.jpg` | **Shopping** | `Coding` | *"10.32.35 8I 0873 YeB ,ll   5G1/86% BEDROOM 13'-3" x10'-8" 248 3 08 Save pochacco"* |
| `ext_028.jpg` | **Education** | `Coding` | *"KALINGA IMSTITUTE @F INDUSTRIAL TECHMOLOGY Deemed t0 be University U/s 3 of the "* |
| `ext_029.jpg` | **Shopping** | `Coding` | *"6:55.47 10 R819 YeB ,ll 561/93% Order Placed You saved <587 Scratch for Supercoi"* |
| `ext_030.jpg` | **Payment** | `Coding` | *"10.31.57 *93  YeB ,ll 56 SI| 93% Checkout zapvi.in Secured by Payshenee Zapvi ZA"* |
| `ext_031.jpg` | **Education** | `Coding` | *"11:14*29 33.30 Xu 28,g YsB ,ll *5G1/68% % ng servicenow.com 23 servicenow @ricee"* |
| `ext_032.jpg` | **Shopping** | `Coding` | *"8.41.58 16 X1]153 VafAB ,l 59% Search for products 1 73% 2,999 <795 4.1 | 97 Sel"* |
| `ext_033.jpg` | **Entertainment** | `Coding` | *"COMBO: 4 90 3/6 Level: 61 B  | N A B A K | n 6 A K N N K | N G B N A K G"* |
| `ext_034.jpg` | **Shopping** | `Coding` | *"11.07.37 43.18 8 13? YeB ,ll *5G1/93% marvel themed Unyks Star Super Hero Action"* |
| `ext_036.jpg` | **Work & Career** | `Coding` | *"4.05:19 488 SeB ,ll ,l 93% k 7 Zycus PPT Rashi Varma 62 ZYcus The offer for Mark"* |
| `ext_037.jpg` | **Shopping** | `Coding` | *"21*32 193 I 24% Search or ask a q .  @ 0 38 Your Orders Order details Arriving T"* |
| `ext_038.jpg` | **General** | `Coding` | *"2135 6 Frl MM Google? ul 24% weather Bhubaneswar Al Mode AII News Images Videos "* |
| `ext_039.jpg` | **General** | `Connectivity` | *"2136 iniate MM 63/9 Il 24% Settings Search Sign in Sign in to use services Aerop"* |
| `ext_040.jpg` | **Communication** | `Coding` | *"21*36 eoutd M 58/9 ul 24% Viki's Terms of Use and Policies Updates Inbox Rakuten"* |
| `ext_041.jpg` | **Work & Career** | `Coding` | *"9.37:14 6 98/8 29 ,l ul50% Search Aayush Kumar Gupta @ Ist Backend & ML Engineer"* |
| `ext_042.jpg` | **Travel** | `Coding` | *"10.24 LTEI ,Ill LTEZ ,ll 66% Silchar Guwahati 21June, Sunday Fri, 19 Sat; 20 Sun"* |
| `ext_043.jpg` | **Education** | `Coding` | *"2.51 61 Loh Lol ,ll 89% ANKUSH LAMBA Just now Reasoning dldcsd Ankush Lamba #sel"* |
| `ext_044.jpg` | **Education** | `Coding` | *"11.57 76Lh ,Il 95% Consent for appearing i docs google.com Consent for appearing"* |
| `ext_045.jpg` | **Shopping** | `Coding` | *"248 Lt LTET ,ll 99% JD-NETMED-S Thursday; 13 August Your Netmeds order FY6AZDZFD"* |
| `ext_047.jpg` | **Travel** | `Coding` | *"8.02 LTET l 83% Choose seat 6eskal In flight Turbulence We're currently navigati"* |
| `ext_048.jpg` | **Travel** | `Coding` | *"8.04 LTEI ,ll 82% IndiGo PNR/Booking Ref:: AgPMQM Status Date of Booking* Paymen"* |
| `ext_049.jpg` | **Payment** | `Coding` | *"9.44 LTEI ,l} 65% %o sap.kiitinteractive.com Contact ushelpdesksapeamekiitacin K"* |
| `ext_050.jpg` | **Entertainment** | `Coding` | *"1/2"* |
| `ext_051.jpg` | **Entertainment** | `Coding` | *"2.09 Fx4 76 Nol ,ll 86% 3.49 Michael Jackson Was a Flirt ForeverMichael 1 lakh v"* |
| `ext_052.jpg` | **Education** | `Coding` | *"10.48 LTEI ,ll 81% examination) It is about following any source yoU are comfort"* |
| `ext_053.jpg` | **Entertainment** | `Coding` | *"11:12 FX 76 Lo1 ,Il 83% DeviantArt ArtyShan Unbox B UILD DRAW PLAY ARTCoM Dark D"* |
| `ext_054.jpg` | **Entertainment** | `Coding` | *"2:14 F x LTCI ,ll 84% YTSTV TV-SERIES RESULTS: MICHAEL JACKSON 4.7 2026 7.4 2026"* |
| `ext_055.jpg` | **Education** | `Coding` | *"6.57 Loh ,ll 98% Dear Student; You are selected KIITEE Mer it Scholarship(Renewa"* |
| `ext_056.jpg` | **Education** | `Coding` | *"10.52 Loh ,ll 87% ociIeuule Dear Sneha Sinha, Exam: Mid Semester (Zth) Exam. Aut"* |
| `ext_057.jpg` | **Shopping** | `Coding` | *"10.25 0 X LTET ,li 91% Exchange Item Exchange Request Submitted Your exchanged i"* |
| `ext_058.jpg` | **Education** | `Coding` | *"9:27 Lt LTET ,ll 97% Group_5678-pdf Gr. Group Members Topic No_ 2305385, 2305493"* |
| `ext_059.jpg` | **Education** | `Coding` | *"NIMISHA DA EDIIORAL_ANAlySIS by BANSAL LIVE 8.00 _ 14-16/98 (?Tacisaryuf ; dgd 3"* |
| `ext_060.jpg` | **Education** | `Coding` | *"Flimsy Pluck Convoluted Chary Prolix Promulgate Embroiled Perennial Bunkum Smidg"* |

---

## Latency & Footprint Audit

- **Classifier CPU Inference Latency**: `41.58 ms/image`
- **Estimated CPU OCR Latency**: `200.0 ms/image`
- **Total End-to-End Latency**: `~241.58 ms/image`
- **Additional Memory Footprint**: `0.0 MB`

---

## Integrity Verification Checklist

- [x] Zero CLIP, TinyCLIP, MobileNet, EfficientNet, or visual encoders used in this experiment.
- [x] Exactly 833 valid training samples evaluated in 5-Fold Stratified CV.
- [x] All 3 zero-byte files (`ss_441.png`, `ss_443.png`, `ss_560.png`) excluded from training/eval.
- [x] 60-image external benchmark kept strictly frozen during CV and model selection.
- [x] Zero external sample data used in fitting TF-IDF, SVM, or tuning weights.
- [x] Canonical 10 categories evaluated without modifying labels or schemas.
- [x] No production code or ChromaDB vector store altered.
