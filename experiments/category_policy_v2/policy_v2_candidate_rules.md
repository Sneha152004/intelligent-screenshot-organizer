# POLICY V2 CANDIDATE RULES SPECIFICATION
=========================================

## Overview

Policy V2 introduces 5 generalized, semantic candidate rules operating over existing OCR text, primary-object intent, and candidate pool scores.

| Rule ID | Target Category | Trigger Evidence | Negative / Excluded Evidence | Confusable Categories | Expected Effect |
|---|---|---|---|---|---|
| **V2_Rule1_Education_Calib** | Education | `weka explorer`, `coursera`, `canvas lms`, `quiz submission`, `homework assignment`, `lecture notes` | `shopping cart`, `payment receipt` | General, Entertainment | Promotes academic software & learning portals to Education. |
| **V2_Rule2_Work_Dashboard_Calib** | Work & Career | `projects`, `tasks`, `crm`, `timeline view`, `business partner` (when lacking hiring words) | `hiring`, `job vacancy`, `job application`, `interview invitation` | General | Suppresses false Work & Career promotion on internal dashboard UIs. |
| **V2_Rule3_General_Penalty_Safeguard** | General | Classifier / MiniLM predicts General | Domain candidate has $\ge 2$ OCR matches or policy boost $\ge 2.0$ | Shopping, Payment, Work & Career | Safeguards General against aggressive multiplier penalties from weak keyword noise. |
| **V2_Rule4_Payment_Calib** | Payment | `sent $`, `usd to`, `invoice amount` (when lacking status) | `payment successful`, `money transferred`, `upi transaction id` | General, Shopping | Calibrates Payment boost to require completed transaction receipts. |
| **V2_Rule5_Shopping_Calib** | Shopping | `explore`, `search or ask`, `home living`, `beauty accessories` (lacking cart intent) | `add to cart`, `add to bag`, `buy now`, `checkout` | General | Calibrates Shopping boost to distinguish active cart checkout from general landing page browsing. |
