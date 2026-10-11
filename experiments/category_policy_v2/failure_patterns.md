# GENERALIZED FAILURE PATTERNS V2

### PAT_01: General homepage / UI browsing misclassified into domain category via weak single keyword
- **Affected Count**: 30
- **Categories**: General, Shopping, Payment, Work & Career
- **Potential Rule**: Rule 3: Calibrated General Safeguard (Require >= 2 OCR matches or strong policy score before penalizing General)

### PAT_02: Coursework and software tool screenshots misclassified as General or Entertainment
- **Affected Count**: 5
- **Categories**: Education, General, Entertainment
- **Potential Rule**: Rule 1: Coursework platform indicator boost for Education

### PAT_03: Dashboard task management misclassified as Work & Career
- **Affected Count**: 7
- **Categories**: Work & Career, General
- **Potential Rule**: Rule 2: Dashboard UI vs Career vacancy disambiguation

### PAT_04: Invoice/Money references misclassified as Payment without completed transaction status
- **Affected Count**: 8
- **Categories**: Payment, General
- **Potential Rule**: Rule 4: Payment completion status vs currency reference disambiguation

