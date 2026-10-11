# Unseen Screenshot Evaluation Dataset (Phase 3G.1)

This directory contains the manifest, images, and manual ground-truth reference annotations for the **Phase 3G.1 Generalization Evaluation Dataset**.

---

## 1. Dataset Overview & Zero Data Leakage Policy

The 22 unseen screenshots in this dataset were **strictly isolated** from the 8-image development benchmark (`ss_001.jpg`, `ss_002.jpg`, `ss_003.jpg`, `ss_008.jpg`, `ss_0011.jpg`, `ss_0021.jpg`, `ss_0028.jpg`, `ss_0030.jpg`) used during Phase 3G baseline design and tuning.

- **No Overlap:** 0% overlap with the development benchmark.
- **No Threshold Tuning:** The classifier rules, prototype anchors, confidence thresholds, and feature extractors were **not modified** after observing unseen results.
- **Manual Ground-Truth Annotations:** Every reference annotation in `evaluation_targets.py` was created through human inspection of the screenshots and OCR text without relying on the classifier under evaluation.

---

## 2. Dataset Size & Functional Category Distribution

Total Target Count: **22 Screenshots**

| Functional Category | Unseen Target Filenames | Image Count | Percentage |
| :--- | :--- | :---: | :---: |
| **Education** | `ss_007.jpg`, `ss_012.jpg`, `ss_018.jpg`, `ss_023.jpg`, `ss_024.jpg`, `ss_027.jpg` | 6 | 27.3% |
| **Payment** | `ss_005.jpg`, `ss_013.jpg`, `ss_017.jpg`, `ss_026.jpg` | 4 | 18.2% |
| **Shopping** | `ss_010.jpg`, `ss_015.jpg`, `ss_020.jpg`, `ss_022.jpg` | 4 | 18.2% |
| **Travel** | `ss_014.jpg`, `ss_016.jpg`, `ss_019.jpg` | 3 | 13.6% |
| **Entertainment** | `ss_004.jpg`, `ss_006.jpg`, `ss_009.jpg`, `ss_025.jpg` | 4 | 18.2% |
| **Coding** | `ss_029.jpg` | 1 | 4.5% |
| **Total** | **22 Unseen Screenshots** | **22** | **100.0%** |

---

## 3. Directory Layout

```text
experiments/intelligence/unseen/
├── README.md                   # This documentation file
├── evaluation_targets.py       # Manifest & hand-crafted reference annotations
└── images/                     # Isolated 22 unseen screenshot image files
    ├── ss_004.jpg
    ├── ss_005.jpg
    ├── ss_006.jpg
    ├── ss_007.jpg
    ├── ss_009.jpg
    ├── ss_010.jpg
    ├── ss_012.jpg
    ├── ss_013.jpg
    ├── ss_014.jpg
    ├── ss_015.jpg
    ├── ss_016.jpg
    ├── ss_017.jpg
    ├── ss_018.jpg
    ├── ss_019.jpg
    ├── ss_020.jpg
    ├── ss_022.jpg
    ├── ss_023.jpg
    ├── ss_024.jpg
    ├── ss_025.jpg
    ├── ss_026.jpg
    ├── ss_027.jpg
    └── ss_029.jpg
```
