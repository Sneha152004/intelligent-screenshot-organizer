"""
Unseen Screenshot Evaluation Dataset Manifest & Ground-Truth Reference Annotations
===================================================================================
Defines 22 genuinely unseen screenshot targets across functional categories.
All reference annotations were manually reviewed and created directly from actual screenshot
images and OCR text, without data leakage or reliance on classifier predictions.
"""

from typing import Any, Dict, List, Set

# The 22 unseen screenshot filenames (strictly disjoint from the 8 development benchmark images)
UNSEEN_TARGET_FILENAMES: List[str] = [
    "ss_004.jpg",
    "ss_005.jpg",
    "ss_006.jpg",
    "ss_007.jpg",
    "ss_009.jpg",
    "ss_010.jpg",
    "ss_012.jpg",
    "ss_013.jpg",
    "ss_014.jpg",
    "ss_015.jpg",
    "ss_016.jpg",
    "ss_017.jpg",
    "ss_018.jpg",
    "ss_019.jpg",
    "ss_020.jpg",
    "ss_022.jpg",
    "ss_023.jpg",
    "ss_024.jpg",
    "ss_025.jpg",
    "ss_026.jpg",
    "ss_027.jpg",
    "ss_029.jpg",
]

# Development Benchmark Filenames for strict leakage check
DEVELOPMENT_BENCHMARK_FILENAMES: Set[str] = {
    "ss_001.jpg",
    "ss_002.jpg",
    "ss_003.jpg",
    "ss_008.jpg",
    "ss_0011.jpg",
    "ss_011.jpg",
    "ss_0021.jpg",
    "ss_021.jpg",
    "ss_0028.jpg",
    "ss_028.jpg",
    "ss_0030.jpg",
    "ss_030.jpg",
}

# Manually reviewed ground-truth reference annotations for the 22 unseen targets
UNSEEN_GROUND_TRUTH_ANNOTATIONS: Dict[str, Dict[str, Any]] = {
    "ss_004.jpg": {
        "category": "Entertainment",
        "tags": ["movie", "ticket", "cinema", "entertainment", "pvr", "zomato"],
        "intent": "Keep cinema ticket booking confirmation",
        "summary": "PVR movie ticket booking confirmation for Spider-Man Brand New Day",
        "importance": 3,
        "entities": [
            {"text": "Spider-Man: Brand New Day", "type": "movie"},
            {"text": "PVR Utkal Galleria", "type": "theater"},
            {"text": "TRAYD98", "type": "booking_id"},
        ],
        "dates": [
            {"text": "Sunday 2 Aug 10:55 AM", "type": "event_date"}
        ],
        "action_items": [
            {"text": "Scan QR code at theater screen 3", "status": "pending"}
        ],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": True,
            "contains_personal_contact": False,
        },
    },
    "ss_005.jpg": {
        "category": "Payment",
        "tags": ["payment", "receipt", "upi", "paytm"],
        "intent": "Keep payment transaction receipt",
        "summary": "Paytm UPI payment receipt of ₹1,800 paid to IPSITA DILIP WAKHARKAR",
        "importance": 4,
        "entities": [
            {"text": "IPSITA DILIP WAKHARKAR", "type": "recipient"},
            {"text": "9881732186@pthdfc", "type": "upi_id"},
        ],
        "dates": [
            {"text": "28 July 2026, 11:34 am", "type": "transaction_date"}
        ],
        "action_items": [],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": True,
            "contains_personal_contact": True,
        },
    },
    "ss_006.jpg": {
        "category": "Entertainment",
        "tags": ["music", "bandish", "classical", "hansdhwani", "teentaal"],
        "intent": "Practice musical bandish notation",
        "summary": "Indian classical music notation sheet for Hansdhwani Teentaal Bandish",
        "importance": 2,
        "entities": [
            {"text": "Hansdhwani", "type": "raag"},
            {"text": "Teentaal", "type": "taal"},
        ],
        "dates": [],
        "action_items": [],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": False,
            "contains_personal_contact": False,
        },
    },
    "ss_007.jpg": {
        "category": "Education",
        "tags": ["academic", "notice", "evaluation", "practical_training", "guidelines"],
        "intent": "Check practical training evaluation guidelines",
        "summary": "Faculty evaluation guidelines notice for B.Tech 7th semester practical training",
        "importance": 3,
        "entities": [
            {"text": "Practical Training", "type": "course"}
        ],
        "dates": [
            {"text": "07/09/26", "type": "deadline"}
        ],
        "action_items": [
            {"text": "Present practical training report before guide on or before 07/09/26", "status": "pending"}
        ],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": False,
            "contains_personal_contact": False,
        },
    },
    "ss_009.jpg": {
        "category": "Entertainment",
        "tags": ["tv_series", "download", "entertainment", "website", "mp4"],
        "intent": "Browse mobile TV series download site details",
        "summary": "O2TvSeries TV series free download website listing and features",
        "importance": 2,
        "entities": [
            {"text": "O2TvSeries", "type": "website"}
        ],
        "dates": [],
        "action_items": [],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": False,
            "contains_personal_contact": False,
        },
    },
    "ss_010.jpg": {
        "category": "Shopping",
        "tags": ["flipkart", "orders", "shopping", "delivery", "credit_card"],
        "intent": "Track e-commerce orders and voucher offer",
        "summary": "Flipkart My Orders page showing delivered mobile stands and SBI card offer",
        "importance": 3,
        "entities": [
            {"text": "Flipkart", "type": "merchant"},
            {"text": "Flipkart SmartBuy NextGen Mobile Stand", "type": "product"},
        ],
        "dates": [
            {"text": "Aug 12", "type": "delivery_date"},
            {"text": "Jul 28", "type": "delivery_date"},
            {"text": "Jul 31", "type": "delivery_date"},
        ],
        "action_items": [
            {"text": "Rate & review delivered mobile stands", "status": "pending"}
        ],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": False,
            "contains_personal_contact": False,
        },
    },
    "ss_012.jpg": {
        "category": "Education",
        "tags": ["kiit", "notice", "practical_training", "computer_engineering", "academics"],
        "intent": "Review official university examination notice",
        "summary": "KIIT School of Computer Engineering official notice on faculty guide assignments for practical training",
        "importance": 4,
        "entities": [
            {"text": "KIIT School of Computer Engineering", "type": "institution"},
            {"text": "Prof. (Dr.) Biswajit Sahoo", "type": "person"},
        ],
        "dates": [
            {"text": "27/07/2026", "type": "notice_date"},
            {"text": "30/08/26", "type": "deadline"},
        ],
        "action_items": [
            {"text": "Submit training report and present before assigned faculty guide by 30/08/26", "status": "pending"}
        ],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": False,
            "contains_personal_contact": True,
        },
    },
    "ss_013.jpg": {
        "category": "Payment",
        "tags": ["payment", "receipt", "razorpay", "kiitconnect", "upi"],
        "intent": "Save payment confirmation receipt",
        "summary": "Razorpay UPI payment receipt of ₹99 for KIITConnect",
        "importance": 4,
        "entities": [
            {"text": "KIITConnect", "type": "merchant"},
            {"text": "pay_T8w4wW3Uiw5nu0", "type": "transaction_id"},
        ],
        "dates": [
            {"text": "Jul 3, 2026, 12:16 PM", "type": "transaction_date"}
        ],
        "action_items": [],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": True,
            "contains_personal_contact": False,
        },
    },
    "ss_014.jpg": {
        "category": "Travel",
        "tags": ["flight", "indigo", "boarding_pass", "guwahati", "bhubaneswar"],
        "intent": "Keep flight boarding pass details",
        "summary": "IndiGo flight boarding pass for MS Sneha Sinha from Guwahati to Bhubaneswar",
        "importance": 5,
        "entities": [
            {"text": "IndiGo", "type": "airline"},
            {"text": "Sneha Sinha", "type": "passenger"},
            {"text": "E6954C", "type": "pnr"},
        ],
        "dates": [
            {"text": "12 Jul, 26", "type": "travel_date"}
        ],
        "action_items": [
            {"text": "Board flight 6E 6914 at 17:10", "status": "pending"}
        ],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": False,
            "contains_personal_contact": True,
        },
    },
    "ss_015.jpg": {
        "category": "Shopping",
        "tags": ["instamart", "swiggy", "reward", "coupon", "shopping"],
        "intent": "Claim Swiggy Instamart reward coupon",
        "summary": "Swiggy Instamart scratch card reward coupon code",
        "importance": 3,
        "entities": [
            {"text": "Swiggy Instamart", "type": "merchant"},
            {"text": "ZPK523503390409", "type": "coupon_code"},
        ],
        "dates": [],
        "action_items": [
            {"text": "Redeem scratch card coupon code ZPK523503390409", "status": "pending"}
        ],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": False,
            "contains_personal_contact": False,
        },
    },
    "ss_016.jpg": {
        "category": "Travel",
        "tags": ["irctc", "train", "ticket", "travel", "tatkal"],
        "intent": "Review train ticket booking details before payment",
        "summary": "IRCTC train ticket booking review for Rny Scl Express AC 3 Tier Tatkal",
        "importance": 4,
        "entities": [
            {"text": "15611 Rny Scl Express", "type": "train"},
            {"text": "Kakali Sinha", "type": "passenger"},
        ],
        "dates": [
            {"text": "14 Jun, Sun, 20:25", "type": "travel_date"}
        ],
        "action_items": [
            {"text": "Proceed to payment for train booking", "status": "pending"}
        ],
        "sensitive_info": {
            "contains_password": True,
            "contains_payment_info": True,
            "contains_personal_contact": True,
        },
    },
    "ss_017.jpg": {
        "category": "Payment",
        "tags": ["payment", "receipt", "kiit", "sap", "netbanking"],
        "intent": "Keep university fee payment receipt",
        "summary": "KIIT SAP Indian Bank netbanking payment receipt for SNEHA SINHA",
        "importance": 4,
        "entities": [
            {"text": "KIIT SAP", "type": "merchant"},
            {"text": "BILL-SAP2606297704460", "type": "order_id"},
            {"text": "SNEHA_SINHA", "type": "person"},
        ],
        "dates": [],
        "action_items": [],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": True,
            "contains_personal_contact": True,
        },
    },
    "ss_018.jpg": {
        "category": "Education",
        "tags": ["education", "coursera", "elective", "course_registration", "email"],
        "intent": "Read open elective course selection guidelines",
        "summary": "Email notice to B.Tech students regarding Coursera Open Elective III course selection",
        "importance": 3,
        "entities": [
            {"text": "Coursera", "type": "platform"}
        ],
        "dates": [],
        "action_items": [
            {"text": "Select Open Elective III course on Coursera", "status": "pending"}
        ],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": False,
            "contains_personal_contact": False,
        },
    },
    "ss_019.jpg": {
        "category": "Travel",
        "tags": ["flight", "indigo", "boarding_pass", "guwahati", "dibrugarh"],
        "intent": "Keep flight boarding pass",
        "summary": "IndiGo flight boarding pass for MS Yoshita Bora from Guwahati to Dibrugarh",
        "importance": 5,
        "entities": [
            {"text": "IndiGo", "type": "airline"},
            {"text": "Yoshita Bora", "type": "passenger"},
            {"text": "AE986H", "type": "pnr"},
        ],
        "dates": [
            {"text": "13 Jun, 26", "type": "travel_date"}
        ],
        "action_items": [
            {"text": "Board flight 6E 6441 at 09:35", "status": "pending"}
        ],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": False,
            "contains_personal_contact": True,
        },
    },
    "ss_020.jpg": {
        "category": "Shopping",
        "tags": ["zapvi", "order", "shopping", "phone_cover", "receipt"],
        "intent": "Keep online phone cover order confirmation",
        "summary": "Zapvi online order completion receipt for Custom Phone Cover ₹259",
        "importance": 3,
        "entities": [
            {"text": "Zapvi", "type": "merchant"},
            {"text": "Custom Phone Cover", "type": "product"},
            {"text": "Yoshita Bora", "type": "person"},
        ],
        "dates": [],
        "action_items": [],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": True,
            "contains_personal_contact": True,
        },
    },
    "ss_022.jpg": {
        "category": "Shopping",
        "tags": ["metro_shoes", "shopping", "product", "shoes", "discount"],
        "intent": "Browse shoe products and promotional discount",
        "summary": "Metro Shoes product page for Women Pumps with prepaid discount offer",
        "importance": 2,
        "entities": [
            {"text": "Metro Shoes", "type": "brand"}
        ],
        "dates": [],
        "action_items": [],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": False,
            "contains_personal_contact": False,
        },
    },
    "ss_023.jpg": {
        "category": "Education",
        "tags": ["kiit", "exam", "question_paper", "statistics", "mapreduce"],
        "intent": "Study semester examination question paper",
        "summary": "KIIT End Semester Examination question paper covering MapReduce and Chi-square statistics",
        "importance": 3,
        "entities": [
            {"text": "KIIT", "type": "institution"}
        ],
        "dates": [],
        "action_items": [],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": False,
            "contains_personal_contact": False,
        },
    },
    "ss_024.jpg": {
        "category": "Education",
        "tags": ["kiit", "email", "lab_viva", "exam", "computer_engineering"],
        "intent": "Prepare for upcoming lab viva exam",
        "summary": "Email notice from Dr. Debanjan Pathak regarding computer engineering Lab Viva examination",
        "importance": 4,
        "entities": [
            {"text": "Dr. Debanjan Pathak", "type": "person"},
            {"text": "KIIT", "type": "institution"},
        ],
        "dates": [
            {"text": "27/10/2025", "type": "viva_date"}
        ],
        "action_items": [
            {"text": "Prepare full syllabus for Lab Viva starting 27/10/2025", "status": "pending"}
        ],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": False,
            "contains_personal_contact": True,
        },
    },
    "ss_025.jpg": {
        "category": "Entertainment",
        "tags": ["power_rangers", "movie", "wikipedia", "poster", "entertainment"],
        "intent": "View Power Rangers movie details",
        "summary": "Wikipedia movie page screenshot for Power Rangers theatrical film",
        "importance": 2,
        "entities": [
            {"text": "Power Rangers", "type": "movie"}
        ],
        "dates": [
            {"text": "MARCH 24", "type": "release_date"}
        ],
        "action_items": [],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": False,
            "contains_personal_contact": False,
        },
    },
    "ss_026.jpg": {
        "category": "Payment",
        "tags": ["kiit", "money_receipt", "tuition_fee", "hostel_fee", "payment"],
        "intent": "Keep official university fee money receipt",
        "summary": "KIIT School of Computer Engineering official Money Receipt for Yoshita Bora ₹288,500",
        "importance": 5,
        "entities": [
            {"text": "KIIT School of Computer Engineering", "type": "institution"},
            {"text": "YOSHITA BORA", "type": "person"},
            {"text": "1000197805", "type": "bp_number"},
        ],
        "dates": [
            {"text": "05.07.2025", "type": "receipt_date"}
        ],
        "action_items": [],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": True,
            "contains_personal_contact": True,
        },
    },
    "ss_027.jpg": {
        "category": "Education",
        "tags": ["formula", "placement_prep", "math", "careerride", "shortcuts"],
        "intent": "Reference math formulas for placement exams",
        "summary": "CareerRide placement exam algebra simplification shortcuts and formulas cheat sheet",
        "importance": 3,
        "entities": [
            {"text": "CareerRide", "type": "website"}
        ],
        "dates": [],
        "action_items": [],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": False,
            "contains_personal_contact": False,
        },
    },
    "ss_029.jpg": {
        "category": "Coding",
        "tags": ["devops", "terminal", "curl", "api", "gunicorn", "python"],
        "intent": "Test DevOps REST API endpoint via cURL",
        "summary": "Linux terminal output testing Flask/Gunicorn DevOps REST API endpoint via cURL",
        "importance": 3,
        "entities": [
            {"text": "devops-capstone-project", "type": "project"},
            {"text": "gunicorn", "type": "server"},
        ],
        "dates": [
            {"text": "2026-06-19", "type": "date"}
        ],
        "action_items": [],
        "sensitive_info": {
            "contains_password": False,
            "contains_payment_info": False,
            "contains_personal_contact": True,
        },
    },
}


def get_unseen_evaluation_targets() -> List[Dict[str, Any]]:
    """
    Retrieves the 22 genuinely unseen screenshot targets paired with their manually
    reviewed reference annotations.

    Returns:
        List of target dictionaries containing:
        - filename: str
        - screenshot_id: str
        - expected: dict (matching ScreenshotMetadata schema)
    """
    targets = []
    for filename in UNSEEN_TARGET_FILENAMES:
        # Strict isolation check
        if filename in DEVELOPMENT_BENCHMARK_FILENAMES:
            raise RuntimeError(f"Data leakage detected: benchmark target '{filename}' found in unseen dataset!")

        sid = filename.split(".")[0]
        expected = UNSEEN_GROUND_TRUTH_ANNOTATIONS.get(filename)
        if expected is None:
            raise ValueError(f"Missing manual reference annotation for unseen target '{filename}'")

        targets.append({
            "filename": filename,
            "screenshot_id": sid,
            "expected": expected,
        })

    return targets
