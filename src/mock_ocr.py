"""
Mock OCR Module
===============
Provides an abstract base interface for OCR engines and a deterministic mock OCR engine
implementation for development and testing without requiring real OCR models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Dict, Optional


class BaseOCREngine(ABC):
    """
    Abstract Base Class for OCR engines.
    Real OCR engines (Tesseract, EasyOCR, PaddleOCR) should implement this interface.
    """

    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from an image file.

        Args:
            image_path: Path to the image file.

        Returns:
            Extracted text content as a string.
        """
        pass


class MockOCREngine(BaseOCREngine):
    """
    Mock OCR engine providing deterministic simulated OCR text output grounded in actual screenshots.
    """

    DEFAULT_MOCK_MAPPINGS: Dict[str, str] = {
        "ss_001.jpg": (
            "Airtel_runu_7550\n\n"
            "Password\n"
            "Air@53054\n\n"
            "Show password\n\n"
            "Advanced options\n\n"
            "Cancel\n"
            "Connect"
        ),
        "ss_002.jpg": (
            "BRAINGROW EDUSERV PRIVATE LIMITED\n\n"
            "₹689.00\n"
            "Paid Successfully\n\n"
            "Payment Id: pay_SdleqofNvlmOZg\n"
            "Method: UPI\n"
            "sinhasneha158-2@okicici\n"
            "Paid On: 15th Apr, 2026 17:55:24 PM IST\n\n"
            "Email: sinhasneha158@gmail.com\n"
            "Mobile Number: +919395163762\n\n"
            "For any order related queries please reach out to BRAINGROW EDUSERV PRIVATE LIMITED at help@yesmock.com or on 9238123890"
        ),
        "ss_003.jpg": (
            "kiit.ac.in\n\n"
            "Course Contents:\n"
            "UNIT I - Deep Networks: Deep Feedforward Networks, Learning XOR, Gradient Based learning, Hidden Units, Back-propagation, Regularization for Deep Learning, Optimization for training Deep Models.\n\n"
            "UNIT II - Convolutional Networks: Convolution operation, Motivation, Pooling, Convolution and Pooling as strong prior, Efficient convolution algorithms, Sequence Modeling: Recurrent and Recursive Nets, LSTM Networks, Applications, Computer Vision, Speech Recognition, Natural Language Processing.\n\n"
            "UNIT III - Linear factor Models: Probabilistic PCA and Factor Analysis, Independent Component Analysis (ICA), Auto encoders, Regularized Auto encoders, Stochastic Auto encoders.\n\n"
            "UNIT IV - Representation Learning: Greedy Layer-wise Unsupervised Pre-Training, Transfer learning, Domain Adaptation, Deep Generative Models.\n\n"
            "UNIT V - Deep Learning with Python: Introduction to Keras and TensorFlow, Deep Learning for computer vision, convnets, Deep Learning for Text and Sequences, Generative Deep Learning, Text Generation with LSTM, DeepDream, Neural Style Transfer, Generative Adversarial Networks (GAN).\n\n"
            "Course Outcomes:\n"
            "CO1: Assess the concept of deep learning\n"
            "CO2: Identify the deep learning algorithms which are more appropriate for various types of learning tasks\n"
            "CO3: Incorporate transfer of knowledge in machine learning algorithms\n"
            "CO4: Implement deep learning algorithms and solve real-world problems\n"
            "CO5: Develop Deep Learning techniques using Python\n"
            "CO6: Represent learning Models\n\n"
            "Textbooks:\n"
            "1. Ian Goodfellow, Yoshua Bengio, Aaron Courville, \"Deep Learning\", The MIT Press, 2016.\n\n"
            "Reference Books:\n"
            "1. Francois Chollet, \"Deep Learning with Python\", Manning Publications, 2017.\n"
            "2. Aurélien Géron, \"Hands-On Machine Learning with Scikit-Learn and TensorFlow\", O'Reilly Media, 2017.\n"
            "3. Josh Patterson, \"Deep Learning: A Practitioner's Approach\", O'Reilly Media."
        ),
        "ss_004.jpg": (
            "Spider-Man: Brand New Day\n"
            "UA13+ | English | 3D\n\n"
            "Sunday 2 Aug | 10:55 AM\n\n"
            "Scan this QR code at theatre\n\n"
            "SCREEN 3\n"
            "QR - L7, L8, L9, L10, L11\n\n"
            "PVR Utkal Galleria, Gautam Nagar, Bhubaneswar\n\n"
            "Booking ID: TRAYD98\n\n"
            "#SeeYouThere\n"
            "district BY ZOMATO"
        ),
        "ss_005.jpg": (
            "₹1,800.00\n\n"
            "Paid to\n"
            "IPSITA DILIP WAKHARKAR\n"
            "Paytm • 9881732186@pthdfc\n\n"
            "28 July 2026, 11:34 am"
        ),
        "ss_006.jpg": (
            "Hansdhwani\n"
            "Teentaal Bandish - 1 [150 BPM]\n\n"
            "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16\n\n"
            "Re ss Ni Pa Sa Ni Re Sa | Sa ss Ni Pa Ga Re Sa Ni\n"
            "Ga Re Ga Pa Ga Re Sa ss | Ga Re Sa Ni Pa Ni Sa Re\n"
            "Sa ss Sa Sa Ni Re Sa ss | Pa ss Ga Pa Ni ss Pa Ni\n"
            "Sa ss Ni Pa Ga Re Sa ss | Ni Sa Ga Re Sa Ni Pa ss"
        ),
        "ss_007.jpg": (
            "The faculty guides for students of 7th semester for the evaluation of Practical Training course are attached herewith. The students must present the practical training before the guide.\n"
            "You are requested to evaluate the Practical Training course on or before examinations (07/09/26). To bring a uniformity in the marking scheme, the following guide referred for awarding the marks:\n\n"
            "Components | Marks\n"
            "Certificate | 10\n"
            "Day-wise Record/Report | 20\n"
            "Content Delivery | 40\n"
            "Question Answer | 20\n"
            "Work Quality | 10\n\n"
            "You are requested to write the date and time of evaluation to the students."
        ),
        "ss_008.jpg": (
            "Taste the Fire, Love the Flavor\n"
            "MD COMBO & KEBAB POINT\n"
            "COMBOS • STICK KEBABS • FRIED SNACKS\n\n"
            "COMBO ITEMS\n"
            "1. 2 Plane Paratha with Chicken Kasa ........ ₹90\n"
            "2. 2 Plane Paratha with Chicken Kima ........ ₹90\n"
            "3. 2 Plane Paratha with Chole Paneer ........ ₹80\n"
            "4. 2 Plane Paratha with Mashrum Masala ........ ₹80\n"
            "5. 2 Plane Paratha with Alu Kasa ........ ₹60\n"
            "+ Extra Normal Paratha ........ ₹20\n"
            "+ Extra Lacha Paratha ........ ₹25\n\n"
            "KEBAB ITEMS\n"
            "1. Chicken Tikka Kebab ........ ₹50\n"
            "2. Chicken Garlic Kebab ........ ₹50\n"
            "3. Chicken Hariyali Kebab ........ ₹50\n"
            "4. Chicken Malai Kebab ........ ₹50\n"
            "5. Prawn Kebab ........ ₹80\n"
            "6. Chicken Tandoori Leg Piece ........ ₹120\n"
            "7. Chicken Tandoori Wings ........ ₹50\n\n"
            "LEMON RICE COMBOS\n"
            "1. Lemon Rice with Chicken Kima ........ ₹100\n"
            "2. Lemon Rice with Chicken Kasa ........ ₹100\n"
            "3. Lemon Rice with Chole Paneer ........ ₹90\n"
            "4. Lemon Rice with Mushroom Masala ........ ₹90\n"
            "5. Lemon Rice with Alu Kasa ........ ₹80\n"
            "+ Extra Rice Plate ........ ₹40\n\n"
            "PURI ITEMS (4 PCS EACH)\n"
            "1. Puri (4 Pcs) + Chicken Kasa ........ ₹90\n"
            "2. Puri (4 Pcs) + Chicken Kima ........ ₹90\n"
            "3. Puri (4 Pcs) + Alu Kasa ........ ₹60\n"
            "4. Puri (4 Pcs) + Chole Paneer ........ ₹80\n"
            "5. Puri (4 Pcs) + Mashrum Masala ........ ₹80\n\n"
            "FOR ORDERS CALL / WHATSAPP: 78948 78844\n"
            "MD COMBO AND KEBAB POINT, IN FRONT OF CAMPUS 25, KIIT, NEAR OTV OFFICE, BHUBANESWAR, PRASHANTI VIHAR - 751024"
        ),
        "ss_009.jpg": (
            "UniConverter Try Now\n\n"
            "1. O2TvSeries\n"
            "URL: https://o2tvseries4.com\n\n"
            "• Legality: Risky / Unclear\n"
            "• Free Plan: Yes\n"
            "• Android Support: Yes, Mobile-Friendly\n"
            "• Full-Season Download: Yes\n"
            "• Formats: MP4 / 3GP\n"
            "• Best For: Mobile users who want lightweight episode or season downloads\n\n"
            "O2TvSeries is a free platform that allows you to download almost any TV series you can think of. Its content library is filled with over 1200 TV series titles complete with most of their released seasons and episodes.\n\n"
            "UniConverter Online Free"
        ),
        "ss_010.jpg": (
            "My Orders\n\n"
            "GET ₹1,250 WELCOME VOUCHERS*\n"
            "With Flipkart SBI Credit Card\n"
            "Apply now >\n\n"
            "Search your... Filters\n"
            "All Flipkart Grocery Minutes\n\n"
            "Delivery expected by Aug 12\n"
            "Your Order has been placed\n\n"
            "Delivered on Jul 28\n"
            "Flipkart SmartBuy NextGen Mobile Stand\n"
            "Rate & Review\n\n"
            "Delivered on Jul 31\n"
            "Sai Essentials Universal Mobile Stand\n"
            "Rate & Review"
        ),
        "ss_011.jpg": (
            "Boarding Pass 6Eskai\n\n"
            "GUWAHATI (T1) GAU\n"
            "01h 45m Non-stop\n"
            "BHUBANESWAR (T1) BBI\n\n"
            "DEPARTS DATE: 12 Jul, 26\n"
            "ARRIVES DATE: 12 Jul, 26\n\n"
            "DEPARTS: 17:55\n"
            "ARRIVES: 19:40\n\n"
            "MS Sneha Sinha\n"
            "TIER: INDIGO BLUCHIP BLU 3\n"
            "FFN: 549114775\n\n"
            "PNR: E6954C\n"
            "Flight: 6E 6914\n"
            "Date: 12 Jul, 26\n\n"
            "Gate: -\n"
            "Boarding Time: 17:10\n"
            "Departure: 17:55\n\n"
            "Seat: 16E\n"
            "Seq. no: 0002\n"
            "Boarding: Zone 2"
        ),
        "ss_012.jpg": (
            "SCHOOL OF COMPUTER ENGINEERING\n"
            "KALINGA INSTITUTE OF INDUSTRIAL TECHNOLOGY (KIIT)\n"
            "(Deemed to be University, u/s 3 of UGC Act 1956)\n\n"
            "Ref No: KIIT-DU/SCOMP/164/26\n"
            "Date: 27/07/2026\n\n"
            "NOTICE\n\n"
            "Sub: Allotment of faculty guides for Practical Training (CS-48001) evaluation\n\n"
            "It is hereby informed to the B.Tech 7th semester students of School of Computer Engineering that faculty guides have been assigned to evaluate the \"Practical Training\". Students are required consult the faculty guide as attached in the annexure to take a suitable date & time for presentation of the practical training before the faculty guide. The presentation must be in-person and to be finished by 30/08/26.\n\n"
            "Distribution of the 100 marks is as follows:\n"
            "• Participation during others presentation under a faculty guide: 10\n"
            "• Preparation of Training Diary/Report: 20\n"
            "• Presentation: 40\n"
            "• Depth of Technical knowhow: 30\n\n"
            "Note, the students must submit the training report/certificates to their respective faculty guides. A student must obtain a written permission (mail accepted) from the faculty guide for an extension, if she/he will not be able to present within the deadline.\n\n"
            "Prof. (Dr.) Biswajit Sahoo\n"
            "DIRECTOR GENERAL\n"
            "School of Computer Engineering\n"
            "Kalinga Institute of Industrial Technology\n\n"
            "Encl: Annexure\n"
            "Copy to: All faculty guides/ Deans\n\n"
            "School of Computer Engineering, KIIT Deemed to be University\n"
            "Campus-25, KIIT, Bhubaneswar, Odisha, India, Pin - 751024\n"
            "Email: director.csit@kiit.ac.in, Website: https://cse.kiit.ac.in/"
        ),
        "ss_013.jpg": (
            "board.kiitconnect.com\n\n"
            "You will be redirected in 3 seconds\n"
            "Payment Successful\n\n"
            "KIITConnect ₹99\n"
            "Jul 3, 2026, 12:16 PM\n"
            "UPI | pay_T8w4wW3Uiw5nu0\n\n"
            "Visit razorpay.com/support for queries\n"
            "Secured by Razorpay"
        ),
        "ss_014.jpg": (
            "Boarding Pass 6Eskai\n\n"
            "GUWAHATI (T1) GAU\n"
            "01h 45m Non-stop\n"
            "BHUBANESWAR (T1) BBI\n\n"
            "DEPARTS DATE: 12 Jul, 26\n"
            "ARRIVES DATE: 12 Jul, 26\n\n"
            "DEPARTS: 17:55\n"
            "ARRIVES: 19:40\n\n"
            "MS Sneha Sinha\n"
            "TIER: INDIGO BLUCHIP BLU 3\n"
            "FFN: 549114775\n\n"
            "PNR: E6954C\n"
            "Flight: 6E 6914\n"
            "Date: 12 Jul, 26\n\n"
            "Gate: -\n"
            "Boarding Time: 17:10\n"
            "Departure: 17:55\n\n"
            "Seat: 16E\n"
            "Seq. no: 0002\n"
            "Boarding: Zone 2"
        ),
        "ss_015.jpg": (
            "Instamart order\n"
            "05:30 PM • 5 items\n\n"
            "5% cashback on all UPI spends\n\n"
            "airtel POSTPAID\n"
            "Up to 2X speeds\n\n"
            "ZPK523503390409\n\n"
            "Claim Now\n\n"
            "SCRATCH CARD\n"
            "Reveal a reward\n\n"
            "Copied to clipboard.\n"
            "Coupon code copied!"
        ),
        "ss_016.jpg": (
            "Review\n\n"
            "15611 Rny Scl Express\n"
            "AC 3 Tier (3A) | Tatkal Quota\n"
            "AVL 0039\n\n"
            "Your Boarding Details\n"
            "GHY Guwahati\n"
            "14 Jun, Sun, 20:25\n"
            "Change\n\n"
            "IRCTC User ID\n"
            "sujay13ji\n\n"
            "Create New ID + | bingu2018\n\n"
            "Your IRCTC password will be required to complete this booking\n\n"
            "Passenger Details\n"
            "Kakali Sinha, 62 (F)\n"
            "Lower Berth\n\n"
            "Proceed to payment"
        ),
        "ss_017.jpg": (
            "KIIT INDIAN BANK PAYMENT GATEWAY\n\n"
            "Please Don't Refresh this page Take Printout & Screenshot for your Records!!!\n"
            "Please Don't Try to go back and pay Again, Always Start Fresh New Payment from KIIT Website or SAP.\n\n"
            "We are now Redirecting to KIIT SAP Payment Website\n"
            "You have 25 seconds to take Print.\n\n"
            "Thank you for Payment with us.\n"
            "Your transaction is Successful.\n\n"
            "Order ID: BILL-SAP2606297704460\n"
            "Tracking ID: DSBIOO51O9USCV\n"
            "Bank Ref No: IGAUEFEPB1\n"
            "Full Name: SNEHA_SINHA\n"
            "Order Status: success\n"
            "Payment Mode: netbanking\n"
            "Status: Transaction Successful"
        ),
        "ss_018.jpg": (
            "Regarding open elective-III booking-Slot 1 External\n"
            "Inbox\n\n"
            "Regis... 5 days ago\n"
            "to Rajat, Anu...\n\n"
            "Dear Students (2023 admitted B.Tech),\n\n"
            "You need to select a course for Open Elective - III in 7th semester from COURSERA. The number of seats in all the courses is unlimited. The students are advised to choose the courses wisely as they will not be allowed to change the courses later.\n\n"
            "COURSERA: Slot 1 students"
        ),
        "ss_019.jpg": (
            "Boarding Pass 6Eskai\n\n"
            "GUWAHATI (T1) GAU\n"
            "00h 55m Non-stop\n"
            "DIBRUGARH (T1) DIB\n\n"
            "DEPARTS DATE: 13 Jun, 26\n"
            "ARRIVES DATE: 13 Jun, 26\n\n"
            "DEPARTS: 10:20\n"
            "ARRIVES: 11:15\n\n"
            "MS YOSHITA BORA\n\n"
            "PNR: AE986H\n"
            "Flight: 6E 6441\n"
            "Date: 13 Jun, 26\n\n"
            "Gate: -\n"
            "Boarding Time: 09:35\n"
            "Departure: 10:20\n\n"
            "Seat: 3B\n"
            "Seq. no: 0062\n"
            "Boarding: Zone 3\n\n"
            "Special Services: CPTR, NUSW"
        ),
        "ss_020.jpg": (
            "zapvi.in\n"
            "Hurry! Add More to Unlock Free Delivery — Ends Today!\n"
            "Zapvi\n\n"
            "ORDER COMPLETE\n\n"
            "Order details\n"
            "PRODUCT | TOTAL\n"
            "Custom Photo Printed Glossy Metal Phone Cover for Realme 11X (5G) × 1 | ₹199.00\n"
            "Subtotal: ₹199.00\n"
            "Shipping: ₹60.00 via Flat rate\n"
            "Payment method: Cashfree Payments\n"
            "Total: ₹259.00\n\n"
            "Billing address\n"
            "Yoshita Bora\n"
            "Qc 20, Campus 25, KIIT School of Computer Engineering(New Block), Prashanti Vihar\n"
            "Bhubaneswar, 751024\n"
            "Odisha\n"
            "919101896722\n\n"
            "yoshita.bora15@gmail.com"
        ),
        "ss_021.jpg": (
            "10:56 Wed, 6 May\n\n"
            "WhatsApp • 4 messages from 3 chats\n\n"
            "3 Idiots 🙈🙈🙈 • 4 minutes ago\n"
            "Shalini\n"
            "Arey 25 boli hai lekin ladka dhundne mein toh lagega na 2-3 saal ....ismein ho jayega Mera kaam 😂\n\n"
            "Reply | Mark as read | Mute\n\n"
            "Milli(Roommate) • Yesterday\n"
            "💀💀yeh to pata nahi kya issue ho raha\n\n"
            "EaseMyTrip • Yesterday\n"
            "📷 Dear Traveller,...\n\n"
            "Telegram • Anu • 21 minutes ago\n"
            "Anu joined Telegram!"
        ),
        "ss_022.jpg": (
            "Buy Metro...\n"
            "metroshoes.com\n"
            "Get 5% Extra Discount On Prepaid Orders\n\n"
            "METRO\n\n"
            "What are you looking for..\n\n"
            "Women Shoes / Women Pumps\n\n"
            "Metro\n\n"
            "ADD TO CART\n"
            "QUICK CHECKOUT\n"
            "Extra 5% discount on prepaid orders"
        ),
        "ss_023.jpg": (
            "KIIT-CONNECT PDF Viewer\n"
            "Page 1 of 7\n\n"
            "ID | Name | City | State\n"
            "1 | Adarsh | Cuttack | Odisha\n"
            "2 | Ahalya | Puri | Odisha\n"
            "3 | Aishwarya | Cuttack | Odisha\n"
            "4 | Akasha | Puri | Odisha\n"
            "5 | Akhila | Baripada | Odisha\n"
            "6 | Salman | Paradeep | Odisha\n"
            "7 | Fatima | Balasore | Odisha\n"
            "8 | Aaden | Sambalpur | Odisha\n\n"
            "The process has to capture input, split, map, shuffle, reduce, and result stages by depicting <key, value> pairs.\n\n"
            "KIIT-DU/2024/SOT/Spring End Semester Examination-2024 (3)\n\n"
            "3. (a) Imagine that in an article you read the statement that babies, on average, play 50 hours per week...\n\n"
            "Table 5: Development of recycling interventions\n"
            "Intervention | Recycles | Does not recycles\n"
            "Flyer | 50 | 75\n"
            "Phone call | 125 | 175\n"
            "Control | 90 | 30\n"
            "Waste | 45 | 10\n\n"
            "Using the Chi-square test for independence, determine whether two categorical variables (intervention and recycles) relate to each other or not by calculating the degree of freedom and articulating the hypotheses."
        ),
        "ss_024.jpg": (
            "Regarding Lab Viva Inbox\n\n"
            "Debanjan Pathak 9:39 am\n"
            "to 2306001, 2306002, 2...\n\n"
            "Dear students,\n"
            "From next lab (27/10/2025) onwards we will have lab viva. viva will be taken with a random Roll Number. Be prepared.\n"
            "Max Mark: 10.\n"
            "Syllabus: Full Syllabus\n\n"
            "--\n"
            "Thanks and Regards\n"
            "Dr. Debanjan Pathak\n"
            "M.Tech CSE, IIT(ISM) Dhanbad ; Ph.D. CSE , NIT Warangal\n"
            "Assistant Professor\n"
            "School Of Computer Engineering\n"
            "KIIT Deemed to be University\n"
            "(Institute of Eminence)\n"
            "Bhubaneswar - 751024. ODISHA\n\n"
            "KALINGA INSTITUTE OF INDUSTRIAL TECHNOLOGY\n\n"
            "Go Green: Kindly don't print this unless so required.\n\n"
            "Visit us @ http://www.kiit.ac.in"
        ),
        "ss_025.jpg": (
            "en.wikipedia.org\n\n"
            "IT'S MORPHIN TIME\n"
            "POWER RANGERS\n\n"
            "IN THEATERS MARCH 24\n\n"
            "Power Rangers (film) - Wikipedia\n"
            "Visit >\n"
            "Images may be subject to copyright. Learn more\n"
            "Share | Save"
        ),
        "ss_026.jpg": (
            "KALINGA INSTITUTE OF INDUSTRIAL TECHNOLOGY (KIIT)\n"
            "DEEMED TO BE UNIVERSITY\n\n"
            "SCHOOL OF COMPUTER ENGINEERING\n"
            "AT/PO-KIIT, Bhubaneshwar - 751024, Odisha\n"
            "Tel. 0674-2725272,2742103,2725347\n"
            "MONEY RECEIPT\n\n"
            "Ref.No: SCSE/00-01/100000154356 Date: 05.07.2025\n"
            "BP No: 1000197805\n\n"
            "Name: YOSHITA BORA\n"
            "Roll No: 2306164\n"
            "Course: B.Tech.\n"
            "Batch: 2023\n"
            "Branch: Information Technology\n\n"
            "Sl.No | Particulars | Semester | Amount (INR)\n"
            "1 | Registration Fee | 2025-Autumn | 1,000.00\n"
            "2 | LAUNDRY FEE | 2024-Spring | 500.00\n"
            "3 | Institutional Fee | 2025-Autumn | 192,000.00\n"
            "4 | Hostel Fee | 2025-Autumn | 70,000.00\n"
            "5 | Mess Fee | 2025-Autumn | 25,000.00\n"
            "Total | 288,500.00\n\n"
            "Amount in Words: Rupees Two Lakh Eighty Eight Thousand Five Hundred Only\n\n"
            "CHQ/DD/UTR No. | Date | Drawn On | Amount (INR)\n"
            "BSBIOB50PH0RNN | 05.07.2025 | Bill Desk | 288,500.00\n"
            "Total | 288,500.00\n\n"
            "Cheques / DDs are subject to realisation.\n"
            "This is a computer generated receipt & doesn't require seal or signature"
        ),
        "ss_027.jpg": (
            "Simplification - Shortcuts & Tricks for 2026 Placement Tests, Job Interviews & Exams\n\n"
            "Basic Formulae\n\n"
            "(a + b)^2\n"
            "• (a + 1/a)^2 = (a^2 + 1/a^2) + 2 = (a - 1/a)^2 + 4\n"
            "• (a - 1/a)^2 = (a^2 + 1/a^2) - 2 = (a + 1/a)^2 - 4\n"
            "• (a + 1/a)^3 = (a^3 + 1/a^3) + 3(a + 1/a)\n"
            "• (a - 1/a)^3 = (a^3 - 1/a^3) - 3(a - 1/a)\n\n"
            "CareerRide.com"
        ),
        "ss_028.jpg": (
            "assetController.js - Internship - Visual Studio Code\n\n"
            "EXPLORER\n"
            "asset-management-system > backend > controllers > JS assetController.js\n\n"
            "const Asset = require(\"../models/Asset\");\n\n"
            "// Get all assets\n"
            "const getAssets = async (req, res) => {\n"
            "    try {\n"
            "        const assets = await Asset.find();\n"
            "        res.json(assets);\n"
            "    } catch (error) {\n"
            "        res.status(500).json({\n"
            "            message: error.message\n"
            "        });\n"
            "    }\n"
            "};\n\n"
            "// Add asset\n"
            "const addAsset = async (req, res) => {\n"
            "    try {\n"
            "        const newAsset = new Asset(req.body);\n"
            "        const savedAsset = await newAsset.save();\n"
            "        res.status(201).json(savedAsset);\n"
            "    } catch (error) {\n"
            "        res.status(500).json({\n"
            "            message: error.message\n"
            "        });\n"
            "    }\n"
            "};"
        ),
        "ss_029.jpg": (
            "theia:devops-capstone-project\n"
            "(venv) theia:project$ curl -i -X POST http://127.0.0.1:5000/accounts \\\n"
            "-H \"Content-Type: application/json\" \\\n"
            "-d '{\"name\":\"John Doe\",\"email\":\"john@doe.com\",\"address\":\"123 Main St.\", \"phone_number\":\"555-1212\"}'\n\n"
            "HTTP/1.1 201 CREATED\n"
            "Server: gunicorn\n"
            "Date: Fri, 19 Jun 2026 15:28:21 GMT\n"
            "Connection: close\n"
            "Content-Type: application/json\n"
            "Content-Length: 128\n"
            "Location: /\n\n"
            "{\"address\":\"123 Main St.\",\"date_joined\":\"2026-06-19\",\"email\":\"john@doe.com\",\"id\":1,\"name\":\"John Doe\",\"phone_number\":\"555-1212\"}"
        ),
        "ss_030.jpg": (
            "Apply now!\n"
            "2nd Global AI Hackathon\n"
            "in collaboration with the MIT Sloan AI Club\n"
            "Aug 9 - 10, 2025\n"
            "Prizes sponsored by OpenAI\n\n"
            "Hi Nafisa,\n\n"
            "We're excited to share that OpenAI is sponsoring the 2nd Global AI Hackathon (Aug 9 - 10) and would love to invite you to apply by July 18 (round 1).\n\n"
            "We had received so much interest for the first hackathon, and it was a great success such that we are excited to provide an additional opportunity.\n\n"
            "Apply here: https://tinyurl.com/APPLY-AI\n\n"
            "We're back with new AI challenges, inspiring speakers, hands-on mentors, and a global community of AI builders.\n\n"
            "Why join?\n"
            "• Win prize money and OpenAI API credits\n"
            "• Gain exposure to top AI companies that are looking to hire\n"
            "• Be selected for an exclusive Demo Day in September - we'll connect you with funding opportunities and VCs to help turn your solution into a venture-backed startup"
        ),
        "screenshot_001.png": (
            "WhatsApp chat with John\n\n"
            "Meeting scheduled at 5 PM tomorrow."
        ),
        "screenshot_002.png": (
            "Amazon Order\n\n"
            "Package delivered successfully."
        ),
    }

    FALLBACK_TEXT: str = "[MOCK OCR] No predefined text."

    def __init__(self, mapping: Optional[Dict[str, str]] = None):
        """
        Initialize Mock OCR Engine with custom or default text mapping.

        Args:
            mapping: Optional dictionary mapping filenames (e.g. "ss_001.jpg") to text.
        """
        self.mapping: Dict[str, str] = (
            mapping if mapping is not None else self.DEFAULT_MOCK_MAPPINGS
        )

    def extract_text(self, image_path: str) -> str:
        """
        Deterministically returns mock OCR text for a given image path based on filename.

        Args:
            image_path: Path to screenshot image.

        Returns:
            Simulated OCR text output string.
        """
        path_obj = Path(image_path)
        filename = path_obj.name
        stem = path_obj.stem

        # Build search keys (exact filename, stem, zero-padding normalized variants)
        keys_to_check = [filename, stem]

        # Normalization logic: ss_0010.jpg -> ss_010.jpg
        match = re.match(r"ss_0*(\d+)(\..+)?", filename)
        if match:
            num = match.group(1)
            ext = match.group(2) or ""
            keys_to_check.append(f"ss_{int(num):03d}{ext}")
            keys_to_check.append(f"ss_{int(num):03d}")
            keys_to_check.append(f"ss_{int(num):02d}{ext}")
            keys_to_check.append(f"ss_{int(num):02d}")

        for key in keys_to_check:
            if key in self.mapping:
                return self.mapping[key]

        return self.FALLBACK_TEXT
