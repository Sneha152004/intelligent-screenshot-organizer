# Category Decision Policy V1 Specification

## 1. Overview & Core Purpose

This document defines **Category Decision Policy V1**, an explicit, rule-based semantic decision framework designed to resolve category boundaries, topic-vs-layout ambiguities, and false-positive domain predictions across the 10 canonical screenshot categories.

---

## 2. Global Reasoning Principle: Primary Object & Purpose

> **Categorize the screenshot according to the primary object and purpose represented by the screenshot, not merely any secondary topic mentioned inside it.**

### Diagnostic Case Framework:

1. **Case A: Job Offer Email / Chat**
   - *Scenario*: Email or chat message from a recruiter discussing a job opportunity.
   - *Primary Object*: Email / Chat Correspondence (`Communication`).
   - *Rule*: Messaging frame (`From:`, `Subject:`, `Inbox`, chat bubbles) defines `Communication`. If the layout is a job vacancy portal (`LinkedIn Jobs`, `Career Portal`, `Job Application Form`), classify as `Work & Career`.

2. **Case B: Product Purchase Email vs Product Catalog**
   - *Scenario*: Email containing a product promotion vs an e-commerce app catalog.
   - *Rule*: E-commerce product page with `Add to cart` / `Buy now` $\rightarrow$ `Shopping`. Email thread $\rightarrow$ `Communication`. Completed financial receipt with `Transaction ID` / `Payment Successful` $\rightarrow$ `Payment`.

3. **Case C: Payment Confirmation vs Shopping Cart**
   - *Scenario*: Shopping checkout vs bank transfer receipt.
   - *Rule*: Product selection prior to payment $\rightarrow$ `Shopping`. Financial receipt / payment confirmation $\rightarrow$ `Payment`.

4. **Case D: Course Chat Discussion vs Textbook/Lecture**
   - *Scenario*: Chat message discussing an assignment vs textbook lecture slides.
   - *Rule*: Chat thread UI $\rightarrow$ `Communication`. Course syllabus, lecture notes, textbook, exam questions $\rightarrow$ `Education`.

---

## 3. Comprehensive Category Definitions & Decision Rules

### Communication
- **Primary Purpose**: Direct messaging, email threads, chat conversations, and social comment threads.
- **Strong Positive Indicators**: `subject:`, `reply to`, `inbox`, `whatsapp`, `telegram`, `from:`, `to:`, `cc:`, `chat bubble`, `unread messages`.
- **Supporting Indicators**: `message`, `email`, `sender`, `recipient`, `conversation`, `mail`, `regards`, `dear`.
- **Negative Indicators**: `job vacancy`, `apply now`, `add to cart`, `payment successful`, `boarding pass`, `source code`, `def `.
- **Exclusion Conditions**: Standalone payment receipt without email frame; standalone job posting page; standalone flight ticket.
- **Tie-Breaking Rule**: If email/chat frame is dominant and intent is correspondence, choose `Communication` over embedded body topic.

---

### Work & Career
- **Primary Purpose**: Job listings, career applications, recruitment, resumes, professional workplace dashboards, and corporate management.
- **Strong Positive Indicators**: `job application`, `job vacancy`, `campus placement`, `interview invitation`, `applicant`, `recruitment`, `resume`, `cv`, `salary`, `job title`.
- **Supporting Indicators**: `job`, `career`, `hiring`, `consultant`, `role`, `offer`, `candidate`, `position`, `workplace`, `employee`.
- **Negative Indicators**: `academic lecture`, `online course`, `add to cart`, `wifi settings`, `boarding pass`.
- **Exclusion Conditions**: Academic university course syllabus or assignment; e-commerce product page.
- **Tie-Breaking Rule**: If the primary view is a job posting or applicant tracking page, choose `Work & Career` over `Education` or `General`.

---

### Shopping
- **Primary Purpose**: E-commerce product listings, shopping carts, product detail pages, and retail catalogs.
- **Strong Positive Indicators**: `add to cart`, `buy now`, `product details`, `free shipping`, `customer reviews`, `order summary`, `delivery address`, `wishlist`.
- **Supporting Indicators**: `cart`, `product`, `price`, `buy`, `order`, `shipping`, `item`, `store`, `size`, `rupees`, `₹`, `discount`, `rating`.
- **Negative Indicators**: `transaction successful`, `payment successful`, `upi ref`, `debited from`, `boarding pass`, `email inbox`.
- **Exclusion Conditions**: Completed bank transaction receipt (belongs to `Payment`); job vacancy posting.
- **Tie-Breaking Rule**: If the screen shows purchase intent before checkout (`Add to cart`, `Buy now`), choose `Shopping` over `Payment`.

---

### Payment
- **Primary Purpose**: Digital payment confirmations, receipts, invoices, bank transfers, and UPI transaction status screens.
- **Strong Positive Indicators**: `payment successful`, `transaction successful`, `money transferred`, `paid successfully`, `upi id`, `ref no`, `transaction id`, `invoice total`, `debited from`.
- **Supporting Indicators**: `paid`, `payment`, `upi`, `transaction`, `receipt`, `transfer`, `bank`, `amount`, `billing`, `invoice`, `account`, `card`.
- **Negative Indicators**: `add to cart`, `buy now`, `product specifications`, `job vacancy`.
- **Exclusion Conditions**: Shopping cart prior to checkout; generic price listing without transaction proof.
- **Tie-Breaking Rule**: If transaction completion status or banking reference number is present, choose `Payment` over `Shopping`.

---

### Education
- **Primary Purpose**: Academic course material, lecture slides, textbooks, study notes, flashcards, exams, and educational quizzes.
- **Strong Positive Indicators**: `online course`, `practice questions`, `lecture notes`, `submit assignment`, `syllabus`, `quiz`, `classroom`, `coursera`, `udemy`, `homework`.
- **Supporting Indicators**: `lecture`, `course`, `university`, `notes`, `exam`, `study`, `student`, `lesson`, `algorithm`, `learn`, `class`, `school`.
- **Negative Indicators**: `job application`, `salary`, `add to cart`, `payment successful`, `git commit`.
- **Exclusion Conditions**: Raw source code editor or IDE terminal view (belongs to `Coding`); corporate job posting page.
- **Tie-Breaking Rule**: If content is academic or tutorial material explaining concepts, choose `Education` over `Work & Career`.

---

### Coding
- **Primary Purpose**: Source code editors, IDE interfaces, developer terminals, git commits, compiler errors, and code tracebacks.
- **Strong Positive Indicators**: `vs code`, `vscode`, `error traceback`, `commit changes`, `pull request`, `source code`, `code editor`, `git commit`, `git push`, `stack trace`, `compiler error`, `def `, `class `, `npm install`, `pip install`.
- **Supporting Indicators**: `python`, `import`, `code`, `terminal`, `git`, `html`, `javascript`, `repository`, `syntax`, `developer`, `ide`, `script`.
- **Negative Indicators**: `add to cart`, `payment successful`, `boarding pass`, `job application`.
- **Exclusion Conditions**: Textbook prose discussing coding conceptually without code syntax/editor layout.
- **Tie-Breaking Rule**: If code syntax, IDE controls, terminal traceback, or git operations are visible, choose `Coding` over `Education`.

---

### Travel
- **Primary Purpose**: Flight tickets, airline bookings, boarding passes, hotel reservations, train/bus tickets, and travel itineraries.
- **Strong Positive Indicators**: `boarding pass`, `pnr`, `flight number`, `travel itinerary`, `flight booking`, `hotel booking`, `gate number`, `seat number`, `irctc`, `indigo`.
- **Supporting Indicators**: `flight`, `booking`, `airline`, `hotel`, `train`, `seat`, `itinerary`, `trip`, `fare`, `terminal`, `passenger`, `ticket`.
- **Negative Indicators**: `add to cart`, `job application`, `source code`, `wifi settings`.
- **Exclusion Conditions**: Generic city news or map screenshot without booking/ticket data.
- **Tie-Breaking Rule**: If screen displays ticket/PNR itinerary details, choose `Travel` over `Payment` or `Communication`.

---

### Connectivity
- **Primary Purpose**: Network configuration, Wi-Fi settings, Bluetooth pairing, IP/MAC addresses, and connection status.
- **Strong Positive Indicators**: `connected to wi-fi`, `network settings`, `bluetooth devices`, `available networks`, `hotspot settings`, `ip address`, `mac address`, `ssid`.
- **Supporting Indicators**: `wifi`, `network`, `signal`, `connected`, `discovering`, `wireless`, `device name`.
- **Negative Indicators**: `job application`, `add to cart`, `payment successful`, `boarding pass`.
- **Exclusion Conditions**: Generic phone system settings without network configuration.
- **Tie-Breaking Rule**: If screen is network/Wi-Fi/Bluetooth setup, choose `Connectivity` over `General`.

---

### Entertainment
- **Primary Purpose**: Video streaming, music playback, media player libraries, gaming, movies, and entertainment media.
- **Strong Positive Indicators**: `watch now`, `play episode`, `listen now`, `now playing`, `official soundtrack`, `spotify`, `youtube`, `deviantart`.
- **Supporting Indicators**: `music`, `video`, `movie`, `play`, `song`, `game`, `artist`, `reel`, `media`, `streaming`.
- **Negative Indicators**: `online course`, `lecture notes`, `job application`, `add to cart`.
- **Exclusion Conditions**: Educational video lecture (belongs to `Education`); product ad video in shopping app.
- **Tie-Breaking Rule**: If media player controls or music/movie streaming UI is visible, choose `Entertainment` over `General`.

---

### General
- **Primary Purpose**: Default fallback for smartphone home screens, clock, calculator, weather, and unclassified screenshots.
- **Strong Positive Indicators**: `weather forecast`, `alarm set for`, `calculator result`, `clock`, `home screen wallpaper`.
- **Supporting Indicators**: `weather`, `calculator`, `alarm`, `calendar`, `device settings`.
- **Negative Indicators**: `add to cart`, `payment successful`, `boarding pass`, `vs code`, `online course`, `job application`, `subject:`.
- **Exclusion Conditions**: Any screenshot containing strong positive indicators for a domain category.
- **Tie-Breaking Rule**: `General` is chosen only when no domain category meets its minimum positive evidence requirement or domain evidence is penalized.
