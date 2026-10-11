# Phase 7H-C: Human Annotation Guidelines & Visual Classification Protocol

**Version**: 3.0 (Phase 7H-C Human Verification Protocol)  
**Date**: 2026-10-02  
**Target Engine**: Visual Screen Understanding Engine (VSUE)  
**Taxonomy Scope**: 13 Canonical Screen Types

---

## 1. Core Principles & Crucial Distinction

### Semantic Category vs. Visual Screen Type

> [!IMPORTANT]
> **Semantic Category** (from `data/metadata.csv`) describes *what domain content* the screenshot discusses (e.g., `Shopping`, `Travel`, `Entertainment`, `Communication`, `Payment`).  
> **Visual Screen Type** (the VSUE taxonomy) describes *how the UI layout is structured* (e.g., `product_listing`, `product_detail`, `cart_checkout_payment`, `messaging_chat`, `email_view`, `media_player_library`, `travel_booking`, `dashboard_settings`, `error_dialog_alert`, `code_editor_terminal`, `document_article_view`, `login_authentication`, `other_screen`).

Never assign a visual screen type based solely on the semantic category or file name. The visual interface structure in the screenshot image must provide the visual evidence.

---

## 2. Detailed Visual Evidence Rules for All 13 Classes

### 1. `product_listing`
* **Definition**: Multi-item catalog, search results grid, or product browsing page.
* **Visual Evidence**: Grid or list layout displaying multiple item thumbnails, prices, star ratings, filter sidebars.
* **Distinguishing Rule**: If the interface focuses on a single item with description and "Add to Cart", assign `product_detail`.

### 2. `product_detail`
* **Definition**: Dedicated product or service page presenting a single item.
* **Visual Evidence**: Prominent primary product photo, product title header, price, quantity/size picker, "Buy Now" or "Add to Cart" button.
* **Distinguishing Rule**: If the interface shows an active checkout summary or payment selection, assign `cart_checkout_payment`.

### 3. `cart_checkout_payment`
* **Definition**: Shopping cart summary, checkout form, shipping details, or payment gateway interface.
* **Visual Evidence**: Order summary card, item quantity breakdown, subtotal/tax/shipping price lines, payment radio buttons (Credit Card, PayPal, Apple Pay), "Place Order" button.

### 4. `login_authentication`
* **Definition**: User sign-in, signup, password reset, or OTP verification interface.
* **Visual Evidence**: Email/username input box, masked password entry field, "Sign In" button, OAuth login buttons ("Continue with Google/Apple").

### 5. `messaging_chat`
* **Definition**: Real-time conversation thread, chat window, or direct message stream.
* **Visual Evidence**: Alternating left/right speech bubbles, contact avatar header, bottom message compose bar with attachment icon.
* **Distinguishing Rule**: Email threads displaying formal email headers (`From:`, `To:`, `Subject:`) are assigned `email_view`.

### 6. `email_view`
* **Definition**: Email inbox, email message reader, or email composition view.
* **Visual Evidence**: Explicit email metadata header (`From:`, `To:`, `Subject:`, timestamp), email action toolbar (Reply, Forward, Archive, Trash).

### 7. `media_player_library`
* **Definition**: Audio/video player, streaming player, or playlist library.
* **Visual Evidence**: Play/Pause controls, progress scrubber bar, volume slider, album art thumbnail, video viewport.

### 8. `code_editor_terminal`
* **Definition**: Active IDE window, text editor with line numbers margin, or terminal console window.
* **Visual Evidence**: Monospaced font rendering, line numbers gutter, syntax color highlighting, command prompt (`$` or `>`).
* **Distinguishing Rule**: Code blocks embedded inside web documentation or articles are assigned `document_article_view`.

### 9. `document_article_view`
* **Definition**: Continuous text content, news articles, academic papers, documentation pages, or wiki entries.
* **Visual Evidence**: Paragraph blocks, section headings, author metadata, scrollbar.

### 10. `travel_booking`
* **Definition**: Flight, hotel, train, or car reservation interface.
* **Visual Evidence**: Date range pickers (Check-in/Check-out), flight origin/destination fields, seating layout chart, passenger selector.

### 11. `dashboard_settings`
* **Definition**: Analytics dashboard, control panel, or application settings interface.
* **Visual Evidence**: Metric KPI tiles, line/bar charts, navigation sidebar, toggle switches, form sliders.

### 12. `error_dialog_alert`
* **Definition**: Modal dialog box, system error popup, or alert notification.
* **Visual Evidence**: Centered popup box, error/warning icon, modal title, "Dismiss" or "OK" buttons, dimmed backdrop.

### 13. `other_screen` *(Fallback)*
* **Definition**: Valid, uncorrupted screenshots that do not fit any of the 12 specific visual classes (e.g. desktop home screen, camera app).
* **Important**: Corrupted or 0-byte empty files are NOT `other_screen` — they must be marked `rejected`.

---

## 3. Review Decision Guidelines for Human Annotators

When inspecting contact sheet thumbnails and filling `phase7hc_human_annotation_queue.csv`:
* `approve`: Proposed visual screen type accurately matches visual evidence. Set `final_screen_type = proposed_screen_type`.
* `correct`: Proposed visual screen type is incorrect. Set `final_screen_type = <corrected_class>` and provide rationale in `review_notes`.
* `ambiguous`: Visual layout is borderline between two classes. Set `final_screen_type = ambiguous`.
* `reject`: Image is corrupted or unreadable. Set `final_screen_type = rejected`.
