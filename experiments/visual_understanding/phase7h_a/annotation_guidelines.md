# Phase 7H-A: Visual Screen-Type Annotation Guidelines

**Version**: 2.0 (Phase 7H-A Expansion Protocol)  
**Date**: 2026-10-02  
**Target Engine**: Visual Screen Understanding Engine (VSUE)  
**Taxonomy Scope**: 13 Canonical Screen Types

---

## 1. Overview & General Principles

The Visual Screen Understanding Engine (VSUE) is designed to recognize the high-level visual structure and functional screen type of user screenshots independently of OCR text content.

### Core Annotation Principles

1. **Visual Dominance Rule**: Assign the primary screen type based on the dominant visual layout, UI component arrangement, and primary interface intent.
2. **No Text-Only Inference**: Do not classify a screenshot based solely on reading embedded OCR words if the visual layout contradicts it.
3. **No Filename Inference**: Never rely on original file names, system paths, or temporary metadata tags to establish screen type.
4. **Strict Status Assignment**: Assign `verified` ONLY when visual evidence clearly confirms the screen type. Mark genuinely ambiguous images as `ambiguous` and unreadable/corrupted files as `rejected`.

---

## 2. Canonical 13-Class Taxonomy Definitions & Visual Evidence Requirements

### 1. `product_listing`
* **Definition**: Interfaces displaying multiple items, search results, product grids, or multi-item catalogs for browsing.
* **Key Visual Features**: Grid or vertical list layout of product thumbnails, price tags, star ratings, filtering sidebars, search bars.
* **Distinguishing Rule**: If the screen focuses primarily on a single item with detailed description and "Add to Cart", classify as `product_detail`.

### 2. `product_detail`
* **Definition**: E-commerce or marketplace page dedicated to presenting a single product, service, or item in detail.
* **Key Visual Features**: Large primary product photo, product title header, price display, color/size selection widgets, "Buy Now" / "Add to Cart" call-to-action buttons, specifications tabs.
* **Distinguishing Rule**: If the screen transitions into an active checkout summary, order review, or payment method selection list, classify as `cart_checkout_payment`.

### 3. `cart_checkout_payment`
* **Definition**: E-commerce shopping cart summary, checkout step, shipping detail form, or payment gateway interface.
* **Key Visual Features**: Order summary card, item quantity pickers, subtotal/tax/shipping price breakdown, payment method radio buttons (Credit Card, PayPal, Apple Pay), "Place Order" / "Proceed to Payment" button.
* **Distinguishing Rule**: Must display clear transactional or checkout workflow elements rather than general product browsing.

### 4. `login_authentication`
* **Definition**: Access control, user login, account registration, password reset, or Multi-Factor Authentication (OTP/MFA) screens.
* **Key Visual Features**: Username/email input fields, password masked entry box, "Sign In" / "Sign Up" primary buttons, OAuth third-party login buttons ("Continue with Google/Apple"), captcha boxes.

### 5. `messaging_chat`
* **Definition**: Real-time conversation threads, direct messages, group chats, or messaging inbox interfaces.
* **Key Visual Features**: Alternating left/right speech bubbles, contact avatar list, bottom message input bar with attachment icons, online status indicators.

### 6. `email_view`
* **Definition**: Electronic mail inbox, individual email reader view, or email composition window.
* **Key Visual Features**: Email header metadata (Sender `From:`, Receiver `To:`, Subject line, Timestamp), email toolbar (Reply, Forward, Archive, Trash), thread view.
* **Distinguishing Rule**: An email reading view contains explicit email header structures distinguishing it from general web articles (`document_article_view`).

### 7. `media_player_library`
* **Definition**: Audio/video playback interface, media streaming app, or playlist library.
* **Key Visual Features**: Play/Pause controls, timeline slider bar, volume control, album art thumbnail, track title, playlist queue.

### 8. `code_editor_terminal`
* **Definition**: Software development IDE, text editor with syntax highlighting, command-line interface, or terminal console.
* **Key Visual Features**: Line numbers margin, monospaced font rendering, syntax color highlighting, code folding arrows, dark-mode terminal window with prompt (`$` or `>`).
* **Distinguishing Rule**: Technical documentation containing formatted code blocks inside a web browser is classified as `document_article_view` unless the screenshot represents an active editor application window.

### 9. `document_article_view`
* **Definition**: Text-centric content including blog posts, news articles, academic papers, PDF readers, documentation pages, or wiki entries.
* **Key Visual Features**: Continuous multi-paragraph body text, article headlines, author metadata, embedded figures with captions, scrollbar.

### 10. `travel_booking`
* **Definition**: Reservation interface for flights, hotels, trains, car rentals, or itinerary planning.
* **Key Visual Features**: Date range pickers (Check-in / Check-out, Departure / Return), origin/destination search fields, seating chart, flight duration timeline, passenger selector.

### 11. `dashboard_settings`
* **Definition**: Application analytics dashboard, control panel, system preferences, account settings, or metrics view.
* **Key Visual Features**: Metric summary cards (KPI tiles), analytics charts (line graphs, bar charts), navigation drawer/sidebar, toggle switches, form sliders.

### 12. `error_dialog_alert`
* **Definition**: Modal dialog popups, system error warnings, permission request popups, or exception alerts.
* **Key Visual Features**: Centered modal dialog window, error icon (warning triangle, red cross), modal title, "OK" / "Cancel" / "Dismiss" buttons, dimmed background backdrop.

### 13. `other_screen` *(Fallback)*
* **Definition**: Interfaces that do not fit into any of the 12 specific functional screen types (e.g., desktop wallpaper, camera view, system home screen).

---

## 3. Allowed Status Enum Values

* `verified`: Human-reviewed and verified with 100% confidence.
* `ambiguous`: Visual layout is genuinely borderline between two screen types.
* `needs_review`: Candidate screenshot requiring human verification.
* `rejected`: Corrupted, 0-byte, unreadable, or invalid image file.

---

## 4. Manifest Schema Specification

All dataset records must adhere strictly to the CSV column schema:
`screenshot_id,image_path,screen_type,annotation_status,annotation_confidence,visual_evidence,annotator_notes,duplicate_group_id,source,review_status,aspect_ratio,orientation,orig_width,orig_height,md5_hash`
