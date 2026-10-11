# Consolidated Screen-Type Taxonomy Specification

**Date**: 2026-10-02  
**Purpose**: Define a distinct, observable, and consolidated taxonomy of interface screen types for the Visual Screen Understanding Engine (VSUE).

---

## 1. Candidate Consolidation & Rationalization

The initial 18 candidate screen-type labels were inspected against the 833-screenshot dataset to eliminate visual overlap and establish distinct boundaries:

| Initial Candidate Labels | Consolidated Taxonomy Label | Consolidation Rationale |
| :--- | :--- | :--- |
| `cart_checkout`, `payment_confirmation`, `payment_form` | **`cart_checkout_payment`** | These 3 screens share common payment input boxes, total amounts, pay buttons, and transaction receipts. |
| `messaging_conversation`, `email_view` | **`messaging_chat`** / **`email_view`** | Retained as distinct: chat uses left/right bubbles; email uses header lists and long text threads. |
| `network_settings`, `dashboard` | **`dashboard_settings`** | Both feature toggle switches, status icons, progress bars, and admin controls. |
| `video_player`, `media_library` | **`media_player_library`** | Both feature video/audio player viewports, play/pause controls, and thumbnail carousels. |
| `learning_interface`, `document_view` | **`document_article_view`** | Both feature structured text blocks, headings, PDF slides, and paragraphs. |

---

## 2. The 12 Consolidated Screen-Type Definitions

### 1. `product_listing`
* **Definition**: Grid or vertical list displaying multiple commercial products or items.
* **Key Visual Elements**: Image thumbnail grid, price tags, item titles, rating stars, filter/sort bar.

### 2. `product_detail`
* **Definition**: Focused single-item product or content page.
* **Key Visual Elements**: Large hero product image, title banner, price badge, "Add to Cart" or "Buy Now" CTA button.

### 3. `cart_checkout_payment`
* **Definition**: Transactional screen displaying order summary, payment input fields, or payment receipt.
* **Key Visual Elements**: Total price amount, payment method icons (UPI, card), item list summary, "Pay" CTA button.

### 4. `login_authentication`
* **Definition**: User sign-in, password entry, or multi-factor authentication screen.
* **Key Visual Elements**: Centered text input fields (username/password), submit/login button, brand logo, OTP prompt.

### 5. `messaging_chat`
* **Definition**: Conversational thread between users or automated agents.
* **Key Visual Elements**: Left/right aligned message bubbles, contact avatar header, text input bar, send icon.

### 6. `email_view`
* **Definition**: Inbox message list or individual email thread.
* **Key Visual Elements**: Sender name, subject line header, timestamp, reply/forward action buttons, text body.

### 7. `media_player_library`
* **Definition**: Audio/video playback screen or media gallery.
* **Key Visual Elements**: Media viewport, play/pause/seek controls, progress bar, media grid thumbnails.

### 8. `code_editor_terminal`
* **Definition**: Integrated Development Environment (IDE), code editor, or command-line terminal window.
* **Key Visual Elements**: Line numbers column, syntax highlighting, file tree sidebar, terminal prompt, code text lines.

### 9. `document_article_view`
* **Definition**: Text-dense reading material, PDF document, wiki page, blog article, or lecture slides.
* **Key Visual Elements**: Multi-paragraph text blocks, section headings, scrollbars, document title header.

### 10. `travel_booking`
* **Definition**: Flight, hotel, train, or cab search results and itinerary confirmation.
* **Key Visual Elements**: Origin/destination tokens, date picker card, flight/hotel summary cards, booking CTA button.

### 11. `dashboard_settings`
* **Definition**: System settings menu, device controls, status dashboard, or analytics dashboard.
* **Key Visual Elements**: Toggle switches, status indicator icons, list items, statistical charts/widgets.

### 12. `error_dialog_alert`
* **Definition**: Modal pop-up window, system warning, or permission prompt.
* **Key Visual Elements**: Centered alert card overlay, warning icon, "OK / Cancel" buttons, dialog text.

### 13. `other_screen` (Fallback)
* **Definition**: Any interface layout that does not cleanly fit the 12 core screen types above.
