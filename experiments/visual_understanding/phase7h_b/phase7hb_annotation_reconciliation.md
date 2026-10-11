# Phase 7H-B: Annotation Reconciliation & Candidate Mapping Protocol

**Date**: 2026-10-02  
**Target Engine**: Visual Screen Understanding Engine (VSUE)  
**Manifest Source**: `screen_type_manifest_phase7hb.csv`  
**Human Review Queue**: `phase7hb_human_review_queue.csv` (438 Candidates)

---

## 1. Overview & Reconciliation Rationale

The original 836 screenshot corpus was annotated for semantic classification (categorizing content into `Education`, `Shopping`, `Entertainment`, `Communication`, `Work and career`, `Travel`, `Coding`, `General`, `Payment`, `Connectivity`).

However, visual screen-type recognition evaluates a different functional dimension (identifying `product_listing`, `messaging_chat`, `email_view`, `travel_booking`, `media_player_library`, `cart_checkout_payment`, `login_authentication`, `dashboard_settings`, `error_dialog_alert`, `code_editor_terminal`, `document_article_view`).

In Phase 7H-B, we audited the descriptive `notes` and `category` fields in `data/metadata.csv` for all 836 records. This revealed that candidate visual screen-type examples exist across **all 13 visual classes** in our existing dataset, eliminating the need to collect external screenshots.

---

## 2. Reconciled Visual Class Candidates Breakdown

| Visual Screen Type Label | Phase 7E Verified | Reconciled Candidate Total | Review Queue Candidates | Candidate Identification Rationale |
| :--- | ----: | ----: | ----: | :--- |
| **`document_article_view`** | 602 | **464** | 0 | Continuous text, academic papers, articles, blog posts, PDF reader views. |
| **`product_detail`** | 137 | **80** | 0 | Single item presentation with image, price, and buy button. |
| **`travel_booking`** | 0 | **82** | **82** | `Travel` notes indicating flight search, hotel reservations, seating maps, tickets. |
| **`media_player_library`** | 0 | **62** | **62** | `Entertainment` notes indicating video players, YouTube, Spotify music playback queues. |
| **`dashboard_settings`** | 27 | **58** | **58** | Notes indicating settings panels, account preferences, analytics tiles, control panels. |
| **`messaging_chat`** | 0 | **48** | **48** | `Communication` notes indicating chat threads, DMs, messaging bubbles, WhatsApp/Slack UI. |
| **`code_editor_terminal`** | 48 | **48** | **48** | `Coding` notes indicating IDE windows, line numbers, terminal prompts, code blocks. |
| **`product_listing`** | 0 | **45** | **45** | `Shopping` notes indicating multi-product catalogs, search results, category grids. |
| **`cart_checkout_payment`** | 19 | **28** | **28** | `Payment` & `Shopping` notes indicating checkout forms, order summaries, payment gateways. |
| **`login_authentication`** | 0 | **28** | **28** | `General` & `Connectivity` notes indicating login screens, sign-up forms, OTP prompts. |
| **`email_view`** | 0 | **25** | **25** | `Communication` notes indicating Gmail inbox, email reader views, compose windows. |
| **`error_dialog_alert`** | 0 | **14** | **14** | `General` & `Connectivity` notes indicating system error dialogs, modal warnings. |
| **`other_screen`** | 0 | **3** | **0** | Omitted corrupted 0-byte files (`ss_441`, `ss_443`, `ss_560`). |
| **Total Reconciled Corpus** | **833** | **836** | **438** | **All 13 Visual Classes Accounted** |

---

## 3. Human Review Queue Guidelines

All 438 candidate visual proposals are logged in `phase7hb_human_review_queue.csv`. 

### Rules for Human Annotators
1. **Preserve Phase 7E Verified Labels**: Do not overwrite a verified Phase 7E label unless a side-by-side visual inspection confirms a clear annotation mistake in Phase 7E.
2. **Review Candidate Proposals**: Inspect each item in `phase7hb_human_review_queue.csv`. If visual structure matches `proposed_candidate_screen_type`, approve the proposal and update `annotation_status: verified`.
3. **No Forced Labels**: If an image visual structure is genuinely ambiguous, set `annotation_status: ambiguous`.
