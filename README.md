# Intelligent Screenshot Organizer

An intelligent screenshot processing and retrieval system using a 3-Layer Storage Architecture (Local File Store, SQLite Metadata Store, ChromaDB Vector Store).

---

## Intelligence Architecture (Phase 4A: Evidence First, Classification Second)

The intelligence layer transforms raw screenshot images into rich, retrievable metadata:

```text
OLD ARCHITECTURE:
  Screenshot ──► OCR ──► Classifier ──► Category + Tags + Intent

NEW PHASE 4A ARCHITECTURE:
  Screenshot
      │
      ▼
   OCR & Lightweight Signal Extraction
      │
      ▼
   Evidence Extraction Layer
   - Entities & Named References
   - Events (purchase, payment, travel, academic, event)
   - Dates & Temporal Context (raw_text, normalized, type)
   - Financial Info (amount, currency, payment_method, tx_id, merchant)
   - Contacts (emails, phone numbers)
   - Web URLs & Domains
   - Sensitive Info Flags & Action Items
   - Text Topics & Tags
      │
      ▼
   Context Resolution & Derived Classification
   - Hierarchical Category & Content Type (e.g. Travel -> flight_ticket)
   - Optional User Intent (None when evidence is weak/absent)
   - System Field Confidence Scores
   - Content-Descriptive Summaries
      │
      ▼
   Structured ScreenshotMetadata Object
      │
      ├───────────────────────────────┐
      ▼                               ▼
SQLite Metadata Store           ChromaDB Vector Index
(Structured SQL Queries)        (Semantic Vector Search)
```

### Key Concepts & Distinctions

1. **Evidence:** Observed factual data extracted from the screenshot (amounts, UPI IDs, dates, email addresses, URLs, named entities, events).
2. **Domain / Category:** Broad functional grouping (`Education`, `Payment`, `Shopping`, `Travel`, `Coding`, `Entertainment`, `Connectivity`, `Work & Career`, `Communication`, `General`).
3. **Content Type:** Hierarchical specialization under category (e.g. `Payment` -> `payment_receipt`, `Travel` -> `flight_ticket`, `Education` -> `course_material`).
4. **User Intent:** Inferred reason why the user saved the screenshot (`Optional[str] = None`). Set to `None` if evidence is weak or absent—no fabricated intent.
5. **Field Confidence:** System confidence scores assigned to extracted fields (`confidence_scores`).

---

## OCR Engines & Swappable Architecture

The system supports a swappable OCR engine abstraction subclassed from `BaseOCREngine`:

1. **`MockOCREngine` (Default for Testing)**:
   - Provides fast, deterministic, offline mock text extraction grounded in sample screenshots.
   - Ideal for unit testing, CI/CD, and fast development iterations without model loading overhead.

2. **`EasyOCREngine` (Production Deep Learning OCR)**:
   - Leverages EasyOCR (PyTorch-based CRAFT detection + ResNet/LSTM recognition).
   - Features lazy initialization (models are only loaded when text extraction is called).
   - Requires zero C++ OS binary installations.

---

## Running Tests

Run the complete test suite:
```bash
python -m pytest -v
```