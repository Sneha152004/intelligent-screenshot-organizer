"""
Semantic Search API router.

Endpoints
---------
GET /api/search?q=<text>&top_k=5&category=<optional>

Uses ChromaDB vector store + MiniLM embedder for dense retrieval,
then enriches results with SQLite metadata.
Falls back gracefully if the vector store or embedder is not available.
"""

import os
from pathlib import Path
import re
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/search", tags=["search"])

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "my", "me", "i", "we", "our", "you", "your", "they",
    "their", "this", "that", "these", "those", "and", "or", "but", "if", "so",
    "find", "show", "get", "list", "search", "screenshot", "screenshots", "image", "images",
    "about", "some", "any", "what", "which", "where", "who", "how", "when",
    "please", "can", "could", "would", "should", "tell", "under", "all"
}

KNOWN_BRANDS = {
    "amazon": ["amazon", "amazon.in", "amazon.com", "amazon pay"],
    "flipkart": ["flipkart", "flipkart.com", "supercoins"],
    "instamart": ["instamart", "swiggy instamart"],
    "swiggy": ["swiggy"],
    "zomato": ["zomato"],
    "metro": ["metroshoes", "metro shoes"],
    "zapvi": ["zapvi", "zapvi.in"],
    "paytm": ["paytm"],
    "phonepe": ["phonepe"],
    "google pay": ["google pay", "gpay"],
    "spotify": ["spotify"],
    "netflix": ["netflix"],
    "airtel": ["airtel"],
}


def _extract_query_keywords(query: str) -> List[str]:
    """Extract informative non-stopword tokens from query."""
    tokens = re.findall(r'[a-zA-Z0-9]+', query.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def _clean_ocr_excerpt(raw_text: str, max_len: int = 220) -> str:
    """
    Cleans raw OCR text for user-facing display:
    - Trims leading status-bar junk (4G, 5G, clock, battery, carrier tokens, noise).
    - Preserves all genuine original words without hallucination.
    """
    if not raw_text:
        return ""

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    STATUS_NOISE = re.compile(
        r'^(?:[0-9oO]{1,2}[:.][0-9oO]{2}(?:\s*[0-9oO])?(?:\s*(?:am|pm))?|[0-9]{1,3}%|[0-9.]+\s*(?:kb|mb)/s|4g\d*|5g|lte\d*|volte\d*|vowifi|3g|2g|mi|wifi|votz|[a-z]{1,2})$',
        re.I
    )

    start_idx = 0
    while start_idx < len(lines):
        line = lines[start_idx]
        if STATUS_NOISE.match(line) or len(line) <= 2:
            start_idx += 1
        else:
            break

    cleaned_lines = lines[start_idx:] if start_idx < len(lines) else lines
    cleaned_text = " - ".join(cleaned_lines) if cleaned_lines else raw_text.replace("\n", " ")

    if len(cleaned_text) > max_len:
        cleaned_text = cleaned_text[:max_len - 3] + "..."

    return cleaned_text


# --- Intent & Purpose Patterns ---
# Query Intent Patterns
INTENT_CHECKOUT_QUERY = re.compile(
    r'\b(?:about\s+to\s+pay|before\s+paying|pre-?checkout|checkout|cart|shopping\s+cart|payment\s+method|select\s+payment|payment\s+selection|choose\s+payment|proceed\s+to\s+pay(?:ment)?|about\s+to\s+checkout)\b',
    re.I
)
INTENT_DISCOVERY_QUERY = re.compile(
    r'\b(?:restaurants?|places\s+to\s+eat|dishes|cuisines?|near\s+me|food\s+options|menu|menus|products?|browse|browsing|catalog|catalogue|recommendations?|explore)\b',
    re.I
)
INTENT_PAYMENT_QUERY = re.compile(
    r'\b(?:payment|payments|paid|transaction|transactions|upi|receipt|receipts|invoice|invoices|transferred|debited)\b',
    re.I
)
INTENT_ORDER_QUERY = re.compile(
    r'\b(?:order|orders|ordered|purchase|purchases|purchased)\b',
    re.I
)

# Candidate Evidence Patterns (evaluated against OCR text / metadata)
EVIDENCE_ORDER_PATTERN = re.compile(
    r'\b(?:'
    r'my\s+orders?'
    r'|order\s+(?:details?|summary|placed|confirmed|complete|completed|history|status|id|#|no|number)'
    r'|ordr\s+(?:details?|summary|placed|confirmed|complete|completed|id|#|no|number)'
    r'|order\s*[\:\#\№]\s*\w+'
    r'|items?\s+ordered'
    r'|\d+\s+(?:items?|ttems?|products?)'
    r'|(?:arriving\s+by|delivered\s+on|delivered\s+to|delivery\s+by|out\s+for\s+delivery|estimated\s+delivery)'
    r'|(?:grand\s+total|order\s+total|total\s+amount|subtotal|invoice\s+(?:no|id|#))'
    r'|view\s+order\s+details?'
    r')\b',
    re.I
)
EVIDENCE_PAYMENT_PATTERN = re.compile(
    r'\b(?:'
    r'payment\s+successful'
    r'|paid\s+successfully'
    r'|transaction\s+successful'
    r'|transfer\s+successful'
    r'|money\s+transferred'
    r'|debited\s+from'
    r'|debited\s+by'
    r'|credited\s+to'
    r'|payment\s+received'
    r'|bill\s+paid'
    r'|upi\s+(?:ref|reference|transaction|id|spends)'
    r'|cashback\s+on\s+(?:all\s+)?upi'
    r'|(?:utr|txn\s+id|transaction\s+id|payment\s+id|receipt\s+no|ref\s+no|reference\s+no|bill-sap)'
    r')\b',
    re.I
)
EVIDENCE_DISCOVERY_PATTERN = re.compile(
    r'(?:'
    r'\bsearch\s+for\s+(?:dishes|restaurants|groceries|products|items?)'
    r'|\bsearch\s+for\s+[\'"][^\'"]+[\'"]'
    r'|\bwhat\s+are\s+you\s+looking\s+for\b'
    r'|\brestaurants\s*&\s*groceries\b'
    r'|\brestaurants\s+near\s+me\b'
    r'|\bexplore\s+restaurants\b'
    r'|\bshowing\s+results\s+in\b'
    r'|\bpure\s+veg\b'
    r'|\btop\s+rated\b'
    r'|\brated\s+4\+\b'
    r'|\bcuisines?\b'
    r'|\bdineout\b'
    r'|\badd\s+to\s+cart\b'
    r'|\badd\s+\+\b'
    r'|\bsort\s+by\b'
    r'|\bfilter\s+by\b'
    r'|\bmins?\s+delivery\b'
    r'|\bfree\s+delivery\s+on\s+orders?\s+above\b'
    r')',
    re.I
)
EVIDENCE_CHECKOUT_PATTERN = re.compile(
    r'\b(?:'
    r'proceed\s+to\s+pay(?:ment)?'
    r'|make\s+payment'
    r'|pay\s+now'
    r'|pay\s+using'
    r'|choose\s+payment'
    r'|select\s+payment'
    r'|select\s+payment\s+method'
    r'|proceed\s+to\s+checkout'
    r'|view\s+cart'
    r'|items?\s+in\s+cart'
    r'|apply\s+coupon'
    r'|delivery\s+address'
    r'|select\s+address'
    r')\b',
    re.I
)


def _calculate_rerank_score(query_text: str, base_sim: float, doc_text: str, meta_obj) -> float:
    """
    Calculate query-aware lexical, entity, and disambiguation adjustments:
    - Enforces explicit brand constraints (Amazon, Flipkart, Instamart, etc.).
    - Disambiguates visiting/business cards from credit/debit cards.
    - Disambiguates completed orders/transactions from product/restaurant discovery and checkout screens.
    - Uses location evidence as supporting evidence, not sole criteria.
    - Enforces keyword overlap ratios and penalizes spurious single-word noise.
    """
    q_lower = query_text.lower()
    keywords = _extract_query_keywords(query_text)

    summary = ((meta_obj.summary if meta_obj else "") or "").lower()
    ocr_evidence = ""
    if meta_obj and meta_obj.evidence and isinstance(meta_obj.evidence, dict):
        ocr_evidence = meta_obj.evidence.get("text_snippet") or ""
    elif meta_obj and meta_obj.summary:
        ocr_evidence = meta_obj.summary
    tags = [t.lower() for t in (meta_obj.tags if meta_obj and meta_obj.tags else [])]
    combined_ocr = f"{doc_text} {ocr_evidence}".lower()
    full_text = f"{combined_ocr} {summary} {' '.join(tags)}"

    score = base_sim

    # 1. Explicit Brand Constraint Validation
    requested_brands = [
        b for b, terms in KNOWN_BRANDS.items()
        if any(re.search(r'\b' + re.escape(t) + r'\b', q_lower) for t in terms)
    ]
    if requested_brands:
        has_req_brand_ocr = any(
            re.search(r'\b' + re.escape(t) + r'\b', combined_ocr)
            for b in requested_brands
            for t in KNOWN_BRANDS[b]
        )
        conflicting_brands = [
            b for b, terms in KNOWN_BRANDS.items()
            if b not in requested_brands and any(re.search(r'\b' + re.escape(t) + r'\b', combined_ocr) for t in terms)
        ]

        if conflicting_brands and not has_req_brand_ocr:
            # Conflicting brand evidence in OCR (e.g. Flipkart or Instamart when Amazon was requested)
            return -1.0
        elif has_req_brand_ocr:
            score += 0.30
        else:
            # Neither requested brand nor conflicting brand in OCR
            score -= 0.40

    # 2. Disambiguation: visiting/business card vs financial credit/debit cards
    is_visiting_card_query = bool(re.search(r'\b(visiting|business)\s+card\b', q_lower)) or "visiting" in keywords
    if is_visiting_card_query:
        has_visiting_evidence = bool(
            re.search(r'\b(visiting\s+card|business\s+card|apnacard|edit\s+card)\b', full_text)
        ) or any("visiting card" in t for t in tags)
        is_financial_card = bool(
            re.search(r'\b(credit\s+card|sbi\s+card|debit\s+card|card\s+payment|voucher|billing\s+address)\b', full_text)
        ) or (meta_obj and meta_obj.category in ["Shopping", "Payment"] and "credit" in full_text)

        if has_visiting_evidence:
            score += 0.30
            # Supporting location evidence (boost if location is present in document)
            for kw in keywords:
                if kw not in ["visiting", "card"] and re.search(r'\b' + re.escape(kw) + r'\b', full_text):
                    score += 0.20
        else:
            if is_financial_card:
                score -= 0.35
            else:
                # Unrelated document matching solely on location (e.g. train ticket with Kanpur)
                score -= 0.30

    # 3. Intent & Document Evidence Detection
    is_chk_query = bool(INTENT_CHECKOUT_QUERY.search(q_lower))
    is_disc_query = bool(INTENT_DISCOVERY_QUERY.search(q_lower))
    is_pay_query = bool(INTENT_PAYMENT_QUERY.search(q_lower)) and not is_chk_query
    is_ord_query = bool(INTENT_ORDER_QUERY.search(q_lower)) and not is_disc_query

    has_ord_ev = bool(EVIDENCE_ORDER_PATTERN.search(combined_ocr)) or any(t in ["order", "receipt", "invoice"] for t in tags)
    has_pay_ev = bool(EVIDENCE_PAYMENT_PATTERN.search(combined_ocr)) or any(t in ["payment", "upi", "receipt"] for t in tags)
    has_disc_ev = bool(EVIDENCE_DISCOVERY_PATTERN.search(combined_ocr))
    has_chk_ev = bool(EVIDENCE_CHECKOUT_PATTERN.search(combined_ocr))

    # 4. General semantic query keyword matching (when not visiting card and no brand queried)
    is_single_token_noise = False
    if not is_visiting_card_query and not requested_brands:
        if keywords:
            matched_kws = [kw for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', full_text)]
            if "order" in keywords and ("order" not in matched_kws) and (has_ord_ev or "ordr" in combined_ocr):
                matched_kws.append("order")
            overlap_ratio = len(matched_kws) / len(keywords)

            if overlap_ratio >= 0.5:
                score += 0.15 * overlap_ratio
            elif len(matched_kws) == 0 and not (has_ord_ev or has_pay_ev or has_disc_ev or has_chk_ev):
                score -= 0.25
            elif len(keywords) >= 3 and len(matched_kws) <= 1:
                # Penalize noisy single-token matches for multi-keyword queries (e.g. "city" in imaginary underwater city on Mars)
                score -= 0.20
                is_single_token_noise = True

    # 5. Intent-aware Document Purpose Adjustments (Generalizable across all queries)
    if is_ord_query:
        if has_ord_ev and not is_single_token_noise:
            if not requested_brands or has_req_brand_ocr:
                score += 0.25
        elif requested_brands:
            # Brand was requested with order intent, but document lacks transaction/order evidence (e.g. Spotify mentioning Amazon Music)
            score -= 0.35

        if has_disc_ev and not has_ord_ev:
            score -= 0.35
        elif has_chk_ev and not has_ord_ev:
            score -= 0.25

    elif is_pay_query:
        if has_pay_ev and not is_single_token_noise:
            if not requested_brands or has_req_brand_ocr:
                score += 0.25
        elif requested_brands:
            # Brand was requested with payment intent, but document lacks payment/transaction evidence
            score -= 0.35

        if has_chk_ev and not has_pay_ev:
            score -= 0.35
        elif has_disc_ev and not has_pay_ev:
            score -= 0.35

    elif is_disc_query:
        if has_disc_ev:
            score += 0.25
        if (has_ord_ev or has_pay_ev) and not has_disc_ev:
            score -= 0.25

    elif is_chk_query:
        if has_chk_ev:
            score += 0.30
        if has_pay_ev and not has_chk_ev:
            score -= 0.25

    return score


def _image_url(image_uri: str) -> Optional[str]:
    """Convert an absolute image_uri to a relative /data/... URL."""
    if not image_uri:
        return None
    if os.path.isabs(image_uri):
        try:
            rel = Path(image_uri).relative_to(Path(os.getcwd()))
            return "/" + str(rel).replace("\\", "/")
        except ValueError:
            return None
    return image_uri


@router.get("")
def semantic_search(
    q: str = Query(..., min_length=1, description="Search query text"),
    top_k: int = Query(10, ge=1, le=50, description="Max results to return"),
    category: Optional[str] = Query(None, description="Optionally filter results to a category"),
    tag: Optional[str] = Query(None, description="Optionally filter results to a tag"),
):
    """
    Perform semantic vector search over indexed screenshot chunks.
    Returns enriched results with metadata from SQLite.
    """
    from ..core.pipeline_instance import get_embedder, get_vector_store, metadata_store, file_store

    embedder = get_embedder()
    vector_store = get_vector_store()

    if embedder is None or vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Search is temporarily unavailable: embedding model or vector store is loading. Please try again shortly.",
        )

    if not q.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    try:
        results = _pipeline_search(q, top_k, category, tag, embedder, vector_store, metadata_store, file_store)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

    return {"query": q, "total": len(results), "results": results}


def _pipeline_search(query_text, top_k, category_filter, tag_filter, embedder, vector_store, metadata_store, file_store):
    """Run vector search and enrich results with SQLite metadata."""
    query_vec = embedder.embed([query_text])[0]
    total_docs = vector_store.count_documents()
    fetch_k = min(total_docs, max(top_k * 25, 150 if not (category_filter or tag_filter) else top_k * 10))
    raw_results = vector_store.similarity_search(query_embedding=query_vec, top_k=fetch_k)

    best_candidates = {}

    for res in raw_results:
        sid = res.get("screenshot_id")
        if not sid:
            continue

        meta_obj = metadata_store.get_metadata(sid)
        image_uri = file_store.get_image_uri(sid) if sid else None

        category = meta_obj.category if meta_obj else res.get("category", "General")
        meta_tags = meta_obj.tags if meta_obj else []

        if category_filter and category.lower() != category_filter.lower():
            continue

        if tag_filter and not any(t.lower() == tag_filter.lower() for t in meta_tags):
            continue

        base_sim = float(res.get("similarity_score", 0.0))
        doc_text = res.get("document", "")

        adjusted_score = _calculate_rerank_score(query_text, base_sim, doc_text, meta_obj)
        if adjusted_score < 0.35:
            continue
        final_sim = round(min(1.0, max(0.0, adjusted_score)), 4)

        if sid not in best_candidates or final_sim > best_candidates[sid]["similarity_score"]:
            resolved_uri = image_uri or (meta_obj.image_uri if meta_obj else "")
            best_candidates[sid] = {
                "screenshot_id": sid,
                "chunk_index": res.get("chunk_index", 0),
                "text": _clean_ocr_excerpt(res.get("document", "")) or res.get("document", ""),
                "raw_text": res.get("document", ""),
                "similarity_score": final_sim,
                "distance": round(1.0 - final_sim, 4),
                "filename": meta_obj.filename if meta_obj else "",
                "category": category,
                "summary": meta_obj.summary if meta_obj else "",
                "tags": meta_obj.tags if meta_obj else [],
                "image_uri": resolved_uri,
                "image_url": _image_url(resolved_uri),
                "created_at": meta_obj.created_at if meta_obj else None,
            }

    enriched = list(best_candidates.values())
    enriched.sort(key=lambda x: x["similarity_score"], reverse=True)
    return enriched[:top_k]
