/**
 * Centralised API helper.
 * All fetch calls go through here so the base URL is in one place.
 * The Vite dev proxy forwards /api/* → http://localhost:8000,
 * so we just use relative paths during development.
 */

const BASE = "";          // empty → relative, works with Vite proxy
const API  = `${BASE}/api`;

async function request(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || body.message || detail;
    } catch (_) {}
    throw new Error(`API ${path} failed (${res.status}): ${detail}`);
  }

  return res.json();
}

// ── Screenshots ──────────────────────────────────────────────────────────────
export function fetchScreenshots({ category, tag, q, limit = 100, offset = 0 } = {}) {
  const params = new URLSearchParams();
  if (category) params.set("category", category);
  if (tag)      params.set("tag", tag);
  if (q)        params.set("q", q);
  params.set("limit",  limit);
  params.set("offset", offset);
  return request(`/screenshots?${params}`);
}

export function fetchScreenshot(id) {
  return request(`/screenshots/${id}`);
}

export async function uploadScreenshots(files) {
  const form = new FormData();
  for (const file of files) form.append("files", file);
  const res = await fetch(`${API}/screenshots/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Upload failed (${res.status})`);
  }
  return res.json();
}

export function deleteScreenshot(id) {
  return request(`/screenshots/${id}`, { method: "DELETE" });
}

// ── Categories ───────────────────────────────────────────────────────────────
export function fetchCategories() {
  return request("/categories");
}

// ── Tags ─────────────────────────────────────────────────────────────────────
export function fetchTags() {
  return request("/tags");
}

// ── Semantic Search ──────────────────────────────────────────────────────────
export function semanticSearch(q, { top_k = 10, category, tag } = {}) {
  const params = new URLSearchParams({ q, top_k });
  if (category) params.set("category", category);
  if (tag)      params.set("tag", tag);
  return request(`/search?${params}`);
}

// ── Chat ─────────────────────────────────────────────────────────────────────
export function fetchConversations() {
  return request("/chat/conversations");
}

export function createConversation(title) {
  return request("/chat/conversations", {
    method: "POST",
    body: JSON.stringify({ title: title || null }),
  });
}

export function fetchConversation(conversationId) {
  return request(`/chat/conversations/${conversationId}`);
}

export function deleteConversation(conversationId) {
  return request(`/chat/conversations/${conversationId}`, {
    method: "DELETE",
  });
}

export function sendChatMessage(message, { top_k = 3, conversation_id = null, history = [] } = {}) {
  const body = { message, top_k };
  if (conversation_id) {
    body.conversation_id = conversation_id;
  }
  if (history && history.length > 0) {
    body.history = history;
  }
  return request("/chat", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// ── Health ───────────────────────────────────────────────────────────────────
export function checkHealth() {
  return request("/health");
}
