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
export function fetchScreenshots({ category, tag, q, importance, min_importance, sort_by, limit = 100, offset = 0 } = {}) {
  const params = new URLSearchParams();
  if (category) params.set("category", category);
  if (tag)      params.set("tag", tag);
  if (q)        params.set("q", q);
  if (importance != null && importance !== "") params.set("importance", importance);
  if (min_importance != null && min_importance !== "") params.set("min_importance", min_importance);
  if (sort_by)  params.set("sort_by", sort_by);
  params.set("limit",  limit);
  params.set("offset", offset);
  return request(`/screenshots?${params}`);
}

export function fetchScreenshot(id) {
  return request(`/screenshots/${id}`);
}

export function fetchScreenshotImportance(id) {
  return request(`/screenshots/${id}/importance`);
}

export function fetchTimeline({ year, month, category, intent, sort = "desc" } = {}) {
  const params = new URLSearchParams();
  if (year != null && year !== "") params.set("year", year);
  if (month != null && month !== "") params.set("month", month);
  if (category) params.set("category", category);
  if (intent) params.set("intent", intent);
  if (sort) params.set("sort", sort);
  return request(`/timeline?${params}`);
}

export function fetchDuplicates({ type, max_distance = 6, category } = {}) {
  const params = new URLSearchParams();
  if (type) params.set("type", type);
  if (max_distance != null) params.set("max_distance", max_distance);
  if (category) params.set("category", category);
  return request(`/screenshots/duplicates?${params}`);
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


// ── Reminders ───────────────────────────────────────────────────────────────
export function fetchReminders({ status, type, date_filter, from_date, to_date, sort = "asc" } = {}) {
  const params = new URLSearchParams();
  if (status) params.set("status", status);
  if (type) params.set("type", type);
  if (date_filter) params.set("date_filter", date_filter);
  if (from_date) params.set("from_date", from_date);
  if (to_date) params.set("to_date", to_date);
  if (sort) params.set("sort", sort);
  return request(`/reminders?${params}`);
}

export function fetchReminder(id) {
  return request(`/reminders/${id}`);
}

export function createReminder(data) {
  return request("/reminders", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export function updateReminder(id, updates) {
  return request(`/reminders/${id}`, {
    method: "PATCH",
    body: JSON.stringify(updates),
  });
}

export function deleteReminder(id) {
  return request(`/reminders/${id}`, {
    method: "DELETE",
  });
}

export function scanReminders() {
  return request("/reminders/scan", {
    method: "POST",
  });
}


// ── Alerts ──────────────────────────────────────────────────────────────────
export function fetchAlerts({ status, severity, type, sort = "desc" } = {}) {
  const params = new URLSearchParams();
  if (status) params.set("status", status);
  if (severity) params.set("severity", severity);
  if (type) params.set("type", type);
  if (sort) params.set("sort", sort);
  return request(`/alerts?${params}`);
}

export function fetchAlertSummary() {
  return request("/alerts/summary");
}

export function updateAlert(id, updates) {
  return request(`/alerts/${id}`, {
    method: "PATCH",
    body: JSON.stringify(updates),
  });
}

export function deleteAlert(id) {
  return request(`/alerts/${id}`, {
    method: "DELETE",
  });
}

export function scanAlerts() {
  return request("/alerts/scan", {
    method: "POST",
  });
}
