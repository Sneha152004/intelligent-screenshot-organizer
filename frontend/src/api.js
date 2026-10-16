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
export function fetchScreenshots({ category, q, limit = 100, offset = 0 } = {}) {
  const params = new URLSearchParams();
  if (category) params.set("category", category);
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

// ── Semantic Search ──────────────────────────────────────────────────────────
export function semanticSearch(q, { top_k = 10, category } = {}) {
  const params = new URLSearchParams({ q, top_k });
  if (category) params.set("category", category);
  return request(`/search?${params}`);
}

// ── Health ───────────────────────────────────────────────────────────────────
export function checkHealth() {
  return request("/health");
}
