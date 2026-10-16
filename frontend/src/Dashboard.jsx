import React, { useState, useEffect, useRef, useCallback } from 'react';
import './Dashboard.css';
import {
  fetchScreenshots,
  fetchScreenshot,
  fetchCategories,
  uploadScreenshots,
  deleteScreenshot,
  semanticSearch,
  checkHealth,
} from './api';

// ── Category icon map (fallback for display) ──────────────────────────────────
const CAT_ICONS = {
  Shopping:      '🛒',
  Travel:        '✈️',
  Education:     '📚',
  Payment:       '💳',
  Communication: '💬',
  Work_Career:   '💼',
  Coding:        '💻',
  Entertainment: '🎮',
  Connectivity:  '📶',
  General:       '📁',
};

const CAT_COLORS = {
  Shopping:      '#f59e0b',
  Travel:        '#06b6d4',
  Education:     '#8b5cf6',
  Payment:       '#10b981',
  Communication: '#3b82f6',
  Work_Career:   '#6366f1',
  Coding:        '#00d4ff',
  Entertainment: '#ec4899',
  Connectivity:  '#f97316',
  General:       '#9ca3af',
};

function catIcon(name)  { return CAT_ICONS[name]  || '🗂️'; }
function catColor(name) { return CAT_COLORS[name] || '#64748b'; }

// ── Small reusable components ─────────────────────────────────────────────────
function Spinner({ size = 20 }) {
  return (
    <svg className="iris-spinner" width={size} height={size} viewBox="0 0 50 50">
      <circle cx="25" cy="25" r="20" fill="none" stroke="currentColor" strokeWidth="5"
        strokeDasharray="31.4 94.2" />
    </svg>
  );
}

function EmptyState({ icon, title, sub }) {
  return (
    <div className="empty-state">
      <div className="empty-state-icon">{icon}</div>
      <h3>{title}</h3>
      <p>{sub}</p>
    </div>
  );
}

function ApiOfflineBanner() {
  return (
    <div className="api-offline-banner">
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10"/>
        <line x1="12" y1="8" x2="12" y2="12"/>
        <line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
      Backend API offline — start the FastAPI server to enable live data.
    </div>
  );
}

// ── Screenshot Thumbnail Card ─────────────────────────────────────────────────
function ScreenshotCard({ item, onClick, onDelete }) {
  const color = catColor(item.category);
  return (
    <div className="gallery-card" onClick={() => onClick(item)} id={`card-${item.screenshot_id}`}>
      <div className="gallery-card-thumb" style={{ borderColor: color + '55' }}>
        {item.image_url ? (
          <img src={item.image_url} alt={item.filename} className="gallery-card-img" />
        ) : (
          <div className="gallery-card-placeholder" style={{ color }}>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <rect x="3" y="3" width="18" height="18" rx="2"/>
              <circle cx="8.5" cy="8.5" r="1.5"/>
              <polyline points="21 15 16 10 5 21"/>
            </svg>
          </div>
        )}
        <div className="gallery-card-cat-badge" style={{ background: color + '22', color, borderColor: color + '55' }}>
          {catIcon(item.category)} {item.category}
        </div>
      </div>
      <div className="gallery-card-body">
        <div className="gallery-card-title" title={item.filename}>{item.filename}</div>
        <div className="gallery-card-meta">
          {item.created_at && <span className="card-date">{new Date(item.created_at).toLocaleDateString()}</span>}
          {item.classification_confidence > 0 && (
            <span className="card-conf">{(item.classification_confidence * 100).toFixed(0)}%</span>
          )}
        </div>
        {item.summary && <p className="gallery-card-summary">{item.summary.slice(0, 80)}{item.summary.length > 80 ? '…' : ''}</p>}
      </div>
      {onDelete && (
        <button className="gallery-card-del" title="Delete" onClick={e => { e.stopPropagation(); onDelete(item); }}>
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/>
          </svg>
        </button>
      )}
    </div>
  );
}

// ── Detail Modal ──────────────────────────────────────────────────────────────
function DetailModal({ item, onClose }) {
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetchScreenshot(item.screenshot_id)
      .then(setDetail)
      .catch(() => setDetail(item))
      .finally(() => setLoading(false));
  }, [item.screenshot_id]);

  const color = catColor(item.category);

  const handleCopy = () => {
    const text = detail?.ocr_text || detail?.summary || '';
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-panel" onClick={e => e.stopPropagation()} id="screenshot-detail-modal">
        <button className="modal-close" onClick={onClose}>✕</button>

        <div className="modal-header" style={{ borderColor: color + '55' }}>
          <span className="modal-cat-icon">{catIcon(item.category)}</span>
          <div>
            <h2 className="modal-title">{item.filename}</h2>
            <div className="modal-cat" style={{ color }}>{item.category}</div>
          </div>
        </div>

        {loading ? (
          <div className="modal-loading"><Spinner size={32} /></div>
        ) : (
          <div className="modal-body">
            {/* Image preview */}
            <div className="modal-preview">
              {item.image_url ? (
                <img src={item.image_url} alt={item.filename} className="modal-img" />
              ) : (
                <div className="modal-no-img">No preview available</div>
              )}
            </div>

            {/* OCR text */}
            <div className="modal-section">
              <div className="modal-section-header">
                <span>EXTRACTED TEXT</span>
                <button className="panel-btn small" onClick={handleCopy}>
                  {copied ? '✓ COPIED' : 'COPY'}
                </button>
              </div>
              <pre className="modal-ocr-text">{detail?.ocr_text || detail?.summary || 'No OCR text available.'}</pre>
            </div>

            {/* Tags */}
            {detail?.tags?.length > 0 && (
              <div className="modal-section">
                <div className="modal-section-header"><span>TAGS</span></div>
                <div className="tag-row">
                  {detail.tags.map((t, i) => (
                    <span key={i} className="tag-pill" style={{ borderColor: color + '66', color }}>{t}</span>
                  ))}
                </div>
              </div>
            )}

            {/* Metadata grid */}
            <div className="modal-section">
              <div className="modal-section-header"><span>METADATA</span></div>
              <div className="meta-grid">
                <MetaRow label="Screenshot ID" value={detail?.screenshot_id} mono />
                <MetaRow label="Created"       value={detail?.created_at ? new Date(detail.created_at).toLocaleString() : '—'} />
                <MetaRow label="OCR Available" value={detail?.ocr_available ? 'Yes' : 'No'} />
                <MetaRow label="Confidence"    value={detail?.classification_confidence ? `${(detail.classification_confidence * 100).toFixed(1)}%` : '—'} />
                {detail?.evidence?.fusion_applied && (
                  <MetaRow label="Fusion"      value={detail.evidence.fusion_reason || 'Applied'} />
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function MetaRow({ label, value, mono }) {
  return (
    <div className="meta-row">
      <span className="meta-label">{label}</span>
      <span className={`meta-val${mono ? ' mono' : ''}`}>{value || '—'}</span>
    </div>
  );
}

// ── Upload Zone ───────────────────────────────────────────────────────────────
function UploadTab({ onUploaded }) {
  const [dragging, setDragging] = useState(false);
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);
  const inputRef = useRef();

  const handleFiles = (incoming) => {
    const valid = Array.from(incoming).filter(f => f.type.startsWith('image/'));
    setFiles(valid);
    setResults([]);
    setError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  const handleUpload = async () => {
    if (!files.length) return;
    setUploading(true);
    setError(null);
    try {
      const data = await uploadScreenshots(files);
      setResults(data.results);
      setFiles([]);
      onUploaded?.();
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  };

  const clearFile = (idx) => setFiles(f => f.filter((_, i) => i !== idx));

  return (
    <div className="upload-tab">
      <div
        id="upload-drop-zone"
        className={`upload-zone${dragging ? ' drag-over' : ''}`}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept="image/*"
          style={{ display: 'none' }}
          onChange={e => handleFiles(e.target.files)}
        />
        <div className="upload-zone-icon">
          <svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
        </div>
        <h3>Drag & Drop Screenshots Here</h3>
        <p>PNG, JPG, WEBP supported — or click to browse</p>
      </div>

      {files.length > 0 && (
        <div className="upload-file-list">
          <div className="upload-file-list-header">
            <span>{files.length} file{files.length > 1 ? 's' : ''} selected</span>
            <button
              className="panel-btn"
              onClick={handleUpload}
              disabled={uploading}
              id="upload-process-btn"
            >
              {uploading ? <><Spinner size={14} /> Processing…</> : `▶ PROCESS ${files.length} FILE${files.length > 1 ? 'S' : ''}`}
            </button>
          </div>
          {files.map((f, i) => (
            <div key={i} className="upload-file-row">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2"/>
                <circle cx="8.5" cy="8.5" r="1.5"/>
                <polyline points="21 15 16 10 5 21"/>
              </svg>
              <span className="upload-file-name">{f.name}</span>
              <span className="upload-file-size">{(f.size / 1024).toFixed(0)} KB</span>
              <button className="upload-file-del" onClick={() => clearFile(i)}>✕</button>
            </div>
          ))}
        </div>
      )}

      {error && <div className="upload-error">⚠ {error}</div>}

      {results.length > 0 && (
        <div className="upload-results">
          <h3 className="upload-results-title">PROCESSING RESULTS</h3>
          {results.map((r, i) => (
            <div key={i} className={`upload-result-row ${r.status}`}>
              <div className="upload-result-name">
                <span className={`result-status-dot ${r.status}`} />
                {r.filename}
              </div>
              {r.status === 'success' ? (
                <div className="upload-result-detail">
                  <span className="cat-badge" style={{ borderColor: catColor(r.category) + '88', color: catColor(r.category) }}>
                    {catIcon(r.category)} {r.category}
                  </span>
                  {r.chunks_indexed > 0 && (
                    <span className="cat-badge">{r.chunks_indexed} chunks indexed</span>
                  )}
                  {r.ocr_text && (
                    <div className="upload-ocr-preview">
                      <span className="upload-ocr-label">OCR PREVIEW:</span>
                      <span className="upload-ocr-text">{r.ocr_text.slice(0, 200)}{r.ocr_text.length > 200 ? '…' : ''}</span>
                    </div>
                  )}
                </div>
              ) : (
                <div className="upload-result-error">⚠ {r.error}</div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Search Tab ────────────────────────────────────────────────────────────────
function SearchTab({ onSelect }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searched, setSearched] = useState(false);

  const handleSearch = async (e) => {
    e?.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await semanticSearch(query.trim(), { top_k: 20 });
      setResults(data.results);
      setSearched(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="search-tab">
      <form className="search-form" onSubmit={handleSearch}>
        <div className="search-input-wrap">
          <svg className="search-form-icon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
          </svg>
          <input
            id="semantic-search-input"
            className="search-form-input"
            type="text"
            placeholder="Search screenshots semantically… e.g. 'wifi password', 'booking confirmation'"
            value={query}
            onChange={e => setQuery(e.target.value)}
            autoFocus
          />
          <button type="submit" className="panel-btn" id="semantic-search-btn" disabled={loading}>
            {loading ? <Spinner size={14} /> : 'SEARCH'}
          </button>
        </div>
      </form>

      {error && <div className="upload-error">⚠ {error}</div>}

      {searched && results.length === 0 && !loading && (
        <EmptyState icon="🔍" title="No results found" sub="Try a different search term or add more screenshots." />
      )}

      {results.length > 0 && (
        <div className="search-results-list">
          <div className="search-results-header">
            <span>{results.length} result{results.length !== 1 ? 's' : ''} for "{query}"</span>
          </div>
          {results.map((r, i) => (
            <div
              key={`${r.screenshot_id}-${r.chunk_index}`}
              className="search-result-row"
              onClick={() => onSelect(r)}
              id={`search-result-${i}`}
            >
              <div className="search-result-left">
                <div className="search-result-score" style={{ color: catColor(r.category) }}>
                  {(r.similarity_score * 100).toFixed(0)}%
                </div>
                <div className="search-result-rank">#{i + 1}</div>
              </div>
              <div className="search-result-body">
                <div className="search-result-file">
                  <span className="search-result-cat-icon">{catIcon(r.category)}</span>
                  {r.filename || r.screenshot_id}
                </div>
                <div className="search-result-snippet">{r.text.slice(0, 200)}{r.text.length > 200 ? '…' : ''}</div>
                <div className="search-result-tags">
                  <span className="tag-pill" style={{ color: catColor(r.category), borderColor: catColor(r.category) + '55' }}>
                    {r.category}
                  </span>
                  {r.tags?.slice(0, 3).map((t, j) => (
                    <span key={j} className="tag-pill">{t}</span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Categories Tab ────────────────────────────────────────────────────────────
function CategoriesTab({ onSelectCategory }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchCategories()
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="tab-loading"><Spinner size={36} /><span>Loading categories…</span></div>;
  if (error)   return <div className="upload-error">⚠ {error}</div>;
  if (!data)   return null;

  const nonEmpty = data.categories.filter(c => c.count > 0);
  const empty    = data.categories.filter(c => c.count === 0);

  return (
    <div className="categories-tab">
      <div className="categories-summary">
        <span className="cat-total-badge">{data.total_screenshots} total screenshots</span>
      </div>

      {nonEmpty.length > 0 && (
        <>
          <h3 className="categories-section-title">YOUR LIBRARY</h3>
          <div className="categories-grid" id="categories-grid">
            {nonEmpty.map(cat => (
              <div
                key={cat.name}
                className="cat-card"
                style={{ '--cat-color': cat.color }}
                onClick={() => onSelectCategory(cat.name)}
                id={`cat-card-${cat.name.toLowerCase()}`}
              >
                <div className="cat-card-icon">{cat.icon}</div>
                <div className="cat-card-label">{cat.label}</div>
                <div className="cat-card-count">{cat.count}</div>
                <div className="cat-card-desc">{cat.description}</div>
                <div className="cat-card-bar">
                  <div
                    className="cat-card-bar-fill"
                    style={{ width: `${Math.min(100, (cat.count / Math.max(...nonEmpty.map(c => c.count))) * 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {empty.length > 0 && (
        <>
          <h3 className="categories-section-title dim">AVAILABLE CATEGORIES</h3>
          <div className="categories-grid dim">
            {empty.map(cat => (
              <div key={cat.name} className="cat-card empty" style={{ '--cat-color': cat.color }}>
                <div className="cat-card-icon">{cat.icon}</div>
                <div className="cat-card-label">{cat.label}</div>
                <div className="cat-card-count dim">0</div>
                <div className="cat-card-desc">{cat.description}</div>
              </div>
            ))}
          </div>
        </>
      )}

      {nonEmpty.length === 0 && empty.length === 0 && (
        <EmptyState icon="📂" title="No categories yet" sub="Upload some screenshots to get started." />
      )}
    </div>
  );
}

// ── Gallery / Overview Content ────────────────────────────────────────────────
function GalleryContent({ title, filterCategory, searchQuery, onSelect, onDelete, onUploaded, refreshTick }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(() => {
    setLoading(true);
    fetchScreenshots({ category: filterCategory, q: searchQuery, limit: 200 })
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [filterCategory, searchQuery, refreshTick]);

  useEffect(() => { load(); }, [load]);

  if (loading) return <div className="tab-loading"><Spinner size={36} /><span>Loading screenshots…</span></div>;
  if (error)   return <div className="upload-error">⚠ {error}</div>;
  if (!data)   return null;

  const items = data.items;

  return (
    <div className="gallery-content">
      {title && <h3 className="gallery-section-title">{title} <span className="gallery-count">({data.total})</span></h3>}
      {items.length === 0 ? (
        <EmptyState
          icon="📷"
          title={searchQuery ? 'No matching screenshots' : 'No screenshots yet'}
          sub={searchQuery ? 'Try a different search term.' : 'Upload screenshots using the Upload tab.'}
        />
      ) : (
        <div className="gallery-grid" id="gallery-grid">
          {items.map(item => (
            <ScreenshotCard key={item.screenshot_id} item={item} onClick={onSelect} onDelete={onDelete} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Stat Card ─────────────────────────────────────────────────────────────────
function StatCard({ title, value, sub, icon, color }) {
  return (
    <div className="stat-card" style={{ '--card-accent': color || 'var(--neon-primary)' }}>
      <div className="stat-card-header">
        <span className="stat-title">{title}</span>
        <span className="stat-icon">{icon}</span>
      </div>
      <div className="stat-value" style={{ color: color || 'var(--neon-primary)' }}>{value}</div>
      <div className="stat-subtext">{sub}</div>
    </div>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────────
const Dashboard = ({ onLogout }) => {
  const [activeTab, setActiveTab]       = useState('overview');
  const [time, setTime]                 = useState(new Date().toLocaleTimeString());
  const [searchQuery, setSearchQuery]   = useState('');
  const [selectedItem, setSelectedItem] = useState(null);
  const [filterCategory, setFilterCat] = useState(null);
  const [apiOnline, setApiOnline]       = useState(null);
  const [statsData, setStatsData]       = useState(null);
  const [refreshTick, setRefreshTick]   = useState(0);

  // Live clock
  useEffect(() => {
    const t = setInterval(() => setTime(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(t);
  }, []);

  // API health check on mount
  useEffect(() => {
    checkHealth()
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false));
  }, []);

  // Load stats for Overview tab
  useEffect(() => {
    if (activeTab !== 'overview') return;
    Promise.all([
      fetchScreenshots({ limit: 1 }),
      fetchCategories(),
    ]).then(([shots, cats]) => {
      setStatsData({ total: shots.total, categories: cats.categories });
    }).catch(() => setStatsData(null));
  }, [activeTab, refreshTick]);

  // Keyboard shortcut Ctrl+K → focus search
  useEffect(() => {
    const handler = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        document.getElementById('header-search-input')?.focus();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const handleSelectItem = (item) => setSelectedItem(item);
  const handleCloseDetail = () => setSelectedItem(null);
  const handleUploaded = () => setRefreshTick(t => t + 1);

  const handleDelete = async (item) => {
    if (!window.confirm(`Delete "${item.filename}"?`)) return;
    try {
      await deleteScreenshot(item.screenshot_id);
      setRefreshTick(t => t + 1);
    } catch (e) {
      alert(`Delete failed: ${e.message}`);
    }
  };

  const handleSelectCategory = (catName) => {
    setFilterCat(catName);
    setActiveTab('gallery');
  };

  const handleSearchSubmit = (e) => {
    e?.preventDefault();
    if (searchQuery.trim()) {
      setActiveTab('search');
    }
  };

  const nonEmptyCategories = statsData?.categories?.filter(c => c.count > 0) || [];
  const topCat = nonEmptyCategories.sort((a, b) => b.count - a.count)[0];

  // nav items
  const NAV = [
    { id: 'overview',    label: 'Overview',    icon: 'M3 3h7v7H3zm11 0h7v7h-7zM3 14h7v7H3zm11 0h7v7h-7z' },
    { id: 'categories',  label: 'Categories',  icon: 'M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82zM7 7h.01' },
    { id: 'gallery',     label: 'Gallery',     icon: 'M3 3h18v18H3z M8.5 8.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zM21 15l-5-5L5 21' },
    { id: 'upload',      label: 'Upload',      icon: 'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12' },
    { id: 'search',      label: 'Search',      icon: 'M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z' },
    { id: 'profile',     label: 'Profile',     icon: 'M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2M12 7a4 4 0 1 0 0-8 4 4 0 0 0 0 8z' },
  ];

  return (
    <div className="iris-dashboard-container" id="iris-dashboard-root">
      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <header className="dash-header" id="dash-header-id">
        <div className="dash-brand">
          <div className="dash-logo-icon">
            <svg viewBox="0 0 100 100" width="36" height="36">
              <circle cx="50" cy="50" r="42" fill="none" stroke="#00d4ff" strokeWidth="2" opacity="0.6"/>
              <circle cx="50" cy="50" r="30" fill="none" stroke="#0066ff" strokeWidth="3" strokeDasharray="12 6"/>
              <circle cx="50" cy="50" r="14" fill="#00d4ff" opacity="0.9"/>
            </svg>
          </div>
          <div className="dash-brand-text">
            <h1>I.R.I.S.</h1>
            <span>Screenshot Organizer</span>
          </div>
          <div className={`dash-status-pill ${apiOnline === true ? 'online' : apiOnline === false ? 'offline' : ''}`}>
            <span className="status-dot"/>
            {apiOnline === null ? 'CONNECTING…' : apiOnline ? 'API ONLINE' : 'API OFFLINE'}
          </div>
        </div>

        <div className="dash-header-center">
          <form className="dash-search-box" onSubmit={handleSearchSubmit}>
            <span className="dash-search-icon">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
              </svg>
            </span>
            <input
              id="header-search-input"
              type="text"
              placeholder="Search screenshots, categories, OCR text…"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
            />
            <span className="dash-shortcut">Ctrl + K</span>
          </form>
        </div>

        <div className="dash-header-right">
          <div className="dash-clock">{time}</div>
          <div className="dash-user-badge" onClick={() => setActiveTab('profile')} style={{ cursor: 'pointer' }}>
            <div className="dash-avatar">G</div>
            <div className="dash-user-info">
              <span className="dash-user-name">GROUP</span>
              <span className="dash-user-role">AUTHORIZED</span>
            </div>
          </div>
          <button className="dash-logout-btn" onClick={onLogout} id="dash-logout-btn-id">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
              <polyline points="16 17 21 12 16 7"/>
              <line x1="21" y1="12" x2="9" y2="12"/>
            </svg>
            LOGOUT
          </button>
        </div>
      </header>

      {/* ── Body ────────────────────────────────────────────────────────────── */}
      <div className="dash-body">
        {/* Sidebar */}
        <aside className="dash-sidebar" id="dash-sidebar-id">
          <ul className="dash-nav-list">
            {NAV.map(nav => (
              <li key={nav.id} className={`dash-nav-item${activeTab === nav.id ? ' active' : ''}`}>
                <button onClick={() => { setActiveTab(nav.id); setFilterCat(null); }}>
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d={nav.icon}/>
                  </svg>
                  <span>{nav.label}</span>
                </button>
              </li>
            ))}
          </ul>

          <div className="dash-sidebar-footer">
            <div className="sidebar-api-status">
              <span className={`sidebar-dot ${apiOnline ? 'online' : 'offline'}`}/>
              <span>{apiOnline ? 'Backend connected' : 'Backend offline'}</span>
            </div>
            {statsData && (
              <div className="sidebar-stats">
                <div className="sidebar-stat-row">
                  <span>SCREENSHOTS</span>
                  <span>{statsData.total}</span>
                </div>
                <div className="sidebar-stat-row">
                  <span>CATEGORIES</span>
                  <span>{nonEmptyCategories.length}</span>
                </div>
              </div>
            )}
          </div>
        </aside>

        {/* Main content */}
        <main className="dash-content" id="dash-content-id">
          {apiOnline === false && <ApiOfflineBanner />}

          {/* ── Overview Tab ─────────────────────────────────────────────── */}
          {activeTab === 'overview' && (
            <div className="overview-tab">
              <div className="stat-cards-grid">
                <StatCard
                  title="TOTAL SCREENSHOTS"
                  value={statsData?.total ?? '—'}
                  sub="Stored in library"
                  color="#00d4ff"
                  icon={<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>}
                />
                <StatCard
                  title="CATEGORIES ACTIVE"
                  value={nonEmptyCategories.length || '—'}
                  sub="Taxonomy classes in use"
                  color="#00f5d4"
                  icon={<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg>}
                />
                <StatCard
                  title="TOP CATEGORY"
                  value={topCat ? catIcon(topCat.name) + ' ' + topCat.label : '—'}
                  sub={topCat ? `${topCat.count} screenshots` : 'Add screenshots first'}
                  color={topCat ? catColor(topCat.name) : '#9ca3af'}
                  icon={<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>}
                />
                <StatCard
                  title="PIPELINE STATUS"
                  value={apiOnline ? 'ONLINE' : 'OFFLINE'}
                  sub={apiOnline ? 'OCR + Classification ready' : 'Start the FastAPI server'}
                  color={apiOnline ? '#10b981' : '#ef4444'}
                  icon={<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>}
                />
              </div>

              <div className="overview-grid">
                <div className="sci-panel overview-recent">
                  <div className="sci-panel-header">
                    <h2 className="sci-panel-title">
                      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                        <rect x="3" y="3" width="18" height="18" rx="2"/>
                        <circle cx="8.5" cy="8.5" r="1.5"/>
                        <polyline points="21 15 16 10 5 21"/>
                      </svg>
                      RECENTLY ADDED
                    </h2>
                    <button className="panel-btn" onClick={() => setActiveTab('gallery')}>VIEW ALL</button>
                  </div>
                  <GalleryContent
                    searchQuery={searchQuery}
                    onSelect={handleSelectItem}
                    onDelete={handleDelete}
                    onUploaded={handleUploaded}
                    refreshTick={refreshTick}
                  />
                </div>

                <div className="sci-panel overview-cats">
                  <div className="sci-panel-header">
                    <h2 className="sci-panel-title">
                      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/>
                        <line x1="7" y1="7" x2="7.01" y2="7"/>
                      </svg>
                      CATEGORIES
                    </h2>
                    <button className="panel-btn" onClick={() => setActiveTab('categories')}>BROWSE</button>
                  </div>
                  {nonEmptyCategories.length === 0 ? (
                    <EmptyState icon="📂" title="No categories yet" sub="Upload screenshots to populate categories." />
                  ) : (
                    <div className="overview-cat-list">
                      {nonEmptyCategories.slice(0, 8).map(cat => (
                        <div
                          key={cat.name}
                          className="overview-cat-row"
                          onClick={() => handleSelectCategory(cat.name)}
                          style={{ '--cat-color': cat.color }}
                        >
                          <span className="overview-cat-icon">{cat.icon}</span>
                          <span className="overview-cat-name">{cat.label}</span>
                          <div className="overview-cat-bar-wrap">
                            <div
                              className="overview-cat-bar"
                              style={{
                                width: `${Math.min(100, (cat.count / nonEmptyCategories[0].count) * 100)}%`,
                                background: cat.color,
                              }}
                            />
                          </div>
                          <span className="overview-cat-count">{cat.count}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ── Categories Tab ───────────────────────────────────────────── */}
          {activeTab === 'categories' && (
            <div className="sci-panel">
              <div className="sci-panel-header">
                <h2 className="sci-panel-title">
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/>
                    <line x1="7" y1="7" x2="7.01" y2="7"/>
                  </svg>
                  CATEGORY BROWSER
                </h2>
              </div>
              <CategoriesTab onSelectCategory={handleSelectCategory} />
            </div>
          )}

          {/* ── Gallery Tab ──────────────────────────────────────────────── */}
          {activeTab === 'gallery' && (
            <div className="sci-panel">
              <div className="sci-panel-header">
                <h2 className="sci-panel-title">
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="3" y="3" width="18" height="18" rx="2"/>
                    <circle cx="8.5" cy="8.5" r="1.5"/>
                    <polyline points="21 15 16 10 5 21"/>
                  </svg>
                  {filterCategory ? `${catIcon(filterCategory)} ${filterCategory}` : 'ALL SCREENSHOTS'}
                </h2>
                <div className="panel-controls">
                  {filterCategory && (
                    <button className="panel-btn" onClick={() => setFilterCat(null)}>CLEAR FILTER</button>
                  )}
                </div>
              </div>
              <GalleryContent
                filterCategory={filterCategory}
                searchQuery={searchQuery}
                onSelect={handleSelectItem}
                onDelete={handleDelete}
                onUploaded={handleUploaded}
                refreshTick={refreshTick}
              />
            </div>
          )}

          {/* ── Upload Tab ───────────────────────────────────────────────── */}
          {activeTab === 'upload' && (
            <div className="sci-panel">
              <div className="sci-panel-header">
                <h2 className="sci-panel-title">
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                  </svg>
                  UPLOAD & PROCESS
                </h2>
              </div>
              <UploadTab onUploaded={handleUploaded} />
            </div>
          )}

          {/* ── Search Tab ───────────────────────────────────────────────── */}
          {activeTab === 'search' && (
            <div className="sci-panel">
              <div className="sci-panel-header">
                <h2 className="sci-panel-title">
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="11" cy="11" r="8"/>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                  </svg>
                  SEMANTIC SEARCH
                </h2>
              </div>
              <SearchTab onSelect={handleSelectItem} />
            </div>
          )}

          {/* ── Profile Tab ──────────────────────────────────────────────── */}
          {activeTab === 'profile' && (
            <div className="profile-container">
              <div className="profile-header-card">
                <div className="profile-header-left">
                  <div className="profile-avatar-outer">
                    <div className="profile-avatar-inner">G</div>
                    <div className="profile-status-ring" title="Active"/>
                  </div>
                  <div className="profile-info">
                    <h1>GROUP</h1>
                    <p>group@gmail.com • +91 1234567890</p>
                    <div className="profile-tags-row">
                      <span className="profile-tag">I.R.I.S. SECURED</span>
                      <span className="profile-tag" style={{ color: 'var(--neon-success)', borderColor: 'var(--neon-success)' }}>
                        {apiOnline ? 'ONLINE' : 'OFFLINE'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="stat-cards-grid">
                <StatCard title="SCREENSHOTS" value={statsData?.total ?? '—'} sub="Total in library" color="#00d4ff"
                  icon={<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>}
                />
                <StatCard title="CATEGORIES" value={nonEmptyCategories.length || '—'} sub="Active taxonomy classes" color="#00f5d4"
                  icon={<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg>}
                />
                <StatCard title="API STATUS" value={apiOnline ? '● ONLINE' : '● OFFLINE'} sub={apiOnline ? 'Port 8000' : 'Start backend'} color={apiOnline ? '#10b981' : '#ef4444'}
                  icon={<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>}
                />
                <StatCard title="OCR ENGINE" value="EasyOCR" sub="+ Multimodal classifier" color="#8b5cf6"
                  icon={<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>}
                />
              </div>

              <div className="sci-panel" style={{ marginTop: '20px' }}>
                <div className="sci-panel-header">
                  <h2 className="sci-panel-title">PIPELINE INFORMATION</h2>
                </div>
                <div className="profile-detail-list">
                  <div className="profile-detail-item"><span className="detail-label">OCR Engine</span><span className="detail-val">EasyOCR (+ Mock fallback)</span></div>
                  <div className="profile-detail-item"><span className="detail-label">Classifier</span><span className="detail-val">FinalRefinedClassifier (Semantic)</span></div>
                  <div className="profile-detail-item"><span className="detail-label">Visual Model</span><span className="detail-val">YOLOv8n-cls (5-class screen type)</span></div>
                  <div className="profile-detail-item"><span className="detail-label">Embedder</span><span className="detail-val">all-MiniLM-L6-v2 (384-dim)</span></div>
                  <div className="profile-detail-item"><span className="detail-label">Vector Store</span><span className="detail-val">ChromaDB</span></div>
                  <div className="profile-detail-item"><span className="detail-label">Metadata Store</span><span className="detail-val">SQLite</span></div>
                  <div className="profile-detail-item"><span className="detail-label">File Store</span><span className="detail-val">Local filesystem (data/screenshots/)</span></div>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* ── Detail Modal ───────────────────────────────────────────────────── */}
      {selectedItem && (
        <DetailModal item={selectedItem} onClose={handleCloseDetail} />
      )}
    </div>
  );
};

export default Dashboard;