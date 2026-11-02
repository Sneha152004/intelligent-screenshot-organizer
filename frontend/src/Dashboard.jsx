import React, { useState, useEffect, useRef, useCallback } from 'react';
import './Dashboard.css';
import ChatTab from './ChatTab';
import {
  fetchScreenshots,
  fetchScreenshot,
  fetchDuplicates,
  fetchTimeline,
  fetchReminders,
  createReminder,
  updateReminder,
  deleteReminder,
  scanReminders,
  fetchAlerts,
  fetchAlertSummary,
  updateAlert,
  deleteAlert,
  scanAlerts,
  fetchCategories,
  fetchTags,
  uploadScreenshots,
  deleteScreenshot,
  semanticSearch,
  checkHealth,
} from './api';

// ── Category icon map (9 canonical categories) ─────────────────────────────────
const CAT_ICONS = {
  Shopping:      '🛒',
  Travel:        '✈️',
  Education:     '📚',
  Payment:       '💳',
  Communication: '💬',
  Work_Career:   '💼',
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
function ScreenshotCard({ item, onClick, onDelete, onTagClick }) {
  const color = catColor(item.category);
  const tags = item.tags || [];
  const displayTags = tags.slice(0, 2);
  const remainingCount = tags.length - displayTags.length;

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
        {item.importance != null && (
          <div
            className={`gallery-card-importance-badge imp-${item.importance}`}
            title={`Importance: ${item.importance}/5${item.importance_reason ? ` — ${item.importance_reason}` : ''}`}
          >
            ★ {item.importance}
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
          {item.importance != null && (
            <span className={`card-imp-label imp-${item.importance}`}>★ {item.importance}/5</span>
          )}
          {item.classification_confidence > 0 && (
            <span className="card-conf">{(item.classification_confidence * 100).toFixed(0)}%</span>
          )}
        </div>
        {item.intent && item.intent !== 'General Info / None' && (
          <div className="gallery-card-intent" title={`Intent: ${item.intent}`}>
            🎯 {item.intent}
          </div>
        )}
        {item.summary && <p className="gallery-card-summary">{item.summary.slice(0, 80)}{item.summary.length > 80 ? '…' : ''}</p>}
        {tags.length > 0 && (
          <div className="gallery-card-tags">
            {displayTags.map((t, idx) => (
              <span
                key={idx}
                className="card-tag-chip"
                onClick={(e) => {
                  e.stopPropagation();
                  onTagClick?.(t);
                }}
                title={`Filter by tag: ${t}`}
              >
                #{t}
              </span>
            ))}
            {remainingCount > 0 && (
              <span className="card-tag-chip more" title={`${remainingCount} more tags`}>
                +{remainingCount}
              </span>
            )}
          </div>
        )}
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
function DetailModal({ item, onClose, onTagClick }) {
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

            {/* Intelligence & Intent */}
            <div className="modal-section">
              <div className="modal-section-header"><span>INTELLIGENCE & INTENT</span></div>
              <div className="meta-grid">
                <MetaRow label="Intent" value={detail?.intent || 'General Info / None'} />
                <MetaRow label="Content Type" value={detail?.content_type || 'unknown'} mono />
                {detail?.importance != null && (
                  <div className="meta-row">
                    <span className="meta-label">Importance Score</span>
                    <span className={`meta-val importance-score-badge imp-${detail.importance}`}>
                      ★ {detail.importance} / 5
                    </span>
                  </div>
                )}
                {detail?.importance_reason && (
                  <MetaRow label="Importance Reason" value={detail.importance_reason} />
                )}
                {detail?.summary && detail.summary !== detail.ocr_text && (
                  <MetaRow label="Summary" value={detail.summary} />
                )}
              </div>
            </div>

            {/* Financial Information */}
            {detail?.financial_info && Object.keys(detail.financial_info).length > 0 && (
              (detail.financial_info.amount != null ||
               (detail.financial_info.amounts && detail.financial_info.amounts.length > 0) ||
               (detail.financial_info.transaction_ids && detail.financial_info.transaction_ids.length > 0) ||
               (detail.financial_info.upi_ids && detail.financial_info.upi_ids.length > 0) ||
               detail.financial_info.payment_method ||
               detail.financial_info.status) ? (
                <div className="modal-section">
                  <div className="modal-section-header"><span>FINANCIAL INFORMATION</span></div>
                  <div className="meta-grid">
                    {detail.financial_info.amount != null && (
                      <MetaRow label="Amount" value={`${detail.financial_info.currency || 'INR'} ${detail.financial_info.amount}`} mono />
                    )}
                    {(!detail.financial_info.amount && detail.financial_info.amounts?.length > 0) && (
                      <MetaRow label="Amount(s)" value={detail.financial_info.amounts.join(', ')} mono />
                    )}
                    {detail.financial_info.payment_method && (
                      <MetaRow label="Payment Method" value={detail.financial_info.payment_method} />
                    )}
                    {detail.financial_info.status && (
                      <MetaRow label="Payment Status" value={detail.financial_info.status} />
                    )}
                    {detail.financial_info.upi_ids?.length > 0 && (
                      <MetaRow label="UPI ID(s)" value={detail.financial_info.upi_ids.join(', ')} mono />
                    )}
                    {detail.financial_info.transaction_ids?.length > 0 && (
                      <MetaRow label="Transaction ID(s)" value={detail.financial_info.transaction_ids.join(', ')} mono />
                    )}
                  </div>
                </div>
              ) : null
            )}

            {/* Extracted Entities */}
            {detail?.entities && detail.entities.length > 0 && (
              <div className="modal-section">
                <div className="modal-section-header"><span>EXTRACTED ENTITIES ({detail.entities.length})</span></div>
                <div className="modal-chip-grid">
                  {detail.entities.map((ent, idx) => {
                    const text = typeof ent === 'string' ? ent : (ent.text || ent.value || JSON.stringify(ent));
                    const type = typeof ent === 'object' && ent.type ? ent.type : (ent.label || 'entity');
                    return (
                      <div key={idx} className="modal-chip">
                        <span className="modal-chip-type">{type}</span>
                        <span className="modal-chip-val">{text}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Dates & Links */}
            {((detail?.dates && detail.dates.length > 0) || (detail?.urls && detail.urls.length > 0)) && (
              <div className="modal-section">
                <div className="modal-section-header"><span>DATES & LINKS</span></div>
                <div className="meta-grid">
                  {detail.dates?.map((d, idx) => {
                    const dateVal = typeof d === 'string' ? d : (d.normalized || d.raw_text || JSON.stringify(d));
                    const dateType = (typeof d === 'object' && d.date_type) ? d.date_type : 'Date';
                    return <MetaRow key={idx} label={dateType} value={dateVal} mono />;
                  })}
                  {detail.urls?.map((u, idx) => (
                    <div key={idx} className="meta-row">
                      <span className="meta-label">URL</span>
                      <a href={u} target="_blank" rel="noopener noreferrer" className="modal-url-link">{u}</a>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Tags */}
            <div className="modal-section">
              <div className="modal-section-header"><span>TAGS</span></div>
              {detail?.tags?.length > 0 ? (
                <div className="tag-row">
                  {detail.tags.map((t, i) => (
                    <span
                      key={i}
                      className="tag-pill clickable"
                      style={{ borderColor: color + '66', color }}
                      onClick={() => {
                        onClose();
                        onTagClick?.(t);
                      }}
                      title={`Filter by tag: ${t}`}
                    >
                      #{t}
                    </span>
                  ))}
                </div>
              ) : (
                <div className="tag-empty-state">No tags generated for this screenshot.</div>
              )}
            </div>

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
                  {r.tags?.length > 0 && (
                    <div className="upload-result-tags" style={{ display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
                      {r.tags.map((t, idx) => (
                        <span key={idx} className="tag-pill" style={{ color: 'var(--neon-accent)', borderColor: 'rgba(0, 245, 212, 0.4)' }}>
                          #{t}
                        </span>
                      ))}
                    </div>
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
function SearchTab({ onSelect, onTagClick }) {
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
                  {r.tags?.map((t, j) => (
                    <span
                      key={j}
                      className="tag-pill clickable"
                      onClick={(e) => {
                        e.stopPropagation();
                        onTagClick?.(t);
                      }}
                      title={`Filter by #${t}`}
                    >
                      #{t}
                    </span>
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



// ── Timeline Tab ──────────────────────────────────────────────────────────────
function TimelineTab({ onSelect, refreshTick }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filterYear, setFilterYear] = useState('');
  const [filterMonth, setFilterMonth] = useState('');
  const [filterCategory, setFilterCategory] = useState('');
  const [filterIntent, setFilterIntent] = useState('');
  const [sortOrder, setSortOrder] = useState('desc');

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    fetchTimeline({
      year: filterYear ? Number(filterYear) : undefined,
      month: filterMonth ? Number(filterMonth) : undefined,
      category: filterCategory || undefined,
      intent: filterIntent || undefined,
      sort: sortOrder,
    })
      .then(res => {
        if (isMounted) {
          setData(res);
          setError(null);
        }
      })
      .catch(err => {
        if (isMounted) setError(err.message);
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => { isMounted = false; };
  }, [filterYear, filterMonth, filterCategory, filterIntent, sortOrder, refreshTick]);

  const handleResetFilters = () => {
    setFilterYear('');
    setFilterMonth('');
    setFilterCategory('');
    setFilterIntent('');
    setSortOrder('desc');
  };

  const hasActiveFilters = Boolean(filterYear || filterMonth || filterCategory || filterIntent || sortOrder !== 'desc');

  if (loading && !data) {
    return <div className="tab-loading"><Spinner size={36} /><span>Loading intelligent timeline…</span></div>;
  }
  if (error) {
    return <div className="upload-error">⚠ {error}</div>;
  }
  if (!data) return null;

  const groups = data.groups || [];
  const years = data.available_years || [];
  const categories = data.available_categories || [];

  return (
    <div className="timeline-container">
      {/* Timeline Controls / Filter Bar */}
      <div className="timeline-toolbar">
        <div className="timeline-filters-row">
          {/* Year Filter */}
          <div className="timeline-filter-item">
            <span className="toolbar-label">YEAR:</span>
            <select
              className="toolbar-select"
              value={filterYear}
              onChange={e => {
                setFilterYear(e.target.value);
                if (!e.target.value) setFilterMonth('');
              }}
              aria-label="Filter by Year"
            >
              <option value="">All Years</option>
              {years.map(y => (
                <option key={y} value={y}>{y}</option>
              ))}
            </select>
          </div>

          {/* Month Filter */}
          <div className="timeline-filter-item">
            <span className="toolbar-label">MONTH:</span>
            <select
              className="toolbar-select"
              value={filterMonth}
              onChange={e => setFilterMonth(e.target.value)}
              aria-label="Filter by Month"
            >
              <option value="">All Months</option>
              {[
                { val: 1, label: 'January' },
                { val: 2, label: 'February' },
                { val: 3, label: 'March' },
                { val: 4, label: 'April' },
                { val: 5, label: 'May' },
                { val: 6, label: 'June' },
                { val: 7, label: 'July' },
                { val: 8, label: 'August' },
                { val: 9, label: 'September' },
                { val: 10, label: 'October' },
                { val: 11, label: 'November' },
                { val: 12, label: 'December' },
              ].map(m => (
                <option key={m.val} value={m.val}>{m.label}</option>
              ))}
            </select>
          </div>

          {/* Category Filter */}
          <div className="timeline-filter-item">
            <span className="toolbar-label">CATEGORY:</span>
            <select
              className="toolbar-select"
              value={filterCategory}
              onChange={e => setFilterCategory(e.target.value)}
              aria-label="Filter by Category"
            >
              <option value="">All Categories</option>
              {categories.map(cat => (
                <option key={cat} value={cat}>{catIcon(cat)} {cat}</option>
              ))}
            </select>
          </div>

          {/* Sort Order Toggle */}
          <div className="timeline-filter-item">
            <span className="toolbar-label">ORDER:</span>
            <select
              className="toolbar-select"
              value={sortOrder}
              onChange={e => setSortOrder(e.target.value)}
              aria-label="Timeline Sort Order"
            >
              <option value="desc">Newest First</option>
              <option value="asc">Oldest First</option>
            </select>
          </div>

          {hasActiveFilters && (
            <button
              type="button"
              className="filter-btn active"
              onClick={handleResetFilters}
              title="Reset all filters"
            >
              Reset Filters ✕
            </button>
          )}
        </div>

        {/* Quick Metrics Bar */}
        <div className="timeline-stats-summary">
          <span className="timeline-stat-badge">
            <strong>{data.total_entries}</strong> screenshots
          </span>
          <span className="timeline-stat-badge">
            <strong>{data.total_groups}</strong> monthly periods
          </span>
        </div>
      </div>

      {/* Main Timeline Stream */}
      {groups.length === 0 ? (
        <EmptyState
          icon="📅"
          title="No events found"
          sub={hasActiveFilters ? "Try clearing or relaxing your timeline filters." : "Your library has no timeline entries yet."}
        />
      ) : (
        <div className="timeline-stream">
          {groups.map(group => (
            <section key={`${group.year}-${group.month}`} className="timeline-month-section">
              {/* Month Header Banner */}
              <div className="timeline-month-header">
                <div className="timeline-month-title">
                  <span className="timeline-month-calendar-icon">📅</span>
                  <h3>{group.month_name}</h3>
                  <span className="timeline-month-count">{group.count} {group.count === 1 ? 'event' : 'events'}</span>
                </div>
              </div>

              {/* Vertical Timeline Track */}
              <div className="timeline-track">
                {group.entries.map((entry, idx) => {
                  const catColor = CAT_COLORS[entry.category] || 'var(--neon-primary)';
                  return (
                    <div key={entry.screenshot_id || idx} className="timeline-entry-row">
                      {/* Timeline Node Dot */}
                      <div
                        className="timeline-track-node"
                        style={{ '--node-color': catColor }}
                        title={`${entry.category} • ${entry.display_date}`}
                      />

                      {/* Timeline Event Card */}
                      <article className="timeline-card">
                        {/* Card Header Row */}
                        <div className="timeline-card-header">
                          <div className="timeline-card-date-group">
                            <span className="timeline-date-badge">
                              {entry.display_date}
                            </span>
                            {entry.is_event_date ? (
                              <span className="timeline-date-type-badge event" title="Extracted from ticket, receipt, or document content">
                                ⭐ Event Date
                              </span>
                            ) : (
                              <span className="timeline-date-type-badge capture" title="Based on screenshot capture time">
                                📷 Captured
                              </span>
                            )}
                          </div>

                          <div className="timeline-card-meta-right">
                            <span
                              className="timeline-cat-badge"
                              style={{
                                background: `${catColor}22`,
                                color: catColor,
                                borderColor: `${catColor}55`,
                              }}
                            >
                              {catIcon(entry.category)} {entry.category}
                            </span>
                            {entry.importance != null && (
                              <span
                                className={`timeline-imp-badge imp-${entry.importance}`}
                                title={`Importance Score: ${entry.importance}/5${entry.importance_reason ? ` — ${entry.importance_reason}` : ''}`}
                              >
                                ★ {entry.importance}
                              </span>
                            )}
                          </div>
                        </div>

                        {/* Card Body Row */}
                        <div className="timeline-card-body">
                          {/* Screenshot Thumbnail */}
                          <div
                            className="timeline-card-thumb"
                            onClick={() => onSelect?.(entry)}
                            title="Click to inspect full screenshot"
                          >
                            <img
                              src={entry.thumbnail_url || entry.image_url}
                              alt={entry.filename}
                              loading="lazy"
                              onError={e => { e.target.style.display = 'none'; }}
                            />
                            <div className="timeline-thumb-overlay">
                              <span>Inspect 🔍</span>
                            </div>
                          </div>

                          {/* Event Details Content */}
                          <div className="timeline-card-content">
                            <div className="timeline-card-title-row">
                              <h4 className="timeline-entry-title" title={entry.filename}>
                                {entry.filename}
                              </h4>
                              {entry.content_type && entry.content_type !== 'unknown' && (
                                <span className="timeline-type-pill">
                                  {entry.content_type.replace('_', ' ')}
                                </span>
                              )}
                            </div>

                            <p className="timeline-card-desc">
                              {entry.short_description}
                            </p>

                            <div className="timeline-card-footer">
                              {entry.intent && (
                                <span className="timeline-intent-tag">
                                  🎯 {entry.intent}
                                </span>
                              )}
                              {entry.tags?.slice(0, 3).map((t, ti) => (
                                <span key={ti} className="tag-pill dim">#{t}</span>
                              ))}

                              <button
                                type="button"
                                className="timeline-inspect-btn"
                                onClick={() => onSelect?.(entry)}
                              >
                                View Details →
                              </button>
                            </div>
                          </div>
                        </div>
                      </article>
                    </div>
                  );
                })}
              </div>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}


// ── Reminders Tab & Modal ─────────────────────────────────────────────────────
const REMINDER_TYPE_ICONS = {
  bill_payment: '💳 Bill Payment',
  travel: '✈️ Travel',
  event: '📅 Event',
  task: '📝 Task',
  subscription: '🔄 Subscription',
  general: '📌 General',
};

function AddEditReminderModal({ isOpen, onClose, onSave, initialData }) {
  const [title, setTitle] = useState(initialData?.title || '');
  const [description, setDescription] = useState(initialData?.description || '');
  const [reminderDate, setReminderDate] = useState(initialData?.reminder_date || new Date().toISOString().split('T')[0]);
  const [reminderTime, setReminderTime] = useState(initialData?.reminder_time || '');
  const [reminderType, setReminderType] = useState(initialData?.reminder_type || 'general');
  const [priority, setPriority] = useState(initialData?.priority || 'medium');
  const [importance, setImportance] = useState(initialData?.importance_score || 3);

  if (!isOpen) return null;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!title.trim() || !reminderDate) {
      alert("Title and Date are required.");
      return;
    }
    onSave({
      title: title.trim(),
      description: description.trim(),
      reminder_date: reminderDate,
      reminder_time: reminderTime || null,
      reminder_type: reminderType,
      priority,
      importance_score: Number(importance),
    });
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content reminder-modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h3 className="modal-title">{initialData ? 'Edit Reminder' : 'Add New Reminder'}</h3>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>
        <form onSubmit={handleSubmit} className="reminder-form">
          <div className="form-group">
            <label>Title *</label>
            <input
              type="text"
              className="form-input"
              value={title}
              onChange={e => setTitle(e.target.value)}
              placeholder="e.g. Pay Electricity Bill"
              required
            />
          </div>
          <div className="form-row">
            <div className="form-group">
              <label>Reminder Date *</label>
              <input
                type="date"
                className="form-input"
                value={reminderDate}
                onChange={e => setReminderDate(e.target.value)}
                required
              />
            </div>
            <div className="form-group">
              <label>Time (optional)</label>
              <input
                type="time"
                className="form-input"
                value={reminderTime}
                onChange={e => setReminderTime(e.target.value)}
              />
            </div>
          </div>
          <div className="form-row">
            <div className="form-group">
              <label>Type</label>
              <select
                className="form-input"
                value={reminderType}
                onChange={e => setReminderType(e.target.value)}
              >
                <option value="bill_payment">💳 Bill Payment</option>
                <option value="travel">✈️ Travel</option>
                <option value="event">📅 Event</option>
                <option value="task">📝 Task</option>
                <option value="subscription">🔄 Subscription</option>
                <option value="general">📌 General</option>
              </select>
            </div>
            <div className="form-group">
              <label>Priority</label>
              <select
                className="form-input"
                value={priority}
                onChange={e => setPriority(e.target.value)}
              >
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>
            </div>
          </div>
          <div className="form-group">
            <label>Importance (1 - 5)</label>
            <input
              type="number"
              min="1"
              max="5"
              className="form-input"
              value={importance}
              onChange={e => setImportance(e.target.value)}
            />
          </div>
          <div className="form-group">
            <label>Notes / Description</label>
            <textarea
              className="form-input"
              rows="3"
              value={description}
              onChange={e => setDescription(e.target.value)}
              placeholder="Additional details..."
            />
          </div>
          <div className="modal-actions">
            <button type="button" className="btn-cancel" onClick={onClose}>Cancel</button>
            <button type="submit" className="btn-save">Save Reminder</button>
          </div>
        </form>
      </div>
    </div>
  );
}

function RemindersTab({ onSelect, refreshTick }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [statusFilter, setStatusFilter] = useState('all');
  const [typeFilter, setTypeFilter] = useState('all');
  const [dateFilter, setDateFilter] = useState('all');
  const [isScanning, setIsScanning] = useState(false);
  const [showAddModal, setShowAddModal] = useState(false);
  const [editingReminder, setEditingReminder] = useState(null);
  const [localTick, setLocalTick] = useState(0);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    fetchReminders({
      status: statusFilter !== 'all' ? statusFilter : undefined,
      type: typeFilter !== 'all' ? typeFilter : undefined,
      date_filter: dateFilter !== 'all' ? dateFilter : undefined,
      sort: 'asc',
    })
      .then(res => {
        if (isMounted) {
          setData(res);
          setError(null);
        }
      })
      .catch(err => {
        if (isMounted) setError(err.message);
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => { isMounted = false; };
  }, [statusFilter, typeFilter, dateFilter, refreshTick, localTick]);

  const handleScan = async () => {
    setIsScanning(true);
    try {
      const res = await scanReminders();
      alert(`Scan complete! ${res.new_reminders_created} new reminders detected, ${res.existing_skipped} existing skipped.`);
      setLocalTick(t => t + 1);
    } catch (e) {
      alert(`Scan failed: ${e.message}`);
    } finally {
      setIsScanning(false);
    }
  };

  const handleToggleStatus = async (reminder) => {
    const nextStatus = reminder.status === 'completed' ? 'pending' : 'completed';
    try {
      await updateReminder(reminder.reminder_id, { status: nextStatus });
      setLocalTick(t => t + 1);
    } catch (e) {
      alert(`Update failed: ${e.message}`);
    }
  };

  const handleDismiss = async (reminder) => {
    try {
      await updateReminder(reminder.reminder_id, { status: 'dismissed' });
      setLocalTick(t => t + 1);
    } catch (e) {
      alert(`Dismiss failed: ${e.message}`);
    }
  };

  const handleDelete = async (reminderId) => {
    if (!window.confirm("Are you sure you want to delete this reminder?")) return;
    try {
      await deleteReminder(reminderId);
      setLocalTick(t => t + 1);
    } catch (e) {
      alert(`Delete failed: ${e.message}`);
    }
  };

  const handleSaveModal = async (formData) => {
    try {
      if (editingReminder) {
        await updateReminder(editingReminder.reminder_id, formData);
        setEditingReminder(null);
      } else {
        await createReminder(formData);
        setShowAddModal(false);
      }
      setLocalTick(t => t + 1);
    } catch (e) {
      alert(`Save failed: ${e.message}`);
    }
  };

  if (loading && !data) {
    return <div className="tab-loading"><Spinner size={36} /><span>Loading intelligent reminders…</span></div>;
  }
  if (error) {
    return <div className="upload-error">⚠ {error}</div>;
  }
  if (!data) return null;

  const reminders = data.reminders || [];
  const summary = data.summary || { total: 0, pending: 0, due_today: 0, upcoming: 0, overdue: 0, completed: 0, dismissed: 0 };
  const todayStr = new Date().toISOString().split('T')[0];

  // Group into Overdue, Today, Upcoming, and Completed/Dismissed
  const overdueList = reminders.filter(r => r.status === 'pending' && r.reminder_date < todayStr);
  const todayList = reminders.filter(r => r.status === 'pending' && r.reminder_date === todayStr);
  const upcomingList = reminders.filter(r => r.status === 'pending' && r.reminder_date > todayStr);
  const completedList = reminders.filter(r => r.status === 'completed' || r.status === 'dismissed');

  const renderReminderCard = (r) => (
    <div key={r.reminder_id} className={`reminder-card priority-${r.priority} status-${r.status}`}>
      <div className="reminder-card-header">
        <div className="reminder-title-area">
          <button
            type="button"
            className={`status-toggle-btn ${r.status}`}
            onClick={() => handleToggleStatus(r)}
            title={r.status === 'completed' ? 'Reopen reminder' : 'Mark as completed'}
          >
            {r.status === 'completed' ? '✓' : '○'}
          </button>
          <div>
            <h4 className={`reminder-title ${r.status === 'completed' ? 'completed-text' : ''}`}>
              {r.title}
            </h4>
            <div className="reminder-badges-row">
              <span className={`reminder-type-badge ${r.reminder_type}`}>
                {REMINDER_TYPE_ICONS[r.reminder_type] || r.reminder_type}
              </span>
              <span className={`priority-badge priority-${r.priority}`}>
                {r.priority.toUpperCase()}
              </span>
              <span className="reminder-date-pill">
                📅 {r.reminder_date} {r.reminder_time ? `• ${r.reminder_time}` : ''}
              </span>
              {r.importance_score && (
                <span className="importance-star">★ {r.importance_score}/5</span>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="reminder-card-body">
        {r.screenshot && (
          <div
            className="reminder-screenshot-thumb"
            onClick={() => onSelect?.(r.screenshot)}
            title="Inspect source screenshot"
          >
            <img
              src={r.screenshot.thumbnail_url || r.screenshot.image_url}
              alt={r.screenshot.filename}
              onError={e => { e.target.style.display = 'none'; }}
            />
            <div className="thumb-hover-overlay">Inspect 🔍</div>
          </div>
        )}
        <div className="reminder-details-area">
          {r.description && <p className="reminder-desc">{r.description}</p>}
          {r.source_text && (
            <p className="reminder-source-snippet">
              <span className="snippet-label">Evidence:</span> "{r.source_text}"
            </p>
          )}
          {r.screenshot && (
            <div className="reminder-source-info">
              <span>Source: <strong>{r.screenshot.filename}</strong></span>
              <span className="mono">[{r.screenshot.category}]</span>
              <button
                type="button"
                className="link-inspect-btn"
                onClick={() => onSelect?.(r.screenshot)}
              >
                View Screenshot ↗
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="reminder-card-actions">
        <button
          type="button"
          className="rem-action-btn complete-btn"
          onClick={() => handleToggleStatus(r)}
        >
          {r.status === 'completed' ? 'Mark Pending' : 'Mark Complete'}
        </button>
        {r.status === 'pending' && (
          <button
            type="button"
            className="rem-action-btn dismiss-btn"
            onClick={() => handleDismiss(r)}
          >
            Dismiss
          </button>
        )}
        <button
          type="button"
          className="rem-action-btn edit-btn"
          onClick={() => setEditingReminder(r)}
        >
          Edit
        </button>
        <button
          type="button"
          className="rem-action-btn delete-btn"
          onClick={() => handleDelete(r.reminder_id)}
        >
          Delete
        </button>
      </div>
    </div>
  );

  return (
    <div className="reminders-container">
      {/* Summary Metrics Bar */}
      <div className="reminders-summary-bar">
        <div className="rem-metric-card">
          <div className="rem-metric-num">{summary.total}</div>
          <div className="rem-metric-label">TOTAL REMINDERS</div>
        </div>
        <div className="rem-metric-card">
          <div className="rem-metric-num active">{summary.pending}</div>
          <div className="rem-metric-label">PENDING</div>
        </div>
        <div className="rem-metric-card">
          <div className="rem-metric-num today">{summary.due_today}</div>
          <div className="rem-metric-label">DUE TODAY</div>
        </div>
        <div className="rem-metric-card">
          <div className="rem-metric-num upcoming">{summary.upcoming}</div>
          <div className="rem-metric-label">UPCOMING</div>
        </div>
        <div className="rem-metric-card">
          <div className="rem-metric-num overdue">{summary.overdue}</div>
          <div className="rem-metric-label">OVERDUE</div>
        </div>
      </div>

      {/* Control Toolbar */}
      <div className="reminders-toolbar">
        <div className="reminders-filters-row">
          <div className="filter-select-group">
            <span className="toolbar-label">STATUS:</span>
            <select
              className="toolbar-select"
              value={statusFilter}
              onChange={e => setStatusFilter(e.target.value)}
            >
              <option value="all">All Statuses</option>
              <option value="pending">Pending</option>
              <option value="completed">Completed</option>
              <option value="dismissed">Dismissed</option>
            </select>
          </div>

          <div className="filter-select-group">
            <span className="toolbar-label">TYPE:</span>
            <select
              className="toolbar-select"
              value={typeFilter}
              onChange={e => setTypeFilter(e.target.value)}
            >
              <option value="all">All Types</option>
              <option value="bill_payment">💳 Bill Payment</option>
              <option value="travel">✈️ Travel</option>
              <option value="event">📅 Event</option>
              <option value="task">📝 Task</option>
              <option value="subscription">🔄 Subscription</option>
              <option value="general">📌 General</option>
            </select>
          </div>

          <div className="filter-select-group">
            <span className="toolbar-label">HORIZON:</span>
            <select
              className="toolbar-select"
              value={dateFilter}
              onChange={e => setDateFilter(e.target.value)}
            >
              <option value="all">All Dates</option>
              <option value="overdue">Overdue Only</option>
              <option value="today">Due Today</option>
              <option value="upcoming">Upcoming</option>
              <option value="completed">Completed</option>
            </select>
          </div>
        </div>

        <div className="reminders-actions-row">
          <button
            type="button"
            className="rem-btn scan-btn"
            disabled={isScanning}
            onClick={handleScan}
          >
            {isScanning ? 'Scanning…' : '⚡ Scan Library'}
          </button>
          <button
            type="button"
            className="rem-btn add-btn"
            onClick={() => setShowAddModal(true)}
          >
            + Add Reminder
          </button>
        </div>
      </div>

      {/* Grouped Sections */}
      {reminders.length === 0 ? (
        <EmptyState
          icon="⏰"
          title="No reminders found"
          sub="Scan your screenshots to detect deadlines, or add custom reminders manually."
        />
      ) : (
        <div className="reminders-sections-list">
          {overdueList.length > 0 && (
            <div className="reminders-group overdue-group">
              <div className="reminders-group-header overdue">
                <h3>⚠️ OVERDUE ({overdueList.length})</h3>
              </div>
              <div className="reminders-group-items">
                {overdueList.map(renderReminderCard)}
              </div>
            </div>
          )}

          {todayList.length > 0 && (
            <div className="reminders-group today-group">
              <div className="reminders-group-header today">
                <h3>🔔 DUE TODAY ({todayList.length})</h3>
              </div>
              <div className="reminders-group-items">
                {todayList.map(renderReminderCard)}
              </div>
            </div>
          )}

          {upcomingList.length > 0 && (
            <div className="reminders-group upcoming-group">
              <div className="reminders-group-header upcoming">
                <h3>📆 UPCOMING ({upcomingList.length})</h3>
              </div>
              <div className="reminders-group-items">
                {upcomingList.map(renderReminderCard)}
              </div>
            </div>
          )}

          {completedList.length > 0 && (
            <div className="reminders-group completed-group">
              <div className="reminders-group-header completed">
                <h3>✓ COMPLETED & DISMISSED ({completedList.length})</h3>
              </div>
              <div className="reminders-group-items">
                {completedList.map(renderReminderCard)}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Add / Edit Modal */}
      <AddEditReminderModal
        isOpen={showAddModal || Boolean(editingReminder)}
        initialData={editingReminder}
        onClose={() => {
          setShowAddModal(false);
          setEditingReminder(null);
        }}
        onSave={handleSaveModal}
      />
    </div>
  );
}


// ── Alerts Tab ────────────────────────────────────────────────────────────────
const ALERT_TYPE_LABELS = {
  overdue_payment: '🚨 Overdue Payment',
  upcoming_payment: '💳 Payment Deadline',
  upcoming_travel: '✈️ Travel Departure',
  upcoming_event: '📅 Scheduled Event',
  approaching_deadline: '⏰ Approaching Deadline',
  subscription_expiry: '🔄 Subscription Renewal',
  high_importance: '⭐ High Importance',
  new_reminder: '🔔 Actionable Reminder',
};

function AlertsTab({ onSelect, refreshTick }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [statusFilter, setStatusFilter] = useState('active');
  const [severityFilter, setSeverityFilter] = useState('all');
  const [typeFilter, setTypeFilter] = useState('all');
  const [isScanning, setIsScanning] = useState(false);
  const [localTick, setLocalTick] = useState(0);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    fetchAlerts({
      status: statusFilter !== 'all' ? statusFilter : undefined,
      severity: severityFilter !== 'all' ? severityFilter : undefined,
      type: typeFilter !== 'all' ? typeFilter : undefined,
      sort: 'desc',
    })
      .then(res => {
        if (isMounted) {
          setData(res);
          setError(null);
        }
      })
      .catch(err => {
        if (isMounted) setError(err.message);
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => { isMounted = false; };
  }, [statusFilter, severityFilter, typeFilter, refreshTick, localTick]);

  const handleScan = async () => {
    setIsScanning(true);
    try {
      const res = await scanAlerts();
      alert(`Alert scan complete! ${res.new_alerts_created} new alerts created, ${res.alerts_updated} updated.`);
      setLocalTick(t => t + 1);
    } catch (e) {
      alert(`Alert scan failed: ${e.message}`);
    } finally {
      setIsScanning(false);
    }
  };

  const handleUpdateStatus = async (alertId, newStatus) => {
    try {
      await updateAlert(alertId, { status: newStatus });
      setLocalTick(t => t + 1);
    } catch (e) {
      alert(`Update failed: ${e.message}`);
    }
  };

  const handleDelete = async (alertId) => {
    if (!window.confirm("Are you sure you want to delete this alert?")) return;
    try {
      await deleteAlert(alertId);
      setLocalTick(t => t + 1);
    } catch (e) {
      alert(`Delete failed: ${e.message}`);
    }
  };

  if (loading && !data) {
    return <div className="tab-loading"><Spinner size={36} /><span>Loading intelligent alerts…</span></div>;
  }
  if (error) {
    return <div className="upload-error">⚠ {error}</div>;
  }
  if (!data) return null;

  const alerts = data.alerts || [];
  const summary = data.summary || { total: 0, active: 0, unread: 0, critical: 0, high: 0, medium: 0, low: 0, resolved: 0, dismissed: 0 };

  return (
    <div className="alerts-container">
      {/* Summary Metrics Bar */}
      <div className="alerts-summary-bar">
        <div className="alert-metric-card">
          <div className="alert-metric-num active">{summary.active}</div>
          <div className="alert-metric-label">ACTIVE ALERTS</div>
        </div>
        <div className="alert-metric-card">
          <div className="alert-metric-num critical">{summary.critical}</div>
          <div className="alert-metric-label">CRITICAL</div>
        </div>
        <div className="alert-metric-card">
          <div className="alert-metric-num high">{summary.high}</div>
          <div className="alert-metric-label">HIGH PRIORITY</div>
        </div>
        <div className="alert-metric-card">
          <div className="alert-metric-num medium">{summary.medium}</div>
          <div className="alert-metric-label">MEDIUM</div>
        </div>
        <div className="alert-metric-card">
          <div className="alert-metric-num resolved">{summary.resolved}</div>
          <div className="alert-metric-label">RESOLVED</div>
        </div>
      </div>

      {/* Control Toolbar */}
      <div className="alerts-toolbar">
        <div className="alerts-filters-row">
          <div className="filter-select-group">
            <span className="toolbar-label">STATUS:</span>
            <select
              className="toolbar-select"
              value={statusFilter}
              onChange={e => setStatusFilter(e.target.value)}
            >
              <option value="active">Active Only</option>
              <option value="all">All Alerts</option>
              <option value="read">Read</option>
              <option value="resolved">Resolved</option>
              <option value="dismissed">Dismissed</option>
            </select>
          </div>

          <div className="filter-select-group">
            <span className="toolbar-label">SEVERITY:</span>
            <select
              className="toolbar-select"
              value={severityFilter}
              onChange={e => setSeverityFilter(e.target.value)}
            >
              <option value="all">All Severities</option>
              <option value="critical">🚨 Critical</option>
              <option value="high">⚠️ High</option>
              <option value="medium">⚡ Medium</option>
              <option value="low">ℹ️ Low</option>
            </select>
          </div>

          <div className="filter-select-group">
            <span className="toolbar-label">TYPE:</span>
            <select
              className="toolbar-select"
              value={typeFilter}
              onChange={e => setTypeFilter(e.target.value)}
            >
              <option value="all">All Types</option>
              <option value="overdue_payment">Overdue Payment</option>
              <option value="upcoming_payment">Payment Deadline</option>
              <option value="upcoming_travel">Travel Departure</option>
              <option value="upcoming_event">Scheduled Event</option>
              <option value="approaching_deadline">Deadline</option>
              <option value="subscription_expiry">Subscription Renewal</option>
              <option value="high_importance">High Importance</option>
            </select>
          </div>
        </div>

        <div className="alerts-actions-row">
          <button
            type="button"
            className="alert-btn scan-btn"
            disabled={isScanning}
            onClick={handleScan}
          >
            {isScanning ? 'Evaluating…' : '⚡ Refresh / Scan Alerts'}
          </button>
        </div>
      </div>

      {/* Alerts Stream List */}
      {alerts.length === 0 ? (
        <EmptyState
          icon="🛡️"
          title="All clear! No alerts requiring attention"
          sub="No active alerts matching your filter criteria. Your deadlines and critical items are under control."
        />
      ) : (
        <div className="alerts-list">
          {alerts.map(alt => (
            <article
              key={alt.alert_id}
              className={`alert-card severity-${alt.severity} status-${alt.status}`}
            >
              <div className="alert-card-header">
                <div className="alert-header-left">
                  <span className={`alert-severity-badge severity-${alt.severity}`}>
                    {alt.severity.toUpperCase()}
                  </span>
                  <span className="alert-type-badge">
                    {ALERT_TYPE_LABELS[alt.alert_type] || alt.alert_type}
                  </span>
                  <span className="alert-date-pill">
                    📅 {alt.trigger_date}
                  </span>
                </div>
                <div className="alert-header-right">
                  <span className={`alert-status-badge status-${alt.status}`}>
                    {alt.status.toUpperCase()}
                  </span>
                </div>
              </div>

              <div className="alert-card-body">
                {alt.screenshot && (
                  <div
                    className="alert-screenshot-thumb"
                    onClick={() => onSelect?.(alt.screenshot)}
                    title="Click to inspect source screenshot"
                  >
                    <img
                      src={alt.screenshot.thumbnail_url || alt.screenshot.image_url}
                      alt={alt.screenshot.filename}
                      onError={e => { e.target.style.display = 'none'; }}
                    />
                    <div className="thumb-hover-overlay">Inspect 🔍</div>
                  </div>
                )}
                <div className="alert-content-area">
                  <h4 className="alert-title">{alt.title}</h4>
                  <p className="alert-message">{alt.message}</p>
                  {alt.reminder && (
                    <div className="alert-linked-info">
                      <span>Linked Reminder: <strong>{alt.reminder.title}</strong></span>
                      <span className="mono">({alt.reminder.reminder_date})</span>
                    </div>
                  )}
                  {alt.screenshot && (
                    <div className="alert-source-info">
                      <span>Source: <strong>{alt.screenshot.filename}</strong></span>
                      <span className="mono">[{alt.screenshot.category}]</span>
                      <button
                        type="button"
                        className="link-inspect-btn"
                        onClick={() => onSelect?.(alt.screenshot)}
                      >
                        Inspect Screenshot ↗
                      </button>
                    </div>
                  )}
                </div>
              </div>

              <div className="alert-card-actions">
                {alt.status === 'active' && (
                  <button
                    type="button"
                    className="alt-action-btn read-btn"
                    onClick={() => handleUpdateStatus(alt.alert_id, 'read')}
                  >
                    Mark as Read
                  </button>
                )}
                {alt.status !== 'resolved' && (
                  <button
                    type="button"
                    className="alt-action-btn resolve-btn"
                    onClick={() => handleUpdateStatus(alt.alert_id, 'resolved')}
                  >
                    ✓ Resolve
                  </button>
                )}
                {alt.status !== 'dismissed' && (
                  <button
                    type="button"
                    className="alt-action-btn dismiss-btn"
                    onClick={() => handleUpdateStatus(alt.alert_id, 'dismissed')}
                  >
                    Dismiss
                  </button>
                )}
                <button
                  type="button"
                  className="alt-action-btn delete-btn"
                  onClick={() => handleDelete(alt.alert_id)}
                >
                  Delete
                </button>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Duplicates Tab ────────────────────────────────────────────────────────────
function DuplicatesTab({ onSelect, refreshTick }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filterType, setFilterType] = useState('all'); // 'all', 'exact', 'visual'
  const [maxDistance, setMaxDistance] = useState(6);
  const [deletingId, setDeletingId] = useState(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState(null);
  const [localTick, setLocalTick] = useState(0);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    fetchDuplicates({
      type: filterType !== 'all' ? filterType : undefined,
      max_distance: maxDistance,
    })
      .then(res => {
        if (isMounted) {
          setData(res);
          setError(null);
        }
      })
      .catch(err => {
        if (isMounted) setError(err.message);
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => { isMounted = false; };
  }, [filterType, maxDistance, refreshTick, localTick]);

  const handleDelete = async (screenshotId) => {
    setDeletingId(screenshotId);
    try {
      await deleteScreenshot(screenshotId);
      setConfirmDeleteId(null);
      setLocalTick(t => t + 1);
    } catch (e) {
      alert(`Failed to delete screenshot: ${e.message}`);
    } finally {
      setDeletingId(null);
    }
  };

  if (loading && !data) {
    return <div className="tab-loading"><Spinner size={36} /><span>Scanning for duplicates…</span></div>;
  }
  if (error) {
    return <div className="upload-error">⚠ {error}</div>;
  }
  if (!data) return null;

  const groups = data.groups || [];

  return (
    <div className="duplicates-container">
      {/* Metrics Bar */}
      <div className="dup-metrics-bar">
        <div className="dup-metric-card">
          <div className="dup-metric-num">{data.total_groups || 0}</div>
          <div className="dup-metric-label">Total Duplicate Groups</div>
        </div>
        <div className="dup-metric-card">
          <div className="dup-metric-num exact">{data.exact_groups_count || 0}</div>
          <div className="dup-metric-label">Exact Matches (SHA-256)</div>
        </div>
        <div className="dup-metric-card">
          <div className="dup-metric-num visual">{data.visual_groups_count || 0}</div>
          <div className="dup-metric-label">Near-Visual Matches (pHash)</div>
        </div>
        <div className="dup-metric-card">
          <div className="dup-metric-num">{data.total_duplicate_candidates || 0}</div>
          <div className="dup-metric-label">Duplicate Copies Found</div>
        </div>
      </div>

      {/* Toolbar */}
      <div className="dup-toolbar">
        <div className="importance-filter-group">
          <span className="toolbar-label">FILTER:</span>
          {[
            { val: 'all', label: 'All Clusters' },
            { val: 'exact', label: 'Exact Only' },
            { val: 'visual', label: 'Visual Only' },
          ].map(opt => (
            <button
              key={opt.val}
              type="button"
              className={`filter-btn${filterType === opt.val ? ' active' : ''}`}
              onClick={() => setFilterType(opt.val)}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <div className="sort-group">
          <span className="toolbar-label">DISTANCE THRESHOLD:</span>
          <select
            className="toolbar-select"
            value={maxDistance}
            onChange={e => setMaxDistance(Number(e.target.value))}
            aria-label="Hamming Distance Threshold"
          >
            <option value={2}>Strict (≤ 2 bits - ~97%)</option>
            <option value={4}>Balanced (≤ 4 bits - ~94%)</option>
            <option value={6}>Standard (≤ 6 bits - ~91%)</option>
            <option value={8}>Relaxed (≤ 8 bits - ~88%)</option>
            <option value={10}>Broad (≤ 10 bits - ~84%)</option>
          </select>
        </div>
      </div>

      {/* Groups List */}
      {groups.length === 0 ? (
        <EmptyState
          icon="✨"
          title="No duplicates found"
          sub={filterType === 'all' ? "Your screenshot library contains no duplicate files." : `No ${filterType} duplicates found with current threshold.`}
        />
      ) : (
        <div className="dup-groups-list">
          {groups.map(group => (
            <div key={group.group_id} className="dup-group-card">
              <div className="dup-group-header">
                <div className="dup-group-title">
                  <span className={`dup-group-badge ${group.type}`}>
                    {group.type === 'exact' ? 'EXACT MATCH' : 'VISUAL SIMILAR'}
                  </span>
                  <span className="dup-hash-snippet" title={group.hash}>
                    {group.type === 'exact' ? `SHA: ${group.hash?.slice(0, 16)}…` : `pHash: ${group.hash}`}
                  </span>
                  <span className="dup-hash-snippet">
                    Score: {(group.similarity_score * 100).toFixed(0)}% (Dist: {group.distance || 0})
                  </span>
                </div>
                <div className="dup-hash-snippet">
                  {group.total_in_group} files in cluster
                </div>
              </div>

              <div className="dup-group-body">
                {/* Primary Screenshot */}
                <div className="dup-column">
                  <div className="dup-column-title primary">PRIMARY / CANONICAL FILE</div>
                  {group.primary && (
                    <div className="dup-item-card">
                      <div className="dup-item-thumb" onClick={() => onSelect?.(group.primary)} style={{ cursor: 'pointer' }}>
                        <img
                          src={group.primary.image_url}
                          alt={group.primary.filename}
                          loading="lazy"
                          onError={e => { e.target.style.display = 'none'; }}
                        />
                      </div>
                      <div className="dup-item-info">
                        <div className="dup-item-fname" title={group.primary.filename}>
                          {group.primary.filename}
                        </div>
                        <div className="dup-item-meta">
                          <span>{catIcon(group.primary.category)} {group.primary.category}</span>
                          <span className="mono">{group.primary.screenshot_id}</span>
                        </div>
                        <div className="dup-actions">
                          <button
                            type="button"
                            className="dup-btn"
                            onClick={() => onSelect?.(group.primary)}
                          >
                            Inspect
                          </button>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Duplicate Copies */}
                <div className="dup-column">
                  <div className="dup-column-title candidate">
                    DUPLICATE COPIES ({group.duplicates?.length || 0})
                  </div>
                  <div className="dup-candidates-grid">
                    {group.duplicates?.map(dup => (
                      <div key={dup.screenshot_id} className="dup-item-card">
                        <div className="dup-item-thumb" onClick={() => onSelect?.(dup)} style={{ cursor: 'pointer' }}>
                          <img
                            src={dup.image_url}
                            alt={dup.filename}
                            loading="lazy"
                            onError={e => { e.target.style.display = 'none'; }}
                          />
                        </div>
                        <div className="dup-item-info">
                          <div className="dup-item-fname" title={dup.filename}>
                            {dup.filename}
                          </div>
                          <div className="dup-item-meta">
                            <span>{catIcon(dup.category)} {dup.category}</span>
                            <span className="mono">
                              {dup.similarity_type === 'exact' ? '100% exact' : `dist ${dup.distance}`}
                            </span>
                          </div>
                          <div className="dup-actions">
                            <button
                              type="button"
                              className="dup-btn"
                              onClick={() => onSelect?.(dup)}
                            >
                              Inspect
                            </button>
                            {confirmDeleteId === dup.screenshot_id ? (
                              <button
                                type="button"
                                className="dup-btn delete"
                                disabled={deletingId === dup.screenshot_id}
                                onClick={() => handleDelete(dup.screenshot_id)}
                              >
                                {deletingId === dup.screenshot_id ? 'Deleting…' : 'Confirm Delete?'}
                              </button>
                            ) : (
                              <button
                                type="button"
                                className="dup-btn delete"
                                onClick={() => setConfirmDeleteId(dup.screenshot_id)}
                              >
                                Delete Copy
                              </button>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Gallery / Overview Content ────────────────────────────────────────────────
function GalleryContent({
  title,
  filterCategory,
  filterTag,
  searchQuery,
  onSelect,
  onDelete,
  onUploaded,
  onTagClick,
  onClearCategory,
  onClearTag,
  refreshTick
}) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [availableTags, setAvailableTags] = useState([]);
  const [filterImportance, setFilterImportance] = useState(null);
  const [sortBy, setSortBy] = useState('default');

  useEffect(() => {
    fetchTags()
      .then(res => {
        if (res && res.tags) {
          setAvailableTags(res.tags);
        }
      })
      .catch(() => setAvailableTags([]));
  }, [refreshTick]);

  const load = useCallback(() => {
    setLoading(true);
    fetchScreenshots({
      category: filterCategory,
      tag: filterTag,
      q: searchQuery,
      importance: filterImportance,
      sort_by: sortBy !== 'default' ? sortBy : undefined,
      limit: 200
    })
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [filterCategory, filterTag, searchQuery, filterImportance, sortBy, refreshTick]);

  useEffect(() => { load(); }, [load]);

  if (loading) return <div className="tab-loading"><Spinner size={36} /><span>Loading screenshots…</span></div>;
  if (error)   return <div className="upload-error">⚠ {error}</div>;
  if (!data)   return null;

  const items = data.items;

  return (
    <div className="gallery-content">
      {title && <h3 className="gallery-section-title">{title} <span className="gallery-count">({data.total})</span></h3>}

      {/* Importance Filter & Sort Toolbar */}
      <div className="gallery-toolbar">
        <div className="importance-filter-group">
          <span className="toolbar-label">IMPORTANCE:</span>
          {[
            { val: null, label: 'All' },
            { val: 5, label: '★5 Critical', cls: 'imp-5' },
            { val: 4, label: '★4 Actionable', cls: 'imp-4' },
            { val: 3, label: '★3 Reference', cls: 'imp-3' },
            { val: 2, label: '★2 General', cls: 'imp-2' },
            { val: 1, label: '★1 Low', cls: 'imp-1' },
          ].map(opt => (
            <button
              key={opt.label}
              type="button"
              className={`filter-btn${filterImportance === opt.val ? ' active' : ''} ${opt.cls || ''}`}
              onClick={() => setFilterImportance(opt.val)}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <div className="sort-group">
          <span className="toolbar-label">SORT:</span>
          <select
            className="toolbar-select"
            value={sortBy}
            onChange={e => setSortBy(e.target.value)}
            aria-label="Sort screenshots"
          >
            <option value="default">Default</option>
            <option value="importance_desc">Importance (High → Low)</option>
            <option value="importance_asc">Importance (Low → High)</option>
            <option value="newest">Newest First</option>
            <option value="oldest">Oldest First</option>
          </select>
        </div>
      </div>

      {/* Active filter bar */}
      {(filterCategory || filterTag) && (
        <div className="active-filters-bar">
          <span className="active-filters-label">ACTIVE FILTERS:</span>
          {filterCategory && (
            <span className="active-filter-pill" style={{ borderColor: catColor(filterCategory) + '88' }}>
              <span className="pill-dot" style={{ background: catColor(filterCategory) }} />
              Category: {filterCategory}
              <button className="pill-clear-btn" onClick={onClearCategory} title="Clear category filter">✕</button>
            </span>
          )}
          {filterTag && (
            <span className="active-filter-pill tag-pill-active">
              <span className="pill-tag-icon">#</span>
              Tag: {filterTag}
              <button className="pill-clear-btn" onClick={onClearTag} title="Clear tag filter">✕</button>
            </span>
          )}
        </div>
      )}

      {/* Popular tag filter chips row */}
      {availableTags.length > 0 && (
        <div className="popular-tags-row">
          <span className="popular-tags-label">POPULAR TAGS:</span>
          <div className="popular-tags-list">
            {availableTags.slice(0, 16).map(t => (
              <button
                key={t.name}
                type="button"
                className={`tag-filter-chip${filterTag === t.name ? ' active' : ''}`}
                onClick={() => onTagClick?.(filterTag === t.name ? null : t.name)}
                title={`${t.count} screenshot${t.count !== 1 ? 's' : ''}`}
              >
                #{t.name} <span className="chip-count">({t.count})</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {items.length === 0 ? (
        <EmptyState
          icon="📷"
          title={searchQuery ? 'No matching screenshots' : (filterTag || filterCategory) ? 'No screenshots match filters' : 'No screenshots yet'}
          sub={searchQuery ? 'Try a different search term.' : (filterTag || filterCategory) ? 'Try clearing or changing your filters.' : 'Upload screenshots using the Upload tab.'}
        />
      ) : (
        <div className="gallery-grid" id="gallery-grid">
          {items.map(item => (
            <ScreenshotCard
              key={item.screenshot_id}
              item={item}
              onClick={onSelect}
              onDelete={onDelete}
              onTagClick={onTagClick}
            />
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

// ── Profile Tab ───────────────────────────────────────────────────────────────
function ProfileTab({ onSelect, onTagClick, onNavigateGallery, apiOnline, statsData, nonEmptyCategories, topCat, refreshTick }) {
  const [recentUploads, setRecentUploads] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    fetchScreenshots({ limit: 5, sort_by: 'newest' })
      .then(res => {
        if (isMounted) setRecentUploads(res.items || []);
      })
      .catch(() => {
        if (isMounted) setRecentUploads([]);
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });
    return () => { isMounted = false; };
  }, [refreshTick]);

  const formatUploadDate = (iso) => {
    if (!iso) return '';
    try {
      const d = new Date(iso);
      return d.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return String(iso).slice(0, 16);
    }
  };

  return (
    <div className="profile-container">
      {/* User Card */}
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

      {/* Metrics Row */}
      <div className="stat-cards-grid">
        <StatCard
          title="SCREENSHOTS"
          value={statsData?.total ?? '—'}
          sub="Total in library"
          color="#00d4ff"
          icon={<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>}
        />
        <StatCard
          title="CATEGORIES"
          value={nonEmptyCategories?.length || '—'}
          sub="Active taxonomy classes"
          color="#00f5d4"
          icon={<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg>}
        />
        <StatCard
          title="TOP CATEGORY"
          value={topCat ? catIcon(topCat.name) + ' ' + topCat.label : '—'}
          sub={topCat ? `${topCat.count} screenshots` : 'Populate library'}
          color={topCat ? catColor(topCat.name) : '#9ca3af'}
          icon={<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>}
        />
        <StatCard
          title="SYSTEM STATUS"
          value={apiOnline ? '● ONLINE' : '● OFFLINE'}
          sub={apiOnline ? 'All services operational' : 'Start backend'}
          color={apiOnline ? '#10b981' : '#ef4444'}
          icon={<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>}
        />
      </div>

      {/* 5 Recent Uploads Panel */}
      <div className="sci-panel profile-recent-panel">
        <div className="sci-panel-header">
          <div className="profile-recent-header-left">
            <h2 className="sci-panel-title">
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>
                <polyline points="17 6 23 6 23 12"/>
              </svg>
              5 RECENT UPLOADS
            </h2>
            <span className="profile-recent-sub">Latest screenshots ingested into I.R.I.S.</span>
          </div>
          <button className="panel-btn" onClick={onNavigateGallery}>
            VIEW ALL IN GALLERY →
          </button>
        </div>

        {loading ? (
          <div className="tab-loading" style={{ minHeight: '180px' }}>
            <Spinner size={32} />
            <span>Loading recent uploads…</span>
          </div>
        ) : recentUploads.length === 0 ? (
          <EmptyState
            icon="📂"
            title="No screenshots uploaded yet"
            sub="Upload screenshots via the Upload tab to populate this section."
          />
        ) : (
          <div className="profile-uploads-list">
            {recentUploads.map((item) => (
              <div
                key={item.screenshot_id}
                className="profile-upload-row"
                onClick={() => onSelect?.(item)}
              >
                {/* Thumbnail */}
                <div className="profile-upload-thumb">
                  <img
                    src={item.thumbnail_url || item.image_url}
                    alt={item.filename}
                    loading="lazy"
                    onError={e => { e.target.style.display = 'none'; }}
                  />
                  <div className="profile-upload-thumb-overlay">
                    <span>🔍</span>
                  </div>
                </div>

                {/* Main Information */}
                <div className="profile-upload-main">
                  <div className="profile-upload-title-row">
                    <span className="profile-upload-name" title={item.filename}>
                      {item.filename}
                    </span>
                    <span className="profile-upload-date">
                      {formatUploadDate(item.created_at)}
                    </span>
                  </div>

                  <div className="profile-upload-meta-row">
                    <span
                      className="cat-badge"
                      style={{
                        borderColor: `${catColor(item.category)}88`,
                        color: catColor(item.category),
                        background: `${catColor(item.category)}18`,
                      }}
                    >
                      {catIcon(item.category)} {item.category}
                    </span>

                    {item.importance != null && (
                      <span
                        className={`timeline-imp-badge imp-${item.importance}`}
                        title={`Importance: ${item.importance}/5`}
                      >
                        ★ {item.importance}
                      </span>
                    )}

                    {item.intent && (
                      <span className="profile-intent-pill">
                        🎯 {item.intent}
                      </span>
                    )}
                  </div>

                  {item.summary && (
                    <p className="profile-upload-summary">
                      {item.summary}
                    </p>
                  )}

                  {item.tags && item.tags.length > 0 && (
                    <div className="profile-upload-tags">
                      {item.tags.slice(0, 4).map((t, idx) => (
                        <span
                          key={idx}
                          className="tag-pill clickable"
                          onClick={(e) => {
                            e.stopPropagation();
                            onTagClick?.(t);
                          }}
                          title={`Filter by #${t}`}
                        >
                          #{t}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Inspect Action */}
                <div className="profile-upload-action">
                  <button
                    type="button"
                    className="profile-inspect-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelect?.(item);
                    }}
                    title="Inspect details"
                  >
                    <span>View Details</span>
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="9 18 15 12 9 6"/>
                    </svg>
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
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
  const [filterTag, setFilterTag]       = useState(null);
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
    setFilterTag(null);
    setActiveTab('gallery');
  };

  const handleTagClick = (tag) => {
    setFilterTag(tag);
    setActiveTab('gallery');
  };

  const handleClearTag = () => setFilterTag(null);
  const handleClearCategory = () => setFilterCat(null);

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
    { id: 'timeline',    label: 'Timeline',    icon: 'M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z' },
    { id: 'reminders',   label: 'Reminders',   icon: 'M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9' },
    { id: 'alerts',      label: 'Alerts',      icon: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z' },
    { id: 'duplicates',  label: 'Duplicates',  icon: 'M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z' },
    { id: 'upload',      label: 'Upload',      icon: 'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12' },
    { id: 'search',      label: 'Search',      icon: 'M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z' },
    { id: 'chat',        label: 'AI Chat',     icon: 'M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z' },
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
                <button onClick={() => { setActiveTab(nav.id); setFilterCat(null); setFilterTag(null); }}>
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
                    onTagClick={handleTagClick}
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

          {/* ── Timeline Tab ─────────────────────────────────────────────── */}
          {activeTab === 'timeline' && (
            <div className="sci-panel">
              <div className="sci-panel-header">
                <h2 className="sci-panel-title">
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <polyline points="12 6 12 12 16 14" />
                  </svg>
                  CHRONOLOGICAL LIFE TIMELINE
                </h2>
              </div>
              <TimelineTab onSelect={handleSelectItem} refreshTick={refreshTick} />
            </div>
          )}

          {/* ── Reminders Tab ────────────────────────────────────────────── */}
          {activeTab === 'reminders' && (
            <div className="sci-panel">
              <div className="sci-panel-header">
                <h2 className="sci-panel-title">
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                  </svg>
                  INTELLIGENT SCREENSHOT REMINDERS
                </h2>
              </div>
              <RemindersTab onSelect={handleSelectItem} refreshTick={refreshTick} />
            </div>
          )}

          {/* ── Alerts Tab ──────────────────────────────────────────────── */}
          {activeTab === 'alerts' && (
            <div className="sci-panel">
              <div className="sci-panel-header">
                <h2 className="sci-panel-title">
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  INTELLIGENT SYSTEM ALERTS
                </h2>
              </div>
              <AlertsTab onSelect={handleSelectItem} refreshTick={refreshTick} />
            </div>
          )}

          {/* ── Duplicates Tab ───────────────────────────────────────────── */}
          {activeTab === 'duplicates' && (
            <div className="sci-panel">
              <div className="sci-panel-header">
                <h2 className="sci-panel-title">
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/>
                  </svg>
                  DUPLICATE SCREENSHOT DETECTION
                </h2>
              </div>
              <DuplicatesTab onSelect={handleSelectItem} refreshTick={refreshTick} />
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
                  {filterCategory
                    ? `${catIcon(filterCategory)} ${filterCategory}`
                    : filterTag
                    ? `#${filterTag}`
                    : 'ALL SCREENSHOTS'}
                </h2>
                <div className="panel-controls">
                  {(filterCategory || filterTag) && (
                    <button
                      className="panel-btn"
                      onClick={() => {
                        setFilterCat(null);
                        setFilterTag(null);
                      }}
                    >
                      CLEAR ALL FILTERS
                    </button>
                  )}
                </div>
              </div>
              <GalleryContent
                filterCategory={filterCategory}
                filterTag={filterTag}
                searchQuery={searchQuery}
                onSelect={handleSelectItem}
                onDelete={handleDelete}
                onUploaded={handleUploaded}
                onTagClick={handleTagClick}
                onClearCategory={handleClearCategory}
                onClearTag={handleClearTag}
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
              <SearchTab onSelect={handleSelectItem} onTagClick={handleTagClick} />
            </div>
          )}

          {/* ── Chat Tab ─────────────────────────────────────────────────── */}
          {activeTab === 'chat' && (
            <div className="sci-panel chat-panel-full">
              <div className="sci-panel-header">
                <h2 className="sci-panel-title">
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>
                  </svg>
                  I.R.I.S. AI CONVERSATIONAL ASSISTANT
                </h2>
              </div>
              <ChatTab onSelect={handleSelectItem} onTagClick={handleTagClick} />
            </div>
          )}

          {/* ── Profile Tab ──────────────────────────────────────────────── */}
          {activeTab === 'profile' && (
            <ProfileTab
              onSelect={handleSelectItem}
              onTagClick={handleTagClick}
              onNavigateGallery={() => setActiveTab('gallery')}
              apiOnline={apiOnline}
              statsData={statsData}
              nonEmptyCategories={nonEmptyCategories}
              topCat={topCat}
              refreshTick={refreshTick}
            />
          )}
        </main>
      </div>

      {/* ── Detail Modal ───────────────────────────────────────────────────── */}
      {selectedItem && (
        <DetailModal
          item={selectedItem}
          onClose={handleCloseDetail}
          onTagClick={handleTagClick}
        />
      )}
    </div>
  );
};

export default Dashboard;